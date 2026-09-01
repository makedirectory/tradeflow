"""Live trading engine - same pipeline as the backtest, but now the money is real
and the bugs are expensive.

Mirrors the backtest pipeline for real time:

    warm-up history (marketdata) -> stream bars -> signals (strategy)
        -> orders (execution)

The engine wires the layers and owns the bar->signal->order loop; it contains no
indicator math, no order-placement detail, and no vendor specifics.
"""

import asyncio
import logging
import math
import time
from datetime import datetime, timedelta
from typing import List, Optional

from tradeflow.engine.barcheck import BarQualityFilter
from tradeflow.execution.ledger import CUMULATIVE, PositionLedger
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.marketdata.base import BarEvent
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.marketdata.timeframe import Timeframe
from tradeflow.strategies import signals
from tradeflow.strategies.base import Strategy
from tradeflow.utils.timeutils import NEW_YORK

logger = logging.getLogger(__name__)

#: How long to wait for streams to close on interrupt before exiting regardless. A
#: shutdown that can hang is a shutdown people learn to send a second interrupt to.
SHUTDOWN_TIMEOUT = 5.0

# Fetch this multiple of the bare lookback, so a few missing bars still leave enough.
_WARMUP_BUFFER = 2

#: Five trading days per seven calendar days.
_CALENDAR_DAYS_PER_TRADING_WEEK = 7 / 5

#: Slack for market holidays, which the ratio above does not model. Cheap insurance:
#: extra history is discarded by the buffer, missing history is silent.
_HOLIDAY_PADDING_DAYS = 5


class BlindStartError(RuntimeError):
    """Raised when no symbol has warm-up history, so every indicator starts blind.

    A guard, not a repair: nothing here can invent the history, and starting anyway
    produces signals that look exactly like valid ones.
    """


class LiveEngine:
    """Streams live bars into a strategy and routes signals to execution."""

    def __init__(
        self,
        strategy: Strategy,
        data_client: MarketDataClient,
        live_trader: LiveTrader,
        *,
        bar_filter: Optional[BarQualityFilter] = None,
        ledger: Optional[PositionLedger] = None,
        reconcile_every: float = 300.0,
        allow_blind_start: bool = False,
    ):
        self.strategy = strategy
        self.data_client = data_client
        self.live_trader = live_trader
        #: Vetoes bars before the strategy sees them. Rejects, never repairs — a
        #: live path that "fixes" its inputs stops being the thing the backtest
        #: validated. Pass ``None`` to run unguarded (what this did before).
        self.bar_filter = bar_filter
        #: Records intent and observed fills so divergence from the broker's actual
        #: account state is detectable. Never authoritative, never remediating.
        self.ledger = ledger
        self.reconcile_every = reconcile_every
        #: Start even when warm-up returned nothing. Off by default: the failure it
        #: guards is silent from inside the loop.
        self._allow_blind_start = allow_blind_start
        #: ``None`` means "never swept", which is deliberately not the same as
        #: "swept at time zero". ``time.monotonic()`` counts from an arbitrary origin
        #: (boot, on Linux), so seeding this with 0.0 made the first sweep depend on
        #: how long the machine had been up — on a freshly booted host it was skipped
        #: entirely for a full interval, which is exactly when reconciling matters
        #: most.
        self._last_reconcile: Optional[float] = None

    async def start(self, symbols: List[str]) -> None:
        """Warm up indicators with history, then stream live bars until canceled.

        When the broker supports it, the account/trade-update stream runs
        concurrently with the market-data stream so fills are logged.
        """
        self.strategy.initialize()
        warmed, _ = self._warm_up(symbols)
        if not warmed and symbols and not self._allow_blind_start:
            # Every indicator would start from nothing. The stream still connects and
            # bars still arrive, so the run looks healthy while every signal it emits
            # is computed from history it never had - indistinguishable, from inside
            # the loop, from a strategy that simply is not triggering.
            raise BlindStartError(
                f"Warm-up returned no history for any of the {len(symbols)} symbols, so "
                f"every indicator would start blind.\n"
                f"  The usual cause is a market-data feed the account is not entitled to: "
                f"historical requests resolve to the full consolidated tape by default, "
                f"while the live stream defaults to a single venue, so an unentitled key "
                f"warms up on nothing and streams normally.\n"
                f"  Pin both halves to a feed the account can read (--feed iex), or pass "
                f"--allow-blind-start to trade without history anyway."
            )
        self._cold_start()

        broker = self.live_trader.broker
        tasks = [self.data_client.stream(symbols, self._on_bar)]
        if broker.supports_trade_updates():
            logger.info("Also streaming trade updates for fill/account feedback")
            tasks.append(broker.stream_trade_updates(self._on_trade_update))

        logger.info("Starting live stream for %d symbols", len(symbols))
        running = [asyncio.ensure_future(task) for task in tasks]
        try:
            # `wait`, not `gather`. On cancellation `gather` propagates to its children
            # and then waits for every one of them to finish unwinding, with no bound -
            # so a stream slow to close its socket held the process open until a second
            # Ctrl-C, and the cleanup below was never even reached. `wait` hands
            # cancellation straight back, leaving the teardown to the bounded wait.
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                task.result()  # re-raise the first stream failure, as gather did
        except asyncio.CancelledError:
            # Ctrl-C cancels this coroutine, not the streams it started. Leaving them
            # to be garbage-collected is what produced the websocket teardown noise:
            # cancel each one and let it finish unwinding before returning.
            logger.info("Shutting down live streams.")
            raise
        finally:
            for task in running:
                task.cancel()
            # Bounded. A stream whose socket teardown blocks used to hold the process
            # open until a second Ctrl-C, which trains people to kill it twice - and
            # the second one arrives during cleanup, when it can interrupt anything.
            #
            # `asyncio.wait`, not `wait_for`: on timeout `wait_for` cancels the thing
            # it was waiting on and then *awaits* it, so a teardown that is slow to
            # answer cancellation hangs the very call meant to bound it. `wait` leaves
            # stragglers alone and returns, which is the whole point here.
            _, pending = await asyncio.wait(running, timeout=SHUTDOWN_TIMEOUT)
            if pending:
                logger.warning(
                    "%d stream(s) did not close within %.0fs; exiting anyway. Nothing "
                    "was sent to the broker during shutdown, and open positions are "
                    "untouched.",
                    len(pending),
                    SHUTDOWN_TIMEOUT,
                )

    def _on_trade_update(self, update) -> None:
        """Log account/order events (fills, cancels, rejects), and record fills.

        This is the only place the ledger learns what actually happened, as opposed
        to what was intended — which is precisely the gap it exists to close.
        """
        logger.info(
            "Trade update: %s %s (order %s, status %s, filled %s)",
            update.event,
            update.symbol,
            update.order_id,
            update.status,
            update.filled_qty,
        )
        if self.ledger is None:
            return
        try:
            if str(update.event).lower() in {"fill", "partial_fill"} and update.filled_qty:
                side = update.side
                if not side:
                    # Never default it. A missing side used to resolve to "buy", which
                    # recorded every short as a long and made the ledger disagree with
                    # the broker by twice the position.
                    logger.error(
                        "Trade update for %s (order %s) carried no side; not recording "
                        "it, because guessing one would put the wrong sign in the ledger",
                        update.symbol,
                        update.order_id,
                    )
                    return
                self.ledger.record_fill(
                    update.symbol,
                    side,
                    float(update.filled_qty),
                    order_id=update.order_id,
                    status=str(update.status),
                    basis=CUMULATIVE,
                )
        except Exception:  # noqa: BLE001 - bookkeeping never breaks the order path
            logger.warning("Could not record a fill in the position ledger", exc_info=True)

    def _cold_start(self) -> None:
        """Teach the strategy what it already holds, before the first bar arrives.

        Warm-up seeds *indicators* from history; it says nothing about the book. A
        process that restarts while holding a position would otherwise believe it was
        flat — and a strategy that believes it is flat cannot emit an exit, because
        its own signal validation rejects one. The position would then be closed only
        by its broker-side bracket legs, if it had any.

        Unconditional, and not governed by ``reconcile_every``: that setting paces the
        ledger sweep, whereas an unhydrated book is a correctness bug rather than a
        cadence choice. Failure here is logged, not fatal — starting flat is wrong but
        recoverable on the next sweep, while refusing to start is not obviously better.
        """
        try:
            adopted = self.live_trader.sync_strategy_book()
        except Exception:  # noqa: BLE001 - bookkeeping never breaks the order path
            logger.warning("Could not read the broker's positions at start-up", exc_info=True)
            return
        # Count the sweep: it just happened, so the first bar need not repeat it.
        self._last_reconcile = time.monotonic()
        if adopted:
            logger.info("Resuming with %d open position(s) adopted from the broker", adopted)

    def warm_up_coverage(self, symbols: List[str]) -> tuple:
        """Run the real warm-up and report ``(with history, fully warmed, asked)``.

        Two counts, not one: a symbol can have bars and still have too few for the
        indicators to be valid. Reporting only presence would let a short warm-up read
        as a pass, and "has history" is not the question a preflight is asked.

        Deliberately the same call the live path makes, not a lighter probe: a
        preflight that fetches differently from the run it precedes confirms nothing
        about that run. Placing no orders, it is safe to call and then exit.
        """
        warmed, sufficient = self._warm_up(symbols)
        return warmed, sufficient, len(symbols)

    def _warm_up(self, symbols: List[str]) -> tuple:
        """Seed each symbol's rolling buffer so indicators are valid on bar one.

        Returns ``(symbols with any history, symbols with the full lookback)``. The
        caller refuses on the first and reports both.
        """
        timeframe = Timeframe.parse(self.strategy.config["timeframe"])
        periods = self.strategy.config.get("required_lookback_periods", 50)
        start = self._lookback_start(timeframe, periods)
        end = datetime.now(NEW_YORK)

        history = self.data_client.get_bars(symbols, timeframe, start, end)
        warmed = sufficient = 0
        for symbol in symbols:
            bars = history.get(symbol)
            # A short warm-up is the failure worth shouting about: the strategy runs
            # anyway, on indicators computed from too little history, and produces
            # confident-looking signals that the backtest never validated. Nothing
            # else in the loop can tell that apart from a quiet market.
            if bars is None or bars.empty:
                logger.error("No warm-up history for %s — its indicators start blind", symbol)
                continue
            warmed += 1
            if len(bars) < periods:
                logger.warning(
                    "Warmed up %s with only %d of the %d bars its indicators need",
                    symbol,
                    len(bars),
                    periods,
                )
            else:
                sufficient += 1
                logger.info("Warmed up %s with %d bars", symbol, len(bars))
            self.strategy.warm_up(symbol, self.strategy.process_data(bars))
        logger.info(
            "Warm-up covered %d of %d symbols; %d have the full %d-bar lookback",
            warmed,
            len(symbols),
            sufficient,
            periods,
        )
        return warmed, sufficient

    def _on_bar(self, event: BarEvent) -> None:
        """Per-bar callback: validate, update the strategy, act on any signal."""
        bar = {
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        }
        if self.bar_filter is not None:
            verdict = self.bar_filter.check(event.symbol, bar, event.timestamp)
            if not verdict.accepted:
                # Skipped, not repaired: the strategy simply never sees this bar, so
                # its state stays exactly what a backtest over clean data would give.
                return

        signal = self.strategy.process_bar(event.symbol, bar, event.timestamp)
        if signal and signal != signals.HOLD:
            logger.info("Signal %s for %s @ $%.4f", signal, event.symbol, event.close)
            decision = self.live_trader.handle_signal(
                event.symbol, signal, event.close, bar_timestamp=event.timestamp
            )
            if not decision:
                logger.info("%s", decision)
            self._record_decision(decision)
            self._record_intent(event.symbol, signal, decision.order)
        self._maybe_reconcile()

    def _record_decision(self, decision) -> None:
        """Record why execution acted or declined.

        Without this, "nothing happened on that bar" is answerable only from logs,
        and only while they still exist. The ledger is already the append-only record
        of what the live path did, so a declined signal belongs in it beside the
        orders it did place.
        """
        if self.ledger is None:
            return
        try:
            self.ledger.record_decision(decision)
        except Exception:  # noqa: BLE001 - bookkeeping never breaks the order path
            logger.warning("Could not record the execution decision", exc_info=True)

    def _record_intent(self, symbol: str, signal: str, order) -> None:
        """Note what we asked for, so a fill that never arrives is detectable."""
        if self.ledger is None:
            return
        try:
            if order is not None:
                self.ledger.record_intent(symbol, order.side.value, order.qty, order_id=order.id)
            elif signal in signals.EXIT_SIGNALS:
                self.ledger.record_close(symbol)
        except Exception:  # noqa: BLE001 - bookkeeping never breaks the order path
            logger.warning("Could not record intent in the position ledger", exc_info=True)

    def _maybe_reconcile(self) -> None:
        """Sweep on a timer, from inside the loop.

        Bounded by construction — one ``list_positions`` call per sweep, never one
        per symbol — because this runs on the trade clock and must not make bar
        processing depend on the size of the universe.
        """
        if self.reconcile_every <= 0:
            return
        now = time.monotonic()
        # The first bar always sweeps. A process that just started is the case most
        # likely to have drifted — it may have missed fills while it was down — so
        # waiting a full interval to find out is backwards.
        if self._last_reconcile is not None and now - self._last_reconcile < self.reconcile_every:
            return
        self._last_reconcile = now
        # Re-read the book first: a bracket leg that filled, or a position closed by
        # hand in the broker's UI, changes what the strategy is entitled to exit.
        try:
            self.live_trader.sync_strategy_book()
        except Exception:  # noqa: BLE001 - bookkeeping never breaks the order path
            logger.warning("Could not refresh the strategy's position book", exc_info=True)
        if self.ledger is None:
            return
        try:
            self.ledger.reconcile(self.live_trader.broker)
        except Exception:  # noqa: BLE001
            logger.warning("Scheduled reconciliation failed", exc_info=True)

    @staticmethod
    def _lookback_start(timeframe: Timeframe, periods: int) -> datetime:
        """How far back to fetch so warm-up actually yields ``periods`` bars.

        This used to convert bars to wall-clock time directly — 50 one-minute bars
        became 100 minutes ago — which silently treats the overnight gap, the weekend,
        and every holiday as tradeable. At 09:35 on a Monday that window reached back
        to 07:55 the same morning and returned five bars for a fifty-bar indicator,
        leaving the strategy warmed up with an eighth of the history it asked for and
        nothing to say so. Daily bars under-fetched too, just less visibly: 100
        calendar days is about 70 sessions.

        So the conversion goes through sessions. Over-fetching is deliberately
        cheap — the buffer keeps only its tail — while under-fetching is invisible,
        so the estimate is padded for holidays rather than made exact. That is also
        why no market calendar is pulled in: precision here buys nothing that a few
        spare days do not.
        """
        bars_needed = max(periods, 1) * _WARMUP_BUFFER
        sessions = math.ceil(bars_needed / timeframe.bars_per_trading_day())
        calendar_days = math.ceil(sessions * _CALENDAR_DAYS_PER_TRADING_WEEK) + _HOLIDAY_PADDING_DAYS
        return datetime.now(NEW_YORK) - timedelta(days=calendar_days)
