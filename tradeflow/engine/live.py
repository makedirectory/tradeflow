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
import time
from datetime import datetime, timedelta
from typing import List, Optional

from tradeflow.engine.barcheck import BarQualityFilter
from tradeflow.execution.ledger import PositionLedger
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.marketdata.base import BarEvent
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.marketdata.timeframe import DAY, HOUR, MINUTE, WEEK, Timeframe
from tradeflow.strategies import signals
from tradeflow.strategies.base import Strategy
from tradeflow.utils.timeutils import NEW_YORK

logger = logging.getLogger(__name__)

# Extra history fetched beyond the bare lookback, to cover non-trading gaps.
_WARMUP_BUFFER = 2

_UNIT_TO_TIMEDELTA = {
    MINUTE: lambda n: timedelta(minutes=n),
    HOUR: lambda n: timedelta(hours=n),
    DAY: lambda n: timedelta(days=n),
    WEEK: lambda n: timedelta(weeks=n),
}


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
        self._warm_up(symbols)

        broker = self.live_trader.broker
        tasks = [self.data_client.stream(symbols, self._on_bar)]
        if broker.supports_trade_updates():
            logger.info("Also streaming trade updates for fill/account feedback")
            tasks.append(broker.stream_trade_updates(self._on_trade_update))

        logger.info("Starting live stream for %d symbols", len(symbols))
        await asyncio.gather(*tasks)

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
                self.ledger.record_fill(
                    update.symbol,
                    getattr(update, "side", "buy"),
                    float(update.filled_qty),
                    order_id=update.order_id,
                    status=str(update.status),
                )
        except Exception:  # noqa: BLE001 - bookkeeping never breaks the order path
            logger.warning("Could not record a fill in the position ledger", exc_info=True)

    def _warm_up(self, symbols: List[str]) -> None:
        """Seed each symbol's rolling buffer so indicators are valid on bar one."""
        timeframe = Timeframe.parse(self.strategy.config["timeframe"])
        periods = self.strategy.config.get("required_lookback_periods", 50)
        start = self._lookback_start(timeframe, periods)
        end = datetime.now(NEW_YORK)

        history = self.data_client.get_bars(symbols, timeframe, start, end)
        for symbol, bars in history.items():
            if not bars.empty:
                self.strategy.warm_up(symbol, self.strategy.process_data(bars))
                logger.info("Warmed up %s with %d bars", symbol, len(bars))

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
            order = self.live_trader.handle_signal(event.symbol, signal, event.close)
            self._record_intent(event.symbol, signal, order)
        self._maybe_reconcile()

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
        if self.ledger is None or self.reconcile_every <= 0:
            return
        now = time.monotonic()
        # The first bar always sweeps. A process that just started is the case most
        # likely to have drifted — it may have missed fills while it was down — so
        # waiting a full interval to find out is backwards.
        if self._last_reconcile is not None and now - self._last_reconcile < self.reconcile_every:
            return
        self._last_reconcile = now
        try:
            self.ledger.reconcile(self.live_trader.broker)
        except Exception:  # noqa: BLE001
            logger.warning("Scheduled reconciliation failed", exc_info=True)

    @staticmethod
    def _lookback_start(timeframe: Timeframe, periods: int) -> datetime:
        units = periods * timeframe.amount * _WARMUP_BUFFER
        delta = _UNIT_TO_TIMEDELTA[timeframe.unit](units)
        return datetime.now(NEW_YORK) - delta
