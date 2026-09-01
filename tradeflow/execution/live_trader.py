"""Live order execution.

Translates strategy signals into concrete broker orders. It is the only thing
in live mode that mutates the account, and it speaks exclusively through the
:class:`Broker` interface - so swapping venues never touches this file.

Sizing and stop/take-profit distances come from the strategy's config; placement
and account/position reads go through the broker.

It is also where the kill switch bites: a halt recorded by
:mod:`tradeflow.execution.halt` refuses new entries here, and deliberately never
refuses an exit.

It also owns the strategy's **position book** - the strategy's belief about what it
holds. That belief is what :meth:`Strategy.validate_signal` consults to decide
whether an exit is legitimate, so a book that is never populated silently converts
every exit into a HOLD. Hydrating it is therefore part of executing, not a
convenience: see :meth:`LiveTrader.sync_strategy_book`.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

from tradeflow.brokers.base import AccountSnapshot, Broker, OrderSide, Position
from tradeflow.brokers.errors import AuthenticationError, BrokerError, DuplicateOrderError
from tradeflow.execution import decision as decisions
from tradeflow.execution.decision import Decision
from tradeflow.execution.halt import HaltState
from tradeflow.execution.order_id import client_order_id
from tradeflow.execution.sizing import PositionSizer, RiskBasedSizer
from tradeflow.strategies import signals
from tradeflow.strategies.base import Strategy
from tradeflow.utils.numeric import round_price, round_quantity

logger = logging.getLogger(__name__)

# Signal -> order side for new entries.
_ENTRY_SIDE = {signals.BUY: OrderSide.BUY, signals.SELL: OrderSide.SELL}

# Cache the market clock briefly so we don't query it on every streamed bar.
_MARKET_STATUS_TTL = 30.0


class LiveTrader:
    """Executes signals against a broker, sizing positions via a PositionSizer."""

    def __init__(
        self,
        broker: Broker,
        strategy: Strategy,
        sizer: Optional[PositionSizer] = None,
        allow_fractional: bool = False,
        respect_market_hours: bool = True,
        halt_state: Optional[HaltState] = None,
        capital: Optional[float] = None,
    ):
        self._broker = broker
        self._strategy = strategy
        #: Durable "stop trading" state. Consulted per actionable signal rather than
        #: cached: promptness is the entire point of a kill switch, and the read is a
        #: small local file on a path that only runs when a signal is not HOLD.
        self._halts = halt_state if halt_state is not None else HaltState()
        # Default to the strategy's own risk-based sizing; callers can inject a
        # portfolio-weight sizer to let the portfolio manager drive live sizing.
        self._sizer = sizer or RiskBasedSizer(strategy)
        self._allow_fractional = allow_fractional
        #: How much of the account this strategy may deploy, or ``None`` for all of it.
        #: A paper account arrives with whatever equity the venue gave it - typically far
        #: more than the capital a config was validated at - and sizing against that
        #: trades a different book from the one that was tested, which would invalidate
        #: the execution telemetry the run exists to gather.
        self._capital = capital
        self._respect_market_hours = respect_market_hours
        self._market_status_cache: Optional[tuple] = None  # (monotonic_ts, is_open)

    @property
    def broker(self) -> Broker:
        return self._broker

    # ------------------------------------------------------------------ #
    # The strategy's position book
    # ------------------------------------------------------------------ #
    def sync_strategy_book(self) -> int:
        """Replace the strategy's position book with what the broker actually holds.

        The strategy decides whether an exit is legitimate by looking itself up in
        its own book (:meth:`Strategy.validate_signal`). Nothing in live mode used to
        write that book, so it was permanently empty and every ``CLOSE_BUY`` /
        ``CLOSE_SELL`` was rewritten to ``HOLD`` before execution ever saw it -
        positions could be opened but never closed by the strategy, only by the
        broker-side bracket legs.

        Broker truth wins outright: the book is rebuilt, not merged. A belief that
        disagrees with the account is not evidence of anything except a stale belief.
        This is a *read* - it reports what is, and never places a corrective order.

        One ``list_positions`` call regardless of universe size, so it is safe to
        call on the trade clock. Returns the number of positions adopted.
        """
        positions = self._broker.list_positions() or []
        book = {}
        for position in positions:
            side = signals.BUY if position.is_long else signals.SELL
            stop_loss, take_profit = self._stop_levels(
                position.avg_entry_price, OrderSide.BUY if position.is_long else OrderSide.SELL
            )
            book[position.symbol] = {
                "side": side,
                "qty": position.qty,
                "entry_price": position.avg_entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        self._strategy.positions = book
        return len(book)

    def handle_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        bar_timestamp: Optional[datetime] = None,
    ) -> Decision:
        """Act on a single signal, and say what was decided and why.

        Returns a :class:`~tradeflow.execution.decision.Decision` rather than an
        order: "no order" has many causes, and collapsing them all into ``None`` made
        the only question worth asking afterwards - why nothing happened on that bar -
        answerable only from logs.

        ``bar_timestamp`` is what distinguishes one decision from a replay of the
        same one; see :mod:`tradeflow.execution.order_id`.
        """
        guards: List[str] = [decisions.HOLD]
        if signal == signals.HOLD:
            return decisions.decline(symbol, signal, "no signal", tuple(guards))

        if self._respect_market_hours:
            guards.append(decisions.MARKET_HOURS)
            if not self._market_open():
                logger.info("Market closed; ignoring %s signal for %s", signal, symbol)
                return decisions.decline(symbol, signal, "market closed", tuple(guards))

        position = self._broker.get_position(symbol)

        if signal in signals.EXIT_SIGNALS:
            return self._handle_exit(symbol, signal, position, guards)

        if signal in signals.ENTRY_SIGNALS:
            # Checked here rather than at the top so a halt can never block an exit.
            # A switch that also trapped the book would be one nobody dares pull.
            guards.append(decisions.HALT)
            halt = self._halts.active(type(self._strategy).__name__)
            if halt is not None:
                logger.warning("HALTED — refusing %s entry for %s: %s", signal, symbol, halt)
                return decisions.decline(symbol, signal, f"halted — {halt}", tuple(guards))
            return self._handle_entry(symbol, signal, price, position, bar_timestamp, guards)

        logger.warning("Ignoring unrecognized signal %r for %s", signal, symbol)
        return decisions.decline(symbol, signal, f"unrecognized signal {signal!r}", tuple(guards))

    # ------------------------------------------------------------------ #
    # Entries & exits
    # ------------------------------------------------------------------ #
    def _handle_entry(
        self,
        symbol: str,
        signal: str,
        price: float,
        position: Optional[Position],
        bar_timestamp: Optional[datetime] = None,
        guards: Optional[List[str]] = None,
    ) -> Decision:
        guards = list(guards or [])

        guards.append(decisions.EXISTING_POSITION)
        if position is not None:
            logger.info("Skipping %s entry for %s: position already open", signal, symbol)
            return decisions.decline(symbol, signal, "position already open", tuple(guards))

        # A cheap local shortcut, not the safety mechanism: it saves a pointless round
        # trip when an order is visibly pending, but it is a check-then-act race and
        # it forgets everything across a restart. The deterministic client order id
        # below is what actually prevents a double submission.
        guards.append(decisions.PENDING_ORDER)
        if self._broker.list_open_orders(symbol):
            logger.info("Skipping %s entry for %s: an order is already pending", signal, symbol)
            return decisions.decline(symbol, signal, "an order is already pending", tuple(guards))

        guards.append(decisions.ACCOUNT)
        try:
            account = self._broker.get_account()
        except BrokerError as exc:
            logger.error("Cannot size %s entry: %s (%s)", symbol, exc, type(exc).__name__)
            return decisions.decline(symbol, signal, f"account unreadable: {exc}", tuple(guards))
        if account is None:
            logger.error("Cannot size %s entry: account unavailable", symbol)
            return decisions.decline(symbol, signal, "account unavailable", tuple(guards))

        # Size against the capital this strategy was given, not the account's whole
        # balance. The affordability check below deliberately keeps using the *real*
        # account: capital is how much of it may be deployed, buying power is what the
        # venue will actually let through, and a cap must never make the second look
        # larger than it is.
        sizing_account = self._deployable(account)
        guards.append(decisions.SIZING)
        qty = round_quantity(
            self._sizer.size(symbol, price, sizing_account),
            allow_fractional=self._allow_fractional,
        )
        if qty <= 0:
            logger.warning("Computed position size <= 0 for %s; skipping", symbol)
            return decisions.decline(symbol, signal, "size rounds to zero", tuple(guards))

        guards.append(decisions.BUYING_POWER)
        cost = qty * price
        if cost > account.buying_power:
            logger.warning(
                "Insufficient buying power for %s: need $%.2f, have $%.2f", symbol, cost, account.buying_power
            )
            return decisions.decline(
                symbol,
                signal,
                f"insufficient buying power: need ${cost:.2f}, have ${account.buying_power:.2f}",
                tuple(guards),
            )

        guards.append(decisions.POSITION_LIMITS)
        # The capped account here too: `max_total_risk` and `max_gross_exposure` are
        # fractions *of equity*, and 5% of a paper account's balance is not 5% of the
        # capital the config was validated at.
        breach = self._limit_breach(symbol, qty, price, sizing_account)
        if breach is not None:
            logger.warning("Refusing %s entry for %s: %s", signal, symbol, breach)
            return decisions.decline(symbol, signal, breach, tuple(guards))

        side = _ENTRY_SIDE[signal]
        stop_loss, take_profit = self._stop_levels(price, side)
        logger.info(
            "Entering %s %s x%s @ ~$%.2f (stop $%.2f / target $%.2f)",
            side.value,
            symbol,
            qty,
            price,
            stop_loss,
            take_profit,
        )
        guards.append(decisions.BROKER)
        order_id = client_order_id(self._strategy, symbol, signal, bar_timestamp)
        try:
            order = self._broker.submit_bracket_order(
                symbol, qty, side, stop_loss, take_profit, client_order_id=order_id
            )
        except DuplicateOrderError:
            # Not a failure: the venue already holds this exact order. Resubmitting is
            # the one thing that must not happen, and the book should still reflect it.
            logger.info("Entry for %s already placed (order id %s); leaving it alone", symbol, order_id)
            return decisions.decline(symbol, signal, "already placed at the broker", tuple(guards))
        except BrokerError as exc:
            logger.error("Entry for %s refused: %s (%s)", symbol, exc, type(exc).__name__)
            return decisions.decline(symbol, signal, f"broker refused: {exc}", tuple(guards))
        if order is None:
            return decisions.decline(symbol, signal, "broker returned no order", tuple(guards))

        # Intent, not truth: the order is submitted, not filled. Recording it now is
        # what lets the strategy recognize its own position on the very next bar and
        # emit an exit for it; the next `sync_strategy_book` replaces this with
        # whatever the broker actually holds.
        self._strategy.positions[symbol] = {
            "side": signal,
            "qty": qty,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        return decisions.allow(symbol, signal, f"entered {qty} @ ~${price:.2f}", tuple(guards), order)

    def _handle_exit(
        self,
        symbol: str,
        signal: str,
        position: Optional[Position],
        guards: Optional[List[str]] = None,
    ) -> Decision:
        guards = list(guards or [])
        guards.append(decisions.EXISTING_POSITION)
        if position is None:
            return decisions.decline(symbol, signal, "no position to close", tuple(guards))

        guards.append(decisions.POSITION_MATCH)
        matches = (signal == signals.CLOSE_BUY and position.is_long) or (
            signal == signals.CLOSE_SELL and not position.is_long
        )
        if not matches:
            return decisions.decline(
                symbol, signal, f"{signal} does not match a {position.side} position", tuple(guards)
            )

        guards.append(decisions.BROKER)
        # Cancel any resting bracket legs first so closing the position can't leave an
        # orphaned stop/take order behind (which could oversell). A leg that cannot be
        # cancelled must not stop the close: an un-exited position is the worse of the
        # two outcomes, and the sweep will surface the orphan.
        for order in self._broker.list_open_orders(symbol):
            try:
                self._broker.cancel_order(order.id)
            except BrokerError as exc:
                logger.warning("Could not cancel resting order %s: %s", order.id, exc)
        logger.info("Closing %s position for %s", position.side, symbol)
        try:
            self._broker.close_position(symbol)
        except BrokerError as exc:
            # Deliberately kept in the book: the position is still open, so the
            # strategy must stay entitled to try again.
            logger.error("Could not close %s: %s (%s)", symbol, exc, type(exc).__name__)
            return decisions.decline(symbol, signal, f"broker could not close it: {exc}", tuple(guards))
        self._strategy.positions.pop(symbol, None)
        return decisions.allow(symbol, signal, f"closed {position.side} position", tuple(guards))

    def _market_open(self) -> bool:
        """Whether the market is open, cached for a short TTL.

        An unreadable clock used to mean "open". That is right for a transient blip -
        the bar stream only delivers during sessions anyway, so this is a secondary
        guard and freezing trading over one failed request would be worse - but it was
        also being applied to a failure that is *never* transient. Expired or revoked
        credentials produced "the market is open", and the system carried on placing
        orders on the strength of an answer nobody had given it.

        So the fallback now depends on the cause: an authentication failure fails
        closed, everything else stays permissive and says so.
        """
        now = time.monotonic()
        if self._market_status_cache and now - self._market_status_cache[0] < _MARKET_STATUS_TTL:
            return self._market_status_cache[1]

        try:
            status = self._broker.get_market_status()
            is_open = status.is_open if status is not None else True
        except AuthenticationError as exc:
            logger.error("Treating the market as closed: credentials rejected (%s)", exc)
            is_open = False
        except BrokerError as exc:
            logger.warning("Market clock unreadable (%s); assuming open", exc)
            is_open = True
        self._market_status_cache = (now, is_open)
        return is_open

    def _deployable(self, account: AccountSnapshot) -> AccountSnapshot:
        """The account as this strategy may use it, capped at its configured capital.

        Returns the account untouched when no capital was set, so an unconfigured run
        behaves exactly as before. Caps rather than replaces: a $8,000 config on an
        account holding $3,000 may deploy $3,000, never the number written in the file.
        """
        if not self._capital:
            return account
        return AccountSnapshot(
            cash=min(account.cash, self._capital),
            equity=min(account.equity, self._capital),
            buying_power=min(account.buying_power, self._capital),
            portfolio_value=min(account.portfolio_value, self._capital),
            trading_blocked=account.trading_blocked,
        )

    def _limit_breach(self, symbol: str, qty: float, price: float, account) -> Optional[str]:
        """Say which portfolio limit this entry would break, or ``None`` if it fits.

        Sizing answers "how big should this position be?" one symbol at a time and
        has no view of the book, so ``max_total_risk`` applied there caps a single
        position against the *whole* budget - N open positions could each consume all
        of it. The backtest enforces these across the book; live did not enforce them
        at all, so a config validated at five positions and a 5% risk budget could run
        unbounded against a margin account whose buying power is a multiple of equity.

        Counted from the strategy's own position book rather than a fresh broker read.
        That book is broker truth at start-up and at every reconciliation, and this
        path runs inside the bar loop - a ``list_positions`` per entry would put a
        broker round trip on a path several symbols can take on the same bar. Two
        consequences, both deliberate:

        * Exposure is measured at entry price, not marked. The trade clock has no
          price for a symbol it is not currently handling, and inventing one is worse
          than measuring at cost. A book that has run up is therefore carrying more
          market exposure than this counts.
        * Freshness is bounded by the reconciliation interval, not the bar. A position
          this trader did not open is invisible until the next sync.

        Reports and rejects; it never resizes an entry to fit.
        """
        limits = self._strategy.position_limits()
        max_positions = limits.get("max_positions")
        max_total_risk = limits.get("max_total_risk")
        max_gross_exposure = limits.get("max_gross_exposure")
        if not (max_positions or max_total_risk or max_gross_exposure):
            return None

        others = [p for held, p in self._strategy.positions.items() if held != symbol]

        if max_positions and len(others) + 1 > max_positions:
            return f"book is full: {len(others)} of {max_positions} positions already open"

        if not (max_total_risk or max_gross_exposure):
            return None

        equity = account.equity
        if equity <= 0:
            return f"cannot check portfolio limits against ${equity:,.2f} of equity"

        held_notional = sum(abs(p["qty"]) * p["entry_price"] for p in others)
        entry_notional = qty * price

        if max_total_risk:
            # Same accounting as the engine: notional x stop distance, one stop
            # fraction for the whole strategy.
            stop_pct = self._strategy.config["stop_loss"]
            open_risk = held_notional * stop_pct
            budget = equity * max_total_risk
            if open_risk + entry_notional * stop_pct > budget:
                return (
                    f"risk budget exhausted: ${open_risk + entry_notional * stop_pct:,.2f} "
                    f"of ${budget:,.2f} at a {stop_pct:.1%} stop"
                )

        if max_gross_exposure:
            cap = equity * max_gross_exposure
            if held_notional + entry_notional > cap:
                return f"gross exposure capped: ${held_notional + entry_notional:,.2f} of ${cap:,.2f}"

        return None

    def _stop_levels(self, entry_price: float, side: OrderSide) -> tuple[float, float]:
        """Compute (stop_loss, take_profit) prices from the strategy's config."""
        stop_pct = self._strategy.config["stop_loss"]
        take_pct = self._strategy.config["take_profit"]
        if side == OrderSide.BUY:
            return round_price(entry_price * (1 - stop_pct)), round_price(entry_price * (1 + take_pct))
        return round_price(entry_price * (1 + stop_pct)), round_price(entry_price * (1 - take_pct))
