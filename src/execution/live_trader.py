"""Live order execution.

Translates strategy signals into concrete broker orders. It is the only thing
in live mode that mutates the account, and it speaks exclusively through the
:class:`Broker` interface - so swapping venues never touches this file.

Sizing and stop/take-profit distances come from the strategy's config; placement
and account/position reads go through the broker.
"""

import logging
import time
from typing import Optional

from src.brokers.base import Broker, OrderResult, OrderSide, Position
from src.execution.sizing import PositionSizer, RiskBasedSizer
from src.strategies import signals
from src.strategies.base import Strategy
from src.utils.numeric import round_price, round_quantity

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
    ):
        self._broker = broker
        self._strategy = strategy
        # Default to the strategy's own risk-based sizing; callers can inject a
        # portfolio-weight sizer to let the portfolio manager drive live sizing.
        self._sizer = sizer or RiskBasedSizer(strategy)
        self._allow_fractional = allow_fractional
        self._respect_market_hours = respect_market_hours
        self._market_status_cache: Optional[tuple] = None  # (monotonic_ts, is_open)

    @property
    def broker(self) -> Broker:
        return self._broker

    def handle_signal(self, symbol: str, signal: str, price: float) -> Optional[OrderResult]:
        """Act on a single signal. Returns the resulting order, if any."""
        if signal == signals.HOLD:
            return None

        if self._respect_market_hours and not self._market_open():
            logger.info("Market closed; ignoring %s signal for %s", signal, symbol)
            return None

        position = self._broker.get_position(symbol)

        if signal in signals.EXIT_SIGNALS:
            self._handle_exit(symbol, signal, position)
            return None

        if signal in signals.ENTRY_SIGNALS:
            return self._handle_entry(symbol, signal, price, position)

        logger.warning("Ignoring unrecognised signal %r for %s", signal, symbol)
        return None

    # ------------------------------------------------------------------ #
    # Entries & exits
    # ------------------------------------------------------------------ #
    def _handle_entry(
        self, symbol: str, signal: str, price: float, position: Optional[Position]
    ) -> Optional[OrderResult]:
        if position is not None:
            logger.info("Skipping %s entry for %s: position already open", signal, symbol)
            return None

        # Guard against double-submitting between order placement and fill.
        if self._broker.list_open_orders(symbol):
            logger.info("Skipping %s entry for %s: an order is already pending", signal, symbol)
            return None

        account = self._broker.get_account()
        if account is None:
            logger.error("Cannot size %s entry: account unavailable", symbol)
            return None

        qty = round_quantity(
            self._sizer.size(symbol, price, account),
            allow_fractional=self._allow_fractional,
        )
        if qty <= 0:
            logger.warning("Computed position size <= 0 for %s; skipping", symbol)
            return None

        cost = qty * price
        if cost > account.buying_power:
            logger.warning(
                "Insufficient buying power for %s: need $%.2f, have $%.2f", symbol, cost, account.buying_power
            )
            return None

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
        return self._broker.submit_bracket_order(symbol, qty, side, stop_loss, take_profit)

    def _handle_exit(self, symbol: str, signal: str, position: Optional[Position]) -> None:
        if position is None:
            return
        matches = (signal == signals.CLOSE_BUY and position.is_long) or (
            signal == signals.CLOSE_SELL and not position.is_long
        )
        if matches:
            # Cancel any resting bracket legs first so closing the position can't
            # leave an orphaned stop/take order behind (which could oversell).
            for order in self._broker.list_open_orders(symbol):
                self._broker.cancel_order(order.id)
            logger.info("Closing %s position for %s", position.side, symbol)
            self._broker.close_position(symbol)

    def _market_open(self) -> bool:
        """Whether the market is open, cached for a short TTL.

        If the clock can't be determined, default to open (permissive): the live
        bar stream only delivers during sessions anyway, so this is a secondary
        guard, and a transient clock error shouldn't freeze trading.
        """
        now = time.monotonic()
        if self._market_status_cache and now - self._market_status_cache[0] < _MARKET_STATUS_TTL:
            return self._market_status_cache[1]

        status = self._broker.get_market_status()
        is_open = status.is_open if status is not None else True
        self._market_status_cache = (now, is_open)
        return is_open

    def _stop_levels(self, entry_price: float, side: OrderSide) -> tuple[float, float]:
        """Compute (stop_loss, take_profit) prices from the strategy's config."""
        stop_pct = self._strategy.config["stop_loss"]
        take_pct = self._strategy.config["take_profit"]
        if side == OrderSide.BUY:
            return round_price(entry_price * (1 - stop_pct)), round_price(entry_price * (1 + take_pct))
        return round_price(entry_price * (1 + stop_pct)), round_price(entry_price * (1 - take_pct))
