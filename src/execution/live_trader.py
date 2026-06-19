"""Live order execution.

Translates strategy signals into concrete broker orders. It is the only thing
in live mode that mutates the account, and it speaks exclusively through the
:class:`Broker` interface - so swapping venues never touches this file.

Sizing and stop/take-profit distances come from the strategy's config; placement
and account/position reads go through the broker.
"""

import logging
from typing import Optional

from src.brokers.base import Broker, OrderResult, OrderSide, Position
from src.strategies import signals
from src.strategies.base import Strategy
from src.utils.numeric import round_price, round_quantity

logger = logging.getLogger(__name__)

# Signal -> order side for new entries.
_ENTRY_SIDE = {signals.BUY: OrderSide.BUY, signals.SELL: OrderSide.SELL}


class LiveTrader:
    """Executes signals against a broker, sizing positions via the strategy."""

    def __init__(self, broker: Broker, strategy: Strategy, allow_fractional: bool = False):
        self._broker = broker
        self._strategy = strategy
        self._allow_fractional = allow_fractional

    def handle_signal(self, symbol: str, signal: str, price: float) -> Optional[OrderResult]:
        """Act on a single signal. Returns the resulting order, if any."""
        if signal == signals.HOLD:
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

        account = self._broker.get_account()
        if account is None:
            logger.error("Cannot size %s entry: account unavailable", symbol)
            return None

        qty = round_quantity(
            self._strategy.calculate_position_size(account.buying_power, price),
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
        logger.info("Entering %s %s x%s @ ~$%.2f (stop $%.2f / target $%.2f)",
                    side.value, symbol, qty, price, stop_loss, take_profit)
        return self._broker.submit_bracket_order(symbol, qty, side, stop_loss, take_profit)

    def _handle_exit(self, symbol: str, signal: str, position: Optional[Position]) -> None:
        if position is None:
            return
        matches = (signal == signals.CLOSE_BUY and position.is_long) or (
            signal == signals.CLOSE_SELL and not position.is_long
        )
        if matches:
            logger.info("Closing %s position for %s", position.side, symbol)
            self._broker.close_position(symbol)

    def _stop_levels(self, entry_price: float, side: OrderSide) -> tuple[float, float]:
        """Compute (stop_loss, take_profit) prices from the strategy's config."""
        stop_pct = self._strategy.config["stop_loss"]
        take_pct = self._strategy.config["take_profit"]
        if side == OrderSide.BUY:
            return round_price(entry_price * (1 - stop_pct)), round_price(entry_price * (1 + take_pct))
        return round_price(entry_price * (1 + stop_pct)), round_price(entry_price * (1 - take_pct))
