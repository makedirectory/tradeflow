"""Pluggable position sizing for live execution.

`LiveTrader` asks a :class:`PositionSizer` "how many units of this symbol should I
buy at this price?" - it doesn't care *how* the answer is reached. Two strategies
ship:

* :class:`RiskBasedSizer` - the default; sizes each entry from the strategy's
  risk-per-trade / stop-loss config (position-by-position).
* :class:`PortfolioWeightSizer` - sizes to a precomputed target weight per symbol,
  e.g. from the OR-Tools :class:`~src.portfolio.allocator.PortfolioAllocator`
  (portfolio-level).

This keeps sizing policy out of `LiveTrader` and lets the portfolio manager drive
live sizing without the executor knowing anything about it.
"""

from abc import ABC, abstractmethod
from typing import Dict

from src.brokers.base import AccountSnapshot
from src.strategies.base import Strategy


class PositionSizer(ABC):
    """Decides the (pre-rounding) size of a new position."""

    @abstractmethod
    def size(self, symbol: str, price: float, account: AccountSnapshot) -> float:
        """Return the desired position size in units (shares)."""


class RiskBasedSizer(PositionSizer):
    """Size each entry from the strategy's risk/stop configuration."""

    def __init__(self, strategy: Strategy):
        self._strategy = strategy

    def size(self, symbol: str, price: float, account: AccountSnapshot) -> float:
        return self._strategy.calculate_position_size(account.buying_power, price)


class PortfolioWeightSizer(PositionSizer):
    """Size to a target portfolio weight per symbol (weight x equity / price).

    Symbols absent from the weight map get zero size, so the live universe is
    effectively the set the allocator chose to fund.
    """

    def __init__(self, weights: Dict[str, float]):
        self._weights = weights

    def size(self, symbol: str, price: float, account: AccountSnapshot) -> float:
        weight = self._weights.get(symbol, 0.0)
        if weight <= 0 or price <= 0:
            return 0.0
        return (weight * account.equity) / price
