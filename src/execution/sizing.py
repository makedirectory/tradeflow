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


class BetaSizer(PositionSizer):
    """Risk-based sizing scaled inversely by each symbol's beta.

    Higher beta (more volatile relative to the benchmark) -> smaller position, so
    risk is more even across names. Betas are precomputed per symbol (sizing has
    no data access); unknown symbols fall back to ``default_beta``. The effective
    beta is clamped to ``[min_abs_beta, max_abs_beta]`` to avoid blow-ups near
    zero and to bound leverage for tiny betas; the sign is ignored (a strongly
    negative beta is still volatile).

    It takes the strategy's normally-sized position (which already respects
    risk-per-trade and position limits) and divides it by the effective beta, so
    the scaling always applies - even when a position-limit cap is the binding
    constraint.
    """

    def __init__(
        self,
        strategy: Strategy,
        betas: Dict[str, float],
        default_beta: float = 1.0,
        min_abs_beta: float = 0.25,
        max_abs_beta: float = 4.0,
    ):
        self._strategy = strategy
        self._betas = betas
        self._default_beta = default_beta
        self._min_abs_beta = min_abs_beta
        self._max_abs_beta = max_abs_beta

    def size(self, symbol: str, price: float, account: AccountSnapshot) -> float:
        beta = self._betas.get(symbol, self._default_beta)
        effective = min(max(abs(beta), self._min_abs_beta), self._max_abs_beta)
        base_size = self._strategy.calculate_position_size(account.buying_power, price)
        return base_size / effective


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
