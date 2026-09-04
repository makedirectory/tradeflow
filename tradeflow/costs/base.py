"""Transaction-cost model core - what it costs to change a position.

Every Sharpe and equity curve the backtester produces is *gross* until a cost model
charges trading. Gross results are the most reliable way to fool yourself: the
strategies that look best in-sample are very often the highest-turnover ones, which
are exactly the ones costs destroy. Active management treats cost as first-class -
it caps the transfer coefficient in ``IR ≈ TC·IC·√BR``.

This module owns the shared types: :class:`Trade` (what is being traded, plus the
liquidity context), :class:`TradeCost` (the commission / spread / impact breakdown),
and the :class:`CostModel` interface. The concrete model lives in
:mod:`tradeflow.costs.parametric`.

**A pure pricing helper, and both clocks may use it.** It said "research-clock only:
the live path gets real fills from the broker" long after that stopped being true -
``execution.live_trader._cost_estimate`` imports it and calls ``cost()`` on the order
path, deliberately, so the modelled number in a live report is comparable with the one
the backtest was judged on. Nothing is broken by that: ``costs/`` is not one of the
packages the trade-clock rule names, and what lives here is arithmetic over its
arguments - no model, no optimizer, no database, no network. But a stale boundary
comment is how the next person justifies a real violation, so the claim is corrected
rather than left to be discovered.

What must stay true is the reason the boundary was drawn: what this computes is a
*prediction*, and a broker fee is an *observation*. The live path records them
separately and must never merge them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    """A trade to be priced, with the trailing liquidity context it's priced against."""

    symbol: str
    shares: float  # signed; cost is charged on the absolute size
    price: float
    adv: float  # trailing average daily volume (shares), as-of the trade
    daily_vol: float  # trailing daily return volatility (fraction), as-of the trade
    spread: float = 0.0005  # quoted spread (fraction, e.g. 0.0005 = 5 bps)

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.price

    @property
    def participation(self) -> float:
        """Fraction of a day's volume this trade demands (`|q| / ADV`)."""
        return abs(self.shares) / self.adv if self.adv > 0 else float("inf")


@dataclass
class TradeCost:
    """The decomposed dollar cost of a trade (commission + spread + impact)."""

    commission: float
    spread_cost: float
    impact_cost: float
    participation: float
    capped: bool  # participation exceeded the realistic cap → fill is optimistic

    @property
    def total(self) -> float:
        return self.commission + self.spread_cost + self.impact_cost


class CostModel(ABC):
    """Prices the cost of changing a position."""

    @abstractmethod
    def cost(self, trade: Trade) -> TradeCost:
        """Return the decomposed dollar cost of ``trade``."""

    def cost_rate(self, trade: Trade) -> float:
        """One-way cost as a fraction of traded notional."""
        notional = trade.notional
        return self.cost(trade).total / notional if notional > 0 else 0.0

    def turnover_cost_rate(self, spread: float = None) -> float:
        """Linear (size-independent) one-way cost per unit of traded notional.

        The size-independent part of a trade's cost — the optimizer's L1 turnover
        coefficient (the cost-aware objective). A model with no linear cost returns 0
        (the default);
        :class:`~tradeflow.costs.parametric.ParametricCostModel` returns commission + s/2.
        """
        return 0.0

    def impact_coefficient(self, daily_vol: float, adv_dollar: float, capital: float) -> float:
        """√-impact coefficient ``k`` for the optimizer's ``Σ kᵢ·|Δwᵢ|^{3/2}`` term.

        The realistic square-root impact is convex in size but not quadratic, so it
        enters the portfolio optimizer's objective as a conic term rather than the
        risk quadratic.
        ``k`` is *per unit of capital, per rebalance*: the impact cost as a fraction of
        capital of trading ``|Δwᵢ|`` (a fraction of the ``capital`` book) is
        ``k·|Δwᵢ|^{3/2}``. A model with no impact returns 0 (the default).
        """
        return 0.0

    def carry_cost(self, notional: float, is_short: bool, holding_years: float) -> float:
        """Financing cost of *holding* a position (borrow on shorts). Default: none."""
        return 0.0

    def borrow_rate(self, override: Optional[float] = None) -> float:
        """Annualized borrow rate (fraction of notional per year) charged on a short
        holding - the portfolio optimizer's per-name short-side cost tilt,
        threaded the same way :meth:`turnover_cost_rate` threads the per-name spread.
        ``override`` is a per-name rate from :class:`~tradeflow.portfolio.optimizer.CostInputs`
        (a locate-desk quote or a manual hard-to-borrow override); a model with no
        borrow cost returns 0 regardless of ``override``.
        """
        return 0.0

    def annual_cost_rate(self, trade: Trade, holding_period_years: float) -> float:
        """Round-trip cost amortized over the holding period - the alpha haircut rate.

        A name held for a month pays its round-trip cost ~12x a year; a +4%/yr alpha
        with a 2% round-trip held one month is deeply negative, not a +4% opportunity.
        """
        if holding_period_years <= 0:
            return float("inf")
        return 2.0 * self.cost_rate(trade) / holding_period_years
