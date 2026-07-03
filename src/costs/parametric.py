"""Parametric cost model: commission + half-spread + square-root market impact.

For a trade of ``q`` shares at price ``p`` in a name with average daily volume
``ADV`` and quoted spread ``s``:

    cost($) = commission + (s/2)·|q|·p + impact_rate·|q|·p

with the empirically robust **square-root law** for impact (Almgren et al.):

    impact_rate = η · σ_daily · √(|q| / ADV)

This is concave in size *per share* (√) but convex in *total* cost
(``|q|·impact_rate ∝ |q|^{3/2}``) - the property that makes an optimizer prefer to
spread a target across names. A linear impact fallback is offered for a fully
convex-quadratic optimizer formulation. Defaults are calibrated to liquid US
equities and are all overridable.
"""

import math

from src.costs.base import CostModel, Trade, TradeCost


class ParametricCostModel(CostModel):
    """Commission + half-spread + (square-root or linear) market impact."""

    def __init__(
        self,
        commission_bps: float = 1.0,
        default_spread_bps: float = 5.0,
        impact_eta: float = 0.3,
        participation_cap: float = 0.10,
        annual_borrow_bps: float = 50.0,
        linear_impact: bool = False,
    ):
        self.commission_rate = commission_bps / 1e4
        self.default_spread = default_spread_bps / 1e4
        self.impact_eta = impact_eta
        self.participation_cap = participation_cap
        self.annual_borrow_rate = annual_borrow_bps / 1e4  # financing cost to hold a short
        self.linear_impact = linear_impact  # linear participation (convex-quadratic) vs √-law

    def cost(self, trade: Trade) -> TradeCost:
        notional = trade.notional
        if notional <= 0:
            return TradeCost(0.0, 0.0, 0.0, 0.0, capped=False)

        spread = trade.spread if trade.spread is not None else self.default_spread
        participation = trade.participation

        commission = self.commission_rate * notional
        spread_cost = (spread / 2.0) * notional
        if math.isinf(participation):
            impact_rate = 0.0  # no ADV info → can't size impact; commission+spread only
        elif self.linear_impact:
            impact_rate = self.impact_eta * trade.daily_vol * participation
        else:
            impact_rate = self.impact_eta * trade.daily_vol * math.sqrt(participation)
        impact_cost = impact_rate * notional

        return TradeCost(
            commission=commission,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            participation=participation,
            capped=participation > self.participation_cap,
        )

    def carry_cost(self, notional: float, is_short: bool, holding_years: float) -> float:
        """Financing cost of *holding* a position - borrow on shorts, accrued over time.

        Long positions have no borrow cost here (margin financing is a separate, later
        concern); a short pays ``borrow_rate · notional · holding_years``. This is what
        stops a short-heavy strategy from being silently flattered.
        """
        if not is_short or holding_years <= 0:
            return 0.0
        return self.annual_borrow_rate * abs(notional) * holding_years

    def turnover_cost_rate(self, spread: float = None) -> float:
        """Linear cost per unit of turnover notional - the optimizer's L1 cost term.

        The size-independent part (commission + half-spread); impact is omitted here
        because it's non-linear in trade size (it belongs in the √-law accounting).
        """
        s = self.default_spread if spread is None else spread
        return self.commission_rate + s / 2.0

    def impact_coefficient(self, daily_vol: float, adv_dollar: float, capital: float) -> float:
        """√-impact coefficient ``k`` (per unit capital) for the optimizer's conic term.

        The √-law impact rate is ``η·σ·√(participation)`` with ``participation =
        |q|/ADV``. Trading ``|Δw|`` of a ``capital``-dollar book is ``q = |Δw|·capital/p``
        shares, so ``participation = |Δw|·capital / (p·ADV) = |Δw|·capital / ADV$``. The
        impact cost as a *fraction of capital* is then ``η·σ·√(capital/ADV$)·|Δw|^{3/2}``,
        so ``k = η·σ·√(capital/ADV$)``. Zero when ADV is unavailable (mirrors
        :meth:`cost`, which charges no impact without volume) or in linear-impact mode
        (that fallback is quadratic-in-size, not conic, and isn't fed to the optimizer).
        """
        # `not (x > 0)` (rather than `x <= 0`) also rejects NaN, so a bad ADV/vol input
        # drops the impact term instead of poisoning the solve with a NaN coefficient.
        if self.linear_impact or not (adv_dollar > 0) or not (capital > 0) or not (daily_vol > 0):
            return 0.0
        return self.impact_eta * daily_vol * math.sqrt(capital / adv_dollar)
