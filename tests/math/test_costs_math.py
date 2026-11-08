"""Math contract: the parametric transaction-cost model and the cost stress.

Related existing tests: ``tests/test_costs.py``, ``tests/test_cost_spread.py``,
``tests/test_cost_aware.py``.

Defects this module guards:

* the configured spread was dead in the cost model (every trade priced at 5 bp whatever
  the model was built with), so a "3x" cost stress cost about 1.6x;
* a cost stress that did not scale every linear term by the same multiple.
"""

import math
from datetime import datetime

import pytest

from quantred.costs import ParametricCostModel, Trade
from quantred.marketdata.client import MarketDataClient
from quantred.services import analysis
from tests.fakes import FakeMarketData

# 1,000 shares at $50 = $50,000 notional; ADV 100,000 shares -> participation 1%
TRADE = Trade("AAA", 1_000, 50.0, adv=100_000.0, daily_vol=0.02)
NOTIONAL = 50_000.0


def test_cost_components_follow_the_square_root_law():
    model = ParametricCostModel(commission_bps=2.0, default_spread_bps=8.0, impact_eta=0.3)
    c = model.cost(TRADE)
    assert c.commission == pytest.approx(NOTIONAL * 2e-4, rel=1e-12)  # 10
    assert c.spread_cost == pytest.approx(NOTIONAL * 8e-4 / 2, rel=1e-12)  # 20
    # impact rate = 0.3 * 0.02 * sqrt(0.01) = 6e-4  ->  $30
    assert c.impact_cost == pytest.approx(30.0, rel=1e-12)
    assert c.total == pytest.approx(60.0, rel=1e-12)
    assert c.participation == pytest.approx(0.01) and c.capped is False


def test_impact_is_convex_in_total_size_but_concave_per_share():
    model = ParametricCostModel(commission_bps=0, default_spread_bps=0)
    small = model.cost(Trade("A", 1_000, 50.0, 100_000.0, 0.02))
    big = model.cost(Trade("A", 4_000, 50.0, 100_000.0, 0.02))
    assert big.impact_cost / small.impact_cost == pytest.approx(4**1.5, rel=1e-12)  # total ~ q^1.5
    assert (big.impact_cost / 4_000) / (small.impact_cost / 1_000) == pytest.approx(
        2.0, rel=1e-12
    )  # per share ~ sqrt(q)


def test_linear_impact_fallback_is_proportional_to_participation():
    model = ParametricCostModel(commission_bps=0, default_spread_bps=0, linear_impact=True)
    assert model.cost(TRADE).impact_cost == pytest.approx(0.3 * 0.02 * 0.01 * NOTIONAL, rel=1e-12)


def test_trade_direction_does_not_change_cost():
    model = ParametricCostModel()
    short = Trade("AAA", -1_000, 50.0, adv=100_000.0, daily_vol=0.02)
    assert model.cost(short).total == pytest.approx(model.cost(TRADE).total, rel=1e-12)


def test_participation_cap_flags_optimistic_fills_at_the_boundary():
    model = ParametricCostModel(participation_cap=0.10)
    assert model.cost(Trade("A", 10_000, 50.0, 100_000.0, 0.02)).capped is False  # exactly at the cap
    assert model.cost(Trade("A", 10_001, 50.0, 100_000.0, 0.02)).capped is True


def test_no_liquidity_information_charges_commission_and_spread_only():
    model = ParametricCostModel(commission_bps=1.0, default_spread_bps=5.0)
    c = model.cost(Trade("A", 100, 10.0, adv=0.0, daily_vol=0.05))
    assert c.impact_cost == 0.0 and c.total == pytest.approx(1_000 * (1e-4 + 5e-4 / 2), rel=1e-12)
    assert math.isinf(c.participation)


def test_zero_notional_trade_costs_nothing():
    assert ParametricCostModel().cost(Trade("A", 0, 10.0, 1e5, 0.02)).total == 0.0


def test_a_named_spread_overrides_and_an_unnamed_one_uses_the_models():
    """Guards the dead-default defect: with the Trade carrying its own 5 bp default, the
    model's configured spread was never consulted."""
    model = ParametricCostModel(commission_bps=0, default_spread_bps=40.0, impact_eta=0.0)
    unnamed = model.cost(Trade("A", 100, 10.0, 0.0, 0.0))
    named = model.cost(Trade("A", 100, 10.0, 0.0, 0.0, spread=0.0010))
    assert unnamed.spread_cost == pytest.approx(1_000 * 40e-4 / 2, rel=1e-12)
    assert named.spread_cost == pytest.approx(1_000 * 10e-4 / 2, rel=1e-12)
    assert Trade("A", 1, 1.0, 1.0, 0.0).spread is None


def test_every_linear_term_is_exactly_proportional_to_the_stress_multiple():
    base = dict(commission_bps=1.0, default_spread_bps=5.0, impact_eta=0.3)
    flat = Trade("A", 500, 20.0, adv=0.0, daily_vol=0.0)  # no impact: only the linear terms
    one = ParametricCostModel(**base).cost(flat).total
    for k in (2.0, 3.0, 10.0):
        scaled = ParametricCostModel(
            commission_bps=base["commission_bps"] * k,
            default_spread_bps=base["default_spread_bps"] * k,
            impact_eta=base["impact_eta"] * k,
        )
        assert scaled.cost(flat).total == pytest.approx(k * one, rel=1e-12)
    # and with impact on, impact scales with eta alone, linearly
    with_impact = ParametricCostModel(**base).cost(TRADE)
    tripled = ParametricCostModel(commission_bps=3.0, default_spread_bps=15.0, impact_eta=0.9).cost(TRADE)
    assert tripled.total == pytest.approx(3 * with_impact.total, rel=1e-12)


def test_borrow_carry_is_rate_times_notional_times_years_and_only_for_shorts():
    model = ParametricCostModel(annual_borrow_bps=75.0)
    assert model.carry_cost(40_000.0, True, 0.5) == pytest.approx(0.0075 * 40_000 * 0.5, rel=1e-12)
    assert model.carry_cost(-40_000.0, True, 0.5) == pytest.approx(0.0075 * 40_000 * 0.5, rel=1e-12)
    assert model.carry_cost(40_000.0, False, 0.5) == 0.0 and model.carry_cost(40_000.0, True, 0.0) == 0.0
    assert model.borrow_rate() == 0.0075 and model.borrow_rate(0.03) == 0.03


def test_linear_turnover_rate_and_alpha_haircut():
    model = ParametricCostModel(commission_bps=1.0, default_spread_bps=6.0)
    assert model.turnover_cost_rate() == pytest.approx(1e-4 + 3e-4, rel=1e-12)
    assert model.turnover_cost_rate(0.0010) == pytest.approx(1e-4 + 5e-4, rel=1e-12)
    flat = Trade("A", 100, 10.0, 0.0, 0.0)
    assert model.cost_rate(flat) == pytest.approx(4e-4, rel=1e-12)
    # a round trip held one month, annualised: 2 * 4bp * 12
    assert model.annual_cost_rate(flat, 1 / 12) == pytest.approx(2 * 4e-4 * 12, rel=1e-12)
    assert math.isinf(model.annual_cost_rate(flat, 0.0))
    assert model.cost_rate(Trade("A", 0, 10.0, 0.0, 0.0)) == 0.0


def test_cost_stress_service_scales_total_cost_with_the_multiple():
    """Run the service sweep with impact and borrow off: every term is linear in the
    multiple, so cost at k x is k x the base (to within the sizing drift of a poorer book).
    It rose ~1.6x at 3x while the spread stayed put."""
    symbols = ["S0", "S1", "S2", "S3"]
    client = MarketDataClient(FakeMarketData([*symbols, "SPY"], n=500, freq="1D"))
    report = analysis.run_cost_stress(
        client,
        "demo_trend",
        symbols,
        datetime(2024, 1, 2),
        datetime(2025, 3, 1),
        capital=100_000.0,
        impact_eta=0.0,
        borrow_bps=0.0,
        multiples=(1.0, 2.0, 5.0),
    )
    cost = {p["multiple"]: p["total_cost"] for p in report["points"]}
    assert cost[1.0] > 0
    assert cost[2.0] / cost[1.0] == pytest.approx(2.0, rel=0.03)
    assert cost[5.0] / cost[1.0] == pytest.approx(5.0, rel=0.05)
