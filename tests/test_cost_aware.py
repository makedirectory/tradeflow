"""Tests for cost-aware portfolio construction (spec 016).

Cost is inside the objective: ``αᵀw − λ·wᵀΣw − Σ cᵢ|Δwᵢ| − Σ kᵢ|Δwᵢ|^{3/2}``. These
tests cover the spec §6 checklist — name-specific tilt, the emergent no-trade band,
w₀ sensitivity, √-impact super-linearity, the linear↔conic gap, and the zero-cost
reduction to spec 008 — plus the proximal-operator primitives, a KKT optimality
certificate, the cost-model coefficients, and the end-to-end service integration.

Offline and deterministic.
"""

from datetime import datetime

import numpy as np
import pytest

from src.alphas.base import Alpha
from src.costs import ParametricCostModel
from src.costs.base import CostModel, Trade
from src.marketdata.client import MarketDataClient
from src.portfolio.optimizer import CostInputs, MeanVarianceOptimizer
from src.risk.base import RiskMatrix
from tests.fakes import DictMarketData, make_ohlcv

AS_OF = datetime(2024, 6, 1)
SYMS = ["A", "B", "C", "D"]
_L = np.array([[0.20, 0, 0, 0], [0.05, 0.18, 0, 0], [0.03, 0.04, 0.22, 0], [0.01, 0.02, 0.03, 0.16]])
SIGMA = _L @ _L.T
ALPHA = np.array([0.06, 0.02, -0.01, 0.04])
LAM = 2.0
H = 1.0 / 12.0


def _alphas(bump: float = 0.0) -> list:
    return [Alpha(s, float(ALPHA[i]) + bump, AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(SYMS)]


def _risk() -> RiskMatrix:
    return RiskMatrix(SYMS, SIGMA)


def _vec(result) -> np.ndarray:
    return np.array([result.weights.get(s, 0.0) for s in SYMS])


def _inputs(spread, adv_dollar=1e12, daily_vol=0.02) -> CostInputs:
    """A CostInputs with a uniform (or per-name dict) spread and deep default liquidity."""
    sp = spread if isinstance(spread, dict) else {s: spread for s in SYMS}
    return CostInputs(
        spread=sp,
        adv_dollar={s: adv_dollar for s in SYMS},
        daily_vol={s: daily_vol for s in SYMS},
    )


# --- closed-form / zero-cost reduction (spec §6 "closed-form sanity") --------
def test_zero_cost_reduces_to_cost_blind_solve():
    free = ParametricCostModel(commission_bps=0.0, default_spread_bps=0.0)
    blind = MeanVarianceOptimizer(max_weight=0.5).optimize(_alphas(), _risk(), risk_aversion=LAM)
    zero = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), risk_aversion=LAM, cost_model=free, cost_inputs=_inputs(0.0), holding_period_years=H
    )
    assert np.allclose(_vec(zero), _vec(blind), atol=1e-9)
    # Unconstrained optimum still the spec-008 closed form, untouched by the (zero) cost.
    expected = (np.linalg.inv(SIGMA) @ ALPHA) / (2 * LAM)
    got = np.array([blind.unconstrained_weights[s] for s in SYMS])
    assert np.allclose(got, expected)


# --- name-specific cost changes weights (spec §6 test 1) ---------------------
def test_uniform_cost_from_cash_is_a_noop():
    # Long-only from cash with a uniform per-name cost: Σcᵢ|wᵢ| = c·Σwᵢ = c is constant,
    # so the optimum is unchanged — the baseline the name-specific case must beat.
    model = ParametricCostModel()
    blind = MeanVarianceOptimizer(max_weight=0.5).optimize(_alphas(), _risk(), risk_aversion=LAM)
    uniform = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=_inputs(0.001), holding_period_years=H
    )
    assert np.allclose(_vec(uniform), _vec(blind), atol=1e-6)


def test_name_specific_cost_tilts_away_from_expensive_names():
    model = ParametricCostModel()
    blind = MeanVarianceOptimizer(max_weight=0.5).optimize(_alphas(), _risk(), risk_aversion=LAM)
    # Make the top-alpha name (A) expensive; the cost-aware optimum must down-weight it.
    expensive = {"A": 0.02, "B": 0.0005, "C": 0.0005, "D": 0.0005}
    tilted = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model,
        cost_inputs=_inputs(expensive), holding_period_years=H,
    )
    assert _vec(tilted)[0] < _vec(blind)[0] - 1e-6  # A shrinks
    assert abs(sum(tilted.weights.values()) - 1.0) < 1e-6  # still fully invested


# --- emergent no-trade band (spec §6 test 2) ---------------------------------
def test_emergent_no_trade_band_suppresses_subthreshold_churn():
    model = ParametricCostModel()
    opt = MeanVarianceOptimizer(max_weight=0.5)  # no_trade_band=0: the band must EMERGE from cost
    base = opt.optimize(_alphas(), _risk(), risk_aversion=LAM)
    # Nudge one name's alpha by +0.02 — below its cost band (a ~100bps spread over a 1/12
    # yr hold gives a ~6%/yr band half-width), but enough that a cost-blind solve chases it.
    nudged = [Alpha(s, float(ALPHA[i]) + (0.02 if s == "D" else 0.0), AS_OF, 0.2, 0.05, 0.0)
              for i, s in enumerate(SYMS)]
    rebal = opt.optimize(
        nudged, _risk(), risk_aversion=LAM, current_weights=base.weights,
        cost_model=model, cost_inputs=_inputs(0.01), holding_period_years=H,
    )
    assert rebal.diagnostics["turnover"] < 1e-9  # machine epsilon: no meaningful trade
    assert rebal.diagnostics["names_traded"] == 0
    # Without the cost, the same nudge chases the alpha — the band is doing the work.
    chase = opt.optimize(nudged, _risk(), risk_aversion=LAM, current_weights=base.weights)
    assert chase.diagnostics["turnover"] > 1e-3


def test_no_trade_band_half_width_is_the_one_way_cost():
    # The band edge is the soft-threshold in _prox_step: no trade iff |m| ≤ thr, where
    # thr = one-way cost cᵢ. Full band width (buy edge − sell edge) = 2cᵢ = round trip.
    m = np.array([-0.05, -0.011, -0.009, 0.009, 0.011, 0.05])
    thr = np.full_like(m, 0.01)
    d = MeanVarianceOptimizer._prox_step(m, thr, np.zeros_like(m))
    assert list(np.abs(d) > 0) == [True, True, False, False, True, True]  # zero strictly inside ±thr

    model = ParametricCostModel(commission_bps=1.0, default_spread_bps=5.0)
    c_one_way = model.turnover_cost_rate() / H  # cᵢ, annualised one-way
    c_lin, _ = MeanVarianceOptimizer._cost_coefficients(model, _inputs(model.default_spread), None, H, SYMS)
    assert np.allclose(c_lin, c_one_way)
    # Full band width = 2·cᵢ equals the annualised round-trip rate (impact→0 for a tiny trade).
    tiny = Trade("A", shares=1e-6, price=100.0, adv=1e12, daily_vol=0.02, spread=model.default_spread)
    assert 2 * c_one_way == pytest.approx(model.annual_cost_rate(tiny, H), rel=1e-6)


# --- w0 sensitivity (spec §6 test 3) -----------------------------------------
def test_turnover_depends_on_current_weights():
    model = ParametricCostModel()
    opt = MeanVarianceOptimizer(max_weight=0.5)
    kw = dict(cost_model=model, cost_inputs=_inputs(0.0005), holding_period_years=H)
    target = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, **kw)
    from_target = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, current_weights=target.weights, **kw)
    from_cash = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, current_weights={}, **kw)
    assert from_target.diagnostics["turnover"] < 1e-6  # already optimal → no trade
    assert from_cash.diagnostics["turnover"] > 0.5  # from cash → fully invest


# --- √-impact super-linearity (spec §6 test 4) -------------------------------
def test_impact_penalty_is_superlinear_in_trade_size():
    # The penalty kᵢ·|Δw|^{3/2} under the optimiser's *own* coefficient: doubling the
    # trade more than doubles it (×2^1.5). Uses the real _cost_coefficients so an
    # annualisation/formula bug in kᵢ would surface here, not just literal arithmetic.
    model = ParametricCostModel()
    _, k_imp = MeanVarianceOptimizer._cost_coefficients(
        model, _inputs(0.0005, adv_dollar=1e8), 1e7, H, SYMS
    )
    assert np.all(k_imp > 0)
    p1 = k_imp[0] * 0.1**1.5
    p2 = k_imp[0] * 0.2**1.5
    assert p2 / p1 == pytest.approx(2**1.5)


def test_sqrt_impact_spreads_size_away_from_illiquid_names():
    model = ParametricCostModel()
    # A (top alpha) is illiquid; the rest are deep. Impact should pull size off A.
    adv = {"A": 5e6, "B": 5e11, "C": 5e11, "D": 5e11}
    ci = CostInputs(spread={s: 0.0005 for s in SYMS}, adv_dollar=adv, daily_vol={s: 0.03 for s in SYMS})
    opt = MeanVarianceOptimizer(max_weight=0.5)
    linear = opt.optimize(  # capital=None → linear only, no √-impact
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, holding_period_years=H
    )
    conic = opt.optimize(  # capital set → √-impact active
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, capital=5e7, holding_period_years=H
    )
    assert conic.diagnostics["impact_cost"] > 0
    assert _vec(conic)[0] < _vec(linear)[0] - 1e-4  # A down-weighted by impact
    assert abs(sum(conic.weights.values()) - 1.0) < 1e-6


# --- linear ↔ conic gap (spec §6 test 5) -------------------------------------
def test_linear_and_conic_agree_when_impact_is_negligible_and_gap_is_reported():
    model = ParametricCostModel()
    ci = _inputs(0.0005, adv_dollar=1e13)  # very deep books
    opt = MeanVarianceOptimizer(max_weight=0.5)
    linear = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, holding_period_years=H)
    conic = opt.optimize(
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, capital=1e5, holding_period_years=H
    )
    # Deep liquidity + small capital → the √-impact term is tiny, so the two solutions
    # coincide within tolerance; the impact charge is surfaced, not hidden.
    assert np.max(np.abs(_vec(linear) - _vec(conic))) < 1e-3
    assert "impact_cost" in conic.diagnostics
    assert conic.diagnostics["impact_cost"] < 1e-4


def test_conic_gap_grows_with_capital():
    model = ParametricCostModel()
    ci = _inputs(0.0005, adv_dollar=1e8)
    opt = MeanVarianceOptimizer(max_weight=0.5)
    small = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, capital=1e5, holding_period_years=H)
    large = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, capital=1e9, holding_period_years=H)
    assert large.diagnostics["impact_cost"] > small.diagnostics["impact_cost"]


# --- KKT optimality certificate ----------------------------------------------
def test_no_feasible_perturbation_beats_the_cost_aware_optimum():
    model = ParametricCostModel()
    ci = _inputs({"A": 0.02, "B": 0.0005, "C": 0.0005, "D": 0.0005})
    opt = MeanVarianceOptimizer(max_weight=0.5)
    res = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, holding_period_years=H)
    w = _vec(res)
    c_lin, k_imp = MeanVarianceOptimizer._cost_coefficients(model, ci, None, H, SYMS)
    w0 = np.zeros(4)

    def util(x):
        dw = x - w0
        return ALPHA @ x - LAM * (x @ SIGMA @ x) - np.sum(c_lin * np.abs(dw)) - np.sum(k_imp * np.abs(dw) ** 1.5)

    u_opt = util(w)
    rng = np.random.default_rng(0)
    beats = 0
    for _ in range(5000):
        i, j = rng.integers(0, 4, 2)
        eps = rng.uniform(0, 0.02)
        wp = w.copy()
        wp[i] += eps
        wp[j] -= eps  # budget-preserving swap
        if wp[i] <= 0.5 + 1e-12 and wp[j] >= -1e-12 and util(wp) > u_opt + 1e-8:
            beats += 1
    assert beats == 0


# --- proximal-operator primitive ---------------------------------------------
def test_prox_step_matches_brute_force_minimizer():
    grid = np.linspace(-0.6, 0.6, 24001)
    for m, thr, kthr in [(0.3, 0.05, 0.0), (0.3, 0.05, 0.4), (-0.25, 0.02, 0.7), (0.01, 0.05, 0.4)]:
        obj = 0.5 * (grid - m) ** 2 + thr * np.abs(grid) + kthr * np.abs(grid) ** 1.5
        brute = grid[int(np.argmin(obj))]
        closed = float(MeanVarianceOptimizer._prox_step(np.array([m]), np.array([thr]), np.array([kthr]))[0])
        assert abs(closed - brute) < 1e-3


def test_prox_step_reduces_to_soft_threshold_without_impact():
    m = np.array([-0.3, -0.02, 0.0, 0.02, 0.3])
    thr = np.full_like(m, 0.05)
    got = MeanVarianceOptimizer._prox_step(m, thr, np.zeros_like(m))
    expected = np.sign(m) * np.maximum(np.abs(m) - thr, 0.0)
    assert np.allclose(got, expected)


# --- cost-model coefficients -------------------------------------------------
def test_impact_coefficient_formula_and_missing_data():
    model = ParametricCostModel(impact_eta=0.3)
    assert model.impact_coefficient(0.02, adv_dollar=1e8, capital=1e6) == pytest.approx(
        0.3 * 0.02 * np.sqrt(1e6 / 1e8)
    )
    assert model.impact_coefficient(0.02, adv_dollar=0.0, capital=1e6) == 0.0  # no ADV → no impact
    assert model.impact_coefficient(0.02, adv_dollar=1e8, capital=0.0) == 0.0  # no capital → no impact
    assert ParametricCostModel(linear_impact=True).impact_coefficient(0.02, 1e8, 1e6) == 0.0


def test_base_cost_model_has_zero_cost_defaults():
    class Null(CostModel):
        def cost(self, trade):  # pragma: no cover - trivial
            from src.costs.base import TradeCost

            return TradeCost(0, 0, 0, 0, capped=False)

    null = Null()
    assert null.turnover_cost_rate() == 0.0
    assert null.impact_coefficient(0.02, 1e8, 1e6) == 0.0


# --- spread proxy ------------------------------------------------------------
def test_spread_proxy_is_monotone_in_range_and_clamped():
    from src.services.analysis import _spread_proxy

    model = ParametricCostModel()
    tight = make_ohlcv(n=60, seed=1, freq="1D")
    wide = tight.assign(high=tight["high"] * 1.5, low=tight["low"] * 0.5)  # a much wider range
    assert _spread_proxy(wide, model) >= _spread_proxy(tight, model)
    assert _spread_proxy(wide, model) <= 0.02  # capped
    assert _spread_proxy(tight, model) >= model.default_spread * 0.2  # floored
    # No OHLC → the model default.
    import pandas as pd

    assert _spread_proxy(pd.DataFrame({"close": [1.0, 2.0]}), model) == model.default_spread


# --- service integration -----------------------------------------------------
def _universe():
    symbols = [f"S{i}" for i in range(8)]
    bars = {s: make_ohlcv(n=300, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    return symbols, MarketDataClient(DictMarketData(bars))


def test_construct_portfolio_cost_aware_reports_the_cost_split():
    from src.services import analysis

    symbols, dc = _universe()
    res = analysis.construct_portfolio(dc, "ma_crossover", symbols, AS_OF, capital=1_000_000.0, cost_aware=True)
    if not res["feasible"]:
        pytest.skip("fixture produced no feasible portfolio")
    d = res["diagnostics"]
    assert d.get("cost_aware") is True
    assert "linear_cost" in d and "impact_cost" in d
    # Headline net is the round-trip haircut; the one-way rebalance figure stays in detail.
    assert d["expected_active_return_net"] == pytest.approx(d["expected_active_return"] - d["round_trip_cost"])
    assert d["expected_active_return_net_oneway"] == pytest.approx(
        d["expected_active_return"] - d["cost_drag"]
    )
    assert d["cost_drag"] == pytest.approx(d["linear_cost"] + d["impact_cost"])  # one-way total


def test_construct_portfolio_gross_objective_uses_ex_post_drag():
    from src.services import analysis

    symbols, dc = _universe()
    res = analysis.construct_portfolio(dc, "ma_crossover", symbols, AS_OF, capital=1_000_000.0, cost_aware=False)
    if not res["feasible"]:
        pytest.skip("fixture produced no feasible portfolio")
    d = res["diagnostics"]
    assert not d.get("cost_aware")
    assert "cost_drag" in d  # ex-post drag still reported on the cost-blind solve


# --- reported-value integrity ------------------------------------------------
def test_reported_linear_cost_equals_independent_sum():
    # Pin the diagnostic against an independently computed Σ cᵢ|Δwᵢ| (from cash, w₀=0),
    # so a coefficient or annualisation bug surfaces (not just the net = gross − cost id).
    model = ParametricCostModel()
    ci = _inputs({"A": 0.02, "B": 0.001, "C": 0.0005, "D": 0.0005})
    res = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, holding_period_years=H
    )
    w = _vec(res)
    c_lin, _ = MeanVarianceOptimizer._cost_coefficients(model, ci, None, H, SYMS)
    assert res.diagnostics["linear_cost"] == pytest.approx(float(np.sum(c_lin * np.abs(w))))


# --- round-trip headline haircut --------------------------------------------
def test_round_trip_headline_is_conservative_and_capacity_aligned():
    model = ParametricCostModel()
    ci = _inputs({"A": 0.02, "B": 0.001, "C": 0.0005, "D": 0.0005}, adv_dollar=1e8)
    res = MeanVarianceOptimizer(max_weight=0.5).optimize(  # from cash → Δw = w
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci,
        capital=1e7, holding_period_years=H,
    )
    d = res.diagnostics
    w = _vec(res)
    c_lin, k_imp = MeanVarianceOptimizer._cost_coefficients(model, ci, 1e7, H, SYMS)
    # Round-trip = 2 × (Σ cᵢwᵢ + Σ kᵢwᵢ^{3/2}) — the same book cost capacity prices, ×2.
    expected_rt = 2.0 * (float(np.sum(c_lin * w)) + float(np.sum(k_imp * w**1.5)))
    assert d["round_trip_cost"] == pytest.approx(expected_rt)
    # From cash the round-trip is exactly twice the one-way rebalance cost, and strictly
    # more conservative than the one-way net.
    assert d["round_trip_cost"] == pytest.approx(2.0 * d["cost_drag"])
    assert d["expected_active_return_net"] < d["expected_active_return_net_oneway"]
    assert d["expected_active_return_net"] == pytest.approx(d["expected_active_return"] - expected_rt)


# --- cardinality / dust re-solve with cost (spec §4 factor 6 interaction) ----
def test_cardinality_liquidation_cost_is_counted():
    # A full-book w₀ with max_names=2 must fully liquidate the dropped names; that
    # liquidation turnover and its cost must appear in the diagnostics (priced over w−w₀).
    model = ParametricCostModel()
    w0 = {s: 0.25 for s in SYMS}
    res = MeanVarianceOptimizer(max_weight=0.5, max_names=2).optimize(
        _alphas(), _risk(), risk_aversion=LAM, current_weights=w0,
        cost_model=model, cost_inputs=_inputs(0.001), holding_period_years=H,
    )
    assert len(res.weights) <= 2
    c_lin, _ = MeanVarianceOptimizer._cost_coefficients(model, _inputs(0.001), None, H, SYMS)
    w = _vec(res)
    w0v = np.array([w0[s] for s in SYMS])
    assert res.diagnostics["turnover"] == pytest.approx(float(np.sum(np.abs(w - w0v))))
    assert res.diagnostics["linear_cost"] == pytest.approx(float(np.sum(c_lin * np.abs(w - w0v))))
    assert res.diagnostics["linear_cost"] > 0  # the two liquidations are charged


def test_dust_floor_liquidation_is_priced_with_cost():
    model = ParametricCostModel()
    res = MeanVarianceOptimizer(max_weight=0.5, min_weight=0.1).optimize(
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=_inputs(0.001), holding_period_years=H
    )
    assert all(v >= 0.1 - 1e-9 for v in res.weights.values())  # no dust survives
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6


# --- missing ADV / zero capital / linear-impact through the solve ------------
def test_missing_adv_name_gets_linear_only_others_get_impact():
    model = ParametricCostModel()
    # Only A lacks ADV; the rest are illiquid enough to accrue impact.
    ci = CostInputs(
        spread={s: 0.0005 for s in SYMS},
        adv_dollar={"B": 5e7, "C": 5e7, "D": 5e7},  # A omitted → kA = 0
        daily_vol={s: 0.03 for s in SYMS},
    )
    c_lin, k_imp = MeanVarianceOptimizer._cost_coefficients(model, ci, 5e7, H, SYMS)
    assert k_imp[0] == 0.0 and np.all(k_imp[1:] > 0)  # A linear-only, others conic
    res = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, capital=5e7, holding_period_years=H
    )
    assert res.feasible and abs(sum(res.weights.values()) - 1.0) < 1e-6


def test_zero_capital_equals_linear_only_solve():
    model = ParametricCostModel()
    ci = _inputs(0.0005, adv_dollar=1e8)
    opt = MeanVarianceOptimizer(max_weight=0.5)
    zero_cap = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, capital=0.0, holding_period_years=H)
    no_cap = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, capital=None, holding_period_years=H)
    assert zero_cap.diagnostics["impact_cost"] == 0.0
    assert np.allclose(_vec(zero_cap), _vec(no_cap))


def test_linear_impact_mode_disables_the_conic_term_in_the_solve():
    linear_model = ParametricCostModel(linear_impact=True)
    ci = _inputs(0.0005, adv_dollar=1e7)  # illiquid → would accrue √-impact if enabled
    opt = MeanVarianceOptimizer(max_weight=0.5)
    lin = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, cost_model=linear_model, cost_inputs=ci, capital=5e7, holding_period_years=H)
    only_linear = opt.optimize(_alphas(), _risk(), risk_aversion=LAM, cost_model=linear_model, cost_inputs=ci, capital=None, holding_period_years=H)
    assert lin.diagnostics["impact_cost"] == 0.0  # conic term not fed to the optimiser
    assert np.allclose(_vec(lin), _vec(only_linear))


# --- robustness: a bad cost input must not poison the solve ------------------
def test_nan_spread_does_not_poison_the_solve():
    model = ParametricCostModel()
    ci = CostInputs(spread={"A": float("nan"), "B": 0.0005, "C": 0.0005, "D": 0.0005})
    res = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), risk_aversion=LAM, cost_model=model, cost_inputs=ci, holding_period_years=H
    )
    assert res.feasible
    assert res.weights  # not an empty book
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6
    assert np.isfinite(res.diagnostics["linear_cost"])
