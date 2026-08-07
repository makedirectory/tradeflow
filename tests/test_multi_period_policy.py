"""Tests for the multi-period trading policy: aim in front of the target.
Offline and deterministic. Works through the checklist in order: the
unit-discipline property, limiting cases (costless reduction,
κ→0/permanent-signal discount extremes), κ's monotonicity, a synthetic
Gârleanu-Pedersen world (the theorem this reproduces numerically), the
fast-signal discount, no-double-damping, and — at the bottom — the service/CLI
integration (``construct_portfolio(policy="aim")`` and ``run_policy_ab``).
"""

import math
from datetime import datetime

import numpy as np

from tests.fakes import DictMarketData, make_ohlcv
from tradeflow.alphas.base import Alpha
from tradeflow.costs import ParametricCostModel
from tradeflow.costs.parametric import cost_curvature
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.portfolio.optimizer import CostInputs, MeanVarianceOptimizer
from tradeflow.portfolio.policy import (
    build_aim_portfolio,
    derive_kappa,
    discount_factor,
    half_life_in_rebalance_units,
    phi_from_half_life,
    trading_half_life,
)
from tradeflow.risk.base import RiskMatrix
from tradeflow.services import analysis


def _alpha(symbol: str, alpha: float) -> Alpha:
    return Alpha(symbol=symbol, alpha=alpha, as_of=datetime(2024, 1, 1), residual_vol=0.2, ic=0.05, raw_z=1.0)


def _risk(symbols, variance=0.04) -> RiskMatrix:
    return RiskMatrix(symbols=symbols, sigma=np.eye(len(symbols)) * variance)


def _cost(symbols, spread=0.001, adv=5e6, vol=0.02):
    return CostInputs(
        spread={s: spread for s in symbols},
        adv_dollar={s: adv for s in symbols},
        daily_vol={s: vol for s in symbols},
    )


# --- cost curvature (tradeflow/costs/parametric.py) --------------------------------
def test_cost_curvature_matches_the_closed_form():
    k, trade = 0.02, 0.01
    assert abs(cost_curvature(k, trade) - 0.75 * k / math.sqrt(trade)) < 1e-12


def test_cost_curvature_is_undefined_without_impact_or_trade_size():
    assert cost_curvature(0.0, 0.01) is None
    assert cost_curvature(0.02, 0.0) is None
    assert cost_curvature(-0.01, 0.01) is None


# --- unit discipline -------------------------------------------------------------
def test_phi_from_half_life_is_the_ln2_over_hl_property():
    for hl in (1.0, 5.0, 21.0, 100.0):
        phi = phi_from_half_life(hl)
        assert abs(hl - math.log(2) / phi) < 1e-9  # half-life in rebalances = ln2/phi, always


def test_phi_from_half_life_floors_at_the_rebalance_frequency():
    # Can't measure decay faster than one rebalance: HL=0.1 floors to HL=1.
    assert phi_from_half_life(0.1) == phi_from_half_life(1.0)


def test_phi_from_half_life_permanent_signal_is_zero():
    assert phi_from_half_life(float("inf")) == 0.0


def test_half_life_in_rebalance_units_converts_bars_to_rebalances():
    # A 63-bar half-life on a 21-bar rebalance cadence is 3 rebalances.
    assert abs(half_life_in_rebalance_units(63.0, 21.0) - 3.0) < 1e-9


def test_half_life_in_rebalance_units_passes_through_degenerate_inputs():
    assert half_life_in_rebalance_units(float("inf"), 21.0) == float("inf")
    assert half_life_in_rebalance_units(10.0, 0.0) == 10.0


def test_trading_half_life_is_ln2_over_kappa():
    assert abs(trading_half_life(0.1) - math.log(2) / 0.1) < 1e-12
    assert trading_half_life(0.0) == float("inf")


# --- limiting cases --------------------------------------------------------------
def test_discount_factor_permanent_signal_has_no_discount_at_any_kappa():
    for kappa in (0.001, 0.1, 0.5, 1.0):
        assert discount_factor(kappa, 0.0) == 1.0


def test_discount_factor_kappa_to_zero_collapses_to_the_exact_limit():
    # discount(kappa, phi) = kappa / (1 - delta*(1-kappa)); as kappa -> 0 this is
    # kappa / (1 - delta) exactly (delta = e^-phi) - the familiar "kappa/phi" is
    # only this formula's own small-phi limit (1-delta ~ phi), so we check the
    # exact one here and the small-phi coincidence separately below.
    phi = 0.2
    tiny_kappa = 1e-6
    delta = math.exp(-phi)
    assert abs(discount_factor(tiny_kappa, phi) - tiny_kappa / (1.0 - delta)) < 1e-9


def test_discount_factor_small_phi_matches_the_textbook_kappa_over_phi():
    phi = 1e-4  # small enough that delta = e^-phi ~ 1-phi to high precision
    kappa = 0.1
    assert abs(discount_factor(kappa, phi) - kappa / (kappa + phi)) < 1e-3


def test_discount_factor_fast_signal_discounted_harder_than_slow():
    kappa = 0.1
    phi_fast, phi_slow = math.log(2) / 5, math.log(2) / 250
    assert discount_factor(kappa, phi_fast) < discount_factor(kappa, phi_slow)


def test_kappa_limits_costless_and_infinite_cost():
    s = 2.0  # lambda_A * sigma^2
    assert abs(derive_kappa(1.0, s, 1e-9) - 1.0) < 1e-4  # c2 -> 0 => kappa -> 1
    assert derive_kappa(1.0, s, 1e9) < 1e-3  # c2 -> inf => kappa -> 0
    assert derive_kappa(1.0, s, None) is None  # undefined curvature => fallback signal
    assert derive_kappa(0.0, s, 1.0) is None  # non-positive risk-aversion*variance


def test_kappa_monotonicity_grid():
    # kappa increases in s = lambda_A*sigma^2, decreases in cost curvature c2.
    for c2 in (0.1, 1.0, 5.0, 20.0):
        s_values = [0.1, 0.5, 1.0, 5.0, 20.0]
        kappas = [derive_kappa(1.0, s, c2) for s in s_values]
        assert all(a < b for a, b in zip(kappas, kappas[1:]))
    for s in (0.1, 1.0, 10.0):
        c2_values = [0.05, 0.5, 2.0, 10.0, 50.0]
        kappas = [derive_kappa(1.0, s, c2) for c2 in c2_values]
        assert all(a > b for a, b in zip(kappas, kappas[1:]))


def test_costless_reduces_to_exact_myopic(monkeypatch=None):
    """The limiting check, made exact (not asymptotic - see the module
    docstring): with no cost model at all, build_aim_portfolio falls back to
    literally the plain myopic solve, byte-identical weights."""
    symbols = ["A", "B", "C"]
    alphas = [_alpha(s, v) for s, v in zip(symbols, [0.10, 0.05, 0.02])]
    risk = _risk(symbols)
    optimizer = MeanVarianceOptimizer(max_weight=0.6)
    w0 = {"A": 0.34, "B": 0.33, "C": 0.33}

    plain = optimizer.optimize(alphas, risk, target_te=0.05, current_weights=w0)
    aim = build_aim_portfolio(optimizer, alphas, risk, phi=0.3, target_te=0.05, current_weights=w0)
    assert aim.diagnostics["aim_degraded"] is True
    assert aim.weights == plain.weights


def test_cost_aware_but_no_capital_also_falls_back():
    """No capital => k_imp is all zero => cost curvature is undefined (only the
    3/2-power term has curvature) => same exact fallback, even though a cost
    model IS supplied (the linear-only regime, a documented case)."""
    symbols = ["A", "B", "C"]
    alphas = [_alpha(s, v) for s, v in zip(symbols, [0.10, 0.05, 0.02])]
    risk = _risk(symbols)
    optimizer = MeanVarianceOptimizer(max_weight=0.6)
    cost_model = ParametricCostModel()
    cost_inputs = _cost(symbols)
    w0 = {"A": 0.34, "B": 0.33, "C": 0.33}

    plain = optimizer.optimize(
        alphas, risk, target_te=0.05, current_weights=w0, cost_model=cost_model, cost_inputs=cost_inputs
    )
    aim = build_aim_portfolio(
        optimizer,
        alphas,
        risk,
        phi=0.3,
        target_te=0.05,
        current_weights=w0,
        cost_model=cost_model,
        cost_inputs=cost_inputs,
    )
    assert aim.diagnostics["aim_degraded"] is True
    assert aim.weights == plain.weights


def test_discount_factor_matches_value_iteration():
    """Cross-check discount_factor's exact closed form against a direct value-
    iteration solve of the Bellman recursion (the module docstring's own claim) -
    at a phi large enough that the small-phi approximation kappa/(kappa+phi)
    would visibly disagree (this is exactly the earlier bug this test guards
    against: the approximation's error grows with phi and can even flip signs
    in a realized-utility comparison)."""
    lam, sigma2, c2 = 5.0, 1.0, 2.0
    phi = 0.4
    delta = math.exp(-phi)
    kappa = derive_kappa(lam, sigma2, c2)

    A, B = 0.0, 0.0
    for _ in range(3000):
        L = lam * sigma2 + A
        omega = 2 * L + c2
        theta = c2 / omega
        mu = (1.0 + B * delta) / omega
        A = L * theta**2 + (c2 / 2) * (theta - 1) ** 2
        B = theta - 2 * L * theta * mu + c2 * (1 - theta) * mu + B * delta * theta
    kappa_from_iteration = 1.0 - theta
    aim_coefficient = mu / kappa_from_iteration  # weight on f in x* = x_prev + kappa*(aim - x_prev)

    assert abs(kappa_from_iteration - kappa) < 1e-6
    predicted_aim_coefficient = discount_factor(kappa, phi) / (2 * lam * sigma2)
    assert abs(aim_coefficient - predicted_aim_coefficient) < 1e-6

    # The approximation this replaced would have been visibly wrong here:
    approx_discount = kappa / (kappa + phi)
    approx_aim_coefficient = approx_discount / (2 * lam * sigma2)
    assert abs(approx_aim_coefficient - aim_coefficient) > 1e-3


# --- synthetic Gârleanu-Pedersen world -----------------------------------------
def test_synthetic_gp_world_aim_beats_myopic_and_gap_grows_with_decay():
    """The theorem this module reproduces numerically: with quadratic cost and an
    exponentially-decaying signal, the aim policy's realized utility >= the
    myopic policy's, and the gap grows as decay gets faster (more to gain from
    anticipating it). Exercises the derived closed forms directly in the exact
    quadratic-cost, single-asset setting the LQ derivation assumed (the module
    docstring's Riccati solution) - the full constrained multi-name optimizer
    with the real (linear + 3/2-power) cost is exercised separately below.
    """
    lam, sigma2, c2 = 5.0, 1.0, 2.0
    kappa = derive_kappa(lam, sigma2, c2)
    assert kappa is not None and 0 < kappa < 1
    T = 60
    gaps = []
    for phi in (0.02, 0.1, 0.4):
        delta = math.exp(-phi)
        discount = discount_factor(kappa, phi)
        w_myopic = w_aim = 0.0
        util_myopic = util_aim = 0.0
        for t in range(T):
            alpha_t = 1.0 * delta**t

            target_myopic = alpha_t / (2 * lam * sigma2)
            trade_myopic = target_myopic - w_myopic
            util_myopic += (
                alpha_t * target_myopic - lam * sigma2 * target_myopic**2 - 0.5 * c2 * trade_myopic**2
            )
            w_myopic = target_myopic

            aim_target = alpha_t * discount / (2 * lam * sigma2)
            trade_aim = kappa * (aim_target - w_aim)
            w_new = w_aim + trade_aim
            util_aim += alpha_t * w_new - lam * sigma2 * w_new**2 - 0.5 * c2 * trade_aim**2
            w_aim = w_new

        assert util_aim >= util_myopic - 1e-9
        gaps.append(util_aim - util_myopic)

    assert gaps[0] <= gaps[1] <= gaps[2]  # gap widens as decay (phi) speeds up


# --- fast-signal discount ------------------------------------------------------
def test_fast_signal_discount_matches_kappa_over_kappa_plus_phi():
    """Two names, one signal each, equal magnitude, half-lives 5 vs 250 rebalances:
    the aim solve's UNCONSTRAINED weight ratio (cash-relative, diagonal Sigma with
    equal variance => the ratio is exactly the alpha ratio) matches the predicted
    kappa/(kappa+phi) discount ratio exactly, and the fast name's turnover
    contribution shrinks more than the slow name's once box/budget engage.
    """
    symbols = ["FAST", "SLOW"]
    risk = _risk(symbols, variance=0.04)
    kappa = 0.1
    phi_fast, phi_slow = math.log(2) / 5, math.log(2) / 250
    d_fast, d_slow = discount_factor(kappa, phi_fast), discount_factor(kappa, phi_slow)

    raw_alphas = [_alpha("FAST", 0.10), _alpha("SLOW", 0.10)]
    discounted_alphas = [_alpha("FAST", 0.10 * d_fast), _alpha("SLOW", 0.10 * d_slow)]

    optimizer = MeanVarianceOptimizer(max_weight=0.9)
    myopic = optimizer.optimize(raw_alphas, risk, risk_aversion=2.0)
    aim = optimizer.optimize(discounted_alphas, risk, risk_aversion=2.0)

    ratio_myopic = myopic.unconstrained_weights["FAST"] / myopic.unconstrained_weights["SLOW"]
    ratio_aim = aim.unconstrained_weights["FAST"] / aim.unconstrained_weights["SLOW"]
    assert abs(ratio_myopic - 1.0) < 1e-9  # equal alpha, equal risk => equal weight, undiscounted
    assert abs(ratio_aim - d_fast / d_slow) < 1e-9  # exactly the predicted discount ratio

    w0 = {"FAST": 0.7, "SLOW": 0.3}
    myopic_w0 = optimizer.optimize(raw_alphas, risk, risk_aversion=2.0, current_weights=w0)
    aim_w0 = optimizer.optimize(discounted_alphas, risk, risk_aversion=2.0, current_weights=w0)
    fast_trade_myopic = abs(myopic_w0.weights["FAST"] - w0["FAST"])
    fast_trade_aim = abs(aim_w0.weights["FAST"] - w0["FAST"])
    slow_trade_myopic = abs(myopic_w0.weights["SLOW"] - w0["SLOW"])
    slow_trade_aim = abs(aim_w0.weights["SLOW"] - w0["SLOW"])
    assert fast_trade_myopic > 1e-9 and slow_trade_myopic > 1e-9
    fast_shrink = fast_trade_aim / fast_trade_myopic
    slow_shrink = slow_trade_aim / slow_trade_myopic
    assert fast_shrink < slow_shrink  # the fast signal's own turnover contribution drops more


# --- no double damping -----------------------------------------------------------
def test_no_double_damping_kappa_and_band_together_dont_over_trade():
    symbols = ["A", "B", "C", "D"]
    alphas = [_alpha(s, v) for s, v in zip(symbols, [0.30, 0.10, 0.10, 0.10])]
    risk = _risk(symbols)
    optimizer = MeanVarianceOptimizer(max_weight=0.6)
    cost_model = ParametricCostModel()
    cost_inputs = _cost(symbols)
    w0 = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    cost_free_jump = optimizer.optimize(alphas, risk, target_te=0.05, current_weights=w0)
    band_only = optimizer.optimize(
        alphas,
        risk,
        target_te=0.05,
        current_weights=w0,
        cost_model=cost_model,
        cost_inputs=cost_inputs,
        capital=1_000_000.0,
    )
    combined = build_aim_portfolio(
        optimizer,
        alphas,
        risk,
        phi=0.0,
        target_te=0.05,
        current_weights=w0,
        cost_model=cost_model,
        cost_inputs=cost_inputs,
        capital=1_000_000.0,
    )
    assert not combined.diagnostics.get("aim_degraded")
    assert combined.diagnostics["turnover"] <= band_only.diagnostics["turnover"] + 1e-9
    assert combined.diagnostics["turnover"] <= cost_free_jump.diagnostics["turnover"] + 1e-9
    assert band_only.diagnostics["turnover"] <= cost_free_jump.diagnostics["turnover"] + 1e-9


# --- service integration: construct_portfolio(policy="aim") -------------------
def _client(symbols, n=500, benchmark="SPY"):
    data = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate([*symbols, benchmark])}
    return MarketDataClient(DictMarketData(data)), data


def test_construct_portfolio_policy_none_is_unchanged():
    symbols = [f"S{i}" for i in range(8)]
    client, data = _client(symbols)
    as_of = data["S0"].index[400].to_pydatetime()

    baseline = analysis.construct_portfolio(client, "volume_spike", symbols, as_of)
    off = analysis.construct_portfolio(client, "volume_spike", symbols, as_of, policy=None)
    assert baseline["weights"] == off["weights"]
    assert "policy_report" not in off


def test_construct_portfolio_policy_aim_feasible_and_reports():
    symbols = [f"S{i}" for i in range(8)]
    client, data = _client(symbols)
    as_of = data["S0"].index[400].to_pydatetime()

    r = analysis.construct_portfolio(
        client, "volume_spike", symbols, as_of, capital=1_000_000.0, policy="aim"
    )
    assert r["feasible"]
    assert r["policy"] == "aim"
    assert abs(sum(r["weights"].values()) - 1.0) < 1e-6
    assert "policy_report" in r
    d = r["diagnostics"]
    assert d["policy"] in ("aim", "myopic_fallback")
    if d["policy"] == "aim":
        assert d["kappa"] > 0
        assert d["trading_half_life_rebalances"] > 0


def test_construct_portfolio_policy_aim_rejects_incompatible_books():
    symbols = [f"S{i}" for i in range(8)]
    client, data = _client(symbols)
    as_of = data["S0"].index[400].to_pydatetime()

    try:
        analysis.construct_portfolio(
            client, "volume_spike", symbols, as_of, policy="aim", book="market_neutral", gross_leverage=1.0
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_construct_portfolio_policy_invalid_value_raises():
    symbols = [f"S{i}" for i in range(4)]
    client, data = _client(symbols)
    as_of = data["S0"].index[300].to_pydatetime()
    try:
        analysis.construct_portfolio(client, "volume_spike", symbols, as_of, policy="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- run_policy_ab (net-of-cost A/B) -------------------------------------------
def test_run_policy_ab_produces_both_variants():
    symbols = [f"S{i}" for i in range(8)]
    client, data = _client(symbols, n=900)
    start = data["S0"].index[100].to_pydatetime()
    end = data["S0"].index[800].to_pydatetime()

    r = analysis.run_policy_ab(
        client, "volume_spike", symbols, start, end, n_points=10, horizon=21, capital=1_000_000.0
    )
    assert r["periods"] >= 2
    assert set(r["summaries"]) == {"myopic", "aim"}
    for name, s in r["summaries"].items():
        assert s["periods"] >= 2
        assert "net_ir" in s
        assert "mean_turnover" in s
    assert r["winner_net_ir"] in {"myopic", "aim"}
    assert isinstance(r["over_damped"], bool)


def test_run_policy_ab_insufficient_window_reports_note():
    symbols = ["S0", "S1"]
    data = {s: make_ohlcv(n=30, seed=i, freq="1D") for i, s in enumerate(["S0", "S1", "SPY"])}
    client = MarketDataClient(DictMarketData(data))
    start = data["S0"].index[0].to_pydatetime()
    end = data["S0"].index[-1].to_pydatetime()

    r = analysis.run_policy_ab(client, "volume_spike", symbols, start, end, horizon=21)
    assert r["periods"] == 0
    assert "note" in r


# --- compute_horizon's blend-superseded note -----------------------------------
def test_compute_horizon_notes_the_aim_policy_supersedes_the_blend():
    symbols = [f"S{i}" for i in range(10)]
    data = {s: make_ohlcv(n=600, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    client = MarketDataClient(DictMarketData(data))
    r = analysis.compute_horizon(
        client, "volume_spike", symbols, datetime(2023, 1, 1), datetime(2024, 12, 31), max_lag=8
    )
    assert "aim-in-front" in r["blend_superseded_by"]
    assert "half_life_upper" in r
