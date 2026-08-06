"""Tests for mean-variance portfolio construction.

Offline and deterministic. Covers the closed-form optimum, target-TE calibration,
transfer-coefficient monotonicity, the no-trade band, turnover-from-w0,
infeasibility detection, and constraint preservation.
"""

from datetime import datetime

import numpy as np
import pytest

from tests.fakes import DictMarketData, make_ohlcv
from tradeflow.alphas.base import Alpha
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.portfolio.optimizer import MeanVarianceOptimizer
from tradeflow.risk.base import RiskMatrix
from tradeflow.services import analysis

AS_OF = datetime(2024, 6, 1)
SYMS = ["A", "B", "C", "D"]
# A fixed positive-definite covariance and alpha vector for the closed-form checks.
_L = np.array([[0.20, 0, 0, 0], [0.05, 0.18, 0, 0], [0.03, 0.04, 0.22, 0], [0.01, 0.02, 0.03, 0.16]])
SIGMA = _L @ _L.T
ALPHA = np.array([0.06, 0.02, -0.01, 0.04])


def _alphas() -> list:
    return [Alpha(s, float(ALPHA[i]), AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(SYMS)]


def _alphas_from(vec) -> list:
    return [Alpha(s, float(vec[i]), AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(SYMS)]


def _risk() -> RiskMatrix:
    return RiskMatrix(SYMS, SIGMA)


# --- closed form -------------------------------------------------------------
def test_unconstrained_matches_closed_form():
    lam = 2.0
    result = MeanVarianceOptimizer(max_weight=1.0).optimize(_alphas(), _risk(), risk_aversion=lam)
    expected = (np.linalg.inv(SIGMA) @ ALPHA) / (2 * lam)
    got = np.array([result.unconstrained_weights[s] for s in SYMS])
    assert np.allclose(got, expected)
    assert abs(result.diagnostics["ir_star"] - np.sqrt(ALPHA @ np.linalg.inv(SIGMA) @ ALPHA)) < 1e-9


def test_target_te_calibration_is_exact():
    result = MeanVarianceOptimizer(max_weight=1.0).optimize(_alphas(), _risk(), target_te=0.04)
    # The unconstrained optimal tracking error ψ* = IR*/(2λ) equals the target exactly.
    assert abs(result.diagnostics["optimal_tracking_error"] - 0.04) < 1e-9


# --- transfer coefficient ----------------------------------------------------
def test_tightening_cardinality_lowers_transfer_coefficient():
    loose = MeanVarianceOptimizer(max_weight=0.5).optimize(_alphas(), _risk(), target_te=0.05)
    tight = MeanVarianceOptimizer(max_weight=0.5, max_names=2).optimize(_alphas(), _risk(), target_te=0.05)
    assert tight.diagnostics["transfer_coefficient"] <= loose.diagnostics["transfer_coefficient"] + 1e-9


# --- no-trade band & turnover ------------------------------------------------
def test_no_trade_band_suppresses_subthreshold_churn():
    base = MeanVarianceOptimizer(max_weight=0.5).optimize(_alphas(), _risk(), target_te=0.05)
    w0 = base.weights
    # A tiny alpha perturbation with a wide band → keep w0, zero turnover.
    nudged = [Alpha(s, float(ALPHA[i]) + 1e-4, AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(SYMS)]
    banded = MeanVarianceOptimizer(max_weight=0.5, no_trade_band=0.05).optimize(
        nudged, _risk(), target_te=0.05, current_weights=w0
    )
    assert banded.diagnostics["turnover"] == 0.0


def test_turnover_is_measured_from_current_weights():
    opt = MeanVarianceOptimizer(max_weight=0.5)
    target = opt.optimize(_alphas(), _risk(), target_te=0.05).weights
    from_target = opt.optimize(_alphas(), _risk(), target_te=0.05, current_weights=target)
    from_cash = opt.optimize(_alphas(), _risk(), target_te=0.05, current_weights={})
    assert from_target.diagnostics["turnover"] < 1e-6  # already there → no trade
    assert from_cash.diagnostics["turnover"] > 0.5  # from cash → fully invest


# --- infeasibility -----------------------------------------------------------
def test_infeasible_constraints_named_not_crashed():
    result = MeanVarianceOptimizer(max_weight=0.3, max_names=1).optimize(_alphas(), _risk(), target_te=0.05)
    assert result.feasible is False
    assert "weight" in result.binding_constraint or "cardinality" in result.binding_constraint
    assert result.weights == {}


# --- constraints preserved ---------------------------------------------------
def test_weights_respect_box_budget_and_cardinality():
    result = MeanVarianceOptimizer(max_weight=0.4, max_names=3).optimize(_alphas(), _risk(), target_te=0.05)
    w = result.weights
    assert len(w) <= 3
    assert all(v <= 0.4 + 1e-9 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-6


# --- min-weight dust floor ---------------------------------------------------
def test_min_weight_leaves_no_dust_positions():
    # Every held name must clear the floor (no 0 < w < min_weight); budget still met.
    result = MeanVarianceOptimizer(max_weight=0.5, min_weight=0.1).optimize(
        _alphas(), _risk(), target_te=0.05
    )
    w = result.weights
    assert all(v >= 0.1 - 1e-9 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_min_weight_must_not_exceed_max_weight():
    import pytest

    with pytest.raises(ValueError):
        MeanVarianceOptimizer(max_weight=0.2, min_weight=0.3)


# --- capacity ------------------------------------------------------------------
def test_capacity_is_liquidity_sensitive():
    from tests.fakes import make_ohlcv
    from tradeflow.services.analysis import _capacity

    syms = [f"S{i}" for i in range(6)]
    weights = {s: 1.0 / len(syms) for s in syms}
    liquid = {s: make_ohlcv(n=200, seed=i, freq="1D") for i, s in enumerate(syms)}
    illiquid = {s: f.assign(volume=f["volume"] * 0.01) for s, f in liquid.items()}

    cap_liquid = _capacity(weights, liquid, gross_alpha=0.05, holding_period_years=1 / 12)
    cap_illiquid = _capacity(weights, illiquid, gross_alpha=0.05, holding_period_years=1 / 12)
    assert cap_liquid > cap_illiquid  # deeper liquidity → more capacity before cost bites
    assert _capacity(weights, liquid, gross_alpha=0.0, holding_period_years=1 / 12) == 0.0  # no edge


# --- integration / leakage ---------------------------------------------------
def test_construct_portfolio_independent_of_post_as_of_bars():
    symbols = [f"S{i}" for i in range(8)]
    full = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    cutoff = full["S0"].index[300]
    as_of = cutoff.to_pydatetime()
    truncated = {s: f.loc[f.index <= cutoff] for s, f in full.items()}

    a = analysis.construct_portfolio(
        MarketDataClient(DictMarketData(truncated)), "ma_crossover", symbols, as_of
    )
    b = analysis.construct_portfolio(MarketDataClient(DictMarketData(full)), "ma_crossover", symbols, as_of)
    assert a["weights"] == b["weights"]
    assert a["diagnostics"] == b["diagnostics"]


# --- benchmark-relative construction --------------------------------------------
W_B = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}


def test_reduction_no_benchmark_matches_todays_behavior():
    """Without benchmark_weights every quantity is byte-identical to the
    plain cash-relative solve - the same reduction pattern."""
    plain = MeanVarianceOptimizer(max_weight=1.0).optimize(_alphas(), _risk(), risk_aversion=2.0)
    zero_bench = MeanVarianceOptimizer(max_weight=1.0).optimize(
        _alphas(), _risk(), risk_aversion=2.0, benchmark_weights={s: 0.0 for s in SYMS}
    )
    assert plain.weights == zero_bench.weights
    assert plain.diagnostics == zero_bench.diagnostics
    assert "has_benchmark" not in plain.diagnostics


def test_round_trip_implied_returns_recovers_the_benchmark():
    """optimize(implied_returns(w_B, Σ, μ_B), Σ, w_B) -> w = w_B."""
    from tradeflow.portfolio.benchmark import implied_returns

    mu = implied_returns(W_B, _risk(), mu_b=0.05)
    result = MeanVarianceOptimizer(max_weight=1.0).optimize(
        _alphas_from([mu[s] for s in SYMS]), _risk(), risk_aversion=1.0, benchmark_weights=W_B
    )
    assert result.feasible
    for s in SYMS:
        assert abs(result.weights.get(s, 0.0) - W_B[s]) < 1e-6


def test_zero_alpha_from_benchmark_holdings_is_a_no_trade():
    """Hidden factor 4: w0 = w_B with zero alphas -> zero trades."""
    zero = [Alpha(s, 0.0, AS_OF, 0.2, 0.05, 0.0) for s in SYMS]
    result = MeanVarianceOptimizer(max_weight=1.0).optimize(
        zero, _risk(), risk_aversion=1.0, benchmark_weights=W_B, current_weights=W_B
    )
    assert result.diagnostics["turnover"] < 1e-6


def test_zero_alpha_from_cash_buys_the_benchmark():
    """Hidden factor 4: w0 = cash with zero alphas -> the solve buys the benchmark,
    paying cost while expected active return stays zero (indexing costs money)."""
    zero = [Alpha(s, 0.0, AS_OF, 0.2, 0.05, 0.0) for s in SYMS]
    result = MeanVarianceOptimizer(max_weight=1.0).optimize(
        zero, _risk(), risk_aversion=1.0, benchmark_weights=W_B, current_weights={}
    )
    for s in SYMS:
        assert abs(result.weights.get(s, 0.0) - W_B[s]) < 1e-6
    assert result.diagnostics["turnover"] > 0.9
    assert abs(result.diagnostics["expected_active_return"]) < 1e-9


def test_neutralization_alpha_dot_wb_is_zero_and_unconstrained_beta_vanishes():
    """After 3.3, α_neutralᵀw_B = 0 exactly, so the *unconstrained* optimum (no
    box/budget) carries no benchmark tilt - box+budget always applies in the
    constrained solve (there's no cardinality-free way to turn them off through
    the public API), so this checks ``unconstrained_weights``, not ``weights``:
    any residual β_a in the constrained result comes from box/budget/cardinality,
    which is exactly what the underweight-bound test below exercises."""
    wb = np.array([W_B[s] for s in SYMS])
    beta = (SIGMA @ wb) / (wb @ SIGMA @ wb)
    alpha_neutral = ALPHA - beta * (ALPHA @ wb)
    assert abs(alpha_neutral @ wb) < 1e-9

    result = MeanVarianceOptimizer(max_weight=1.0).optimize(
        _alphas(), _risk(), risk_aversion=1.0, benchmark_weights=W_B
    )
    w_star = np.array([result.unconstrained_weights[s] for s in SYMS])
    active_beta_unconstrained = beta @ (w_star - wb)
    assert abs(active_beta_unconstrained) < 1e-9


def test_te_truth_and_the_beta_residual_split():
    """Reported ψ equals √(w_aᵀΣw_a) recomputed independently; β_a²σ_B² + ω² sums
    exactly to ψ² (the "TE truth" property)."""
    result = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), target_te=0.05, benchmark_weights=W_B
    )
    w = np.array([result.weights.get(s, 0.0) for s in SYMS])
    wb = np.array([W_B[s] for s in SYMS])
    wa = w - wb
    psi = float(np.sqrt(wa @ SIGMA @ wa))
    assert abs(result.diagnostics["predicted_tracking_error"] - psi) < 1e-9

    d = result.diagnostics
    split = d["active_beta"] ** 2 * d["benchmark_variance"] + d["residual_risk"] ** 2
    assert abs(split - psi**2) < 1e-9


def test_underweight_bound_pins_active_weight_to_minus_benchmark():
    """A strongly negative alpha on a small-w_B name pins w = 0, hence
    w_a = -w_B (the constraint a market-neutral book relaxes)."""
    hostile = ALPHA.copy()
    hostile[3] = -1.0  # symbol D, w_B=0.1: make it maximally unattractive
    result = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas_from(hostile), _risk(), target_te=0.05, benchmark_weights=W_B
    )
    assert result.weights.get("D", 0.0) < 1e-9  # pinned to 0 -> w_a = -w_B,D


def test_self_benchmark_degeneracy_warns():
    """Hidden factor 5: w0 == w_B makes ψ measure "distance from myself"."""
    result = MeanVarianceOptimizer(max_weight=1.0).optimize(
        _alphas(), _risk(), risk_aversion=1.0, benchmark_weights=W_B, current_weights=W_B
    )
    assert result.diagnostics["self_benchmark_warning"] is True

    elsewhere = MeanVarianceOptimizer(max_weight=1.0).optimize(
        _alphas(), _risk(), risk_aversion=1.0, benchmark_weights=W_B, current_weights={}
    )
    assert elsewhere.diagnostics["self_benchmark_warning"] is False


# --- construct_portfolio integration ---------------------------------------------
def test_construct_portfolio_reduction_without_benchmark_holdings():
    """No benchmark_holdings -> byte-identical to the cash-relative solve (the
    same reduction pattern, at the service layer)."""
    symbols = [f"S{i}" for i in range(8)]
    bars = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    as_of = bars["S0"].index[-1].to_pydatetime()
    client = MarketDataClient(DictMarketData(bars))

    plain = analysis.construct_portfolio(client, "ma_crossover", symbols, as_of)
    assert plain["benchmark_portfolio"] is None
    assert "has_benchmark" not in plain["diagnostics"]


def test_construct_portfolio_equal_benchmark_is_fully_covered():
    symbols = [f"S{i}" for i in range(8)]
    bars = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    as_of = bars["S0"].index[-1].to_pydatetime()
    client = MarketDataClient(DictMarketData(bars))

    result = analysis.construct_portfolio(
        client, "ma_crossover", symbols, as_of, benchmark_holdings="equal", benchmark_premium=0.05
    )
    assert result["feasible"]
    bp = result["benchmark_portfolio"]
    assert bp["coverage"] == pytest.approx(1.0)
    assert bp["uncovered_weight"] == pytest.approx(0.0)
    assert result["diagnostics"]["has_benchmark"] is True
    assert set(bp["consensus_returns"]) == set(symbols)

    # Value-added identity (SR_P² ≈ SR_B² + IR²) is reported and self-consistent.
    vai = bp["value_added_identity"]
    assert vai["sr_portfolio_predicted"] == pytest.approx((vai["sr_benchmark"] ** 2 + vai["ir"] ** 2) ** 0.5)


def test_construct_portfolio_file_benchmark_reports_partial_coverage(tmp_path):
    symbols = [f"S{i}" for i in range(8)]
    bars = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    as_of = bars["S0"].index[-1].to_pydatetime()
    client = MarketDataClient(DictMarketData(bars))

    path = tmp_path / "bench.csv"
    path.write_text("symbol,weight\nS0,0.5\nS1,0.3\nNOT_IN_UNIVERSE,0.2\n")
    result = analysis.construct_portfolio(
        client, "ma_crossover", symbols, as_of, benchmark_holdings=str(path), benchmark_premium=0.06
    )
    assert result["feasible"]
    bp = result["benchmark_portfolio"]
    assert bp["coverage"] == pytest.approx(0.8)
    assert bp["uncovered_weight"] == pytest.approx(0.2)
    assert "NOT_IN_UNIVERSE" not in bp["consensus_returns"]


# --- CLI --------------------------------------------------------------------------
def test_cli_allocate_utility_with_benchmark_holdings(monkeypatch, tmp_path, capsys):
    import main

    symbols = [f"S{i}" for i in range(8)]
    bars = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    as_of = bars["S0"].index[-1].to_pydatetime()
    client = MarketDataClient(DictMarketData(bars))
    monkeypatch.setattr(main, "build_data_and_broker", lambda: (None, client))

    args = main.build_parser().parse_args(
        [
            "allocate",
            "--objective",
            "utility",
            "--strategy",
            "ma_crossover",
            "--symbols",
            ",".join(symbols),
            "--as-of",
            as_of.strftime("%Y-%m-%d"),
            "--benchmark-holdings",
            "equal",
            "--benchmark-premium",
            "0.06",
        ]
    )
    args.func(args)
    out = capsys.readouterr().out
    assert "benchmark 'equal'" in out
    assert "consensus returns" in out
    assert "active beta" in out
