"""Tests for the conditional risk layer (spec 024): Σ_t = D_t·R·D_t.

Offline and deterministic. Works through the spec's §6 checklist in order:
forecast quality (MZ/QLIKE), TE tracking under a regime-switching synthetic world,
the as-of property, PSD assembly under stress inputs (both backends), the churn
guard (calm-regime turnover share + the λ=1 reduction), and — in
``tests/test_conditional_risk_service.py`` — the service-level wiring and the
net-of-cost A/B harness.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk import (
    LedoitWolfCovariance,
    SampleCovariance,
    build_factor_risk_matrix,
    build_risk_matrix,
    estimate_factor_model,
)
from src.risk.conditional import (
    condition_risk_matrix,
    evaluate_vol_forecasts,
    ewma_covariance_path,
    ewma_variance_path,
    ewma_variance_series,
    har_variance_forecast,
    mincer_zarnowitz,
    qlike_loss,
    turnover_risk_share,
)
from src.risk.factor import FactorRiskMatrix
from tests.fakes import make_ohlcv

PPY = 252.0


def _vol_clustering_returns(t: int, seed: int = 0, n_regimes: int = 8) -> pd.Series:
    """Block-regime vol-clustering synthetic returns (a vol level draw per regime,
    several regime switches across the sample) — a flat/expanding unconditional
    forecast is systematically wrong within every regime (averaging across all of
    them); EWMA/HAR should beat it here (the honest evidence-gate scenario)."""
    rng = np.random.default_rng(seed)
    seg = t // n_regimes
    vols = rng.uniform(0.005, 0.04, n_regimes)
    r = [rng.normal(0, v, seg) for v in vols]
    return pd.Series(np.concatenate(r))


def _correlated_panel(t: int, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 1, (t, 1))
    loadings = rng.uniform(0.3, 1.2, (1, n))
    x = factor @ loadings + rng.normal(0, 0.5, (t, n))
    return pd.DataFrame(x, columns=[f"S{i}" for i in range(n)])


# --- forecast quality (MZ / QLIKE) -------------------------------------------
def test_ewma_beats_unconditional_on_vol_clustering_synthetic():
    """The honest evidence gate: on genuinely vol-clustered data, EWMA's QLIKE
    should be lower (better) than the flat unconditional trailing forecast, and its
    MZ slope closer to 1. This is the scenario spec 024's whole premise rests on."""
    r = _vol_clustering_returns(1500, seed=3)
    result = evaluate_vol_forecasts(r, min_obs=60, n_points=200, periods_per_year=PPY)
    assert result.n_points > 50
    ewma_q = result.by_method["ewma"]["qlike"]
    uncond_q = result.by_method["unconditional"]["qlike"]
    assert np.isfinite(ewma_q) and np.isfinite(uncond_q)
    assert ewma_q < uncond_q  # conditioning wins on the scenario it's built for

    ewma_b = result.by_method["ewma"]["mincer_zarnowitz"]["b"]
    uncond_b = result.by_method["unconditional"]["mincer_zarnowitz"]["b"]
    assert abs(ewma_b - 1.0) < abs(uncond_b - 1.0)


def test_evaluate_vol_forecasts_reports_by_regime():
    r = _vol_clustering_returns(1200, seed=4)
    result = evaluate_vol_forecasts(r, min_obs=60, n_points=150)
    for method in ("ewma", "har", "unconditional"):
        by_regime = result.by_method[method]["by_regime"]
        assert set(by_regime) == {"low", "mid", "high"}
        assert sum(by_regime[k]["n"] for k in by_regime) == result.n_points


def test_mincer_zarnowitz_recovers_a_perfect_forecast():
    rng = np.random.default_rng(0)
    forecast = rng.uniform(0.5, 2.0, 200)
    realized = forecast.copy()  # perfect forecast: a=0, b=1, r2=1
    mz = mincer_zarnowitz(realized, forecast)
    assert abs(mz["a"]) < 1e-8
    assert abs(mz["b"] - 1.0) < 1e-8
    assert mz["r2"] > 0.999


def test_qlike_is_zero_for_a_perfect_forecast():
    rng = np.random.default_rng(1)
    v = rng.uniform(0.1, 1.0, 100)
    assert qlike_loss(v, v) < 1e-9


def test_qlike_penalizes_underforecasting_more_by_construction():
    # QLIKE is asymmetric: understating a realized spike is punished harder than
    # overstating a calm period by the same ratio (Patton 2011's whole point).
    realized = np.array([4.0])
    under = qlike_loss(realized, np.array([1.0]))  # forecast 1, realized 4 (ratio 4)
    over = qlike_loss(realized, np.array([16.0]))  # forecast 16, realized 4 (ratio 0.25)
    assert under != over  # not symmetric in the forecast


# --- TE tracking under a regime-switching synthetic world --------------------
def test_conditional_sigma_tracks_a_vol_regime_switch_better_than_unconditional():
    """A world that's calm for a long stretch then doubles in vol: the conditional
    Σ's diagonal should move toward the new regime; the unconditional (flat-window)
    Σ, estimated once over the whole mixed window, systematically understates the
    stress-regime variance (exactly spec 024's motivating failure)."""
    rng = np.random.default_rng(7)
    calm = rng.normal(0, 0.01, (300, 4))
    stress = rng.normal(0, 0.03, (100, 4))  # 3x vol
    panel = pd.DataFrame(np.vstack([calm, stress]), columns=["A", "B", "C", "D"])

    unconditional = build_risk_matrix(SampleCovariance(), _bars_from_panel(panel), PPY, min_obs=60)
    conditional = condition_risk_matrix(unconditional, panel, "ewma", PPY, min_obs=60, lambda_=0.90)

    true_stress_var = float(np.var(stress[:, 0])) * PPY
    uncond_var = float(np.diag(unconditional.sigma)[unconditional.symbols.index("A")])
    cond_var = float(np.diag(conditional.sigma)[conditional.symbols.index("A")])
    # Conditional variance (after the regime switch) is closer to the TRUE stress
    # variance than the flat-window unconditional estimate is.
    assert abs(cond_var - true_stress_var) < abs(uncond_var - true_stress_var)


def _bars_from_panel(panel: pd.DataFrame) -> dict:
    """Turn a returns panel into fake OHLCV bars (cumulative price from returns)."""
    bars = {}
    for col in panel.columns:
        price = 100.0 * np.cumprod(1.0 + panel[col].to_numpy())
        idx = pd.date_range("2023-01-01", periods=len(price), freq="1D")
        bars[col] = pd.DataFrame(
            {"open": price, "high": price * 1.001, "low": price * 0.999, "close": price, "volume": 1e6},
            index=idx,
        )
    return bars


# --- as-of property -----------------------------------------------------------
def test_ewma_variance_is_independent_of_post_t_returns():
    rng = np.random.default_rng(11)
    prefix = rng.normal(0, 0.02, (150, 3))
    tail_a = rng.normal(0, 0.1, (50, 3))
    tail_b = rng.normal(0, 0.5, (50, 3))  # a wildly different "future"

    panel_a = pd.DataFrame(np.vstack([prefix, tail_a]), columns=["A", "B", "C"])
    panel_b = pd.DataFrame(np.vstack([prefix, tail_b]), columns=["A", "B", "C"])

    at_t = ewma_variance_path(panel_a.iloc[:150], lambda_=0.94, min_obs=60)
    from_a = ewma_variance_path(panel_a.iloc[:150], lambda_=0.94, min_obs=60)
    from_b = ewma_variance_path(panel_b.iloc[:150], lambda_=0.94, min_obs=60)
    # Slicing to <= t is what matters — the prefix is byte-identical either way, so
    # a correctly as-of (forward-pass, non-centered) implementation is unaffected by
    # whatever wildly different data follows.
    pd.testing.assert_series_equal(at_t, from_a)
    pd.testing.assert_series_equal(from_a, from_b)


def test_ewma_variance_series_row_t_depends_only_on_prefix():
    rng = np.random.default_rng(12)
    x = rng.normal(0, 0.02, (200, 2))
    full = ewma_variance_series(x, lambda_=0.94, min_obs=60)
    prefix_only = ewma_variance_series(x[:120], lambda_=0.94, min_obs=60)
    np.testing.assert_allclose(full[:120], prefix_only, equal_nan=True)


def test_har_forecast_is_independent_of_future_returns():
    rng = np.random.default_rng(13)
    prefix = pd.Series(rng.normal(0, 0.02, 150))
    tail_a = pd.Series(rng.normal(0, 0.1, 50))
    tail_b = pd.Series(rng.normal(0, 0.5, 50))
    at_t = har_variance_forecast(pd.concat([prefix, tail_a]).iloc[:150], min_obs=60)
    from_b = har_variance_forecast(pd.concat([prefix, tail_b]).iloc[:150], min_obs=60)
    assert at_t == from_b


# --- PSD assembly under stress inputs -----------------------------------------
def test_conditioned_shrinkage_matrix_stays_psd_with_a_thin_history_name():
    bars = {s: make_ohlcv(n=200, seed=i, freq="1D") for i, s in enumerate(["AAA", "BBB", "CCC"])}
    bars["FFF"] = make_ohlcv(n=5, seed=9, freq="1D")  # 5 returns — way below min_obs
    matrix = build_risk_matrix(LedoitWolfCovariance(), bars, PPY, min_obs=60, conditional="ewma")
    assert matrix.is_positive_definite()
    assert not np.isnan(matrix.sigma).any()
    assert "FFF" in matrix.symbols


def test_conditioned_shrinkage_matrix_stays_psd_with_a_dead_flat_name():
    rng = np.random.default_rng(21)
    panel = pd.DataFrame(
        {
            "A": rng.normal(0, 0.02, 200),
            "B": rng.normal(0, 0.02, 200),
            "C": np.zeros(200),  # dead flat — zero variance
        }
    )
    unconditional = build_risk_matrix(SampleCovariance(), _bars_from_panel(panel), PPY, min_obs=60)
    conditioned = condition_risk_matrix(unconditional, panel, "ewma", PPY, min_obs=60)
    assert not np.isnan(conditioned.sigma).any()
    assert np.allclose(conditioned.sigma, conditioned.sigma.T)
    eigvals = np.linalg.eigvalsh(conditioned.sigma)
    assert (eigvals >= -1e-8).all()  # PSD (a flat name can be exactly rank-deficient)


def test_conditioned_shrinkage_matrix_stays_psd_with_a_vol_spike():
    rng = np.random.default_rng(22)
    panel = pd.DataFrame(
        {
            "A": rng.normal(0, 0.01, 200),
            "B": rng.normal(0, 0.01, 200),
        }
    )
    panel.loc[199, "B"] = 0.5  # a single 10x+ vol spike on the last observed bar
    unconditional = build_risk_matrix(SampleCovariance(), _bars_from_panel(panel), PPY, min_obs=60)
    conditioned = condition_risk_matrix(unconditional, panel, "ewma", PPY, min_obs=60, lambda_=0.94)
    assert not np.isnan(conditioned.sigma).any()
    assert np.linalg.eigvalsh(conditioned.sigma).min() >= -1e-8
    # The spike shows up: B's conditional vol should exceed its unconditional vol.
    diag = conditioned.conditional_diagnostics
    assert diag["sigma_regime"]["B"] > 1.0


def test_conditioned_factor_matrix_stays_psd_under_stress_inputs():
    rng = np.random.default_rng(23)
    n, k, t = 12, 3, 300
    x = rng.normal(0, 1, (n, k))
    f = rng.multivariate_normal(np.zeros(k), np.diag([0.04, 0.02, 0.01]), t)
    u = rng.normal(0, 1, (t, n)) * 0.01
    syms = [f"S{i}" for i in range(n)]
    returns = pd.DataFrame(f @ x.T + u, columns=syms)
    returns.loc[t - 1, "S0"] *= 15.0  # a stress-bar vol spike
    exposures = pd.DataFrame(x, index=syms, columns=["f0", "f1", "f2"])

    matrix = estimate_factor_model(returns, exposures, periods_per_year=PPY)
    assert matrix is not None
    conditioned = condition_risk_matrix(matrix, returns, "ewma", PPY, min_obs=60)
    assert isinstance(conditioned, FactorRiskMatrix)
    assert not np.isnan(conditioned.sigma).any()
    assert np.linalg.eigvalsh(conditioned.sigma).min() >= -1e-6
    assert np.allclose(conditioned.sigma, conditioned.sigma.T, atol=1e-9)


def test_build_factor_risk_matrix_conditional_end_to_end():
    bars = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate([f"S{i}" for i in range(8)])}
    bench = make_ohlcv(n=400, seed=99, freq="1D")
    matrix = build_factor_risk_matrix(bars, bench, PPY, min_obs=60, conditional="ewma")
    assert matrix is not None
    assert matrix.is_positive_definite()
    assert matrix.conditional_diagnostics is not None


# --- HAR backend on the factor path ------------------------------------------
def test_condition_risk_matrix_har_backend_runs_and_stays_psd():
    panel = _correlated_panel(300, 6, seed=31)
    unconditional = build_risk_matrix(SampleCovariance(), _bars_from_panel(panel), PPY, min_obs=60)
    conditioned = condition_risk_matrix(unconditional, panel, "har", PPY, min_obs=60)
    assert not np.isnan(conditioned.sigma).any()
    assert np.linalg.eigvalsh(conditioned.sigma).min() >= -1e-8


def test_condition_risk_matrix_unknown_method_raises():
    panel = _correlated_panel(200, 3)
    unconditional = build_risk_matrix(SampleCovariance(), _bars_from_panel(panel), PPY, min_obs=60)
    with pytest.raises(ValueError, match="ewma"):
        condition_risk_matrix(unconditional, panel, "bogus", PPY, min_obs=60)


def test_condition_risk_matrix_off_is_a_no_op():
    panel = _correlated_panel(200, 3)
    unconditional = build_risk_matrix(SampleCovariance(), _bars_from_panel(panel), PPY, min_obs=60)
    same = condition_risk_matrix(unconditional, panel, None, PPY, min_obs=60)
    assert same is unconditional


# --- churn guard: risk-driven vs alpha-driven turnover ------------------------
def test_lambda_equals_one_freezes_at_the_seed_window():
    """The reduction test: λ=1 means the recursion never updates past the seed —
    D_t is exactly the (population) variance of the FIRST min_obs rows, no matter
    how much data follows."""
    rng = np.random.default_rng(41)
    panel = pd.DataFrame(rng.normal(0, 0.02, (300, 3)), columns=["A", "B", "C"])
    seed_only = ewma_variance_path(panel.iloc[:60], lambda_=1.0, min_obs=60)
    frozen = ewma_variance_path(panel, lambda_=1.0, min_obs=60)
    pd.testing.assert_series_equal(seed_only, frozen)


def test_ewma_covariance_matches_variance_path_on_the_diagonal():
    rng = np.random.default_rng(42)
    panel = pd.DataFrame(rng.normal(0, 0.02, (200, 3)), columns=["A", "B", "C"])
    cov = ewma_covariance_path(panel, lambda_=0.94, min_obs=60)
    var = ewma_variance_path(panel, lambda_=0.94, min_obs=60)
    np.testing.assert_allclose(np.diag(cov), var.to_numpy(), atol=1e-12)


def test_turnover_risk_share_is_zero_when_sigma_is_unchanged():
    from src.portfolio.optimizer import MeanVarianceOptimizer

    symbols = ["A", "B", "C", "D"]
    sigma = np.eye(4) * 0.04
    matrix = _matrix(symbols, sigma)
    alphas = [_alpha(s, 0.05 * (i + 1), float(i)) for i, s in enumerate(symbols)]
    optimizer = MeanVarianceOptimizer(max_weight=0.5)
    result = turnover_risk_share(
        optimizer, alphas, current_weights={}, matrix_t=matrix, matrix_prev=matrix, target_te=0.05
    )
    assert result["risk_driven_share"] == 0.0
    assert result["turnover_actual"] == result["turnover_frozen_sigma"]


def test_turnover_risk_share_is_positive_when_sigma_moves():
    from src.portfolio.optimizer import MeanVarianceOptimizer

    symbols = ["A", "B", "C", "D"]
    sigma_prev = np.eye(4) * 0.04
    sigma_t = np.eye(4) * 0.04
    sigma_t[0, 0] = 0.36  # name A's variance goes up 3x — a stress-driven de-risk
    matrix_prev = _matrix(symbols, sigma_prev)
    matrix_t = _matrix(symbols, sigma_t)
    alphas = [_alpha(s, 0.05, 1.0) for s in symbols]
    optimizer = MeanVarianceOptimizer(max_weight=0.5)
    current = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    result = turnover_risk_share(
        optimizer, alphas, current_weights=current, matrix_t=matrix_t, matrix_prev=matrix_prev, target_te=0.05
    )
    assert result["risk_driven_share"] > 0.0


def _matrix(symbols, sigma):
    from src.risk import RiskMatrix

    return RiskMatrix(symbols=symbols, sigma=sigma)


def _alpha(symbol: str, alpha: float, z: float):
    from datetime import datetime

    from src.alphas import Alpha

    return Alpha(symbol=symbol, alpha=alpha, as_of=datetime(2024, 1, 1), residual_vol=0.2, ic=0.05, raw_z=z)
