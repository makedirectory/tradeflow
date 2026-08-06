"""Tests for the risk model: shrinkage covariance + the quantities built on Σ.

Offline and deterministic. Covers positive-definiteness/invertibility, the
Ledoit–Wolf shrinkage bounds, the tracking-error closed form, the as-of leakage
guard, and the ragged-panel fallback.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tests.fakes import DictMarketData, make_ohlcv
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.risk import LedoitWolfCovariance, RiskMatrix, SampleCovariance, build_risk_matrix
from tradeflow.risk.base import build_return_panel
from tradeflow.services import analysis

AS_OF = datetime(2024, 6, 1)


def _correlated_returns(t: int, n: int, seed: int = 0) -> pd.DataFrame:
    """T×N returns with a common factor (so names genuinely co-move)."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 1, (t, 1))
    loadings = rng.uniform(0.3, 1.2, (1, n))
    x = factor @ loadings + rng.normal(0, 0.5, (t, n))
    return pd.DataFrame(x, columns=[f"S{i}" for i in range(n)])


# --- positive-definite / invertible -----------------------------------------
def test_shrinkage_is_positive_definite_when_t_below_n():
    r = _correlated_returns(t=20, n=30, seed=1)  # T < N: raw sample is singular
    sigma, delta = LedoitWolfCovariance().estimate(r)
    m = RiskMatrix(list(r.columns), sigma * 252)
    assert m.is_positive_definite()
    np.linalg.inv(m.sigma)  # Σ⁻¹ exists — the whole point for the optimizer
    assert 0.0 <= delta <= 1.0


def test_shrinkage_intensity_decreases_with_more_data():
    lw = LedoitWolfCovariance()

    # Heterogeneous correlations so the constant-correlation target is genuinely
    # biased — then δ → 0 as the sample becomes reliable.
    def panel(t):
        rng = np.random.default_rng(t)
        f1, f2 = rng.normal(0, 1, (t, 1)), rng.normal(0, 1, (t, 1))
        tight = f1 @ np.ones((1, 5)) + rng.normal(0, 0.3, (t, 5))
        loose = f2 @ (np.ones((1, 5)) * 0.3) + rng.normal(0, 1.0, (t, 5))
        return pd.DataFrame(np.hstack([tight, loose]), columns=[f"S{i}" for i in range(10)])

    small = lw.estimate(panel(15))[1]
    large = lw.estimate(panel(3000))[1]
    assert 0.0 <= large <= small <= 1.0
    assert large < 0.1 < small  # shrink hard when noisy, barely when reliable


# --- tracking error ----------------------------------------------------------
def test_tracking_error_matches_closed_form():
    sigma = np.array([[0.04, 0.006], [0.006, 0.09]])
    m = RiskMatrix(["A", "B"], sigma)
    w, w_bench = {"A": 0.7, "B": 0.3}, {"A": 0.5, "B": 0.5}
    active = np.array([0.2, -0.2])
    assert abs(m.tracking_error(w, w_bench) - float(np.sqrt(active @ sigma @ active))) < 1e-12


def test_mcr_aggregates_to_portfolio_volatility():
    sigma = np.array([[0.04, 0.006], [0.006, 0.09]])
    m = RiskMatrix(["A", "B"], sigma)
    w = {"A": 0.6, "B": 0.4}
    mcr = m.marginal_contribution_to_risk(w)
    total = sum(w[s] * mcr[s] for s in ["A", "B"])  # Σ wᵢ·MCRᵢ == σ_P
    assert abs(total - m.volatility(w)) < 1e-12


# --- implied (reverse-optimization) beta ----------------------------------------
def test_implied_beta_matches_closed_form():
    sigma = np.array([[0.04, 0.006, 0.002], [0.006, 0.09, 0.003], [0.002, 0.003, 0.05]])
    m = RiskMatrix(["A", "B", "C"], sigma)
    wb = {"A": 0.5, "B": 0.3, "C": 0.2}
    wb_vec = np.array([0.5, 0.3, 0.2])
    expected = (sigma @ wb_vec) / (wb_vec @ sigma @ wb_vec)
    beta = m.implied_beta(wb)
    assert np.allclose([beta[s] for s in ["A", "B", "C"]], expected)
    # β is the beta that makes the benchmark self-consistent: βᵀw_B == 1 exactly.
    assert abs(float(beta.reindex(["A", "B", "C"]) @ wb_vec) - 1.0) < 1e-9


def test_implied_beta_is_zero_without_benchmark_risk():
    sigma = np.array([[0.04, 0.0], [0.0, 0.09]])
    m = RiskMatrix(["A", "B"], sigma)
    beta = m.implied_beta({"A": 0.0, "B": 0.0})
    assert (beta == 0.0).all()


# --- as-of / leakage ---------------------------------------------------------
def test_risk_matrix_independent_of_post_as_of_bars():
    symbols = [f"S{i}" for i in range(6)]
    full = {s: make_ohlcv(n=300, seed=i, freq="1D") for i, s in enumerate(symbols)}
    cutoff = full["S0"].index[180]
    as_of = cutoff.to_pydatetime()
    truncated = {s: f.loc[f.index <= cutoff] for s, f in full.items()}

    a = analysis.compute_risk(MarketDataClient(DictMarketData(truncated)), symbols, as_of)
    b = analysis.compute_risk(MarketDataClient(DictMarketData(full)), symbols, as_of)
    assert a["top_risk_contributors"] == b["top_risk_contributors"]
    assert a["shrinkage"] == b["shrinkage"]
    assert a["condition_number"] == b["condition_number"]


# --- ragged panel ------------------------------------------------------------
def test_ragged_panel_uses_fallback_not_drop():
    # FFF has only a short history (< min_obs); it must survive with a fallback row,
    # not be dropped or NaN-propagated.
    bars = {s: make_ohlcv(n=200, seed=i, freq="1D") for i, s in enumerate(["AAA", "BBB", "CCC"])}
    bars["FFF"] = make_ohlcv(n=10, seed=9, freq="1D")
    panel, under = build_return_panel(bars, min_obs=60)
    assert under == ["FFF"]

    matrix = build_risk_matrix(LedoitWolfCovariance(), bars, periods_per_year=252.0, min_obs=60)
    assert "FFF" in matrix.symbols  # kept via fallback
    assert matrix.is_positive_definite()
    assert not np.isnan(matrix.sigma).any()
    # The fallback name is independent (zero covariance with the others).
    i = matrix.symbols.index("FFF")
    off_diag = np.delete(matrix.sigma[i], i)
    assert np.allclose(off_diag, 0.0)


def test_sample_covariance_runs_and_is_symmetric():
    r = _correlated_returns(t=120, n=5, seed=2)
    sigma, shrink = SampleCovariance().estimate(r)
    assert shrink is None
    assert np.allclose(sigma, sigma.T)


# --- factor model --------------------------------------------------------------
def test_factor_model_recovers_known_factor_covariance():
    from tradeflow.risk import FactorRiskMatrix, estimate_factor_model

    rng = np.random.default_rng(0)
    n, k, t = 20, 3, 3000
    x = rng.normal(0, 1, (n, k))
    f_true = np.diag([0.04, 0.02, 0.01])
    spec_true = rng.uniform(0.5, 1.5, n) * 1e-3
    f = rng.multivariate_normal(np.zeros(k), f_true, t)
    u = rng.normal(0, 1, (t, n)) * np.sqrt(spec_true)
    syms = [f"S{i}" for i in range(n)]
    returns = pd.DataFrame(f @ x.T + u, columns=syms)
    exposures = pd.DataFrame(x, index=syms, columns=["f0", "f1", "f2"])

    m = estimate_factor_model(returns, exposures, periods_per_year=1.0)
    assert isinstance(m, FactorRiskMatrix)
    assert m.is_positive_definite()
    assert np.allclose(np.diag(m.factor_cov), np.diag(f_true), atol=0.01)


def test_factor_and_specific_variance_sum_to_total():
    from tradeflow.risk import estimate_factor_model

    rng = np.random.default_rng(1)
    n, k, t = 10, 2, 500
    x = rng.normal(0, 1, (n, k))
    returns = pd.DataFrame(rng.normal(0, 0.01, (t, n)), columns=[f"S{i}" for i in range(n)])
    exposures = pd.DataFrame(x, index=returns.columns, columns=["a", "b"])
    m = estimate_factor_model(returns, exposures, periods_per_year=252.0)

    w = {s: 1.0 / n for s in returns.columns}
    assert abs((m.factor_variance(w) + m.specific_variance(w)) - m.variance(w)) < 1e-12


def test_compute_risk_factor_model_reports_split():
    symbols = [f"S{i}" for i in range(8)]
    data = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    r = analysis.compute_risk(MarketDataClient(DictMarketData(data)), symbols, AS_OF, model="factor")
    assert r["positive_definite"]
    assert "factor_risk_share" in r
    assert abs(r["factor_risk_share"] + r["specific_risk_share"] - 1.0) < 1e-9
    assert set(r["factor_names"]) == {"market", "momentum", "volatility", "size"}


def test_factor_exposures_subset_relaxes_history_requirement():
    """A subset without momentum keeps names the full four-factor build must drop."""
    from tradeflow.risk.exposures import build_factor_exposures

    # ~90 bars: enough for volatility/size (60-bar windows), far short of 12-1 momentum.
    bars = {s: make_ohlcv(n=90, seed=i, freq="1D") for i, s in enumerate(["AAA", "BBB", "CCC"])}
    bench = make_ohlcv(n=90, seed=9, freq="1D")

    full = build_factor_exposures(bars, bench)
    subset = build_factor_exposures(bars, bench, factors=["market", "volatility", "size"])
    assert full.empty
    assert list(subset.index) == ["AAA", "BBB", "CCC"]
    assert list(subset.columns) == ["market", "volatility", "size"]
    # Cross-sectionally standardized: mean 0, unit dispersion per factor.
    assert np.allclose(subset.mean().values, 0.0, atol=1e-12)
    assert np.allclose(subset.std(ddof=0).values, 1.0, atol=1e-12)


def test_factor_exposures_unknown_factor_raises():
    from tradeflow.risk.exposures import build_factor_exposures

    bars = {"AAA": make_ohlcv(n=200, seed=0, freq="1D")}
    with pytest.raises(ValueError, match="value"):
        build_factor_exposures(bars, None, factors=["market", "value"])


def test_factor_exposures_empty_factor_list_returns_empty():
    from tradeflow.risk.exposures import build_factor_exposures

    bars = {s: make_ohlcv(n=200, seed=i, freq="1D") for i, s in enumerate(["AAA", "BBB"])}
    assert build_factor_exposures(bars, None, factors=[]).empty


def test_factor_exposures_reuse_precomputed_betas():
    """A supplied beta Series is used verbatim for the market column (no re-regression)."""
    from tradeflow.risk.exposures import build_factor_exposures

    symbols = ["AAA", "BBB", "CCC"]
    bars = {s: make_ohlcv(n=90, seed=i, freq="1D") for i, s in enumerate(symbols)}
    bench = make_ohlcv(n=90, seed=9, freq="1D")
    betas = pd.Series({"AAA": 0.5, "BBB": 1.0, "CCC": 1.5})
    frame = build_factor_exposures(bars, bench, factors=["market"], betas=betas)
    # Standardized column must be the z-score of the supplied betas exactly.
    expected = (betas - betas.mean()) / betas.std(ddof=0)
    assert np.allclose(frame["market"].reindex(symbols).values, expected.values)
