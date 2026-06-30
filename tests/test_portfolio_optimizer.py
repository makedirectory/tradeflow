"""Tests for mean-variance portfolio construction (spec 008).

Offline and deterministic. Covers the closed-form optimum, target-TE calibration,
transfer-coefficient monotonicity, the no-trade band, turnover-from-w0,
infeasibility detection, and constraint preservation.
"""

from datetime import datetime

import numpy as np

from src.alphas.base import Alpha
from src.marketdata.client import MarketDataClient
from src.portfolio.optimizer import MeanVarianceOptimizer
from src.risk.base import RiskMatrix
from src.services import analysis
from tests.fakes import DictMarketData, make_ohlcv

AS_OF = datetime(2024, 6, 1)
SYMS = ["A", "B", "C", "D"]
# A fixed positive-definite covariance and alpha vector for the closed-form checks.
_L = np.array([[0.20, 0, 0, 0], [0.05, 0.18, 0, 0], [0.03, 0.04, 0.22, 0], [0.01, 0.02, 0.03, 0.16]])
SIGMA = _L @ _L.T
ALPHA = np.array([0.06, 0.02, -0.01, 0.04])


def _alphas() -> list:
    return [Alpha(s, float(ALPHA[i]), AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(SYMS)]


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
