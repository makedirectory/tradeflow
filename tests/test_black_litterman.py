"""Tests for the Black-Litterman posterior.

Offline and deterministic. Works through the checklist in order: no-views ⇒
prior (and the optimizer round-trip through it), the calibration identity across an
IC x T_eff grid, propagation sign/magnitude to a correlated name, the Ω/τ limits,
the K=N degenerate blend identity, and the no-double-shrink audit trail at the
service layer.
"""

from datetime import datetime

import numpy as np
import pytest

from src.alphas.base import Alpha
from src.alphas.refine import level_shrink_factor
from src.marketdata.client import MarketDataClient
from src.portfolio.optimizer import MeanVarianceOptimizer
from src.portfolio.posterior import (
    black_litterman,
    black_litterman_from_ic,
    calibration_gap,
    expected_single_view_weight,
    view_variance,
)
from src.risk.base import RiskMatrix
from tests.fakes import DictMarketData, make_ohlcv

AS_OF = datetime(2024, 6, 1)
SYMS = ["A", "B", "C", "D"]


def _diag_risk(variances) -> RiskMatrix:
    symbols = SYMS[: len(variances)]
    return RiskMatrix(symbols, np.diag(variances))


# --- no views ⇒ prior ---------------------------------------------------------
def test_no_views_returns_prior_exactly():
    risk = _diag_risk([0.04, 0.09, 0.16, 0.25])
    post = black_litterman_from_ic({}, risk, ic=0.05, t_eff=60)
    assert post.mu_post == {s: 0.0 for s in SYMS}
    assert post.source == {s: "prior" for s in SYMS}


def test_no_views_optimizer_round_trip_reduces_to_benchmark():
    """empty P -> pi (here 0, residual space) -> optimize returns w = w_B (the
    benchmark-relative round-trip, now through the BL posterior path)."""
    w_b = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    risk = RiskMatrix(
        SYMS,
        np.array(
            [
                [0.20, 0.02, 0.01, 0.00],
                [0.02, 0.18, 0.01, 0.01],
                [0.01, 0.01, 0.22, 0.02],
                [0.00, 0.01, 0.02, 0.16],
            ]
        ),
    )
    post = black_litterman_from_ic({}, risk, ic=0.05, t_eff=60)
    alphas = [Alpha(s, post.mu_post[s], AS_OF, 0.2, 0.05, 0.0) for s in SYMS]
    result = MeanVarianceOptimizer(max_weight=1.0).optimize(
        alphas, risk, risk_aversion=1.0, benchmark_weights=w_b
    )
    assert result.feasible
    for s in SYMS:
        assert abs(result.weights.get(s, 0.0) - w_b[s]) < 1e-6


# --- calibration identity ------------------------------------------------------
@pytest.mark.parametrize(
    "ic,t_eff",
    [(0.05, 60), (0.10, 120), (0.03, 40), (0.08, 300), (0.02, 20), (0.15, 500)],
)
def test_calibration_identity_matches_level_shrink(ic, t_eff):
    """Single-view posterior on name A must equal the refinement's level-shrunk
    alpha exactly (to float precision) - the identity that decides Omega, not a
    preference."""
    risk = _diag_risk([0.04, 0.09, 0.16, 0.25])
    q_a = 0.05
    post = black_litterman_from_ic({"A": q_a}, risk, ic=ic, t_eff=t_eff)
    expected_mu = q_a * level_shrink_factor(ic, t_eff)
    assert post.mu_post["A"] == pytest.approx(expected_mu, rel=1e-9, abs=1e-12)
    assert calibration_gap(post) == pytest.approx(0.0, abs=1e-9)


def test_calibration_identity_anchors_match_known_shrink_values():
    """Anchors shared with the refinement's level_shrink_factor table."""
    r1 = RiskMatrix(["A"], np.diag([0.04]))
    post1 = black_litterman_from_ic({"A": 1.0}, r1, ic=0.05, t_eff=60)
    assert post1.mu_post["A"] == pytest.approx(0.13, abs=0.01)
    post2 = black_litterman_from_ic({"A": 1.0}, r1, ic=0.10, t_eff=120)
    assert post2.mu_post["A"] == pytest.approx(0.55, abs=0.01)


def test_expected_single_view_weight_is_level_shrink_factor():
    for ic, t_eff in [(0.04, 90), (0.12, 250)]:
        assert expected_single_view_weight(ic, t_eff) == level_shrink_factor(ic, t_eff)


# --- propagation ----------------------------------------------------------------
def test_propagation_sign_and_magnitude_to_correlated_name():
    rho_ab = 0.5
    var_a, var_b = 0.04, 0.09
    cov_ab = rho_ab * np.sqrt(var_a * var_b)
    risk = RiskMatrix(["A", "B"], np.array([[var_a, cov_ab], [cov_ab, var_b]]))
    post = black_litterman_from_ic({"A": 0.05}, risk, ic=0.05, t_eff=60)

    assert post.source == {"A": "view", "B": "propagated"}
    assert np.sign(post.mu_post["B"]) == np.sign(post.mu_post["A"])
    expected_ratio = rho_ab * np.sqrt(var_b / var_a)  # mu_B = rho_AB*(omega_B/omega_A)*mu_A
    actual_ratio = post.mu_post["B"] / post.mu_post["A"]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-9)


def test_uncorrelated_name_stays_at_prior():
    risk = RiskMatrix(["A", "B"], np.array([[0.04, 0.0], [0.0, 0.09]]))
    post = black_litterman_from_ic({"A": 0.05}, risk, ic=0.05, t_eff=60)
    assert post.mu_post["B"] == pytest.approx(0.0, abs=1e-12)
    assert post.source["B"] == "prior"


# --- limits (Omega -> inf / 0, tau up) ----------------------------------------
def test_omega_to_infinity_gives_prior():
    risk = _diag_risk([0.04, 0.09])
    post = black_litterman({"A": 0.05}, risk, omega={"A": 1e12}, tau=1.0)
    assert post.mu_post["A"] == pytest.approx(0.0, abs=1e-9)


def test_omega_to_zero_honors_view_exactly():
    risk = _diag_risk([0.04, 0.09])
    post = black_litterman({"A": 0.05}, risk, omega={"A": 1e-15}, tau=1.0)
    assert post.mu_post["A"] == pytest.approx(0.05, rel=1e-6)


def test_tau_up_moves_posterior_toward_view():
    risk = _diag_risk([0.04, 0.09])
    low = black_litterman({"A": 0.05}, risk, omega={"A": 0.01}, tau=1e-4)
    high = black_litterman({"A": 0.05}, risk, omega={"A": 0.01}, tau=1e4)
    assert abs(low.mu_post["A"]) < abs(high.mu_post["A"])
    assert high.mu_post["A"] == pytest.approx(0.05, rel=1e-3)


def test_black_litterman_tau_sensitivity_reported():
    risk = _diag_risk([0.04, 0.09])
    post = black_litterman({"A": 0.05}, risk, omega={"A": 0.01}, tau=1.0)
    assert "tau_half" in post.tau_sensitivity and "tau_double" in post.tau_sensitivity
    # tau/2 shrinks toward prior (0), 2*tau moves toward the view relative to tau=1.
    assert abs(post.tau_sensitivity["tau_half"]["A"]) < abs(post.mu_post["A"])
    assert abs(post.tau_sensitivity["tau_double"]["A"]) > abs(post.mu_post["A"])


def test_ic_zero_view_is_dropped_not_infinite():
    """An IC=0 view carries zero information - equivalent to no view at all, not a
    literal infinite Omega fed through the linear solve."""
    risk = _diag_risk([0.04, 0.09])
    post = black_litterman_from_ic({"A": 0.05}, risk, ic=0.0, t_eff=60)
    assert post.views == {}
    assert post.mu_post["A"] == 0.0
    assert post.source["A"] == "prior"
    assert not np.isfinite(view_variance(0.2, 0.0, 60))  # inf, as documented


# --- K = N degenerate case -------------------------------------------------------
def test_k_equals_n_degenerate_blend_matches_calibration_identity():
    risk = _diag_risk([0.04, 0.09, 0.16, 0.25])
    ic, t_eff = 0.06, 80
    views = {"A": 0.05, "B": -0.03, "C": 0.02, "D": 0.01}
    post = black_litterman_from_ic(views, risk, ic=ic, t_eff=t_eff)
    expected_w = expected_single_view_weight(ic, t_eff)
    for s in SYMS:
        assert post.mu_post[s] / views[s] == pytest.approx(expected_w, rel=1e-9)


# --- service integration: no double shrink ------------------------------------
def _universe():
    symbols = [f"S{i}" for i in range(8)]
    bars = {s: make_ohlcv(n=300, seed=i, freq="1D") for i, s in enumerate([*symbols, "SPY"])}
    return symbols, MarketDataClient(DictMarketData(bars))


def test_construct_portfolio_bl_requires_t_eff():
    from src.services import analysis

    symbols, dc = _universe()
    with pytest.raises(ValueError, match="posterior_t_eff"):
        analysis.construct_portfolio(dc, "ma_crossover", symbols, AS_OF, posterior="bl")


def test_construct_portfolio_bl_shrink_chain_applies_ic_uncertainty_once():
    from src.services import analysis

    symbols, dc = _universe()
    res = analysis.construct_portfolio(
        dc, "ma_crossover", symbols, AS_OF, posterior="bl", posterior_t_eff=60.0
    )
    if not res["feasible"]:
        pytest.skip("fixture produced no feasible portfolio")
    chain = res["shrink_chain"]
    ic_uncertainty_steps = [s for s in chain if s.get("step") == "ic_uncertainty"]
    bl_steps = [s for s in chain if s.get("step") == "bl"]
    assert len(ic_uncertainty_steps) == 0  # the refine step's level_shrink stayed off
    assert len(bl_steps) == 1  # IC-uncertainty is owned exactly once, by bl


def test_construct_portfolio_bl_reports_posterior_section():
    from src.services import analysis

    symbols, dc = _universe()
    res = analysis.construct_portfolio(
        dc, "ma_crossover", symbols, AS_OF, posterior="bl", posterior_t_eff=60.0
    )
    if not res["feasible"]:
        pytest.skip("fixture produced no feasible portfolio")
    post = res["posterior"]
    assert post["method"] == "bl"
    assert post["t_eff"] == 60.0
    assert "tau_sensitivity" in post
    per_name = {row["symbol"]: row for row in post["per_name"]}
    assert set(per_name) == set(res["weights"]) | set(per_name)  # every held name is in the table
    for row in per_name.values():
        assert row["source"] in ("view", "propagated", "prior")


def test_construct_portfolio_without_posterior_omits_section():
    from src.services import analysis

    symbols, dc = _universe()
    res = analysis.construct_portfolio(dc, "ma_crossover", symbols, AS_OF)
    assert res["posterior"] is None
