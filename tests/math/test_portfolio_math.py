"""Math contract: mean-variance construction, the transfer coefficient, the proximal cost
step, the Bayesian posterior and the multi-period trading rate.

Independent references: closed forms written out with numpy linear algebra, a
general-purpose constrained solver (``scipy.optimize``) for the capped programme, brute
force 1-D minimisation for the cost step, and a value-iteration solve of the
single-asset control problem for the trading rate.

Related existing tests: ``tests/test_transfer_coefficient.py``,
``tests/test_portfolio_optimizer.py``, ``tests/test_black_litterman.py``,
``tests/test_multi_period_policy.py``.

Defects this module guards:

* the transfer coefficient was the plain Pearson correlation of alpha with the weights,
  which ignores the covariance, so ``tc * IR*`` did not reproduce the book's predicted IR;
* the Bayesian-posterior view variance not matching the refinement's own level shrink.
"""

import math
from datetime import datetime

import numpy as np
import pytest
from scipy import optimize

from quantred.alphas.base import Alpha
from quantred.costs.parametric import ParametricCostModel, cost_curvature
from quantred.portfolio import policy
from quantred.portfolio.optimizer import CostInputs, MeanVarianceOptimizer, long_only_cap_shortfall
from quantred.portfolio.posterior import (
    black_litterman,
    black_litterman_from_ic,
    calibration_gap,
    view_variance,
)
from quantred.risk.base import RiskMatrix

AS_OF = datetime(2024, 6, 1)
N = 8
SYMS = [f"N{i}" for i in range(N)]


def _world(seed=2, n=N):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0, 0.2, (n, 3))
    sigma = loadings @ loadings.T + np.diag(rng.uniform(0.02, 0.06, n))
    alpha = rng.normal(0.0, 0.03, n)
    syms = [f"N{i}" for i in range(n)]
    return syms, alpha, sigma, RiskMatrix(syms, sigma)


def _alphas(syms, alpha):
    return [Alpha(s, float(a), AS_OF, 0.2, 0.05, 0.0) for s, a in zip(syms, alpha)]


def _vec(result, syms):
    return np.array([result.weights.get(s, 0.0) for s in syms])


# --------------------------------------------------------------------------- #
# Unconstrained optimum and the diagnostics built on it
# --------------------------------------------------------------------------- #
def _interior_case():
    """Alpha = Sigma v with v orthogonal to the budget and benchmark-beta directions, so
    around an equal-weight benchmark the optimum w_B + v/(2 lambda) violates no bound."""
    syms, _, sigma, risk = _world(3)
    v = np.random.default_rng(4).normal(0, 1, N)
    ones = np.ones(N)
    basis = np.column_stack([ones, sigma @ ones])
    v -= basis @ np.linalg.lstsq(basis, v, rcond=None)[0]
    alpha = sigma @ v * 1e-3
    return syms, alpha, sigma, risk, np.full(N, 1.0 / N)


def test_unconstrained_optimum_weights_ir_star_and_transfer_coefficient_one():
    syms, alpha, sigma, risk, wb = _interior_case()
    lam = 1.7
    r = MeanVarianceOptimizer(max_weight=1.0).optimize(
        _alphas(syms, alpha), risk, risk_aversion=lam, benchmark_weights=dict(zip(syms, wb))
    )
    w = _vec(r, syms)
    assert w == pytest.approx(wb + np.linalg.solve(sigma, alpha) / (2 * lam), abs=1e-7)
    d = r.diagnostics
    ir_star = math.sqrt(alpha @ np.linalg.solve(sigma, alpha))
    assert d["ir_star"] == pytest.approx(ir_star, rel=1e-9)
    assert d["transfer_coefficient"] == pytest.approx(1.0, abs=1e-6)
    assert d["predicted_ir"] == pytest.approx(ir_star, rel=1e-6)
    assert d["optimal_tracking_error"] == pytest.approx(ir_star / (2 * lam), rel=1e-9)
    # looser: the optimum is reached by an iterative projection, converged to ~1e-6 relative
    assert d["predicted_tracking_error"] == pytest.approx(ir_star / (2 * lam), rel=1e-4)
    # value added IR*^2 / (4 lambda) is also alpha'w_a - lambda w_a' Sigma w_a at the optimum
    wa = w - wb
    assert d["value_added"] == pytest.approx(ir_star**2 / (4 * lam), rel=1e-9)
    assert d["value_added"] == pytest.approx(alpha @ wa - lam * wa @ sigma @ wa, rel=1e-5)


def test_target_tracking_error_sets_lambda_to_ir_star_over_twice_the_target():
    syms, alpha, sigma, risk, wb = _interior_case()
    target = 0.03
    r = MeanVarianceOptimizer(max_weight=1.0).optimize(
        _alphas(syms, alpha), risk, target_te=target, benchmark_weights=dict(zip(syms, wb))
    )
    ir_star = math.sqrt(alpha @ np.linalg.solve(sigma, alpha))
    assert r.diagnostics["risk_aversion"] == pytest.approx(ir_star / (2 * target), rel=1e-12)
    assert r.diagnostics["predicted_tracking_error"] == pytest.approx(target, rel=1e-5)


def test_capped_book_tc_is_the_covariance_cosine_and_tc_times_ir_star_is_the_predicted_ir():
    """With a cap binding, tc must be alpha'w / (IR* psi): the Pearson form agrees with
    neither this nor predicted_ir."""
    syms, alpha, sigma, risk = _world(5)
    alpha = alpha.copy()
    alpha[0], alpha[1] = 0.12, 0.10  # dominant forecasts so the cap binds
    r = MeanVarianceOptimizer(max_weight=0.2).optimize(_alphas(syms, alpha), risk, risk_aversion=2.0)
    w = _vec(r, syms)
    assert w.max() == pytest.approx(0.2, abs=1e-6)  # the cap is genuinely active
    ir_star = math.sqrt(alpha @ np.linalg.solve(sigma, alpha))
    psi = math.sqrt(w @ sigma @ w)
    d = r.diagnostics
    assert d["transfer_coefficient"] == pytest.approx(alpha @ w / (ir_star * psi), rel=1e-9)
    assert d["predicted_ir"] == pytest.approx(alpha @ w / psi, rel=1e-9)
    assert d["transfer_coefficient"] * ir_star == pytest.approx(d["predicted_ir"], rel=1e-9)
    assert d["ir_achieved"] == pytest.approx(d["predicted_ir"], rel=1e-9)
    assert 0 < d["transfer_coefficient"] < 1
    pearson = float(np.corrcoef(alpha, w)[0, 1])
    assert abs(d["transfer_coefficient"] - pearson) > 0.01  # the two definitions are distinguishable here


def test_capped_long_only_solution_matches_a_general_purpose_solver():
    syms, alpha, sigma, risk = _world(6)
    lam, cap = 3.0, 0.25
    r = MeanVarianceOptimizer(max_weight=cap).optimize(_alphas(syms, alpha), risk, risk_aversion=lam)
    ref = optimize.minimize(
        lambda w: -(alpha @ w - lam * w @ sigma @ w),
        np.full(N, 1.0 / N),
        method="SLSQP",
        bounds=[(0, cap)] * N,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert ref.success
    assert _vec(r, syms) == pytest.approx(ref.x, abs=2e-5)
    assert sum(r.weights.values()) == pytest.approx(1.0, abs=1e-8)


def test_benchmark_aware_solve_neutralises_alpha_and_reports_the_beta_split():
    syms, alpha, sigma, risk = _world(7)
    wb = np.full(N, 1.0 / N)
    r = MeanVarianceOptimizer(max_weight=1.0, min_weight=0.0).optimize(
        _alphas(syms, alpha), risk, risk_aversion=4.0, benchmark_weights=dict(zip(syms, wb))
    )
    beta = sigma @ wb / (wb @ sigma @ wb)
    neutral = alpha - beta * (alpha @ wb)
    assert neutral @ wb == pytest.approx(0.0, abs=1e-15)
    assert r.diagnostics["ir_star"] == pytest.approx(
        math.sqrt(neutral @ np.linalg.solve(sigma, neutral)), rel=1e-9
    )
    wa = _vec(r, syms) - wb
    d = r.diagnostics
    assert d["active_beta"] == pytest.approx(beta @ wa, abs=1e-9)
    # psi^2 = beta_a^2 sigma_B^2 + residual^2
    assert d["predicted_tracking_error"] ** 2 == pytest.approx(
        d["active_beta"] ** 2 * (wb @ sigma @ wb) + d["residual_risk"] ** 2, rel=1e-9
    )


def test_fully_invested_cap_feasibility_boundary():
    assert long_only_cap_shortfall(10, 0.1, None) is None  # 10 * 0.1 == 1: exactly fundable
    assert long_only_cap_shortfall(10, 0.099, None) is not None
    assert long_only_cap_shortfall(10, 0.3, 3) is not None  # 3 names x 0.3 < 1
    assert long_only_cap_shortfall(10, 0.34, 3) is None
    r = MeanVarianceOptimizer(max_weight=0.1).optimize(
        _alphas(SYMS[:5], np.ones(5)), RiskMatrix(SYMS[:5], np.eye(5)), risk_aversion=1.0
    )
    assert r.feasible is False and r.weights == {}


def test_market_neutral_book_is_dollar_neutral_and_within_the_leverage_cap():
    syms, alpha, sigma, risk = _world(8)
    r = MeanVarianceOptimizer(max_weight=0.3).optimize(
        _alphas(syms, alpha),
        risk,
        risk_aversion=2.0,
        book="market_neutral",
        short_max_weight=0.3,
        gross_leverage=1.5,
    )
    w = _vec(r, syms)
    assert w.sum() == pytest.approx(0.0, abs=1e-6)
    assert np.abs(w).sum() <= 1.5 + 1e-6
    assert w.min() >= -0.3 - 1e-9 and w.max() <= 0.3 + 1e-9
    assert alpha @ w > 0  # it is paid for the risk it takes


# --------------------------------------------------------------------------- #
# Cost terms
# --------------------------------------------------------------------------- #
def test_prox_step_is_the_argmin_of_its_stated_objective():
    """d* = argmin 0.5 (d-m)^2 + thr |d| + kthr |d|^1.5, checked by brute force."""
    grid = np.linspace(-3, 3, 600_001)
    for m_, thr, k in [(1.0, 0.2, 0.0), (1.0, 0.2, 0.5), (-2.0, 0.3, 1.2), (0.1, 0.2, 0.7), (0.25, 0.0, 2.0)]:
        obj = 0.5 * (grid - m_) ** 2 + thr * np.abs(grid) + k * np.abs(grid) ** 1.5
        brute = grid[np.argmin(obj)]
        got = MeanVarianceOptimizer._prox_step(np.array([m_]), np.array([thr]), np.array([k]))[0]
        assert got == pytest.approx(brute, abs=2e-5), (m_, thr, k)


def test_prox_step_is_a_soft_threshold_without_impact():
    m_ = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
    got = MeanVarianceOptimizer._prox_step(m_, np.full(5, 0.2), np.zeros(5))
    assert got == pytest.approx(np.sign(m_) * np.maximum(np.abs(m_) - 0.2, 0.0), abs=1e-15)


def test_cost_coefficients_follow_the_commission_spread_and_sqrt_impact_formulas():
    model = ParametricCostModel(commission_bps=2.0, default_spread_bps=10.0, impact_eta=0.4)
    inputs = CostInputs(
        spread={"A": 0.0004}, adv_dollar={"A": 5e6, "B": 2e6}, daily_vol={"A": 0.02, "B": 0.03}
    )
    h, capital = 1.0 / 12.0, 1e6
    c, k = MeanVarianceOptimizer._cost_coefficients(model, inputs, capital, h, ["A", "B"])
    # one-way linear rate = commission + spread/2, annualised by the holding period;
    # a name with no quoted spread falls back to the model's own
    assert c == pytest.approx([(2e-4 + 0.0004 / 2) / h, (2e-4 + 0.0010 / 2) / h], rel=1e-12)
    # impact: eta * sigma * sqrt(capital / ADV$) per unit of |dw|^1.5
    assert k == pytest.approx(
        [0.4 * 0.02 * math.sqrt(1e6 / 5e6) / h, 0.4 * 0.03 * math.sqrt(1e6 / 2e6) / h], rel=1e-12
    )
    _, k_none = MeanVarianceOptimizer._cost_coefficients(model, inputs, None, h, ["A", "B"])
    assert not k_none.any()  # no capital, no impact term: linear first


def test_cost_aware_diagnostics_charge_exactly_what_the_coefficients_say():
    syms, alpha, sigma, risk = _world(9)
    model = ParametricCostModel()
    inputs = CostInputs(
        spread={s: 0.0006 for s in syms},
        adv_dollar={s: 3e6 for s in syms},
        daily_vol={s: 0.02 for s in syms},
    )
    w0 = {s: 1.0 / N for s in syms}
    h, capital = 1.0 / 12.0, 5e5
    r = MeanVarianceOptimizer(max_weight=0.3).optimize(
        _alphas(syms, alpha),
        risk,
        risk_aversion=2.0,
        current_weights=w0,
        cost_model=model,
        cost_inputs=inputs,
        capital=capital,
        holding_period_years=h,
    )
    w, dw = _vec(r, syms), _vec(r, syms) - 1.0 / N
    c = (1e-4 + 0.0006 / 2) / h
    k = 0.3 * 0.02 * math.sqrt(capital / 3e6) / h
    d = r.diagnostics
    assert d["linear_cost"] == pytest.approx(c * np.abs(dw).sum(), rel=1e-9)
    assert d["impact_cost"] == pytest.approx(k * (np.abs(dw) ** 1.5).sum(), rel=1e-9)
    assert d["round_trip_cost"] == pytest.approx(
        2 * (c * np.abs(w).sum() + k * (np.abs(w) ** 1.5).sum()), rel=1e-9
    )
    assert d["expected_active_return_net"] == pytest.approx(alpha @ w - d["round_trip_cost"], rel=1e-9)


def test_cost_curvature_closed_form_and_undefined_cases():
    # d2/dw2 of k |w|^1.5 is 0.75 k |w|^-0.5
    assert cost_curvature(0.8, 0.04) == pytest.approx(0.75 * 0.8 / 0.2, rel=1e-12)
    assert cost_curvature(0.0, 0.04) is None and cost_curvature(0.8, 0.0) is None


def test_impact_coefficient_is_zero_without_the_inputs_it_needs():
    model = ParametricCostModel()
    assert model.impact_coefficient(0.02, 0.0, 1e6) == 0.0
    assert model.impact_coefficient(0.02, 1e6, 0.0) == 0.0
    assert model.impact_coefficient(float("nan"), 1e6, 1e6) == 0.0
    assert ParametricCostModel(linear_impact=True).impact_coefficient(0.02, 1e6, 1e6) == 0.0


# --------------------------------------------------------------------------- #
# Bayesian posterior
# --------------------------------------------------------------------------- #
def test_black_litterman_posterior_matches_the_matrix_formula():
    syms, _, sigma, risk = _world(10, 6)
    views = {syms[0]: 0.04, syms[2]: -0.02, syms[3]: 0.03}
    omega = {syms[0]: 0.002, syms[2]: 0.004, syms[3]: 0.003}
    tau = 0.25
    p = np.zeros((3, 6))
    for row, s in enumerate(views):
        p[row, syms.index(s)] = 1.0
    q = np.array(list(views.values()))
    om = np.diag(list(omega.values()))
    mu = tau * sigma @ p.T @ np.linalg.inv(p @ (tau * sigma) @ p.T + om) @ q
    post = black_litterman(views, risk, omega, tau)
    assert np.array([post.mu_post[s] for s in syms]) == pytest.approx(mu, rel=1e-10)
    assert [post.source[s] for s in syms][0] == "view" and "propagated" in post.source.values()
    # tau sensitivity is the same formula at tau/2 and 2 tau
    half = tau / 2 * sigma @ p.T @ np.linalg.inv(p @ (tau / 2 * sigma) @ p.T + om) @ q
    assert np.array([post.tau_sensitivity["tau_half"][s] for s in syms]) == pytest.approx(half, rel=1e-10)


def test_black_litterman_without_views_is_the_zero_prior():
    syms, _, _, risk = _world(11, 4)
    post = black_litterman({}, risk, {}, 0.5)
    assert set(post.mu_post.values()) == {0.0} and set(post.source.values()) == {"prior"}


def test_single_view_calibration_reproduces_the_level_shrink_weight():
    """One covered name, no information elsewhere, tau = 1/T_eff: the posterior is the
    view scaled by g/(g+1) with g = T_eff * IC^2, and the derived view variance is
    omega^2 / (T_eff^2 IC^2)."""
    sigma = np.array([[0.09, 0.02], [0.02, 0.04]])
    risk = RiskMatrix(["A", "B"], sigma)
    ic, t_eff, q = 0.08, 150.0, 0.05
    g = t_eff * ic**2
    assert view_variance(0.3, ic, t_eff) == pytest.approx(0.09 / (t_eff**2 * ic**2), rel=1e-12)
    post = black_litterman_from_ic({"A": q}, risk, ic, t_eff)
    assert post.mu_post["A"] / q == pytest.approx(g / (g + 1), rel=1e-10)
    assert calibration_gap(post) == pytest.approx(0.0, abs=1e-10)
    # B is not covered: it inherits through the covariance, scaled by sigma_AB / sigma_AA
    assert post.mu_post["B"] == pytest.approx(post.mu_post["A"] * 0.02 / 0.09, rel=1e-10)


def test_uninformative_or_nan_views_are_dropped_not_propagated():
    risk = RiskMatrix(["A", "B"], np.eye(2) * 0.04)
    assert view_variance(0.2, 0.0, 100) == math.inf and view_variance(0.2, 0.1, 0) == math.inf
    post = black_litterman_from_ic({"A": 0.05, "B": float("nan")}, risk, 0.0, 100.0)
    assert all(v == 0.0 for v in post.mu_post.values())


# --------------------------------------------------------------------------- #
# Multi-period trading rate and aim discount
# --------------------------------------------------------------------------- #
def _kappa_by_value_iteration(s, c2):
    """Single-asset control: cost-to-go A w^2; each step minimise
    (s + A)(w + d)^2 + (c2/2) d^2 over the trade d."""
    a = 0.0
    for _ in range(100_000):
        ell = s + a
        nxt = ell * (c2 / 2) / (ell + c2 / 2)
        if abs(nxt - a) < 1e-15:
            break
        a = nxt
    ell = s + a
    return 2 * ell / (2 * ell + c2)


@pytest.mark.parametrize(
    "risk_aversion, variance, c2", [(1.0, 0.04, 0.5), (5.0, 0.09, 0.02), (0.5, 0.01, 3.0)]
)
def test_trading_rate_matches_value_iteration_and_the_riccati_root(risk_aversion, variance, c2):
    s = risk_aversion * variance
    kappa = policy.derive_kappa(risk_aversion, variance, c2)
    assert kappa == pytest.approx(_kappa_by_value_iteration(s, c2), rel=1e-9)
    root = max(np.roots([1.0, -s, -s * c2 / 2]))  # L^2 - sL - s c2/2 = 0
    assert kappa == pytest.approx(2 * root / (2 * root + c2), rel=1e-9)
    assert 0 < kappa < 1


def test_trading_rate_limits_and_undefined_cases():
    assert policy.derive_kappa(1.0, 0.04, 1e-9) == pytest.approx(1.0, abs=1e-4)
    assert policy.derive_kappa(1.0, 0.04, 1e9) < 1e-3
    assert policy.derive_kappa(1.0, 0.04, None) is None and policy.derive_kappa(1.0, 0.04, 0.0) is None
    assert policy.derive_kappa(0.0, 0.04, 1.0) is None


def _aim_discount_by_bellman(s, c2, delta):
    """Solve the control problem by value iteration, sharing no formula with the package.

    Each period choose the new position x (trade d = x - w) to maximise
    ``f x - s x^2 - (c2/2) d^2 + V(x, delta f)`` with ``V(w, f) = -A w^2 + B w f - C f^2``.
    A is iterated to its fixed point first (it does not depend on the signal). Then B is
    iterated by *numerically* maximising over x at the four corners of a (w, f) square and
    reading the w*f coefficient as a mixed second difference, so no closed form for the
    maximiser is used. The policy's weight on f is the finite-difference d x*/d f at w = 0;
    the aim portfolio is f/(2s) (the one-period Markowitz solve), so the discount is that
    weight over (kappa * f/(2s)).
    """
    from scipy.optimize import minimize_scalar

    a = 0.0
    for _ in range(100_000):
        ell = s + a
        nxt = ell * (c2 / 2) / (ell + c2 / 2)
        if abs(nxt - a) < 1e-15:
            break
        a = nxt
    kappa = 2 * (s + a) / (2 * (s + a) + c2)

    def best(w, f, b):
        def neg(x):
            return -(f * x - s * x * x - (c2 / 2) * (x - w) ** 2 - a * x * x + b * delta * x * f)

        res = minimize_scalar(neg, bounds=(-60, 60), method="bounded", options={"xatol": 1e-13})
        return res.x, -res.fun

    b = 0.0
    for _ in range(500):
        v = {(w, f): best(w, f, b)[1] for w in (-1.0, 1.0) for f in (-1.0, 1.0)}
        # mixed second difference of V isolates the w*f coefficient: B = (V++ - V+- - V-+ + V--)/4
        b_new = (v[(1.0, 1.0)] - v[(1.0, -1.0)] - v[(-1.0, 1.0)] + v[(-1.0, -1.0)]) / 4
        if abs(b_new - b) < 1e-13:
            break
        b = b_new
    h = 1e-3
    weight_on_f = (best(0.0, h, b)[0] - best(0.0, -h, b)[0]) / (2 * h)
    return kappa, weight_on_f * 2 * s / kappa


@pytest.mark.parametrize("s, c2, phi", [(0.08, 0.5, 0.35), (0.02, 0.3, 1.2), (0.3, 0.05, 0.05)])
def test_aim_discount_matches_a_bellman_value_iteration(s, c2, phi):
    """The discount on a decaying signal comes from solving the dynamic program, not from
    restating ``kappa / (1 - delta (1 - kappa))``: the value function is iterated to its
    fixed point and the closed form is read against the resulting policy. The common
    small-phi shortcut kappa/(kappa+phi) is also shown to be off at large phi."""
    kappa_ref, discount_ref = _aim_discount_by_bellman(s, c2, math.exp(-phi))
    kappa = policy.derive_kappa(1.0, s, c2)
    assert kappa == pytest.approx(kappa_ref, rel=1e-9)
    assert policy.discount_factor(kappa, phi) == pytest.approx(discount_ref, rel=1e-6)
    if phi >= 1.0:
        assert abs(kappa / (kappa + phi) - discount_ref) > 0.03 * discount_ref
    assert policy.discount_factor(kappa, 0.0) == 1.0 and policy.discount_factor(0.0, phi) == 0.0


def test_decay_units_and_half_life_conversions():
    assert policy.phi_from_half_life(10.0) == pytest.approx(math.log(2) / 10)
    assert policy.phi_from_half_life(0.2) == pytest.approx(math.log(2) / 1.0)  # floored at one rebalance
    assert policy.phi_from_half_life(math.inf) == 0.0
    assert policy.half_life_in_rebalance_units(42.0, 21.0) == 2.0
    assert policy.trading_half_life(0.25) == pytest.approx(math.log(2) / 0.25)
    assert math.isinf(policy.trading_half_life(0.0))
