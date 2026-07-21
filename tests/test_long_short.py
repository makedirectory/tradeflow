"""Tests for long/short portfolio construction (Spec 018).

Covers, in the spec's own §6 order: the reduction (book="long_only" is byte-for-byte
the pre-018 solve), the market-neutral book's neutrality/leverage/box invariants, the
dominance of the relaxed constraint (measured via the transfer coefficient - see the
docstring on ``test_dominance_*`` for why that metric and not a raw IR ratio), the
long-only size bias vs long/short, carry (borrow) monotonically shrinking shorts, the
mandatory gross-leverage/short-cap guards, and a KKT optimality certificate over the
box + budget + leverage + borrow program.

Offline and deterministic. Fixed SIGMA/ALPHA fixtures shared with
tests/test_portfolio_optimizer.py and tests/test_cost_aware.py.
"""

from datetime import datetime

import numpy as np
import pytest

from src.alphas.base import Alpha
from src.costs.parametric import ParametricCostModel
from src.portfolio.optimizer import CostInputs, MeanVarianceOptimizer
from src.risk.base import RiskMatrix

AS_OF = datetime(2024, 6, 1)
SYMS = ["A", "B", "C", "D"]
_L = np.array([[0.20, 0, 0, 0], [0.05, 0.18, 0, 0], [0.03, 0.04, 0.22, 0], [0.01, 0.02, 0.03, 0.16]])
SIGMA = _L @ _L.T
ALPHA = np.array([0.06, 0.02, -0.01, 0.04])
LAM = 2.0
H = 1.0 / 12.0
W_B = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}


def _alphas_from(vec) -> list:
    return [Alpha(s, float(vec[i]), AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(SYMS)]


def _alphas() -> list:
    return _alphas_from(ALPHA)


def _risk() -> RiskMatrix:
    return RiskMatrix(SYMS, SIGMA)


def _vec(result) -> np.ndarray:
    return np.array([result.weights.get(s, 0.0) for s in SYMS])


# --- reduction: book="long_only" is the untouched pre-018 code path -----------
def test_book_long_only_matches_omitting_book():
    default = MeanVarianceOptimizer(max_weight=0.5).optimize(_alphas(), _risk(), risk_aversion=LAM)
    explicit = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), risk_aversion=LAM, book="long_only"
    )
    assert default.weights == explicit.weights
    assert default.diagnostics == explicit.diagnostics


def test_book_long_only_matches_the_closed_form():
    # Same closed-form check as test_portfolio_optimizer.py's unconstrained test -
    # book="long_only" must still reproduce it exactly (008/016's own math untouched).
    result = MeanVarianceOptimizer(max_weight=1.0).optimize(
        _alphas(), _risk(), risk_aversion=LAM, book="long_only"
    )
    expected = (np.linalg.inv(SIGMA) @ ALPHA) / (2 * LAM)
    got = np.array([result.unconstrained_weights[s] for s in SYMS])
    assert np.allclose(got, expected)


def test_unknown_book_raises():
    with pytest.raises(ValueError):
        MeanVarianceOptimizer(max_weight=0.5).optimize(_alphas(), _risk(), risk_aversion=LAM, book="130/30")


# --- mandatory guards (hidden factor 1) ---------------------------------------
def test_market_neutral_requires_gross_leverage():
    with pytest.raises(ValueError, match="gross_leverage"):
        MeanVarianceOptimizer(max_weight=0.5).optimize(
            _alphas(), _risk(), risk_aversion=LAM, book="market_neutral", short_max_weight=0.5
        )


def test_market_neutral_requires_positive_short_max_weight():
    with pytest.raises(ValueError, match="short_max_weight"):
        MeanVarianceOptimizer(max_weight=0.5).optimize(
            _alphas(), _risk(), risk_aversion=LAM, book="market_neutral", gross_leverage=1.0
        )


# --- neutrality + leverage + box invariants -----------------------------------
def test_market_neutral_satisfies_dollar_neutrality_leverage_and_box():
    result = MeanVarianceOptimizer(max_weight=0.5).optimize(
        _alphas(), _risk(), risk_aversion=LAM, book="market_neutral", short_max_weight=0.5, gross_leverage=1.0
    )
    w = _vec(result)
    assert abs(w.sum()) < 1e-8
    assert np.sum(np.abs(w)) <= 1.0 + 1e-6
    assert np.all(w >= -0.5 - 1e-9) and np.all(w <= 0.5 + 1e-9)
    assert result.diagnostics["dollar_neutral_residual"] < 1e-8
    assert result.diagnostics["gross_leverage"] == pytest.approx(np.sum(np.abs(w)))
    assert result.diagnostics["book"] == "market_neutral"


def test_gross_leverage_binds_when_tight_and_not_when_loose():
    opt = MeanVarianceOptimizer(max_weight=0.5)
    loose = opt.optimize(
        _alphas(),
        _risk(),
        risk_aversion=0.5,
        book="market_neutral",
        short_max_weight=0.5,
        gross_leverage=100.0,
    )
    tight = opt.optimize(
        _alphas(), _risk(), risk_aversion=0.5, book="market_neutral", short_max_weight=0.5, gross_leverage=0.3
    )
    assert loose.diagnostics["gross_leverage"] > 0.3  # unconstrained-by-leverage natural level
    assert tight.diagnostics["gross_leverage"] == pytest.approx(0.3, abs=1e-6)  # cap actually binds


def test_gross_leverage_is_monotone_in_the_cap():
    opt = MeanVarianceOptimizer(max_weight=0.5)
    caps = [0.2, 0.5, 1.0, 100.0]
    realized = [
        opt.optimize(
            _alphas(),
            _risk(),
            risk_aversion=0.5,
            book="market_neutral",
            short_max_weight=0.5,
            gross_leverage=cap,
        ).diagnostics["gross_leverage"]
        for cap in caps
    ]
    assert all(r <= cap + 1e-6 for r, cap in zip(realized, caps))  # never exceeds its own cap
    assert all(a <= b + 1e-9 for a, b in zip(realized, realized[1:]))  # non-decreasing as the cap loosens


# --- dominance (spec 018 §6) ---------------------------------------------------
def test_dominance_transfer_coefficient_long_short_at_least_long_only():
    """On identical alpha/Σ/λ, relaxing the long-only floor can only help - but
    ``predicted_ir`` (expected active return / the BOOK'S OWN realized TE) isn't the
    right metric to compare directly: market-neutral's wider box lets it take a
    bigger, riskier active bet at the same λ, which can raise its *realized* TE
    enough that the ratio doesn't move as expected. ``ir_star`` (the fully
    unconstrained IR bound) is identical for both books since it depends only on
    alpha/Σ, not the box/budget; ``transfer_coefficient`` (equivalently
    ``ir_achieved = tc * ir_star``) measures how much of that SHARED bound survives
    each book's own constraint set - the apples-to-apples dominance metric.
    """
    hostile = ALPHA.copy()
    hostile[3] = -0.3  # symbol D, w_B=0.1: a small-benchmark-weight name made unattractive
    opt = MeanVarianceOptimizer(max_weight=0.5)
    lo = opt.optimize(_alphas_from(hostile), _risk(), target_te=0.05, benchmark_weights=W_B)
    ls = opt.optimize(
        _alphas_from(hostile),
        _risk(),
        target_te=0.05,
        benchmark_weights=W_B,
        book="market_neutral",
        short_max_weight=0.5,
        gross_leverage=2.0,
    )
    assert lo.diagnostics["ir_star"] == pytest.approx(ls.diagnostics["ir_star"])  # same λ-calibration input
    assert lo.weights.get("D", 0.0) < 1e-9  # long-only: D pinned to 0 (the floor the spec relaxes)
    assert ls.weights["D"] < 0  # long/short: D can be genuinely shorted instead
    assert ls.diagnostics["transfer_coefficient"] >= lo.diagnostics["transfer_coefficient"] - 1e-9
    assert ls.diagnostics["ir_achieved"] >= lo.diagnostics["ir_achieved"] - 1e-9


# --- size bias (spec 018 §6, §2 goal) ------------------------------------------
def test_long_only_shows_more_negative_size_exposure_than_long_short():
    """A cap-weighted w_B (log-normal caps) with alphas drawn independent of size
    (so any size tilt is *incidental*, not chosen): the long-only book's forced
    underweight in the smallest names should show up as a more negative size
    exposure than the long/short book's, which can short those names directly
    instead of being pinned at their tiny w_B floor (spec 018 §1's "structural
    negative size bias [long-only books] never chose").
    """
    rng = np.random.default_rng(6)
    n = 20
    syms = [f"S{i}" for i in range(n)]
    caps = np.exp(rng.uniform(0, 4, n))
    wb = caps / caps.sum()
    size = np.log(caps)
    size = (size - size.mean()) / size.std()  # cross-sectionally standardized, like src/risk/exposures.py
    alpha = rng.normal(0, 0.05, n)
    chol = rng.normal(0, 0.05, (n, n))
    sigma = chol @ chol.T + np.eye(n) * 0.02

    risk = RiskMatrix(syms, sigma)
    alphas = [Alpha(s, float(alpha[i]), AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(syms)]
    wb_dict = {s: float(wb[i]) for i, s in enumerate(syms)}
    opt = MeanVarianceOptimizer(max_weight=0.5)
    lo = opt.optimize(alphas, risk, target_te=0.05, benchmark_weights=wb_dict)
    ls = opt.optimize(
        alphas,
        risk,
        target_te=0.05,
        benchmark_weights=wb_dict,
        book="market_neutral",
        short_max_weight=0.5,
        gross_leverage=3.0,
    )

    def size_exposure(result) -> float:
        w = np.array([result.weights.get(s, 0.0) for s in syms])
        return float(size @ (w - wb))

    lo_size, ls_size = size_exposure(lo), size_exposure(ls)
    assert lo_size < -0.05  # long-only: a real negative size tilt
    assert ls_size > lo_size + 0.05  # long/short: materially less negative


# --- carry bites (spec 018 §6) --------------------------------------------------
def test_raising_borrow_monotonically_shrinks_the_short_book():
    opt = MeanVarianceOptimizer(max_weight=0.5)
    ci = CostInputs(
        spread={s: 0.0005 for s in SYMS},
        adv_dollar={s: 1e12 for s in SYMS},
        daily_vol={s: 0.02 for s in SYMS},
    )

    def short_mass(borrow_bps: float) -> float:
        model = ParametricCostModel(annual_borrow_bps=borrow_bps)
        result = opt.optimize(
            _alphas(),
            _risk(),
            risk_aversion=0.5,
            book="market_neutral",
            short_max_weight=0.5,
            gross_leverage=2.0,
            cost_model=model,
            cost_inputs=ci,
            holding_period_years=H,
        )
        w = _vec(result)
        return float(np.sum(np.maximum(-w, 0.0)))

    masses = [short_mass(bps) for bps in (0.0, 50.0, 500.0, 5000.0)]
    assert all(a >= b - 1e-9 for a, b in zip(masses, masses[1:]))  # monotone non-increasing
    assert masses[0] > 0.5  # a real short book with no borrow
    assert masses[-1] < 1e-6  # extreme borrow: shorting (hence, under Σw=0, the whole book) is priced out


def test_borrow_cost_diagnostic_matches_independent_sum():
    model = ParametricCostModel(annual_borrow_bps=200.0)
    ci = CostInputs(
        spread={s: 0.0005 for s in SYMS},
        adv_dollar={s: 1e12 for s in SYMS},
        daily_vol={s: 0.02 for s in SYMS},
    )
    opt = MeanVarianceOptimizer(max_weight=0.5)
    result = opt.optimize(
        _alphas(),
        _risk(),
        risk_aversion=0.5,
        book="market_neutral",
        short_max_weight=0.5,
        gross_leverage=2.0,
        cost_model=model,
        cost_inputs=ci,
        holding_period_years=H,
    )
    w = _vec(result)
    borrow = MeanVarianceOptimizer._borrow_coefficients(model, ci, SYMS)
    expected = float(np.sum(borrow * np.maximum(-w, 0.0)))
    assert result.diagnostics["borrow_cost"] == pytest.approx(expected)


def test_borrow_rate_override_and_default():
    model = ParametricCostModel(annual_borrow_bps=50.0)
    assert model.borrow_rate(None) == pytest.approx(0.005)
    assert model.borrow_rate(0.10) == pytest.approx(0.10)
    from src.costs.base import CostModel, TradeCost

    class Null(CostModel):
        def cost(self, trade):  # pragma: no cover - trivial
            return TradeCost(0, 0, 0, 0, capped=False)

    assert Null().borrow_rate(0.10) == 0.0  # a model with no borrow ignores the override


# --- prox_step_short primitive --------------------------------------------------
def test_prox_step_short_matches_brute_force_minimizer():
    grid = np.linspace(-1.0, 1.0, 20001)
    cases = [
        (0.3, 0.05, 0.0, 0.0, 0.0, 0.0),
        (0.3, 0.05, 0.4, 0.1, 0.0, 0.2),
        (-0.4, 0.02, 0.0, 0.15, 0.05, -0.1),
        (0.5, 0.03, 0.2, 0.05, 0.1, -0.2),
        (-0.2, 0.01, 0.0, 0.0, 0.08, 0.3),
    ]
    for m, thr, kthr, borrow, mu, w0 in cases:
        dw = grid - w0
        obj = (
            0.5 * (grid - (m + w0)) ** 2
            + thr * np.abs(dw)
            + kthr * np.abs(dw) ** 1.5
            + borrow * np.maximum(-grid, 0.0)
            + mu * np.abs(grid)
        )
        brute = grid[int(np.argmin(obj))]
        got = MeanVarianceOptimizer._prox_step_short(
            np.array([m]), np.array([thr]), np.array([kthr]), np.array([borrow]), mu, np.array([w0])
        )[0]
        assert abs(got - brute) < 2e-3


def test_prox_step_short_reduces_without_borrow_or_leverage():
    # borrow = mu = 0: the six candidates collapse to _prox_step's two branches.
    m = np.array([-0.3, -0.02, 0.0, 0.02, 0.3])
    thr = np.full_like(m, 0.05)
    kthr = np.full_like(m, 0.1)
    w0 = np.array([0.1, -0.1, 0.05, 0.0, -0.2])
    got = MeanVarianceOptimizer._prox_step_short(m, thr, kthr, np.zeros_like(m), 0.0, w0)
    expected = w0 + MeanVarianceOptimizer._prox_step(m, thr, kthr)
    assert np.allclose(got, expected)


# --- KKT optimality certificate (spec 018 §6/§7) --------------------------------
def test_no_feasible_perturbation_beats_the_market_neutral_optimum():
    """Verifies the FOUND point is optimal under box + budget + leverage + borrow,
    empirically certifying the nested (nu, tau) bisection (spec §7's open risk) -
    not just that it converges, but that it converges to the right answer."""
    model = ParametricCostModel(annual_borrow_bps=200.0)
    ci = CostInputs(
        spread={s: 0.001 for s in SYMS}, adv_dollar={s: 1e12 for s in SYMS}, daily_vol={s: 0.02 for s in SYMS}
    )
    opt = MeanVarianceOptimizer(max_weight=0.5)
    lam = 0.5
    gross_cap = 0.3
    result = opt.optimize(
        _alphas(),
        _risk(),
        risk_aversion=lam,
        book="market_neutral",
        short_max_weight=0.5,
        gross_leverage=gross_cap,
        cost_model=model,
        cost_inputs=ci,
        holding_period_years=H,
    )
    w = _vec(result)
    assert result.diagnostics["gross_leverage"] == pytest.approx(
        gross_cap, abs=1e-6
    )  # confirms it binds here

    c_lin, k_imp = MeanVarianceOptimizer._cost_coefficients(model, ci, None, H, SYMS)
    borrow = MeanVarianceOptimizer._borrow_coefficients(model, ci, SYMS)
    w0 = np.zeros(4)

    def util(x):
        dw = x - w0
        return (
            ALPHA @ x
            - lam * (x @ SIGMA @ x)
            - np.sum(c_lin * np.abs(dw))
            - np.sum(k_imp * np.abs(dw) ** 1.5)
            - np.sum(borrow * np.maximum(-x, 0.0))
        )

    u_opt = util(w)
    rng = np.random.default_rng(0)
    beats = 0
    for _ in range(5000):
        i, j = rng.integers(0, 4, 2)
        eps = rng.uniform(0, 0.02)
        wp = w.copy()
        wp[i] += eps
        wp[j] -= eps  # budget-preserving (Σw=0 unchanged)
        box_ok = bool(np.all(wp <= 0.5 + 1e-12) and np.all(wp >= -0.5 - 1e-12))
        leverage_ok = bool(np.sum(np.abs(wp)) <= gross_cap + 1e-9)
        if box_ok and leverage_ok and util(wp) > u_opt + 1e-8:
            beats += 1
    assert beats == 0
