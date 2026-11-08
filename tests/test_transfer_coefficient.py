"""Transfer coefficient in the risk metric, so ``tc * IR*`` is the book's predicted IR.

The earlier form was a plain Pearson correlation between alpha and the weights, which
ignores the covariance: on a capped ten-name book it read near 0.9 and implied an
achieved IR several times the book's own predicted IR.
"""

import numpy as np
import pytest

from quantred.alphas.base import Alpha
from quantred.portfolio.optimizer import MeanVarianceOptimizer
from quantred.risk.base import RiskMatrix
from tests.test_portfolio_optimizer import AS_OF

N = 10
SYMS = [f"N{i}" for i in range(N)]


def _world():
    rng = np.random.default_rng(2)
    loadings = rng.normal(0, 0.2, (N, 3))
    sigma = loadings @ loadings.T + np.diag(rng.uniform(0.02, 0.06, N))
    alpha = rng.normal(0.0, 0.03, N)
    alpha[0], alpha[1] = 0.12, 0.10  # a couple of dominant forecasts so a cap binds
    alphas = [Alpha(s, float(alpha[i]), AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(SYMS)]
    return alphas, alpha, sigma, RiskMatrix(SYMS, sigma)


def test_unconstrained_optimum_has_transfer_coefficient_one():
    _, _, sigma, risk = _world()
    # Alpha built as Sigma @ v with v summing to zero, so the optimum's active weights
    # (v / 2*lambda) satisfy the budget and no floor or cap binds around an equal-weight
    # benchmark: the book *is* the unconstrained optimum.
    v = np.random.default_rng(4).normal(0, 1, N)
    ones = np.ones(N)
    basis = np.column_stack([ones, sigma @ ones])  # budget and benchmark-beta directions
    v -= basis @ np.linalg.lstsq(basis, v, rcond=None)[0]  # also makes alpha . w_B = 0
    alpha = sigma @ v * 1e-3
    alphas = [Alpha(s, float(alpha[i]), AS_OF, 0.2, 0.05, 0.0) for i, s in enumerate(SYMS)]
    bench = {s: 1.0 / N for s in SYMS}
    r = MeanVarianceOptimizer(max_weight=1.0).optimize(
        alphas, risk, risk_aversion=1.0, benchmark_weights=bench
    )
    w = np.array([r.weights[s] for s in SYMS])
    assert (w > 1e-6).all() and (w < 1.0 - 1e-6).all()
    assert r.diagnostics["transfer_coefficient"] == pytest.approx(1.0, abs=1e-6)


def test_constrained_tc_times_ir_star_is_the_predicted_ir():
    alphas, alpha, sigma, risk = _world()
    r = MeanVarianceOptimizer(max_weight=0.2).optimize(alphas, risk, risk_aversion=2.0)
    d = r.diagnostics
    w = np.array([r.weights.get(s, 0.0) for s in SYMS])

    # Independent closed form from alpha, w and sigma alone.
    ir_star = np.sqrt(alpha @ np.linalg.solve(sigma, alpha))
    tc = (alpha @ w) / (ir_star * np.sqrt(w @ sigma @ w))
    assert d["transfer_coefficient"] == pytest.approx(tc, abs=1e-9)
    assert d["ir_achieved"] == pytest.approx(d["predicted_ir"], abs=1e-9)
    assert d["transfer_coefficient"] < 0.999  # the cap really binds here
