"""Portfolio construction as mean-variance utility maximization.

The OR-Tools :class:`~src.portfolio.allocator.PortfolioAllocator` maximises a scalar
score subject to constraints - it has no notion of risk, so it piles weight onto the
highest-scoring names regardless of how correlated they are. Active-management
construction instead maximises a **risk-adjusted utility**, trading expected residual
return (alpha, Spec 005) against active risk (covariance, Spec 006):

    U(w) = αᵀw − λ_A · wᵀΣw            (long-only absolute: benchmark w_B = 0)

The output is the portfolio that maximises the *information ratio you can actually
implement*, plus the Fundamental-Law diagnostics that make the cost of every
constraint visible. This is **research-clock**: it proposes target weights (a config a
human promotes), it never places an order - so it lives apart from the operational
position sizer.

Pure numpy: the unconstrained optimum is closed-form; the constrained (long-only, box,
budget, cardinality) optimum is a small convex QP solved by projected gradient with a
capped-simplex projection - no heavyweight QP dependency.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.alphas.base import Alpha
from src.risk.base import RiskMatrix


@dataclass
class PortfolioResult:
    """A proposed portfolio plus the quantities the Fundamental Law cares about."""

    weights: Dict[str, float]
    feasible: bool = True
    binding_constraint: Optional[str] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)
    unconstrained_weights: Dict[str, float] = field(default_factory=dict)


class MeanVarianceOptimizer:
    """Maximise ``αᵀw − λ·wᵀΣw`` over long-only, box-bounded, budgeted weights."""

    def __init__(
        self,
        max_weight: float = 0.25,
        max_names: Optional[int] = None,
        no_trade_band: float = 0.0,
        max_iter: int = 2000,
        tol: float = 1e-10,
    ):
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        self.max_weight = max_weight
        self.max_names = max_names
        self.no_trade_band = no_trade_band
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        alphas: List[Alpha],
        risk: RiskMatrix,
        *,
        target_te: Optional[float] = None,
        risk_aversion: Optional[float] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioResult:
        """Construct the utility-maximising portfolio over the alpha/risk universe.

        Supply *either* ``target_te`` (the intuitive knob - "run at 4% tracking
        error") *or* ``risk_aversion`` (``λ_A`` directly). ``current_weights`` (``w_0``)
        is where turnover is measured from. Returns the weights, the diagnostics
        (``IR*``, predicted TE/IR, transfer coefficient, value added, turnover), and a
        feasibility verdict naming the binding constraint when infeasible.
        """
        symbols, alpha = self._align(alphas, risk)
        if len(symbols) == 0:
            return PortfolioResult(weights={}, feasible=False, binding_constraint="empty universe")

        sigma = self._submatrix(risk, symbols)
        n = len(symbols)

        # Cardinality + box can make the budget unreachable: name the binding bound.
        k = min(self.max_names or n, n)
        if k * self.max_weight < 1.0 - 1e-9:
            return PortfolioResult(
                weights={},
                feasible=False,
                binding_constraint=f"max_weight*{k} < 1 (cardinality/weight cap can't fund the book)",
            )

        sigma_inv = np.linalg.inv(sigma)
        ir_star = float(np.sqrt(max(alpha @ sigma_inv @ alpha, 0.0)))

        lam = self._risk_aversion(ir_star, target_te, risk_aversion)
        unconstrained = (sigma_inv @ alpha) / (2.0 * lam)

        # Constrained solve, with optional cardinality (two-stage: select then weight).
        w = self._solve(alpha, sigma, lam, np.arange(n))
        if self.max_names is not None and int(np.sum(w > 1e-9)) > self.max_names:
            keep = np.argsort(w)[::-1][: self.max_names]
            w = self._solve(alpha, sigma, lam, keep)

        w0 = self._vector(current_weights or {}, symbols)
        # No-trade band: if every name moves less than the band, don't churn.
        if self.no_trade_band > 0 and np.max(np.abs(w - w0)) < self.no_trade_band:
            w = w0.copy()

        diagnostics = self._diagnostics(alpha, sigma, w, w0, lam, ir_star)
        return PortfolioResult(
            weights={s: float(w[i]) for i, s in enumerate(symbols) if w[i] > 1e-9},
            feasible=True,
            diagnostics=diagnostics,
            unconstrained_weights={s: float(unconstrained[i]) for i, s in enumerate(symbols)},
        )

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def _solve(self, alpha: np.ndarray, sigma: np.ndarray, lam: float, active: np.ndarray) -> np.ndarray:
        """Projected-gradient QP over the active names; full-length weight vector out."""
        n = len(alpha)
        a, s = alpha[active], sigma[np.ix_(active, active)]
        # Lipschitz constant of ∇f = -a + 2λ·s·w is 2λ·λmax(s); step = 1/L.
        lmax = float(np.linalg.eigvalsh(s).max())
        step = 1.0 / (2.0 * lam * lmax) if lmax > 0 else 1.0
        w = self._project(np.ones(len(active)) / len(active))
        for _ in range(self.max_iter):
            grad = -a + 2.0 * lam * (s @ w)
            nxt = self._project(w - step * grad)
            if np.max(np.abs(nxt - w)) < self.tol:
                w = nxt
                break
            w = nxt
        full = np.zeros(n)
        full[active] = w
        return full

    def _project(self, y: np.ndarray) -> np.ndarray:
        """Euclidean projection onto {0 ≤ w ≤ max_weight, Σw = 1} (capped simplex)."""
        c, target = self.max_weight, 1.0
        lo, hi = float(y.min() - c), float(y.max())
        for _ in range(100):
            tau = 0.5 * (lo + hi)
            total = np.clip(y - tau, 0.0, c).sum()
            if total > target:
                lo = tau
            else:
                hi = tau
        return np.clip(y - 0.5 * (lo + hi), 0.0, c)

    # ------------------------------------------------------------------ #
    # Calibration & diagnostics
    # ------------------------------------------------------------------ #
    def _risk_aversion(self, ir_star, target_te, risk_aversion) -> float:
        if risk_aversion is not None:
            return float(risk_aversion)
        if target_te is not None and target_te > 0 and ir_star > 0:
            return ir_star / (2.0 * target_te)  # λ_A = IR* / (2·ψ_target)
        return 1.0  # neutral default when neither is usable

    def _diagnostics(self, alpha, sigma, w, w0, lam, ir_star) -> Dict[str, float]:
        sig_w = sigma @ w
        active_variance = float(w @ sig_w)
        te = float(np.sqrt(max(active_variance, 0.0)))
        expected = float(alpha @ w)
        # Transfer coefficient: corr(α, Σ-adjusted active weights) ∈ [-1, 1].
        tc = self._corr(alpha, sig_w)
        return {
            "ir_star": ir_star,
            "risk_aversion": float(lam),
            "expected_active_return": expected,
            "predicted_tracking_error": te,
            "predicted_ir": expected / te if te > 0 else 0.0,
            "transfer_coefficient": tc,
            "ir_achieved": tc * ir_star,
            "optimal_tracking_error": ir_star / (2.0 * lam) if lam > 0 else 0.0,
            "value_added": (ir_star**2) / (4.0 * lam) if lam > 0 else 0.0,
            "turnover": float(np.sum(np.abs(w - w0))),
        }

    # ------------------------------------------------------------------ #
    # Alignment helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _align(alphas: List[Alpha], risk: RiskMatrix):
        """Common symbols (in Σ order) and the aligned alpha vector."""
        by_symbol = {a.symbol: a.alpha for a in alphas}
        symbols = [s for s in risk.symbols if s in by_symbol]
        return symbols, np.array([by_symbol[s] for s in symbols], dtype=float)

    @staticmethod
    def _submatrix(risk: RiskMatrix, symbols: List[str]) -> np.ndarray:
        idx = [risk.symbols.index(s) for s in symbols]
        return risk.sigma[np.ix_(idx, idx)]

    @staticmethod
    def _vector(weights: Dict[str, float], symbols: List[str]) -> np.ndarray:
        return np.array([weights.get(s, 0.0) for s in symbols], dtype=float)

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
