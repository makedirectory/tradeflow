"""Portfolio construction as mean-variance utility maximization.

The OR-Tools :class:`~src.portfolio.allocator.PortfolioAllocator` maximizes a scalar
score subject to constraints - it has no notion of risk, so it piles weight onto the
highest-scoring names regardless of how correlated they are. Active-management
construction instead maximizes a **risk-adjusted utility**, trading expected residual
return (alpha) against active risk (covariance) - and, once cost is in the objective
(the cost-aware solve), against the *name-specific* cost of getting there:

    U(w) = αᵀw − λ_A · wᵀΣw − Σᵢ cᵢ·|Δwᵢ| − Σᵢ kᵢ·|Δwᵢ|^{3/2}      (Δw = w − w₀)
           └ value ┘  └ active risk ┘  └ linear turnover ┘  └ √-impact (conic) ┘

The output is the portfolio that maximizes the *information ratio you can actually
implement net of cost*, plus the Fundamental-Law diagnostics that make the cost of
every constraint visible. This is **research-clock**: it proposes target weights (a
config a human promotes), it never places an order - so it lives apart from the
operational position sizer.

Pure numpy - no heavyweight QP/conic dependency. The unconstrained cost-free optimum is
closed-form; the constrained problem (long-only, box, budget, cardinality) is a small
convex program solved by **proximal gradient**. The cost term is nonsmooth but convex,
so it enters through the proximal step: after each gradient step on the smooth part we
apply the exact proximal operator of ``cᵢ·|Δwᵢ| + kᵢ·|Δwᵢ|^{3/2}`` *composed with* the
capped-simplex projection. That proximal operator is separable per name given a single
budget dual, and each coordinate has a closed form (a soft-threshold *around w₀* for the
linear cost; a quadratic-in-√ root for the √-impact) - so the whole conic problem is
solved by the same 1-D budget bisection the cost-free projection already used, with no
external solver. When ``cᵢ = kᵢ = 0`` it reduces *exactly* to the cost-free projected
gradient, so the cost-blind behavior is unchanged.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.alphas.base import Alpha
from src.costs.base import CostModel
from src.risk.base import RiskMatrix


@dataclass
class CostInputs:
    """Per-name, as-of liquidity context the cost model prices trades against.

    Threaded from the cost model: a fractional ``spread`` (quoted, or a high-low-range proxy),
    trailing ``adv_dollar`` (ADV in dollars = price · share-ADV), and trailing
    ``daily_vol`` (daily return volatility). All keyed by symbol; missing names fall
    back to the cost model's defaults (spread) or drop the √-impact term (ADV/vol).
    """

    spread: Dict[str, float] = field(default_factory=dict)
    adv_dollar: Dict[str, float] = field(default_factory=dict)
    daily_vol: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioResult:
    """A proposed portfolio plus the quantities the Fundamental Law cares about."""

    weights: Dict[str, float]
    feasible: bool = True
    binding_constraint: Optional[str] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)
    unconstrained_weights: Dict[str, float] = field(default_factory=dict)


class MeanVarianceOptimizer:
    """Maximize ``αᵀw − λ·wᵀΣw − cost(w − w₀)`` over long-only, box, budgeted weights."""

    def __init__(
        self,
        max_weight: float = 0.25,
        min_weight: float = 0.0,
        max_names: Optional[int] = None,
        no_trade_band: float = 0.0,
        max_iter: int = 2000,
        tol: float = 1e-10,
    ):
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 <= min_weight <= max_weight:
            raise ValueError("min_weight must be in [0, max_weight]")
        self.max_weight = max_weight
        self.min_weight = min_weight  # minimum weight for a *held* name (a dust floor)
        self.max_names = max_names
        # Manual override retained for backward-compat / the cost-free case; when a cost
        # model is supplied the no-trade band *emerges* from the cost (see optimize()).
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
        cost_model: Optional[CostModel] = None,
        cost_inputs: Optional[CostInputs] = None,
        capital: Optional[float] = None,
        holding_period_years: float = 1.0 / 12.0,
    ) -> PortfolioResult:
        """Construct the utility-maximizing portfolio over the alpha/risk universe.

        Supply *either* ``target_te`` (the intuitive knob - "run at 4% tracking
        error") *or* ``risk_aversion`` (``λ_A`` directly). ``current_weights`` (``w₀``)
        is where turnover is measured from.

        Pass a ``cost_model`` + ``cost_inputs`` to make the solve **cost-aware** (Spec
        016): the objective gains a name-specific linear turnover penalty and, when
        ``capital`` is given, the √-impact term. Cost coefficients are *annualized* to
        the same units as the (annualized) alpha by dividing the one-way rate by
        ``holding_period_years`` - matching the cost model's alpha-haircut and the
        ex-post cost drag. Without a cost model the solve is cost-blind (unchanged).

        Returns the weights, the diagnostics (``IR*``, predicted TE/IR, transfer
        coefficient, value added, turnover, and - when cost-aware - the linear/impact
        cost split and net expected return), and a feasibility verdict naming the
        binding constraint when infeasible.
        """
        symbols, alpha = self._align(alphas, risk)
        if len(symbols) == 0:
            return PortfolioResult(weights={}, feasible=False, binding_constraint="empty universe")

        sigma = self._submatrix(risk, symbols)
        n = len(symbols)

        # Cardinality + box can make the budget unreachable: name the binding bound.
        k_cap = min(self.max_names or n, n)
        if k_cap * self.max_weight < 1.0 - 1e-9:
            return PortfolioResult(
                weights={},
                feasible=False,
                binding_constraint=f"max_weight*{k_cap} < 1 (cardinality/weight cap can't fund the book)",
            )

        sigma_inv = np.linalg.inv(sigma)
        ir_star = float(np.sqrt(max(alpha @ sigma_inv @ alpha, 0.0)))

        lam = self._risk_aversion(ir_star, target_te, risk_aversion)
        unconstrained = (sigma_inv @ alpha) / (2.0 * lam)

        w0 = self._vector(current_weights or {}, symbols)
        # Per-name cost coefficients (annualized), aligned to `symbols`. Both zero when
        # cost-blind -> the solve reduces exactly to the cost-blind projected gradient.
        c_lin, k_imp = self._cost_coefficients(
            cost_model, cost_inputs, capital, holding_period_years, symbols
        )
        cost_aware = bool(np.any(c_lin > 0) or np.any(k_imp > 0))

        w = self._constrained_solve(alpha, sigma, lam, n, c_lin, k_imp, w0)

        # Manual no-trade band (cost-free override): if every name moves less than the
        # band, don't churn. When cost-aware the band instead *emerges* from c_lin (a
        # name stays at w₀ⁱ while its marginal value is inside ±cᵢ), so this is a no-op.
        if self.no_trade_band > 0 and np.max(np.abs(w - w0)) < self.no_trade_band:
            w = w0.copy()

        diagnostics = self._diagnostics(alpha, sigma, w, w0, lam, ir_star, c_lin, k_imp, cost_aware)
        return PortfolioResult(
            weights={s: float(w[i]) for i, s in enumerate(symbols) if w[i] > 1e-9},
            feasible=True,
            diagnostics=diagnostics,
            unconstrained_weights={s: float(unconstrained[i]) for i, s in enumerate(symbols)},
        )

    # ------------------------------------------------------------------ #
    # Cost coefficients
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cost_coefficients(cost_model, cost_inputs, capital, holding_period_years, symbols):
        """Build the annualized per-name linear (``cᵢ``) and √-impact (``kᵢ``) vectors.

        ``cᵢ = turnover_cost_rate(spreadᵢ) / h`` (one-way commission+half-spread, per
        year); ``kᵢ = impact_coefficient(σᵢ, ADV$ᵢ, capital) / h`` (per year), and
        ``kᵢ = 0`` when ``capital`` is absent - the "ship linear first" default. Zero
        vectors when no cost model is supplied.
        """
        n = len(symbols)
        c_lin = np.zeros(n)
        k_imp = np.zeros(n)
        if cost_model is None or cost_inputs is None:
            return c_lin, k_imp
        h = max(holding_period_years, 1e-9)
        have_capital = capital is not None and capital > 0
        for i, s in enumerate(symbols):
            spread = cost_inputs.spread.get(s)
            c_lin[i] = cost_model.turnover_cost_rate(spread) / h
            if have_capital:
                adv_dollar = cost_inputs.adv_dollar.get(s, 0.0)
                daily_vol = cost_inputs.daily_vol.get(s, 0.0)
                k_imp[i] = cost_model.impact_coefficient(daily_vol, adv_dollar, capital) / h
        # A non-finite input (e.g. a NaN spread proxy) must drop that name's cost term,
        # not poison the whole solve with a NaN coefficient (which would empty the book).
        c_lin = np.where(np.isfinite(c_lin), c_lin, 0.0)
        k_imp = np.where(np.isfinite(k_imp), k_imp, 0.0)
        return c_lin, k_imp

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def _constrained_solve(self, alpha, sigma, lam, n, c_lin, k_imp, w0) -> np.ndarray:
        """Box+budget convex program, plus the two non-convex constraints by re-solving.

        Cardinality (``‖w‖₀ ≤ max_names``) and the semi-continuous dust floor
        (``w ∈ {0} ∪ [min_weight, max_weight]``) are both non-convex, so each is
        enforced the same pragmatic way: solve the convex (now cost-aware) program, drop
        the offending names (the smallest beyond the cardinality cap, or any stuck in
        the ``(0, min_weight)`` hole), and re-solve on the survivors until stable. A
        dropped held name is a full liquidation of ``w₀ⁱ``; its cost is captured in the
        final diagnostics, which price ``w − w₀`` over the full universe.
        """
        active = np.arange(n)
        w = self._solve(alpha, sigma, lam, active, c_lin, k_imp, w0)

        if self.max_names is not None and int(np.sum(w > 1e-9)) > self.max_names:
            active = np.argsort(w)[::-1][: self.max_names]
            w = self._solve(alpha, sigma, lam, active, c_lin, k_imp, w0)

        # Dust floor: drop held names below min_weight and re-solve, while the budget
        # stays fundable (enough remaining names at max_weight to reach 1).
        while self.min_weight > 0:
            held = np.where(w > 1e-9)[0]
            keep = held[w[held] >= self.min_weight]
            if len(keep) == len(held):
                break  # every held name clears the floor
            if len(keep) == 0 or len(keep) * self.max_weight < 1.0 - 1e-9:
                break  # dropping further would make the budget infeasible
            w = self._solve(alpha, sigma, lam, keep, c_lin, k_imp, w0)
        return w

    def _solve(self, alpha, sigma, lam, active, c_lin, k_imp, w0) -> np.ndarray:
        """Proximal-gradient solve over the active names; full-length weight vector out.

        Minimizes ``f(w) + g(w)`` with smooth ``f = −αᵀw + λwᵀΣw`` and nonsmooth convex
        ``g = Σ cᵢ|wᵢ−w₀ⁱ| + Σ kᵢ|wᵢ−w₀ⁱ|^{3/2} + ι_{box∩budget}``. Each iteration is a
        gradient step on ``f`` followed by the exact proximal operator of ``g`` (see
        :meth:`_prox_project`). Step ``1/L`` with ``L = 2λ·λmax(Σ)`` guarantees
        convergence; ``λmin(Σ) > 0`` makes ``f`` strongly convex, so it is linear.
        """
        n = len(alpha)
        a = alpha[active]
        s = sigma[np.ix_(active, active)]
        c = c_lin[active]
        k = k_imp[active]
        w0a = w0[active]
        # Lipschitz constant of ∇f = -a + 2λ·s·w is 2λ·λmax(s); step = 1/L.
        lmax = float(np.linalg.eigvalsh(s).max())
        step = 1.0 / (2.0 * lam * lmax) if lmax > 0 else 1.0
        thr = step * c
        kthr = step * k
        # Warm-start from w₀ (projected) when it holds weight, else uniform.
        start = w0a if w0a.sum() > 1e-9 else np.ones(len(active)) / len(active)
        w = self._prox_project(start, np.zeros_like(thr), np.zeros_like(kthr), w0a)
        for _ in range(self.max_iter):
            grad = -a + 2.0 * lam * (s @ w)
            nxt = self._prox_project(w - step * grad, thr, kthr, w0a)
            if np.max(np.abs(nxt - w)) < self.tol:
                w = nxt
                break
            w = nxt
        full = np.zeros(n)
        full[active] = w
        return full

    def _prox_project(self, y: np.ndarray, thr: np.ndarray, kthr: np.ndarray, w0: np.ndarray) -> np.ndarray:
        """Exact prox of the cost + capped-simplex indicator, via a budget bisection.

        Solves ``argmin_w ½‖w−y‖² + Σ thrᵢ|wᵢ−w₀ⁱ| + Σ kthrᵢ|wᵢ−w₀ⁱ|^{3/2}`` over
        ``{0 ≤ w ≤ cap, Σw = 1}``. With a single dual ``τ`` for the budget it separates
        per coordinate: ``wᵢ(τ) = clip(w₀ⁱ + d*((yᵢ−τ)−w₀ⁱ), 0, cap)`` where ``d*`` is
        the per-name proximal step (:meth:`_prox_step`). Each ``wᵢ(τ)`` is monotone
        non-increasing in ``τ``, so ``Σwᵢ(τ) = 1`` is found by bisection - the same
        1-D structure the cost-free projection used. With ``thr = kthr = 0`` the
        coordinate map is ``clip(yᵢ − τ, 0, cap)`` and this is exactly that projection.
        """
        cap = self.max_weight

        def total(tau: float) -> float:
            m = (y - tau) - w0
            return float(np.clip(w0 + self._prox_step(m, thr, kthr), 0.0, cap).sum())

        pad = float(thr.max()) if thr.size else 0.0
        lo = float(y.min() - cap - pad)
        hi = float(y.max() + pad)
        # Guarantee the budget root is bracketed (total is monotone non-increasing in τ).
        guard = 0
        while total(lo) < 1.0 and guard < 200:
            lo -= abs(lo) + 1.0
            guard += 1
        guard = 0
        while total(hi) > 1.0 and guard < 200:
            hi += abs(hi) + 1.0
            guard += 1
        for _ in range(100):
            tau = 0.5 * (lo + hi)
            if total(tau) > 1.0:
                lo = tau
            else:
                hi = tau
        tau = 0.5 * (lo + hi)
        m = (y - tau) - w0
        return np.clip(w0 + self._prox_step(m, thr, kthr), 0.0, cap)

    @staticmethod
    def _prox_step(m: np.ndarray, thr: np.ndarray, kthr: np.ndarray) -> np.ndarray:
        """Per-name trade ``d* = argmin_d ½(d−m)² + thr|d| + kthr|d|^{3/2}`` (unclipped).

        Closed form: ``d* = 0`` when ``|m| ≤ thr`` (the emergent no-trade band - a name
        is untraded while its move is inside the linear cost); otherwise, with residual
        ``r = |m| − thr`` and ``b = 3·kthr/2``, the magnitude solves ``u² + b·u = r``
        (``u = √|d|``), i.e. ``u = (−b + √(b² + 4r))/2`` and ``d* = sign(m)·u²``. With
        ``kthr = 0`` this is the ordinary soft-threshold ``sign(m)·max(|m|−thr, 0)``.
        """
        residual = np.maximum(np.abs(m) - thr, 0.0)
        b = 1.5 * kthr
        u = (-b + np.sqrt(b * b + 4.0 * residual)) / 2.0
        return np.sign(m) * u * u

    # ------------------------------------------------------------------ #
    # Calibration & diagnostics
    # ------------------------------------------------------------------ #
    def _risk_aversion(self, ir_star, target_te, risk_aversion) -> float:
        if risk_aversion is not None:
            return float(risk_aversion)
        if target_te is not None and target_te > 0 and ir_star > 0:
            return ir_star / (2.0 * target_te)  # λ_A = IR* / (2·ψ_target)
        return 1.0  # neutral default when neither is usable

    def _diagnostics(self, alpha, sigma, w, w0, lam, ir_star, c_lin, k_imp, cost_aware) -> Dict[str, float]:
        sig_w = sigma @ w
        active_variance = float(w @ sig_w)
        te = float(np.sqrt(max(active_variance, 0.0)))
        expected = float(alpha @ w)
        # Transfer coefficient: corr(α, Σ-adjusted active weights) ∈ [-1, 1].
        tc = self._corr(alpha, sig_w)
        dw = w - w0
        diagnostics = {
            "ir_star": ir_star,
            "risk_aversion": float(lam),
            "expected_active_return": expected,
            "predicted_tracking_error": te,  # realized TE at the (cost-bent) optimum
            "predicted_ir": expected / te if te > 0 else 0.0,
            "transfer_coefficient": tc,
            "ir_achieved": tc * ir_star,
            "optimal_tracking_error": ir_star / (2.0 * lam) if lam > 0 else 0.0,
            "value_added": (ir_star**2) / (4.0 * lam) if lam > 0 else 0.0,
            "turnover": float(np.sum(np.abs(dw))),
        }
        if cost_aware:
            # One-way cost of *this rebalance's* turnover - what the objective charged,
            # kept for continuity with the prior (pre-016) ex-post drag. Detail.
            linear_cost = float(np.sum(c_lin * np.abs(dw)))
            impact_cost = float(np.sum(k_imp * np.abs(dw) ** 1.5))
            rebalance_cost = linear_cost + impact_cost
            # Round-trip haircut on the *held book* (the capacity convention):
            # entering AND exiting each position, amortized over the holding period. This
            # is the conservative headline net figure - the same cost model _capacity
            # prices, so the net return and the capacity number agree.
            book = np.abs(w)
            round_trip_cost = float(2.0 * (np.sum(c_lin * book) + np.sum(k_imp * book**1.5)))
            diagnostics.update(
                {
                    "cost_aware": True,
                    "linear_cost": linear_cost,  # one-way Σ cᵢ|Δwᵢ| (rebalance turnover)
                    "impact_cost": impact_cost,  # one-way Σ kᵢ|Δwᵢ|^{3/2} (rebalance √-impact)
                    "cost_drag": rebalance_cost,  # one-way rebalance total (continuity)
                    "round_trip_cost": round_trip_cost,  # round-trip book haircut (headline)
                    "expected_active_return_net": expected - round_trip_cost,  # headline: round-trip
                    "expected_active_return_net_oneway": expected - rebalance_cost,  # detail: one-way
                    "names_traded": int(np.sum(np.abs(dw) > 1e-9)),
                }
            )
        return diagnostics

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
