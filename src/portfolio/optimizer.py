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

``book="market_neutral"`` (Spec 018) relaxes the long-only box to ``[−s, cap]`` and the
budget to ``Σw = 0``, adds a mandatory gross-leverage cap ``‖w‖₁ ≤ L`` (a second,
outer dual - the inner budget bisection now runs once per trial value of the leverage
dual, "one loop deeper" per name), and prices per-name borrow carry as a linear tilt on
the short side only (``Σ borrowᵢ·max(−wᵢ, 0)``) - a *second* zero-anchored kink that
composes with the turnover kink at ``w₀`` (see :meth:`_prox_step_short`). This is a
wholly separate solve path from the long-only one above: ``book="long_only"`` (the
default) runs the exact pre-018 code, unchanged, so every existing result is
byte-for-byte identical.
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
    #: Per-name annualized borrow rate override (Spec 018) - a locate-desk quote or a
    #: manual hard-to-borrow rate; missing names fall back to the cost model's flat
    #: default (:meth:`~src.costs.base.CostModel.borrow_rate`). Only priced for
    #: ``book="market_neutral"`` (long-only books hold no shorts to carry).
    borrow: Dict[str, float] = field(default_factory=dict)


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
        benchmark_weights: Optional[Dict[str, float]] = None,
        book: str = "long_only",
        short_max_weight: float = 0.0,
        gross_leverage: Optional[float] = None,
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

        Pass ``benchmark_weights`` (``w_B``) to make the solve **benchmark-aware**
        (Spec 017): risk and expected return are measured in *active* space
        (``w_a = w − w_B``) rather than against cash, alpha is neutralized against
        the Σ-implied benchmark beta (``αᵀw_B = 0``) so the optimizer carries no
        implicit benchmark-timing view, and ``predicted_tracking_error`` becomes the
        real thing instead of ``√(wᵀΣw)`` on a cash-relative book. Cost stays
        anchored at ``w₀`` (current holdings) - risk and cost intentionally read
        two different reference points. Without ``benchmark_weights`` (or with an
        all-zero one) every quantity below reduces byte-for-byte to the cash-relative
        (pre-017) behavior - the same "cost-blind reduces to today" pattern as 016.

        Pass ``book="market_neutral"`` (Spec 018) to relax the long-only box
        ``[0, cap]`` to ``[−short_max_weight, cap]`` and the budget from ``Σw = 1``
        to ``Σw = 0``. ``gross_leverage`` (``‖w‖₁ ≤ L``) is then *mandatory* - an
        unconstrained long/short mean-variance book on a noisy Σ is a leverage
        machine (error maximization un-truncated by the long-only bound) - and
        ``cost_inputs.borrow`` prices a per-name annualized carry rate on the short
        side only, composing with the turnover cost as a second, asymmetric kink at
        ``w = 0`` (see :meth:`_prox_step_short`). ``book="long_only"`` (the default)
        runs the exact pre-018 code path, so every existing result is unaffected.

        Returns the weights, the diagnostics (``IR*``, predicted TE/IR, transfer
        coefficient, value added, turnover, and - when cost-aware - the linear/impact
        cost split and net expected return; when benchmark-aware, the active-beta /
        residual-risk split; when market-neutral, the dollar-neutral residual and
        realized gross leverage), and a feasibility verdict naming the binding
        constraint when infeasible.
        """
        if book == "market_neutral":
            if gross_leverage is None:
                raise ValueError(
                    "gross_leverage is mandatory for book='market_neutral' - an "
                    "unconstrained long/short book on a noisy Σ is a leverage machine "
                    "(spec 018 hidden factor 1); pass an explicit ‖w‖₁ cap"
                )
            if short_max_weight <= 0:
                raise ValueError(
                    "short_max_weight must be > 0 for book='market_neutral' - "
                    "otherwise no name can be shorted and Σw=0 forces an all-cash book"
                )
        elif book != "long_only":
            raise ValueError(f"book must be 'long_only' or 'market_neutral', got {book!r}")

        symbols, alpha = self._align(alphas, risk)
        if len(symbols) == 0:
            return PortfolioResult(weights={}, feasible=False, binding_constraint="empty universe")

        sigma = self._submatrix(risk, symbols)
        n = len(symbols)

        if book == "long_only":
            # Cardinality + box can make the budget unreachable: name the binding bound.
            k_cap = min(self.max_names or n, n)
            if k_cap * self.max_weight < 1.0 - 1e-9:
                return PortfolioResult(
                    weights={},
                    feasible=False,
                    binding_constraint=f"max_weight*{k_cap} < 1 (cardinality/weight cap can't fund the book)",
                )

        sigma_inv = np.linalg.inv(sigma)

        # Benchmark portfolio (Spec 017): β = Σw_B/(w_Bᵀ Σ w_B) is the one canonical
        # beta (§4.3) - zero vector (and sigma_b2 = 0) when there is no benchmark, so
        # every step below is a no-op and this reduces exactly to the cash-relative
        # solve. Names in benchmark_weights but absent from `symbols` are silently
        # excluded by `_vector` - the caller (spec 017 §4.1/§4.2) is responsible for
        # restricting/renormalizing w_B to Σ's covered universe before calling in.
        w_b = self._vector(benchmark_weights or {}, symbols)
        sigma_b2 = float(w_b @ sigma @ w_b)
        has_benchmark = sigma_b2 > 0
        beta = (sigma @ w_b) / sigma_b2 if has_benchmark else np.zeros(n)
        # α ← α − β·(αᵀw_B): after this, αᵀw_B = 0 exactly (§3.3) - no incentive to
        # lean the book long/short of the benchmark; only constraints can now
        # produce a nonzero active beta (surfaced below, not fought here).
        alpha_neutral = alpha - beta * float(alpha @ w_b) if has_benchmark else alpha

        ir_star = float(np.sqrt(max(alpha_neutral @ sigma_inv @ alpha_neutral, 0.0)))

        lam = self._risk_aversion(ir_star, target_te, risk_aversion)
        # The unconstrained total holding: benchmark plus the unconstrained active
        # bet (§3.1) - w_B + Σ⁻¹α_neutral/2λ. Reduces to the cash-relative closed
        # form exactly when w_B = 0.
        unconstrained = w_b + (sigma_inv @ alpha_neutral) / (2.0 * lam)

        w0 = self._vector(current_weights or {}, symbols)
        # Per-name cost coefficients (annualized), aligned to `symbols`. Both zero when
        # cost-blind -> the solve reduces exactly to the cost-blind projected gradient.
        c_lin, k_imp = self._cost_coefficients(
            cost_model, cost_inputs, capital, holding_period_years, symbols
        )
        cost_aware = bool(np.any(c_lin > 0) or np.any(k_imp > 0))

        # Risk anchored at the benchmark, cost anchored at w₀ (§3.1, hidden factor 4):
        # absorb the whole w_B risk-anchor effect into a shifted alpha so the
        # existing box/budget/cost solver runs completely unchanged (the "two
        # reference points, one solver" trick - see the module docstring's cost
        # term for the analogous pattern in 016). Shift is exactly zero when there
        # is no benchmark.
        alpha_shifted = alpha_neutral + 2.0 * lam * (sigma @ w_b) if has_benchmark else alpha_neutral

        borrow = None
        if book == "long_only":
            w = self._constrained_solve(alpha_shifted, sigma, lam, n, c_lin, k_imp, w0)
        else:
            borrow = self._borrow_coefficients(cost_model, cost_inputs, symbols)
            w = self._constrained_solve_market_neutral(
                alpha_shifted, sigma, lam, n, c_lin, k_imp, borrow, w0, short_max_weight, gross_leverage
            )

        # Manual no-trade band (cost-free override): if every name moves less than the
        # band, don't churn. When cost-aware the band instead *emerges* from c_lin (a
        # name stays at w₀ⁱ while its marginal value is inside ±cᵢ), so this is a no-op.
        if self.no_trade_band > 0 and np.max(np.abs(w - w0)) < self.no_trade_band:
            w = w0.copy()

        diagnostics = self._diagnostics(
            alpha_neutral, sigma, w, w0, lam, ir_star, c_lin, k_imp, cost_aware, w_b, beta, sigma_b2
        )
        if book == "market_neutral":
            # Dollar- vs beta-neutrality (spec 018 §3.3, hidden factor 3): Σw and
            # (when a benchmark/β source is supplied) βᵀw_a are different residuals
            # that diverge whenever betas are dispersed - `active_beta` above already
            # reports the latter via the one canonical β (017's implied_beta), reused
            # rather than redefined; this block adds the former plus the mandatory
            # leverage accounting and the short-side borrow carry.
            borrow_cost = float(np.sum(borrow * np.maximum(-w, 0.0)))
            diagnostics.update(
                {
                    "book": "market_neutral",
                    "dollar_neutral_residual": float(abs(np.sum(w))),
                    "gross_leverage": float(np.sum(np.abs(w))),
                    "gross_leverage_cap": float(gross_leverage),
                    "borrow_cost": borrow_cost,
                }
            )
            if "expected_active_return_net" in diagnostics:
                diagnostics["expected_active_return_net"] -= borrow_cost
                diagnostics["expected_active_return_net_oneway"] -= borrow_cost

        return PortfolioResult(
            weights={s: float(w[i]) for i, s in enumerate(symbols) if abs(w[i]) > 1e-9},
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

    @staticmethod
    def _borrow_coefficients(cost_model, cost_inputs, symbols) -> np.ndarray:
        """Per-name annualized borrow rate (Spec 018 §5): ``CostInputs.borrow``
        overrides the cost model's flat default, threaded the same way
        :meth:`_cost_coefficients` threads spread/ADV/vol. Already annualized (no
        ``/h`` - it is a per-year *holding* rate, not a one-way turnover rate); zero
        when there is no cost model.
        """
        n = len(symbols)
        borrow = np.zeros(n)
        if cost_model is None:
            return borrow
        overrides = cost_inputs.borrow if cost_inputs is not None else {}
        for i, s in enumerate(symbols):
            borrow[i] = cost_model.borrow_rate(overrides.get(s))
        return np.where(np.isfinite(borrow), borrow, 0.0)

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
    # Market-neutral solve (Spec 018) - a separate path from the long-only one
    # above; book="long_only" never touches any of this.
    # ------------------------------------------------------------------ #
    def _constrained_solve_market_neutral(
        self, alpha, sigma, lam, n, c_lin, k_imp, borrow, w0, short_cap, gross_leverage
    ) -> np.ndarray:
        """Box ``[−short_cap, cap]`` + budget ``Σw=0`` + gross-leverage program, plus
        cardinality/dust-floor by re-solving - the same re-solve pattern as
        :meth:`_constrained_solve`, generalized to magnitude (``|w|``) since a large
        short is as much a "position" as a large long.
        """
        active = np.arange(n)
        w = self._solve_leveraged(
            alpha, sigma, lam, active, c_lin, k_imp, borrow, w0, short_cap, gross_leverage
        )

        if self.max_names is not None and int(np.sum(np.abs(w) > 1e-9)) > self.max_names:
            active = np.argsort(np.abs(w))[::-1][: self.max_names]
            w = self._solve_leveraged(
                alpha, sigma, lam, active, c_lin, k_imp, borrow, w0, short_cap, gross_leverage
            )

        while self.min_weight > 0:
            held = np.where(np.abs(w) > 1e-9)[0]
            keep = held[np.abs(w[held]) >= self.min_weight]
            if len(keep) == len(held):
                break  # every held name clears the floor
            if len(keep) == 0:
                return np.zeros(n)  # dust floor emptied the book - a valid (if useless) all-cash result
            active = keep
            w = self._solve_leveraged(
                alpha, sigma, lam, active, c_lin, k_imp, borrow, w0, short_cap, gross_leverage
            )
        return w

    def _solve_leveraged(
        self, alpha, sigma, lam, active, c_lin, k_imp, borrow, w0, short_cap, gross_leverage
    ) -> np.ndarray:
        """The outer gross-leverage dual (spec 018 §3.1/§7): solve the box/budget/
        borrow program at ``μ=0`` first: since the leverage cap "only sometimes
        binds", most solves stop here. Only when ‖w‖₁ already exceeds the cap does
        the outer bisection on ``μ≥0`` run - each trial ``μ`` re-solves the *entire*
        inner proximal-gradient program to convergence ("one loop deeper" than the
        long-only path). Monotonicity of ‖w(μ)‖₁ (a stiffer zero-anchored threshold
        can only shrink every name toward 0) is verified empirically by the KKT
        certificate test, not just assumed - spec 018 §7 flags this as the open risk.
        """
        w = self._solve_market_neutral(alpha, sigma, lam, active, c_lin, k_imp, borrow, w0, short_cap, 0.0)
        gross = float(np.sum(np.abs(w)))
        if gross <= gross_leverage + 1e-9:
            return w

        def gross_at(mu: float) -> float:
            wm = self._solve_market_neutral(
                alpha, sigma, lam, active, c_lin, k_imp, borrow, w0, short_cap, mu
            )
            return float(np.sum(np.abs(wm)))

        lo_mu, hi_mu = 0.0, 1.0
        guard = 0
        while gross_at(hi_mu) > gross_leverage and guard < 60:
            hi_mu *= 2.0
            guard += 1
        for _ in range(40):
            mid = 0.5 * (lo_mu + hi_mu)
            if gross_at(mid) > gross_leverage:
                lo_mu = mid
            else:
                hi_mu = mid
        return self._solve_market_neutral(
            alpha, sigma, lam, active, c_lin, k_imp, borrow, w0, short_cap, 0.5 * (lo_mu + hi_mu)
        )

    def _solve_market_neutral(
        self, alpha, sigma, lam, active, c_lin, k_imp, borrow, w0, short_cap, mu
    ) -> np.ndarray:
        """Proximal-gradient solve at a FIXED leverage dual ``μ`` - otherwise the same
        iteration structure as :meth:`_solve`, over the market-neutral box/budget."""
        n = len(alpha)
        if len(active) == 0:
            return np.zeros(n)
        a = alpha[active]
        s = sigma[np.ix_(active, active)]
        c = c_lin[active]
        k = k_imp[active]
        b = borrow[active]
        w0a = w0[active]
        cap = self.max_weight
        lower = -short_cap
        lmax = float(np.linalg.eigvalsh(s).max())
        step = 1.0 / (2.0 * lam * lmax) if lmax > 0 else 1.0
        thr = step * c
        kthr = step * k
        bthr = step * b
        # Warm-start via a cost-blind box+budget projection (mu=0), same spirit as _solve.
        start = w0a if np.abs(w0a).sum() > 1e-9 else np.zeros(len(active))
        w = self._prox_project_market_neutral(
            start, np.zeros_like(thr), np.zeros_like(kthr), np.zeros_like(bthr), 0.0, w0a, lower, cap
        )
        for _ in range(self.max_iter):
            grad = -a + 2.0 * lam * (s @ w)
            nxt = self._prox_project_market_neutral(w - step * grad, thr, kthr, bthr, mu, w0a, lower, cap)
            if np.max(np.abs(nxt - w)) < self.tol:
                w = nxt
                break
            w = nxt
        full = np.zeros(n)
        full[active] = w
        return full

    def _prox_project_market_neutral(self, y, thr, kthr, borrow, mu, w0, lower, cap) -> np.ndarray:
        """Budget-dual (Σw=0) bisection over the market-neutral box, at a fixed
        leverage dual ``μ`` - mirrors :meth:`_prox_project` exactly, just with
        budget 0 instead of 1, box ``[lower, cap]`` instead of ``[0, cap]``, and the
        short/leverage-aware per-name step (:meth:`_prox_step_short`).
        """

        def total(tau: float) -> float:
            w = self._prox_step_short(y - tau, thr, kthr, borrow, mu, w0)
            return float(np.clip(w, lower, cap).sum())

        pad = float(thr.max()) if thr.size else 0.0
        lo = float(y.min() - cap - pad)
        hi = float(y.max() + pad)
        guard = 0
        while total(lo) < 0.0 and guard < 200:
            lo -= abs(lo) + 1.0
            guard += 1
        guard = 0
        while total(hi) > 0.0 and guard < 200:
            hi += abs(hi) + 1.0
            guard += 1
        for _ in range(100):
            tau = 0.5 * (lo + hi)
            if total(tau) > 0.0:
                lo = tau
            else:
                hi = tau
        tau = 0.5 * (lo + hi)
        w = self._prox_step_short(y - tau, thr, kthr, borrow, mu, w0)
        return np.clip(w, lower, cap)

    @classmethod
    def _prox_step_short(cls, m, thr, kthr, borrow, mu, w0) -> np.ndarray:
        """Per-name proximal step for the market-neutral book: minimizes, over ``w``,

            ½(w−y)² + cᵢ|w−w₀ⁱ| + kᵢ|w−w₀ⁱ|^{3/2} + borrowᵢ·max(−w,0) + μ|w|

        (``y = m + w₀``, ``μ`` the leverage dual). Unlike :meth:`_prox_step` this
        has *two* kinks - the turnover kink at ``w = w₀`` and a second, zero-anchored
        one where the short-side borrow and the (symmetric) leverage threshold turn
        on at ``w = 0``. The function is still convex and piecewise, so the true
        minimizer is one of a small set of closed-form candidates: the unconstrained
        stationary point of each of the (up to 4) smooth pieces the two kinks carve
        out, plus the two kinks themselves. Evaluating the objective at all six and
        keeping the best is exact (no iterative search) and, unlike hand-selecting
        which piece is "active", needs no case analysis on the sign of ``w₀`` -
        convexity guarantees the argmin is correct regardless of which candidates
        happen to be infeasible for their own piece.

        With ``borrow = μ = 0`` the six candidates collapse to exactly
        :meth:`_prox_step`'s two branches (plus dominated duplicates), but this
        method is never called from the long-only path, so that path's numerics are
        untouched.
        """
        w1 = w0 + cls._branch_pos(m, thr + mu, kthr)  # region w >= max(w0, 0)
        w2 = w0 + cls._branch_neg(m, thr + borrow + mu, kthr)  # region w <= min(w0, 0)
        w3 = np.where(w0 > 0, w0 + cls._branch_neg(m, thr - mu, kthr), w0)  # region 0 <= w <= w0 (w0>0 only)
        w4 = np.where(
            w0 < 0, w0 + cls._branch_pos(m, thr - borrow - mu, kthr), w0
        )  # region w0 <= w <= 0 (w0<0 only)
        zeros = np.zeros_like(w0)
        candidates = np.stack([w1, w2, w3, w4, zeros, w0])

        y_eff = m + w0
        dw = candidates - w0
        obj = (
            0.5 * (candidates - y_eff) ** 2
            + thr * np.abs(dw)
            + kthr * np.abs(dw) ** 1.5
            + borrow * np.maximum(-candidates, 0.0)
            + mu * np.abs(candidates)
        )
        best = np.argmin(obj, axis=0)
        return np.take_along_axis(candidates, best[None, :], axis=0)[0]

    @staticmethod
    def _branch_pos(m, thr, kthr) -> np.ndarray:
        """Stationary point (``d = w−w₀ ≥ 0`` branch, extended to all reals) of
        ``½(d−m)² + thr·d + kthr·d^{3/2}`` - the same closed form as
        :meth:`_prox_step`'s ``m ≥ 0`` branch, factored out so :meth:`_prox_step_short`
        can reuse it with a shifted threshold (borrow/leverage add to ``thr`` here).
        """
        b = 1.5 * kthr
        residual = np.maximum(m - thr, 0.0)
        u = (-b + np.sqrt(b * b + 4.0 * residual)) / 2.0
        return u * u

    @staticmethod
    def _branch_neg(m, thr, kthr) -> np.ndarray:
        """Stationary point (``d = w−w₀ ≤ 0`` branch) - the mirror of
        :meth:`_branch_pos`, matching :meth:`_prox_step`'s ``m < 0`` branch."""
        b = 1.5 * kthr
        residual = np.maximum(-(m + thr), 0.0)
        u = (-b + np.sqrt(b * b + 4.0 * residual)) / 2.0
        return -(u * u)

    # ------------------------------------------------------------------ #
    # Calibration & diagnostics
    # ------------------------------------------------------------------ #
    def _risk_aversion(self, ir_star, target_te, risk_aversion) -> float:
        if risk_aversion is not None:
            return float(risk_aversion)
        if target_te is not None and target_te > 0 and ir_star > 0:
            return ir_star / (2.0 * target_te)  # λ_A = IR* / (2·ψ_target)
        return 1.0  # neutral default when neither is usable

    def _diagnostics(
        self, alpha, sigma, w, w0, lam, ir_star, c_lin, k_imp, cost_aware, w_b, beta, sigma_b2
    ) -> Dict[str, float]:
        """``alpha`` here is already benchmark-neutralized when ``w_b`` is nonzero
        (§3.3) - the caller passes ``alpha_neutral``, never the raw forecast."""
        # Active weights w_a = w - w_B (§3.1). w_b is the zero vector without a
        # benchmark, so w_a = w and every quantity below is the pre-017 cash-relative
        # one - unchanged.
        w_a = w - w_b
        sig_wa = sigma @ w_a
        active_variance = float(w_a @ sig_wa)
        te = float(np.sqrt(max(active_variance, 0.0)))  # ψ, the real tracking error
        expected = float(alpha @ w_a)
        # Transfer coefficient: corr(α, Σ-adjusted active weights) ∈ [-1, 1].
        tc = self._corr(alpha, sig_wa)
        dw = w - w0  # cost is anchored at w₀, not w_B (§3.1 hidden factor 4) - unchanged
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
        if sigma_b2 > 0:
            # ψ² = β_a²·σ_B² + ω² (§3.3): the active-beta (benchmark-timing) share of
            # tracking error vs the residual (stock-selection) share. β_a should be
            # ~0 by construction (alpha is neutralized); a nonzero value here comes
            # only from a binding constraint (box/cardinality), which is exactly what
            # this diagnostic is for surfacing.
            active_beta = float(beta @ w_a)
            residual_risk = float(np.sqrt(max(te**2 - active_beta**2 * sigma_b2, 0.0)))
            diagnostics.update(
                {
                    "has_benchmark": True,
                    "benchmark_variance": sigma_b2,
                    "active_beta": active_beta,
                    "residual_risk": residual_risk,
                    # A benchmark equal to current holdings makes ψ measure "distance
                    # from myself" - legal but meaningless (§4.5).
                    "self_benchmark_warning": bool(np.sum(np.abs(w0 - w_b)) < 1e-6),
                }
            )
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
