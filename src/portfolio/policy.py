"""Multi-period trading: aim in front of the target (Spec 022).

Spec 016 made the single-period solve honestly cost-aware, but it is still
**myopic**: every rebalance it pays real cost to reach *this period's* optimum,
ignoring that (1) alphas decay, so some of what it trades into is gone by the time
it fully arrives, and (2) today's trade sets tomorrow's starting point, so a chain
of myopic solves over-trades relative to the true dynamic optimum whenever alphas
are persistent.

Gârleanu & Pedersen (2013) showed that with quadratic trading costs and
exponentially-decaying signals, the dynamic program collapses to two rules:

- **The aim portfolio** - the Markowitz solve on alphas *discounted* by
  ``κ/(κ+φ)`` per signal (``φ`` its decay rate, ``κ`` the trading rate below) -
  "aim in front of the target": construct the portfolio you'll still want once you
  get there, not the one you want today.
- **Partial adjustment** - trade a fraction ``κ ∈ (0,1]`` of the gap each period,
  ``w_t = w_{t-1} + κ·(aim_t - w_{t-1})``, ``κ`` increasing in risk (urgency) and
  decreasing in cost.

This module derives ``κ`` from the book's own risk-aversion/variance and cost
curvature (a closed form checked against GP's limiting cases below), applies the
decay discount, and composes the partial-adjustment step with 016's own no-trade
band via 016's *existing* proximal-projection machinery - it wraps
:class:`~src.portfolio.optimizer.MeanVarianceOptimizer`, never modifies it. A cost
curvature that can't be estimated (no capital, or a trade too small to pin the
√-impact term) is not an error - the policy degrades gracefully to exactly 016's
myopic, cost-aware solve (see :func:`derive_kappa`).

**κ derivation (the PR's own math).** Single-asset discrete-time LQ control with
quadratic risk ``λσ²w²`` and quadratic trading cost ``(c₂/2)Δw²`` (our real cost is
linear + 3/2-power; ``c₂`` is its local curvature at the book's typical trade size,
see :func:`src.costs.parametric.cost_curvature` - hidden factor 4, an approximation
validated by the net-of-cost A/B, not asserted exact). The Bellman value function
``V(w,f) = -Aw² + Bwf - Cf²`` for a signal decaying as ``f_{t+1} = δf_t`` gives, by
matching quadratic coefficients (worked in full in the spec's PR discussion):
``L := λσ² + A`` solves ``L² - λσ²·L - λσ²c₂/2 = 0``, so with ``s = λσ²``:

    L = (s + √(s² + 2·s·c₂)) / 2         κ = 2L/(2L + c₂)

which simplifies to the closed form in :func:`derive_kappa` - with **no dependence
on φ** in κ itself (φ only enters the discount, below): the trading *speed*
depends on risk/cost alone; *what you aim at* depends on decay. Limiting cases,
both exact and tested (``test_kappa_limits``): ``c₂ → 0 ⇒ κ → 1`` (costless: no
curvature to solve against - see the fallback note below); ``c₂ → ∞ ⇒ κ → 0``;
monotone (``∂κ/∂c₂ < 0`` for all ``c₂ ≥ 0``, proof in the PR) ⇒ κ increasing in
``s = λ_A·σ²``, decreasing in cost.

Solving the *other* coefficient-matching equation (the ``B``/``wf`` coefficient)
exactly - using the algebraic identity ``2Lθ = c₂·a`` that falls out of the same
fixed point - gives the *aim* coefficient (the weight on ``f`` in the optimal
policy) **exactly**: ``μ/a = a / [2λσ²·(1 − δ(1−a))]`` where ``δ = e^(−φ)`` is the
per-rebalance persistence - i.e. the discount on alpha is ``κ / (1 − δ(1−κ))``
(:func:`discount_factor`), not merely its small-φ limit ``κ/(κ+φ)`` (which is
what a first pass at this derivation gives, and is commonly quoted this way, but
whose error *grows* with φ - a real bug in an earlier draft of this module,
caught by ``test_synthetic_gp_world...`` disagreeing with a direct value-
iteration solve of the Bellman recursion at large φ). Since the exact form costs
nothing extra to evaluate, :func:`discount_factor` uses it directly.

**Why "costless ⇒ exact 016" is a fallback, not a limit.** Plugging ``c₂ = 0``
into the closed form gives ``κ = 1`` exactly, and at ``κ=1`` the exact discount
formula above *does* correctly give 1 (``δ(1−1) = 0``) for any φ - so the
costless case is consistent either way here. The fallback is still needed for a
different reason: :func:`derive_kappa` has no way to fit ``c₂`` at all without an
impact term (no capital, or a linear-only cost model) - "undefined" is not the
same number as "zero", and treating it as zero would silently invent a κ=1 from
a curvature that was never actually measured. So the policy's job is to
recognize "no curvature to fit" and hand
back the exact myopic solve, undiscounted, rather than apply a discount formula
whose derivation assumed a nonzero cost. :func:`derive_kappa` returns ``None``
whenever the cost curvature is undefined (no impact term, or too small a typical
trade to pin it), and :func:`build_aim_portfolio` treats ``None`` as "run 016
exactly, band-only" - this is what makes the costless limiting case an *exact*,
tested reduction (``test_costless_reduces_to_exact_016``) instead of an asymptote.
"""

import math
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from src.alphas.base import Alpha
from src.portfolio.optimizer import MeanVarianceOptimizer, PortfolioResult
from src.risk.base import RiskMatrix

#: Floor on the typical per-name trade size fed to the curvature fit (Spec 022
#: §3.2) - avoids a near-zero denominator manufacturing an absurd κ when the myopic
#: reference solve traded almost nothing (e.g. it was already near its no-trade band).
MIN_TYPICAL_TRADE = 1e-4


# --------------------------------------------------------------------------- #
# Pure math: decay units, the discount, and the trading rate κ
# --------------------------------------------------------------------------- #
def phi_from_half_life(half_life_periods: float, min_half_life: float = 1.0) -> float:
    """Per-rebalance decay rate ``φ = ln2 / HL``, floored at ``min_half_life`` (Spec
    022 hidden factor 1: can't measure decay faster than the rebalance frequency
    itself - a floor of 1 rebalance is the fastest a half-life can mean anything).
    A permanent signal (``HL = inf``) has ``φ = 0`` - no discount at any κ.
    """
    if not math.isfinite(half_life_periods):
        return 0.0
    return math.log(2) / max(half_life_periods, min_half_life)


def half_life_in_rebalance_units(half_life_periods: float, periods_per_rebalance: float) -> float:
    """Convert a half-life measured in raw bars (012's unit) into rebalance units
    (022's unit) - the unit-discipline hidden factor: silently mixing bars and
    rebalances would misprice every discount by the cadence factor.
    ``periods_per_rebalance`` is the bar count between rebalances (e.g. ~21 for a
    monthly cadence on daily bars). A non-finite half-life (permanent signal) or a
    non-positive cadence passes through unchanged.
    """
    if not math.isfinite(half_life_periods) or periods_per_rebalance <= 0:
        return half_life_periods
    return half_life_periods / periods_per_rebalance


def discount_factor(kappa: float, phi: float) -> float:
    """The aim-alpha decay discount (Spec 022 §3.1), the **exact** closed form:

        discount = κ / (1 − δ·(1−κ)),   δ = e^(−φ)

    (derived from the same Riccati fixed point as :func:`derive_kappa` - see the
    module docstring - by solving the *other* coefficient-matching equation,
    ``B(1−δθ) = θ·β``, exactly, with no small-φ linearization). This is
    numerically verified against a direct value-iteration solve of the Bellman
    recursion (see ``test_discount_factor_matches_value_iteration``): the
    frequently-quoted ``κ/(κ+φ)`` form is only this formula's small-φ limit
    (``δ ≈ 1−φ`` ⇒ ``1−δ(1−κ) ≈ φ+κ``) and its error *grows* with φ - exact is no
    more expensive to compute, so there is no reason to use the approximation.

    ``φ = 0`` (a permanent signal) ⇒ ``δ = 1`` ⇒ discount ``= 1`` at any κ,
    exactly - no discount ever applies to a signal that doesn't decay. ``κ → 0``
    (trading nearly frozen) ⇒ discount ``→ κ/(1−δ)`` for any φ > 0, which
    coincides with the familiar ``κ/φ`` in the same small-φ limit.
    """
    if phi <= 0:
        return 1.0
    if kappa <= 0:
        return 0.0
    delta = math.exp(-phi)
    denom = 1.0 - delta * (1.0 - kappa)
    return kappa / denom if denom > 0 else 1.0


def derive_kappa(risk_aversion: float, variance: float, cost_curvature: Optional[float]) -> Optional[float]:
    """Gârleanu-Pedersen's trading rate, closed form (derived in this module's
    docstring): ``s = λ_A·σ²``, ``κ = (s + √(s²+2sc₂)) / (s + c₂ + √(s²+2sc₂))``.

    Monotone increasing in ``s`` (risk/urgency), decreasing in ``c₂`` (cost
    curvature) - ``test_kappa_monotonicity`` grids this. ``κ → 1`` as ``c₂ → 0``;
    ``κ → 0`` as ``c₂ → ∞`` (``test_kappa_limits``).

    Returns ``None`` - the "cost curvature ill-defined" signal - when ``c₂`` is
    missing/non-positive (no impact term, or a trade too small to pin it) or when
    ``risk_aversion·variance`` itself is non-positive. The caller
    (:func:`build_aim_portfolio`) treats ``None`` as "fall back to 016 exactly,
    band-only" (Spec 022 §3.2's own documented fallback), not an error.
    """
    if cost_curvature is None or not (cost_curvature > 0):
        return None
    s = risk_aversion * variance
    if not (s > 0):
        return None
    root = math.sqrt(s * s + 2.0 * s * cost_curvature)
    return (s + root) / (s + cost_curvature + root)


def trading_half_life(kappa: float) -> float:
    """Implied trading half-life, in rebalances: ``ln2/κ`` (Spec 022 §3.2's own
    reported diagnostic). Infinite when κ is non-positive (frozen book)."""
    return math.log(2) / kappa if kappa > 0 else float("inf")


# --------------------------------------------------------------------------- #
# The policy: aim construction + partial adjustment, wrapping the optimizer
# --------------------------------------------------------------------------- #
def build_aim_portfolio(
    optimizer: MeanVarianceOptimizer,
    alphas: List[Alpha],
    risk: RiskMatrix,
    *,
    phi: float = 0.0,
    trade_rate: Optional[float] = None,
    target_te: Optional[float] = None,
    risk_aversion: Optional[float] = None,
    current_weights: Optional[Dict[str, float]] = None,
    cost_model=None,
    cost_inputs=None,
    capital: Optional[float] = None,
    holding_period_years: float = 1.0 / 12.0,
) -> PortfolioResult:
    """The Spec 022 policy: derive κ, discount the alphas by ``κ/(κ+φ)``, solve the
    cost-free aim (016 with the cost term zeroed), take the κ-fraction step from
    ``current_weights``, and pass the result through 016's own proximal band
    projection (the *same* box/budget/no-trade-band machinery a real cost-aware
    016 solve would apply - reused via :meth:`MeanVarianceOptimizer._prox_project`,
    never re-derived) so the band and κ compose rather than stack independently
    (Spec 022 hidden factor 3).

    ``phi`` is the (already per-rebalance, already CI-adjusted) decay rate for
    this alpha vector - a single scalar, since v1 is a single combined signal per
    call (Spec 022 §7: scalar κ/φ is the v1 lean; per-signal only matters upstream
    of 013's combination, where :func:`discount_factor` is the same one-line vector
    scale applied per signal before ``combine_scores`` - no change needed here).

    ``trade_rate`` overrides the derived κ (the CLI's ``--trade-rate``). Without a
    usable cost curvature (no capital, or the myopic reference solve traded too
    little to pin one) and no override, this **falls back to exactly the plain
    016 cost-aware solve** - the costless/undefined-curvature reduction is exact,
    not asymptotic (see the module docstring). Position constraints (box,
    cardinality, min-weight) are enforced on the *aim* solve itself (it is a full
    016 solve), satisfying hidden factor 6 - the final κ-adjusted-and-banded trade
    then respects the box via the proximal projection, though a rebalance whose
    ``current_weights`` and aim disagree sharply on *which* names to hold can, in
    principle, transiently exceed ``max_names`` until the next full re-solve (a
    documented v1 limitation, not exercised when ``max_names`` is unset).

    Book scope: **long-only only** in v1 (the spec is silent on market-neutral;
    the final banding step reuses the long-only proximal projection). Benchmark-
    relative (Spec 017) books are also out of v1's scope - the aim/myopic split
    here works in cash-relative space only.
    """
    myopic = optimizer.optimize(
        alphas,
        risk,
        target_te=target_te,
        risk_aversion=risk_aversion,
        current_weights=current_weights,
        cost_model=cost_model,
        cost_inputs=cost_inputs,
        capital=capital,
        holding_period_years=holding_period_years,
    )
    if not myopic.feasible:
        return myopic

    symbols, alpha_vec = MeanVarianceOptimizer._align(alphas, risk)
    if len(symbols) == 0:
        return myopic

    sigma = MeanVarianceOptimizer._submatrix(risk, symbols)
    lam = float(myopic.diagnostics["risk_aversion"])
    c_lin, k_imp = MeanVarianceOptimizer._cost_coefficients(
        cost_model, cost_inputs, capital, holding_period_years, symbols
    )

    names_traded = myopic.diagnostics.get("names_traded")
    turnover = float(myopic.diagnostics.get("turnover", 0.0))
    typical_trade = (turnover / names_traded) if names_traded else MIN_TYPICAL_TRADE
    typical_trade = max(typical_trade, MIN_TYPICAL_TRADE)

    mean_k = float(np.mean(k_imp)) if k_imp.size else 0.0
    typical_variance = float(np.mean(np.diag(sigma)))
    from src.costs.parametric import cost_curvature

    c2 = cost_curvature(mean_k, typical_trade)
    kappa_derived = derive_kappa(lam, typical_variance, c2)
    kappa = trade_rate if trade_rate is not None else kappa_derived

    if kappa is None or not (kappa > 0):
        result = PortfolioResult(
            weights=dict(myopic.weights),
            feasible=True,
            binding_constraint=myopic.binding_constraint,
            diagnostics={
                **myopic.diagnostics,
                "policy": "myopic_fallback",
                "aim_degraded": True,
                "fallback_reason": "cost curvature undefined (no capital/impact term, or "
                "trade too small to pin c2) and no --trade-rate override; degraded to "
                "016's plain cost-aware solve",
                "cost_curvature": c2,
                "kappa_derived": kappa_derived,
                "typical_trade_size": typical_trade,
            },
            unconstrained_weights=dict(myopic.unconstrained_weights),
        )
        return result

    discount = discount_factor(kappa, phi)
    discounted_alphas = [replace(a, alpha=a.alpha * discount) for a in alphas]

    aim = optimizer.optimize(
        discounted_alphas,
        risk,
        risk_aversion=lam,
        current_weights=current_weights,
        cost_model=None,
        cost_inputs=None,
        capital=None,
    )
    if not aim.feasible:
        return aim

    w_prev = MeanVarianceOptimizer._vector(current_weights or {}, symbols)
    w_aim = MeanVarianceOptimizer._vector(aim.weights, symbols)
    raw_target = w_prev + kappa * (w_aim - w_prev)

    lmax = float(np.linalg.eigvalsh(sigma).max())
    step = 1.0 / (2.0 * lam * lmax) if lmax > 0 else 1.0
    thr = step * c_lin
    kthr = step * k_imp
    w_final = optimizer._prox_project(raw_target, thr, kthr, w_prev)

    sigma_inv = np.linalg.inv(sigma)
    ir_star = float(np.sqrt(max(alpha_vec @ sigma_inv @ alpha_vec, 0.0)))
    zeros = np.zeros(len(symbols))
    cost_aware = bool(np.any(c_lin > 0) or np.any(k_imp > 0))
    diagnostics = optimizer._diagnostics(
        alpha_vec, sigma, w_final, w_prev, lam, ir_star, c_lin, k_imp, cost_aware, zeros, zeros, 0.0
    )
    diagnostics.update(
        {
            "policy": "aim",
            "aim_degraded": False,
            "kappa": kappa,
            "kappa_derived": kappa_derived,
            "kappa_overridden": trade_rate is not None,
            "trading_half_life_rebalances": trading_half_life(kappa),
            "cost_curvature": c2,
            "typical_trade_size": typical_trade,
            "phi_per_rebalance": phi,
            "signal_discount": discount,
        }
    )

    return PortfolioResult(
        weights={s: float(w_final[i]) for i, s in enumerate(symbols) if abs(w_final[i]) > 1e-9},
        feasible=True,
        diagnostics=diagnostics,
        unconstrained_weights=dict(aim.weights),
    )
