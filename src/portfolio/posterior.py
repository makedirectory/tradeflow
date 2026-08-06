"""Black–Litterman posterior: blend our views with Σ's implied propagation.

The reverse-optimized consensus (``src/portfolio/benchmark.py``) backs out the
expected returns for which the benchmark is itself optimal; our alphas are
*deviations from* that consensus. Left alone, the optimizer reads a name with no
signal as a *hard* zero-view ("this name earns exactly consensus") rather than as
ignorance, and treats every view's confidence as baked into its magnitude alone.
Black–Litterman fixes both: a single precision-weighted blend of the (zero, in
residual space) prior and our views, where prior precision comes from ``(τΣ)⁻¹``
and view precision from measured IC. Names we never scored still get a real,
nonzero posterior when they're correlated with a name we *did* score ("never zero
the correlated forecasts") — the propagation is the entire point.

Pure numpy, no I/O. Two layers:

- :func:`black_litterman` — the update itself, given ``Ω`` directly (a diagonal
  view-covariance dict). This is the primitive the limit tests (Ω→0, Ω→∞) drive.
- :func:`black_litterman_from_ic` — derives ``Ω`` from IC via the calibration
  identity (:func:`view_variance`) and calls the primitive. This is what
  ``construct_portfolio`` calls.

Everything here is in **residual/active space**: the prior mean is exactly 0, not
because our alphas happen to be benchmark-neutral, but because the
reverse-optimization consensus is *defined* as the return explained by beta — by
construction there is zero consensus residual return left over. ``q`` (the views)
must be the *unshrunk* refined alphas (the refinement's own level-shrink stage
left off upstream) — Ω below already carries the full IC-uncertainty haircut, so
shrinking twice would halve it twice.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.alphas.refine import level_shrink_factor
from src.risk.base import RiskMatrix


@dataclass
class BLPosterior:
    """The BL posterior over Σ's full universe, plus the audit trail (τ-sensitivity, sourcing)."""

    mu_post: Dict[str, float]  # symbol -> posterior residual return, EVERY Σ symbol
    source: Dict[str, str]  # symbol -> "view" | "propagated" | "prior"
    views: Dict[str, float] = field(default_factory=dict)  # symbol -> q, covered names only
    omega: Dict[str, float] = field(default_factory=dict)  # symbol -> view variance, covered only
    tau: float = 0.0
    tau_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Set only by :func:`black_litterman_from_ic` (the low-level update takes Ω
    #: directly and has no notion of a single IC/T_eff behind it).
    ic: Optional[float] = None
    t_eff: Optional[float] = None


def view_variance(residual_vol: float, ic: float, t_eff: float) -> float:
    """The calibration-forced view-error variance ``Ω_n = ω_n² / (T_eff²·IC²)``.

    Not the naive ``ω²·(1−IC²)`` — that formula alone prices only "the signal
    isn't perfectly correlated with truth," and ignores that IC itself is
    *estimated* over a finite ``T_eff``. The calibration identity (a single
    covered-name view, uninformative prior, pinned ``τ = 1/T_eff``, must reproduce
    the refinement's own level-shrunk alpha exactly — the ``g/(g+1)`` weight
    :func:`~src.alphas.refine.level_shrink_factor` already computes, ``g =
    T_eff·IC²``) forces this closed form: the BL blend weight on a single view is
    ``τω²/(τω²+Ω)``; setting that equal to ``g/(g+1)`` with ``τ = 1/T_eff`` and
    solving for ``Ω`` gives exactly ``ω²/(T_eff²·IC²)``.

    Returns ``inf`` when ``ic`` is zero/non-finite or ``t_eff`` is non-positive (an
    uninformative view carries zero precision, i.e. infinite variance) — callers
    that build a diagonal Ω matrix must drop such views rather than pass ``inf``
    through a linear solve (see :func:`black_litterman_from_ic`).
    """
    if not np.isfinite(ic) or ic == 0 or t_eff <= 0:
        return float("inf")
    return float((residual_vol**2) / (t_eff**2 * ic**2))


def expected_single_view_weight(ic: float, t_eff: float) -> float:
    """The blend weight a lone covered-name view must land on (the calibration identity).

    Reuses :func:`~src.alphas.refine.level_shrink_factor` directly (not a second,
    independently-derived formula) so the calibration test is a tautology by
    construction if :func:`view_variance` is right, and a real check if it isn't.
    """
    return level_shrink_factor(ic, t_eff)


def calibration_gap(posterior: "BLPosterior") -> Optional[float]:
    """For a single-view posterior, ``actual_weight − expected_weight``.

    ``actual_weight = μ_post,n / q_n`` for the one covered name; ``expected_weight``
    is :func:`expected_single_view_weight`. Near 0 is the pass condition — anything
    else means Ω is mis-derived relative to the refinement's level shrink (double- or
    under-shrunk). Returns ``None`` when the posterior isn't single-view (the K=N
    matrix-form generalization of this same identity is a future extension) or
    when ``ic``/``t_eff`` weren't recorded (a raw :func:`black_litterman` call).
    """
    if len(posterior.views) != 1 or posterior.ic is None or posterior.t_eff is None:
        return None
    ((sym, q),) = posterior.views.items()
    if q == 0:
        return None
    actual = posterior.mu_post[sym] / q
    expected = expected_single_view_weight(posterior.ic, posterior.t_eff)
    return float(actual - expected)


def black_litterman(
    views: Dict[str, float],
    risk: RiskMatrix,
    omega: Dict[str, float],
    tau: float,
) -> BLPosterior:
    """The BL update in residual/active space:
    ``μ_post = τΣPᵀ(PτΣPᵀ+Ω)⁻¹q``.

    Zero-prior-mean, one ``(K×K)`` solve (``K`` = number of covered names — cheap at
    this scale). ``views`` (``q``) and ``omega`` (``Ω``, diagonal, one entry per
    view — exactly one view per name, already decorrelated/combined upstream by
    the multi-signal combination) are keyed by symbol; any symbol in ``views``
    absent from ``risk.symbols`` is ignored. Also computes the τ-sensitivity
    (posterior recomputed at ``τ/2`` and ``2τ``).
    """
    symbols = list(risk.symbols)
    idx = {s: i for i, s in enumerate(symbols)}
    sigma = risk.sigma

    covered = [s for s in views if s in idx and s in omega]
    cov_idx = [idx[s] for s in covered]
    q = np.array([views[s] for s in covered], dtype=float)
    omega_diag = np.array([omega[s] for s in covered], dtype=float)

    tau_eff = max(float(tau), 0.0)
    mu_post_vec = _update(sigma, cov_idx, q, omega_diag, tau_eff)

    tol = 1e-12
    source: Dict[str, str] = {}
    for s in symbols:
        if s in covered:
            source[s] = "view"
        elif abs(mu_post_vec[idx[s]]) > tol:
            source[s] = "propagated"
        else:
            source[s] = "prior"

    sensitivity: Dict[str, Dict[str, float]] = {}
    for label, t in (("tau_half", tau_eff / 2.0), ("tau_double", tau_eff * 2.0)):
        vec = _update(sigma, cov_idx, q, omega_diag, t)
        sensitivity[label] = {s: float(vec[idx[s]]) for s in symbols}

    return BLPosterior(
        mu_post={s: float(mu_post_vec[idx[s]]) for s in symbols},
        source=source,
        views={s: float(views[s]) for s in covered},
        omega={s: float(omega[s]) for s in covered},
        tau=tau_eff,
        tau_sensitivity=sensitivity,
    )


def black_litterman_from_ic(
    views: Dict[str, float],
    risk: RiskMatrix,
    ic: float,
    t_eff: float,
    tau: Optional[float] = None,
) -> BLPosterior:
    """:func:`black_litterman` with Ω derived from IC and τ pinned to
    ``1/T_eff`` — what ``construct_portfolio`` calls.

    A view whose derived Ω is non-finite (``ic`` zero/non-finite, or ``t_eff`` ≤ 0 —
    an uninformative signal or unmeasurable sample) is dropped from the covered set
    entirely rather than passed through as ``inf`` (which would poison the linear
    solve); its name still gets a posterior, either 0 (prior) or a nonzero
    propagated value from OTHER views, exactly as if it had never had a signal.
    """
    symbols = list(risk.symbols)
    idx = {s: i for i, s in enumerate(symbols)}
    filtered_views: Dict[str, float] = {}
    omega: Dict[str, float] = {}
    for s, q in views.items():
        if s not in idx or q != q:  # not covered by Σ, or NaN
            continue
        vol = float(np.sqrt(max(risk.sigma[idx[s], idx[s]], 0.0)))
        om = view_variance(vol, ic, t_eff)
        if np.isfinite(om):
            filtered_views[s] = q
            omega[s] = om

    tau_eff = float(tau) if tau is not None else (1.0 / t_eff if t_eff > 0 else 0.0)
    posterior = black_litterman(filtered_views, risk, omega, tau_eff)
    posterior.ic = float(ic)
    posterior.t_eff = float(t_eff)
    return posterior


def _update(
    sigma: np.ndarray, cov_idx: List[int], q: np.ndarray, omega_diag: np.ndarray, tau: float
) -> np.ndarray:
    """``μ_post = τΣPᵀ(PτΣPᵀ+Ω)⁻¹q`` for one ``τ`` — the shared inner solve."""
    n = sigma.shape[0]
    if not cov_idx:
        return np.zeros(n)
    tau_sigma_pt = tau * sigma[:, cov_idx]  # (n x K): τΣPᵀ
    m = tau_sigma_pt[cov_idx, :] + np.diag(omega_diag)  # (K x K): PτΣPᵀ + Ω
    x = np.linalg.solve(m, q)
    return tau_sigma_pt @ x
