"""Information horizon - how fast an alpha decays, and what to do about it.

Every alpha is perishable: its forecasting power (IC) decays over time at a rate
that is an intrinsic property of the signal. Measuring that decay yields two free
wins a research engine should surface:

1. **The rebalance cadence is derivable** - trading much faster than the signal
   regenerates pays cost for noise; much slower throws away breadth. The half-life
   pins the sweet spot.
2. **The freshest signal is rarely optimal** - blending the current signal with a
   lagged copy systematically raises the information ratio (a second partly-
   independent read, or a hedge that cancels noise).

This module is the pure math: fit the decay from an IC-vs-lag profile (measured by
the information-analysis layer), and compute the optimal current/lagged blend.
"""

import math
from typing import Dict, Mapping, Tuple

import numpy as np

#: horizon/half-life ratio at which signal-return correlation peaks (≈ 0.638).
PEAK_HORIZON_RATIO = 1.2566


def fit_decay(ic_by_lag: Mapping[int, float], ci_z: float = 1.96) -> Dict[str, float]:
    """Fit the per-period decay ``δ`` from an IC-vs-lag profile by log-linear regression.

    Models ``IC(n) = IC₀ · δ^n`` ⇒ ``ln IC(n) = ln IC₀ + n·ln δ``, fit on the lags with
    a positive IC (the reliable, same-sign region). Returns ``δ``, the half-life
    ``HL = −ln2 / ln δ``, the fitted ``IC₀``, and the fit ``R²`` (low R² ⇒ the decay
    isn't a clean exponential - don't over-trust the half-life).

    Also fits the OLS standard error of the decay slope and reports the
    ``ci_z``-sigma confidence band on the half-life (``half_life_lower``,
    ``half_life_upper``) - short histories give wide CIs on the half-life, and
    the multi-period trading policy's aim discount uses this band rather than
    trusting the point estimate. ``half_life_upper`` (a less-negative slope, i.e. slower
    decay) is the direction that avoids prematurely discounting a genuinely
    persistent signal on a noisy short history; ``half_life_lower`` is the
    faster-decay end. Both collapse to the point estimate when the slope's SE isn't
    defined (fewer than 3 usable lags).
    """
    lags = sorted(n for n, ic in ic_by_lag.items() if ic > 0 and n >= 0)
    if len(lags) < 2:
        return {
            "delta": float("nan"),
            "half_life": float("nan"),
            "ic0": float("nan"),
            "r_squared": 0.0,
            "half_life_lower": float("nan"),
            "half_life_upper": float("nan"),
            "decay_slope_se": float("nan"),
        }

    x = np.array(lags, dtype=float)
    y = np.log(np.array([ic_by_lag[n] for n in lags], dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    delta = float(math.exp(slope))
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    half_life = -math.log(2) / math.log(delta) if 0 < delta < 1 else float("inf")

    n = len(lags)
    sxx = float(np.sum((x - x.mean()) ** 2))
    slope_se = math.sqrt(ss_res / (n - 2) / sxx) if n > 2 and sxx > 0 else float("nan")

    def _half_life_at(s: float) -> float:
        return -math.log(2) / s if s < 0 else float("inf")

    if slope_se == slope_se:  # not NaN
        half_life_upper = _half_life_at(slope + ci_z * slope_se)  # less negative -> slower decay
        half_life_lower = _half_life_at(slope - ci_z * slope_se)  # more negative -> faster decay
    else:
        half_life_upper = half_life
        half_life_lower = half_life

    return {
        "delta": delta,
        "half_life": half_life,
        "ic0": float(math.exp(intercept)),
        "r_squared": r_squared,
        "half_life_lower": half_life_lower,
        "half_life_upper": half_life_upper,
        "decay_slope_se": slope_se,
    }


def peak_return_horizon(half_life: float) -> float:
    """The forward-return horizon a signal of this half-life predicts best (≈ 1.26·HL).

    Scoring a short-horizon signal against a long-horizon return (or vice versa) is a
    common way to understate real skill; this is the horizon to match.
    """
    return PEAK_HORIZON_RATIO * half_life if math.isfinite(half_life) else float("inf")


def blend_weights(gamma: float, rho: float) -> Tuple[float, float]:
    """IR-maximizing weights on the current vs lagged signal: ``(w_now, w_lag)``.

    ``w_now = (1 − γ·ρ) / (1 + γ² − 2·γ·ρ)`` (decay ``γ``, current/lagged
    autocorrelation ``ρ``); ``w_lag = 1 − w_now``. Three regimes:

    - ``γ > ρ`` → **diversify**: ``w_lag > 0`` (the lag carries independent info).
    - ``γ < ρ`` → **hedge**: ``w_lag < 0`` (it mostly cancels current noise).
    - ``γ = ρ`` → the latest signal alone is sufficient (``w_lag = 0``).
    """
    denom = 1.0 + gamma * gamma - 2.0 * gamma * rho
    if denom == 0:
        return 1.0, 0.0
    w_now = (1.0 - gamma * rho) / denom
    return w_now, 1.0 - w_now


def frequency_ir_curve(ic_by_lag: Mapping[int, float]) -> Dict[int, float]:
    """The accuracy-vs-frequency curve ``IR(Δt) = IC(Δt)·√(1/Δt)`` over candidate cadences.

    Faster rebalancing raises ``√BR`` but acts on less-confirmed information (lower
    ``IC``); the optimum is interior. Returns ``{cadence: IR_proxy}`` so the cadence is
    *chosen*, not assumed.
    """
    curve = {}
    for dt, ic in ic_by_lag.items():
        if dt >= 1:
            curve[dt] = float(ic * math.sqrt(1.0 / dt))
    return curve


def recommended_cadence(ic_by_lag: Mapping[int, float]) -> int:
    """The cadence (in periods) that maximizes the IR proxy ``IC(Δt)·√(1/Δt)``."""
    curve = frequency_ir_curve(ic_by_lag)
    return max(curve, key=curve.get) if curve else 1


def effective_sample_size(n_obs: float, horizon: int, spacing: float = 1.0) -> float:
    """Effective *independent* observations when each spans ``horizon`` periods.

    An IC measured from returns over a ``horizon``-period holding, sampled every
    ``spacing`` periods, overlaps: consecutive observations share ``horizon − spacing``
    periods of return and so are not independent. The independent count is
    ``n_obs / overlap`` with ``overlap = max(1, horizon/spacing)`` — daily rows with a
    21-day horizon (``spacing=1``) are ~21× overlapped, so ``T_eff ≈ n/21``; rebalances
    already spaced ``≥ horizon`` apart (``spacing ≥ horizon``) are independent and
    ``T_eff = n``. This is the honest ``T`` the IC-uncertainty shrink
    (:func:`tradeflow.alphas.refine.level_shrink_factor`) needs: using the raw row count
    under-shrinks by exactly the overlap factor.
    """
    overlap = max(1.0, horizon / max(spacing, 1e-9))
    return float(n_obs / overlap) if overlap > 0 else float(n_obs)
