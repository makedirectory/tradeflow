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


def fit_decay(ic_by_lag: Mapping[int, float]) -> Dict[str, float]:
    """Fit the per-period decay ``δ`` from an IC-vs-lag profile by log-linear regression.

    Models ``IC(n) = IC₀ · δ^n`` ⇒ ``ln IC(n) = ln IC₀ + n·ln δ``, fit on the lags with
    a positive IC (the reliable, same-sign region). Returns ``δ``, the half-life
    ``HL = −ln2 / ln δ``, the fitted ``IC₀``, and the fit ``R²`` (low R² ⇒ the decay
    isn't a clean exponential - don't over-trust the half-life).
    """
    lags = sorted(n for n, ic in ic_by_lag.items() if ic > 0 and n >= 0)
    if len(lags) < 2:
        return {"delta": float("nan"), "half_life": float("nan"), "ic0": float("nan"), "r_squared": 0.0}

    x = np.array(lags, dtype=float)
    y = np.log(np.array([ic_by_lag[n] for n in lags], dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    delta = float(math.exp(slope))
    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    half_life = -math.log(2) / math.log(delta) if 0 < delta < 1 else float("inf")
    return {
        "delta": delta,
        "half_life": half_life,
        "ic0": float(math.exp(intercept)),
        "r_squared": r_squared,
    }


def peak_return_horizon(half_life: float) -> float:
    """The forward-return horizon a signal of this half-life predicts best (≈ 1.26·HL).

    Scoring a short-horizon signal against a long-horizon return (or vice versa) is a
    common way to understate real skill; this is the horizon to match.
    """
    return PEAK_HORIZON_RATIO * half_life if math.isfinite(half_life) else float("inf")


def blend_weights(gamma: float, rho: float) -> Tuple[float, float]:
    """IR-maximising weights on the current vs lagged signal: ``(w_now, w_lag)``.

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
    """The cadence (in periods) that maximises the IR proxy ``IC(Δt)·√(1/Δt)``."""
    curve = frequency_ir_curve(ic_by_lag)
    return max(curve, key=curve.get) if curve else 1
