"""Bootstrap skill inference (spec 023): simulate the null instead of assuming it.

The rest of the stack's skill-vs-luck machinery is parametric: PSR/DSR
(:mod:`src.analytics.metrics`) assume the Sharpe estimator's asymptotic
distribution (with skew/kurtosis corrections) and, for DSR, an *assumed* effective
trial count and trial-Sharpe variance. Those are good, cheap, always-on defaults —
this module is the heavier, definitive check behind ``--bootstrap-skill``, not a
replacement.

Two questions, two functions:

1. **Is this one track record skill?** :func:`bootstrap_null` imposes the null
   (demean by the estimated alpha), block-resamples the residual with the
   Politis-Romano stationary bootstrap (never i.i.d. - active returns are
   autocorrelated), and reports the realized IR's rank in the empirical null
   distribution: an *own* p-value with no assumption about the return
   distribution's shape.
2. **Is the best of the K configs actually tried skill?** :func:`reality_check`
   (White 2000) resamples the same block indices across every trial's column at
   once, so the cross-trial correlation structure is preserved by construction
   (DSR has to *assume* a trial-Sharpe variance to approximate this; this replays
   the actual trials from the 026 trial store) - the null distribution of "the best
   IR among K correlated tries," compared against the actual observed best.

Both report a Monte-Carlo standard error on the p-value (``sqrt(p(1-p)/B)``) and a
block-length sensitivity check (p at L/2 and 2L) - a p-value that flips across that
range is not a result (spec 023 hidden factor 3).
"""

import math
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from src.analytics.metrics import TRADING_DAYS_PER_YEAR

Numbers = Union[Sequence[float], np.ndarray]

__all__ = [
    "politis_white_block_length",
    "stationary_bootstrap_indices",
    "bootstrap_null",
    "reality_check",
]


def _as_array(values: Numbers) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _annualized_ir(x: np.ndarray, periods_per_year: float, axis: int = -1) -> np.ndarray:
    """``mean/std * sqrt(periods_per_year)`` along ``axis``; zero-std slices get IR
    0 (never inf/nan - a null distribution must stay finite to rank against)."""
    mean = x.mean(axis=axis)
    std = x.std(axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ir = np.where(std > 0, mean / std * math.sqrt(periods_per_year), 0.0)
    return np.asarray(ir, dtype=float)


# --------------------------------------------------------------------------- #
# Block-length selection (Politis & White 2004 / Patton, Politis & White 2009)
# --------------------------------------------------------------------------- #
def politis_white_block_length(returns: Numbers, *, max_lag: Optional[int] = None) -> float:
    """The automatic expected block length for the stationary bootstrap.

    Estimates the series' own decay time from its correlogram (a flat-top lag
    window over the sample autocorrelations), then sets the MSE-optimal average
    block length for the stationary bootstrap (Politis & White 2004 eq. 4.1-4.2,
    the ``D_SB = 2*Ghat**2`` variant). Too short a block leaks autocorrelation into
    the resample and makes the p-value anti-conservative - the dangerous direction
    (spec 023 hidden factor 3) - so this is the *reported*, always-overridable
    default, never a silent assumption.

    This is an engineering-faithful implementation of the published rule (the same
    flat-top kernel, the same "insignificant for K_n consecutive lags" bandwidth
    rule) - not a byte-for-byte port of any particular reference implementation.
    No test in this codebase depends on matching one exactly, only on the
    calibration/power properties the block length is supposed to buy (see
    ``tests/test_bootstrap_skill.py``).
    """
    r = _as_array(returns)
    n = len(r)
    if n < 20:
        # Too short to estimate a correlogram meaningfully; fall back to a quarter
        # of the sample, floored at 1 - a conservative (long) block for tiny n.
        return max(float(n) / 4.0, 1.0)

    r = r - r.mean()
    var = float(np.dot(r, r) / n)
    if var <= 0:
        return 1.0

    k_max = max_lag or min(n // 2, 100)
    rho = np.ones(k_max + 1, dtype=float)
    for k in range(1, k_max + 1):
        rho[k] = float(np.dot(r[: n - k], r[k:]) / n) / var

    # Bandwidth m: the first lag beyond which the correlogram is "insignificant"
    # for K_n consecutive lags (Politis & White's own significance band, not a
    # generic 95% CI).
    k_n = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    band = 2.0 * math.sqrt(math.log10(n) / n)
    m = k_max
    for k in range(1, max(k_max - k_n + 1, 2)):
        if np.all(np.abs(rho[k : k + k_n]) < band):
            m = k
            break
    bandwidth = int(np.clip(2 * m, 1, k_max))

    def _flat_top(x: float) -> float:
        ax = abs(x)
        if ax <= 0.5:
            return 1.0
        if ax <= 1.0:
            return 2.0 * (1.0 - ax)
        return 0.0

    g_hat = 0.0
    cap_g_hat = rho[0]
    for k in range(1, bandwidth + 1):
        w = _flat_top(k / bandwidth)
        g_hat += 2.0 * w * k * rho[k]
        cap_g_hat += 2.0 * w * rho[k]

    if cap_g_hat <= 0:
        return float(max(bandwidth, 1))
    d_sb = 2.0 * cap_g_hat**2
    if d_sb <= 0:
        return float(max(bandwidth, 1))
    b = ((2.0 * g_hat**2) / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, max(n // 4, 1)))


# --------------------------------------------------------------------------- #
# Stationary (Politis-Romano 1994) block bootstrap
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(
    t: int, block_length: float, b: int, rng: np.random.Generator
) -> np.ndarray:
    """``B`` rows of ``T`` circular resample indices with geometric block lengths
    (expected length ``block_length``), Politis & Romano's (1994) stationary
    bootstrap.

    Implemented as its equivalent recurrence: at each step, "restart" to a fresh
    uniform index with probability ``p = 1/block_length``, else continue the
    previous index + 1 (mod T) - a geometric run length with mean
    ``block_length``. Vectorized over ``B`` (one restart draw per time step, not
    per ``(round, step)`` pair): the only large array actually materialized is the
    ``(B, T)`` index gather this feeds, plus a ``T``-length loop of ``B``-wide
    numpy ops.
    """
    if t <= 0:
        return np.empty((b, 0), dtype=np.int64)
    p = 1.0 / max(float(block_length), 1.0)
    idx = np.empty((t, b), dtype=np.int64)
    idx[0] = rng.integers(0, t, size=b)
    if t > 1:
        restarts = rng.random((t - 1, b)) < p
        fresh = rng.integers(0, t, size=(t - 1, b))
        for step in range(1, t):
            prev = idx[step - 1]
            idx[step] = np.where(restarts[step - 1], fresh[step - 1], (prev + 1) % t)
    return idx.T  # (B, T)


def _p_value(null: np.ndarray, observed: float, b: int) -> "tuple[float, float]":
    """One-sided p = P(null >= observed), with the standard +1 continuity
    correction (never reports p=0 from a finite B) and its Monte-Carlo SE."""
    p = float((1 + np.sum(null >= observed)) / (b + 1))
    se = float(math.sqrt(max(p * (1.0 - p), 0.0) / b))
    return p, se


def _flips_significance(p: float, p_half: float, p_double: float, level: float = 0.05) -> bool:
    sig = p < level
    return (sig != (p_half < level)) or (sig != (p_double < level))


# --------------------------------------------------------------------------- #
# 1. Single track record: zero-alpha bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_null(
    returns: Numbers,
    *,
    B: int = 2000,
    block_length: Optional[float] = None,
    seed: int = 0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Zero-alpha stationary block bootstrap for one active-return track record.

    Demeans by the full-sample estimated alpha (Fama-French: ``r_tilde = r_a -
    alpha_hat``; ``alpha_hat`` defaults to the sample mean - pass ``alpha=`` for a
    regression-estimated intercept instead) to impose H0: no skill, block-resamples
    the residual ``B`` times (stationary bootstrap; never the equity curve, never
    an i.i.d. shuffle - active returns are autocorrelated), recomputes the
    annualized IR each round -> the empirical null distribution. The p-value is
    one-sided (``P(null IR >= observed IR)``): the hypothesis under test is
    "skill", so a negative track record correctly returns a high p (no evidence of
    skill needed to explain a loss).

    Reports the Monte-Carlo SE of p, the block length used (default:
    :func:`politis_white_block_length`, always overridable and always reported),
    and block-length sensitivity at L/2 and 2L (hidden factor 3: a p whose
    significance flips across that range is not a result, per
    ``block_sensitivity_flag``).
    """
    r = _as_array(returns)
    t = len(r)
    if t < 8:
        return {
            "ir_observed": 0.0,
            "p_value": 1.0,
            "p_se": 0.0,
            "B": 0,
            "block_length": 0.0,
            "seed": seed,
            "n_obs": t,
            "block_sensitivity": {},
            "block_sensitivity_flag": False,
            "insufficient_data": True,
        }

    a_hat = float(r.mean()) if alpha is None else float(alpha)
    ir_observed = float(_annualized_ir(r, periods_per_year))
    l_default = float(block_length) if block_length is not None else politis_white_block_length(r)
    resid = r - a_hat

    def _run(length: float):
        rng = np.random.default_rng(seed)
        idx = stationary_bootstrap_indices(t, length, B, rng)
        resampled = resid[idx]  # (B, T)
        null_ir = _annualized_ir(resampled, periods_per_year, axis=-1)
        p, se = _p_value(null_ir, ir_observed, B)
        return p, se, null_ir

    p_value, p_se, null_ir = _run(l_default)
    p_half, _, _ = _run(max(l_default / 2.0, 1.0))
    p_double, _, _ = _run(l_default * 2.0)

    return {
        "ir_observed": ir_observed,
        "p_value": p_value,
        "p_se": p_se,
        "B": B,
        "block_length": l_default,
        "seed": seed,
        "n_obs": t,
        "alpha_hat": a_hat,
        "null_ir_mean": float(np.mean(null_ir)),
        "null_ir_std": float(np.std(null_ir, ddof=1)) if B > 1 else 0.0,
        "block_sensitivity": {"half": p_half, "double": p_double},
        "block_sensitivity_flag": _flips_significance(p_value, p_half, p_double),
        "insufficient_data": False,
    }


# --------------------------------------------------------------------------- #
# 2. Best-of-K: White's Reality Check
# --------------------------------------------------------------------------- #
def reality_check(
    trial_returns_matrix: Any,
    *,
    B: int = 2000,
    block_length: Optional[float] = None,
    seed: int = 0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    trial_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """White's (2000) Reality Check over the T x K joint panel of trials' OOS
    return series.

    Every trial column is demeaned by its OWN mean (impose H0 independently per
    trial - each trial's own null is "no skill for that config"), then resampled
    with ONE set of block indices per round applied to *every* column at once -
    cross-trial correlation is preserved by construction, which is the entire
    point: DSR has to *assume* a trial-Sharpe variance to approximate this: this
    replays the actual trials. The per-round max IR across K columns is the null
    distribution of "the best IR among K correlated tries"; the family p-value is
    the rank of the actual observed max REAL (non-resampled) IR in it - the
    question a researcher who tried K configs and kept the best actually faces.

    Reports which trial achieved the observed max (``selected_trial``), every
    trial's own real IR, the Monte-Carlo SE of the family p, and the same
    block-length sensitivity check as :func:`bootstrap_null`.
    """
    m = np.asarray(trial_returns_matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
        return {
            "family_p": 1.0,
            "family_p_se": 0.0,
            "B": 0,
            "k_trials": 0 if m.ndim != 2 else int(m.shape[1]),
            "block_sensitivity": {},
            "block_sensitivity_flag": False,
            "insufficient_data": True,
        }
    t, k = m.shape
    if t < 8:
        return {
            "family_p": 1.0,
            "family_p_se": 0.0,
            "B": 0,
            "k_trials": k,
            "n_obs": t,
            "block_sensitivity": {},
            "block_sensitivity_flag": False,
            "insufficient_data": True,
        }

    real_ir = _annualized_ir(m, periods_per_year, axis=0)  # (K,)
    best_idx = int(np.argmax(real_ir))
    observed_max_ir = float(real_ir[best_idx])
    ids = list(trial_ids) if trial_ids is not None else list(range(k))

    resid = m - m.mean(axis=0, keepdims=True)
    l_default = (
        float(block_length) if block_length is not None else politis_white_block_length(m[:, best_idx])
    )

    def _run(length: float):
        rng = np.random.default_rng(seed)
        idx = stationary_bootstrap_indices(t, length, B, rng)  # (B, T)
        resampled = resid[idx]  # (B, T, K) - one gather, one set of block indices/round, shared by every column
        ir = _annualized_ir(resampled, periods_per_year, axis=1)  # (B, K)
        null_max = ir.max(axis=1)  # (B,)
        p, se = _p_value(null_max, observed_max_ir, B)
        return p, se, null_max

    family_p, family_p_se, null_max = _run(l_default)
    p_half, _, _ = _run(max(l_default / 2.0, 1.0))
    p_double, _, _ = _run(l_default * 2.0)

    return {
        "family_p": family_p,
        "family_p_se": family_p_se,
        "B": B,
        "block_length": l_default,
        "seed": seed,
        "k_trials": k,
        "n_obs": t,
        "observed_max_ir": observed_max_ir,
        "selected_trial": ids[best_idx],
        "selected_trial_index": best_idx,
        "per_trial_real_ir": {ids[i]: float(real_ir[i]) for i in range(k)},
        "null_max_ir_mean": float(np.mean(null_max)),
        "null_max_ir_std": float(np.std(null_max, ddof=1)) if B > 1 else 0.0,
        "block_sensitivity": {"half": p_half, "double": p_double},
        "block_sensitivity_flag": _flips_significance(family_p, p_half, p_double),
        "insufficient_data": False,
    }
