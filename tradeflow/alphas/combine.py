"""Multi-signal alpha combination & shrinkage.

One signal becomes one alpha (:mod:`tradeflow.alphas`). Real research has several - a
trend read, a volume read, a mean-reversion read - and they are **correlated**.
Two mistakes follow if you ignore that:

1. **Naive weighting double-counts.** Weighting by raw IC over-weights redundant
   signals (three flavors of trend look like three bets but are one) and
   under-weights a weak-but-independent signal that adds the most.
2. **Estimated ICs are themselves uncertain**, and that uncertainty should shrink
   the contribution toward zero - more for short histories and low ICs.

This module adds both correctives: optimal combination via the signal
**correlation matrix**, and **Bayesian IC-uncertainty shrinkage**. The output is a
single combined score (with its combined IC), which flows through the same
:func:`~tradeflow.alphas.base.refine_alpha` pipeline - so the per-signal assumed IC is
replaced by one measured, shrunk, redundancy-aware number, never applied twice.

The ICs and the correlation matrix must be **measured** (here, over a trailing
window of realized residual returns), not assumed - in-sample/assumed ICs would let
the combination over-fit its own weights.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from tradeflow.alphas import refine
from tradeflow.indicators import indicators

Scorer = Callable[[pd.DataFrame], float]

#: Tiny ridge added to the signal correlation matrix before inverting it. Keeps the
#: GLS combination finite when two signals are near-duplicates (ρ → 1 ⇒ singular).
DEFAULT_RIDGE = 1e-4


# --------------------------------------------------------------------------- #
# The combination math (pure, closed-form)
# --------------------------------------------------------------------------- #
def shrink_ic(ic: float, n_obs: int) -> float:
    """Bayesian IC-uncertainty shrinkage: ``ic' = ic · g/(g+1)``, ``g = n·ic²``.

    Confidence grows with the number of observations ``n`` and the signal's own
    explanatory power ``ic²``. So ``ic' → 0`` as ``n`` is small or ``ic → 0``, and
    ``ic' → ic`` as ``n·ic² → ∞``. This is what stops a noisy, short-history IC from
    being trusted at face value.
    """
    g = n_obs * ic * ic
    return float(ic * g / (g + 1.0)) if g > 0 else 0.0


def effective_ic(ic1: float, ic2: float, rho: float) -> float:
    """Redundancy-corrected effective IC of signal 1 given signal 2 (two-signal).

    ``IC₁' = (IC₁ − ρ·IC₂) / (1 − ρ²)`` - the part of signal 1's skill that is not
    already explained by signal 2.
    """
    denom = 1.0 - rho * rho
    if denom <= 0:
        return 0.0
    return (ic1 - rho * ic2) / denom


def _regularized(corr: np.ndarray, ridge: float) -> np.ndarray:
    c = np.asarray(corr, dtype=float)
    return c + ridge * np.eye(c.shape[0])


def combination_weights(ics: np.ndarray, corr: np.ndarray, ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    """Optimal signal weights ``w = Ω⁻¹ · IC`` (GLS on the signal correlation matrix).

    These maximize the combined IC. Because ``Ω⁻¹`` accounts for correlation, two
    redundant signals **split** a weight rather than each getting full credit - the
    built-in defense against double-counting. A ridge regularizes the inverse so a
    near-duplicate pair (ρ → 1) stays finite.
    """
    ic = np.asarray(ics, dtype=float)
    return np.linalg.solve(_regularized(corr, ridge), ic)


def combined_ic(ics: np.ndarray, corr: np.ndarray, ridge: float = DEFAULT_RIDGE) -> float:
    """The combined information coefficient ``√(ICᵀ Ω⁻¹ IC)``.

    For two signals this is exactly ``√((IC₁² + IC₂² − 2ρ·IC₁·IC₂)/(1 − ρ²))``.
    Adding a weak *independent* signal raises it; adding a strong *redundant* one
    barely does.
    """
    ic = np.asarray(ics, dtype=float)
    val = float(ic @ np.linalg.solve(_regularized(corr, ridge), ic))
    return float(np.sqrt(max(val, 0.0)))


def combine_scores(score_frame: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted sum of per-signal cross-sectional z-scores → one combined score.

    Each signal column is standardized cross-sectionally first (so the weights act
    on comparable units), then combined by ``weights``. The result's absolute scale
    is irrelevant - :func:`refine_alpha` re-standardizes it - so only the relative
    weights matter.
    """
    if score_frame.empty:
        return pd.Series(dtype=float)
    z = score_frame.apply(refine.zscore, axis=0)
    combined = z.to_numpy() @ np.asarray(weights, dtype=float)
    return pd.Series(combined, index=score_frame.index)


# --------------------------------------------------------------------------- #
# IC measurement (over a trailing window of realized residual returns)
# --------------------------------------------------------------------------- #
@dataclass
class SignalMeasurement:
    """Measured inputs for the combination, plus their provenance."""

    signals: List[str]
    ics: Dict[str, float]  # mean cross-sectional IC per signal
    shrunk_ics: Dict[str, float]  # after Bayesian shrinkage
    correlation: pd.DataFrame  # K×K signal correlation matrix
    weights: Dict[str, float]  # Ω⁻¹ · shrunk_IC, per signal
    combined_ic: float
    n_periods: int  # rebalances measured (the shrinkage sample size)


def measure_signals(
    bars: Dict[str, pd.DataFrame],
    scorers: Dict[str, Scorer],
    benchmark_bars: pd.DataFrame,
    as_of: datetime,
    *,
    horizon: int = 5,
    n_points: int = 12,
    warmup: int = 30,
    min_names: int = 5,
    ridge: float = DEFAULT_RIDGE,
) -> SignalMeasurement:
    """Measure each signal's IC and the signal correlation matrix over history.

    At each of ``n_points`` rebalance dates within ``(warmup, as_of − horizon]``, score
    every signal on the cross-section (using only bars ``≤ t``) and correlate it with
    the subsequent ``horizon``-bar realized **residual** return (return minus
    ``β·benchmark`` - rewarding skill, not beta). The mean over rebalances is the IC;
    the mean cross-sectional correlation between signals is ``Ω``. ICs are then shrunk
    and combined into GLS weights.
    """
    names = [s for s in bars if not bars[s].empty]
    signal_names = list(scorers)
    ref_index = benchmark_bars.index
    usable = ref_index[ref_index <= _to_ts(as_of, ref_index)]
    # Rebalance points: evenly spaced, leaving room for the forward horizon.
    last = len(usable) - horizon - 1
    if last <= warmup:
        return _empty_measurement(signal_names)
    points = np.linspace(warmup, last, num=min(n_points, last - warmup), dtype=int)

    per_signal_ic: Dict[str, List[float]] = {s: [] for s in signal_names}
    corr_accum: List[pd.DataFrame] = []

    for j in points:
        t = usable[j]
        t_fwd = usable[j + horizon]
        scores, fwd = _cross_section(bars, names, scorers, benchmark_bars, t, t_fwd, min_names)
        if scores is None:
            continue
        for s in signal_names:
            ic = scores[s].corr(fwd)
            if pd.notna(ic):
                per_signal_ic[s].append(float(ic))
        corr_accum.append(scores.corr())

    n_periods = len(corr_accum)
    ics = {s: float(np.mean(v)) if v else 0.0 for s, v in per_signal_ic.items()}
    if corr_accum:
        correlation = sum(corr_accum) / n_periods
    else:
        correlation = pd.DataFrame(np.eye(len(signal_names)), index=signal_names, columns=signal_names)

    shrunk = {s: shrink_ic(ics[s], n_periods) for s in signal_names}
    ic_vec = np.array([shrunk[s] for s in signal_names])
    corr_mat = correlation.loc[signal_names, signal_names].to_numpy()
    w = combination_weights(ic_vec, corr_mat, ridge) if n_periods else np.zeros(len(signal_names))
    ic_comb = combined_ic(ic_vec, corr_mat, ridge) if n_periods else 0.0

    return SignalMeasurement(
        signals=signal_names,
        ics=ics,
        shrunk_ics=shrunk,
        correlation=correlation,
        weights={s: float(w[i]) for i, s in enumerate(signal_names)},
        combined_ic=ic_comb,
        n_periods=n_periods,
    )


def _cross_section(bars, names, scorers, benchmark_bars, t, t_fwd, min_names):
    """Per-signal score cross-section at ``t`` and the realized residual return to ``t_fwd``."""
    bench_close = benchmark_bars["close"]
    if t not in bench_close.index or t_fwd not in bench_close.index:
        return None, None
    bench_ret = bench_close.loc[t_fwd] / bench_close.loc[t] - 1.0

    score_rows: Dict[str, Dict[str, float]] = {s: {} for s in scorers}
    resid: Dict[str, float] = {}
    for sym in names:
        frame = bars[sym]
        hist = frame.loc[frame.index <= t]
        if len(hist) < 2 or t not in frame.index or t_fwd not in frame.index:
            continue
        for s, scorer in scorers.items():
            val = scorer(hist)
            if val is not None and not pd.isna(val):
                score_rows[s][sym] = float(val)
        r = frame["close"].loc[t_fwd] / frame["close"].loc[t] - 1.0
        beta = indicators.calculate_beta(hist["close"], bench_close.loc[bench_close.index <= t])
        resid[sym] = r - beta * bench_ret

    scores = pd.DataFrame(score_rows)
    fwd = pd.Series(resid)
    common = scores.dropna().index.intersection(fwd.dropna().index)
    if len(common) < min_names:
        return None, None
    return scores.loc[common], fwd.loc[common]


def _to_ts(as_of: datetime, index: pd.Index) -> pd.Timestamp:
    """Localize a possibly-naive ``as_of`` to a (possibly tz-aware) index's timezone."""
    ts = pd.Timestamp(as_of)
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        ts = ts.tz_localize(index.tz) if ts.tzinfo is None else ts.tz_convert(index.tz)
    return ts


def _empty_measurement(signal_names: List[str]) -> SignalMeasurement:
    return SignalMeasurement(
        signals=signal_names,
        ics={s: 0.0 for s in signal_names},
        shrunk_ics={s: 0.0 for s in signal_names},
        correlation=pd.DataFrame(np.eye(len(signal_names)), index=signal_names, columns=signal_names),
        weights={s: 0.0 for s in signal_names},
        combined_ic=0.0,
        n_periods=0,
    )


def combined_score(bars, scorers, measurement: SignalMeasurement, as_of: datetime) -> pd.Series:
    """Score each name with every signal at ``as_of`` and combine by the GLS weights."""
    names = [s for s in bars if not bars[s].empty]
    rows: Dict[str, Dict[str, float]] = {s: {} for s in scorers}
    for sym in names:
        frame = bars[sym]
        for s, scorer in scorers.items():
            val = scorer(frame)
            if val is not None and not pd.isna(val):
                rows[s][sym] = float(val)
    score_frame = pd.DataFrame(rows).reindex(columns=measurement.signals)
    weights = np.array([measurement.weights[s] for s in measurement.signals])
    return combine_scores(score_frame.dropna(), weights)
