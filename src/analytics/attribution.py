"""Performance attribution - where the realized active return came from, and
whether the track record means anything.

The information-analysis module measures skill **ex ante** (IC, breadth,
predicted IR) and splits realized return **once**, pooled, into factor vs
specific. This module generalizes that split **per period, per factor**, adds a
systematic benchmark-timing split (expected/surprise/timing), and applies the
same honesty gates (t-stats, Bayesian-blended risk, multiple-testing inflation)
to the attributed series themselves - so a ranked table of ~8 return buckets
can't be read the way a single lucky backtest can't.

Everything here is pure math on already-computed exposures/weights/returns; the
data wiring (sampling rebalances, building leakage-safe cross-sections) lives in
the service layer (:func:`src.services.analysis.compute_attribution`), mirroring
how :mod:`src.analytics.information` stays pure math while
``compute_information`` does the wiring. Research-clock only.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.analytics.metrics import norm_cdf


# --------------------------------------------------------------------------- #
# Per-period attribution (the regression identity, exact by construction)
# --------------------------------------------------------------------------- #
def cross_sectional_regression(x: np.ndarray, r: np.ndarray):
    """OLS ``b = (xᵀx)⁻¹xᵀr``; returns ``(b, residual)`` with ``r = x·b + residual``
    exactly (up to floating point). ``x`` is ``N×K``, ``r`` is length-``N``.

    Degenerate inputs (fewer names than columns, or no columns) return an
    all-zero ``b`` and pass ``r`` through unchanged as the residual - the
    honest "this stage explained nothing" answer, not a crash.
    """
    if x.size == 0 or x.shape[1] == 0 or x.shape[0] < x.shape[1] + 1:
        k = x.shape[1] if x.ndim == 2 else 0
        return np.zeros(k), r.copy()
    b = np.linalg.pinv(x.T @ x) @ x.T @ r
    return b, r - x @ b


@dataclass
class PeriodAttribution:
    """One rebalance's attributed active return - every field sums exactly to
    ``r_active`` (the regression-identity adding-up this module tests for)."""

    r_active: float
    beta_a: float
    r_bench: float
    systematic: float  # beta_a * r_bench - the WHOLE benchmark-timing bucket
    factor_returns: Dict[str, float]  # b_j(t): the cross-sectional factor RETURN
    factor_contributions: Dict[str, float]  # w_a . (x_j * b_j): the portfolio's share
    signal_returns: Dict[str, float]
    signal_contributions: Dict[str, float]
    specific: float
    residual: float  # numerical adding-up slack only; ~0 by construction


def attribute_period(
    w_active: pd.Series,
    risk_x: pd.DataFrame,
    r_raw: pd.Series,
    beta_per_name: pd.Series,
    r_bench: float,
    signal_x: Optional[pd.DataFrame] = None,
) -> Optional[PeriodAttribution]:
    """Attribute one period's realized active return by exact regression identity.

    ``w_active`` = active weights ``w_a`` known at the *start* of the period;
    ``risk_x`` = start-of-period risk-factor exposures (market/momentum/
    volatility/size, :func:`src.risk.exposures.build_factor_exposures`);
    ``beta_per_name`` = the canonical Σ-implied per-name beta
    (:meth:`~src.risk.base.RiskMatrix.implied_beta`, the one canonical β
    used everywhere), known at the start of the period; ``r_raw`` = realized
    per-name raw returns over the period; ``r_bench`` = realized benchmark
    return; ``signal_x`` = optional per-name combined-signal z-scores.

    Two stages, each an exact regression identity so nothing is double-counted
    or lost:

    1. **Systematic (known, predetermined) benchmark-timing**: strip
       ``β_i(t)·r_B(t)`` per name using the *same* canonical β, so its portfolio
       aggregate is exactly ``β_a(t)·r_B(t)`` - not fit to this period's data,
       just the already-known tilt times the realized benchmark move.
    2. **Risk factors + signals, JOINTLY**: one cross-sectional regression of
       the beta-adjusted return on ``[risk_x, signal_x]`` together. Signals as a
       *second pass* on the risk-factor residual ("cleaner ownership") was tried
       and reverted - a second pass leaves an
       **omitted-variable bias** that does not vanish with more names: when
       active weights are built from a signal that is not yet among the
       regressors doing the fitting (exactly how a paper book is built
       upstream - its own alpha z-score), whichever risk factor happens to
       correlate with that signal *in this period's finite sample* picks up a
       persistent, same-sign share of the signal's true return - a genuine
       integrity bug, not sampling noise around zero (measured at |t| > 60 in
       a 2000-trial check during development). Fitting both blocks in one
       regression removes the omission and the bias disappears (measured
       |t| < 1). The remaining tradeoff (signals correlated with risk
       factors destabilize a joint fit) is real but smaller and disclosable
       (report the condition number); the omitted-variable bias was not
       disclosable - it looks exactly like real factor timing.

    Returns ``None`` when too few names are common to the inputs to regress
    (fewer than ``len(risk_x.columns) + (len(signal_x.columns) if any) + 2``).
    """
    common = (
        w_active.index.intersection(risk_x.index)
        .intersection(r_raw.dropna().index)
        .intersection(beta_per_name.dropna().index)
    )
    has_signals = signal_x is not None and not signal_x.empty
    if has_signals:
        common = common.intersection(signal_x.dropna(how="any").index)
    n_cols = len(risk_x.columns) + (len(signal_x.columns) if has_signals else 0)
    if len(common) < n_cols + 2:
        return None

    w = w_active.loc[common].to_numpy(dtype=float)
    x = risk_x.loc[common].to_numpy(dtype=float)
    r = r_raw.loc[common].to_numpy(dtype=float)
    beta = beta_per_name.loc[common].to_numpy(dtype=float)

    r_active = float(w @ r)
    beta_a = float(w @ beta)
    systematic = beta_a * r_bench

    # Stage 1: per-name systematic strip (same canonical beta, so it sums to
    # beta_a*r_bench exactly: w @ (beta*r_bench) == (w@beta)*r_bench).
    u1 = r - beta * r_bench

    # Stage 2: risk factors and signals, jointly (see docstring for why).
    joint_x = np.column_stack([x, signal_x.loc[common].to_numpy(dtype=float)]) if has_signals else x
    b_joint, specific_series = cross_sectional_regression(joint_x, u1)
    n_risk = len(risk_x.columns)
    b_factors, b_signals = b_joint[:n_risk], b_joint[n_risk:]

    factor_returns = {col: float(b_factors[j]) for j, col in enumerate(risk_x.columns)}
    factor_contributions = {col: float(w @ (x[:, j] * b_factors[j])) for j, col in enumerate(risk_x.columns)}
    signal_returns: Dict[str, float] = {}
    signal_contributions: Dict[str, float] = {}
    if has_signals:
        sx = signal_x.loc[common].to_numpy(dtype=float)
        signal_returns = {col: float(b_signals[k]) for k, col in enumerate(signal_x.columns)}
        signal_contributions = {
            col: float(w @ (sx[:, k] * b_signals[k])) for k, col in enumerate(signal_x.columns)
        }

    specific = float(w @ specific_series)
    reconstructed = (
        systematic + sum(factor_contributions.values()) + sum(signal_contributions.values()) + specific
    )
    residual = r_active - reconstructed

    return PeriodAttribution(
        r_active=r_active,
        beta_a=beta_a,
        r_bench=r_bench,
        systematic=systematic,
        factor_returns=factor_returns,
        factor_contributions=factor_contributions,
        signal_returns=signal_returns,
        signal_contributions=signal_contributions,
        specific=specific,
        residual=residual,
    )


# --------------------------------------------------------------------------- #
# The systematic split: expected / surprise / timing
# --------------------------------------------------------------------------- #
def systematic_split(
    beta_a_series: Sequence[float], r_bench_series: Sequence[float], mu_b_period: float
) -> Dict[str, float]:
    """Split ``Σ_t β_a(t)·r_B(t)`` into expected-beta / benchmark-surprise / timing.

    ``mu_b_period`` is the ASSUMED (ex-ante) benchmark return *per rebalance
    period* - not the realized mean, which would make "expected" indistinguishable
    from "surprise" after the fact. The identity (exact, by construction):
    ``expected + surprise + timing == Σ_t β_a(t)·r_B(t)``, because
    ``Σ(β_bar+δβ)(rb_bar+δr) = β_bar·rb_bar·T + β_bar·Σδr + rb_bar·Σδβ + Σδβδr``
    and the middle two terms vanish (deviations from the mean sum to zero).

    Only ``timing`` is a genuine per-period series with its own sampling
    variation (the ``timing_series`` of ``δβ_a(t)·δr_B(t)``); expected/surprise
    are aggregate-only numbers (they use ``β_bar``, only known after the fact) -
    the report shows them without a t-stat, marked "not skill".
    """
    beta = np.asarray(beta_a_series, dtype=float)
    rb = np.asarray(r_bench_series, dtype=float)
    t = len(beta)
    if t == 0:
        return {
            "expected": 0.0,
            "surprise": 0.0,
            "timing": 0.0,
            "timing_series": [],
            "beta_bar": 0.0,
            "r_bench_bar": 0.0,
        }
    beta_bar = float(beta.mean())
    rb_bar = float(rb.mean())
    expected = beta_bar * mu_b_period * t
    surprise = beta_bar * (rb_bar - mu_b_period) * t
    timing_series = (beta - beta_bar) * (rb - rb_bar)
    return {
        "expected": float(expected),
        "surprise": float(surprise),
        "timing": float(timing_series.sum()),
        "timing_series": timing_series.tolist(),
        "beta_bar": beta_bar,
        "r_bench_bar": rb_bar,
    }


# --------------------------------------------------------------------------- #
# Bayesian blend - the honest risk for a short attributed series
# --------------------------------------------------------------------------- #
def prior_weight_t0(min_obs: int, bars_per_period: float) -> float:
    """``T₀`` in REBALANCE-PERIOD units, derived from the risk model's own
    estimation window.

    The risk model needs ``min_obs`` *bars* of history before it trusts Σ
    (:func:`src.risk.base.build_risk_matrix`, default 60); at this attribution's
    rebalance cadence of ``bars_per_period`` bars per period, that's
    ``min_obs / bars_per_period`` PERIODS worth of prior confidence - i.e. "the
    risk model itself wouldn't trust an estimate from fewer periods than this,
    so neither should the attribution". This mapping isn't specified anywhere
    else in the codebase; it is a documented judgment call.
    """
    return float(min_obs) / max(bars_per_period, 1e-9)


def bayesian_blend_variance(sigma2_prior: float, sigma2_realized: float, t: float, t0: float) -> float:
    """``σ² = σ²_prior·T₀/(T+T₀) + σ²_real·T/(T+T₀)`` (17A.12).

    ``T → 0`` recovers the prior exactly; ``T → ∞`` recovers the realized
    variance exactly; the two weights always sum to 1.
    """
    denom = t + t0
    if denom <= 0:
        return float(sigma2_prior)
    return float(sigma2_prior * (t0 / denom) + sigma2_realized * (t / denom))


def series_stats(
    values: Sequence[float],
    periods_per_year_series: float,
    sigma2_prior: float,
    t0: float,
) -> Dict[str, float]:
    """Mean / Bayesian-blended-risk IR / t-stat for one attributed series.

    The per-period realized variance is blended with ``sigma2_prior`` (17A.12)
    so a short sample leans on the risk model's structural prior instead of a
    wild few-point sample SD; ``t → ∞`` (long samples) recovers the plain
    realized-variance t-stat.
    """
    arr = np.array([v for v in values if v == v], dtype=float)  # drop NaNs
    t = len(arr)
    mean = float(arr.mean()) if t else 0.0
    sigma2_real = float(arr.var(ddof=1)) if t > 1 else sigma2_prior
    sigma2 = bayesian_blend_variance(sigma2_prior, sigma2_real, t, t0)
    sd = math.sqrt(max(sigma2, 0.0))
    ann_mean = mean * periods_per_year_series
    ann_sd = sd * math.sqrt(periods_per_year_series)
    ir = float(ann_mean / ann_sd) if ann_sd > 0 else 0.0
    tstat = float((mean / sd) * math.sqrt(t)) if sd > 0 and t > 0 else 0.0
    return {
        "mean": mean,
        "annualized_mean": ann_mean,
        "vol_blended": sd,
        "annualized_vol_blended": ann_sd,
        "ir": ir,
        "t_stat": tstat,
        "periods": t,
    }


# --------------------------------------------------------------------------- #
# Track-record calculus
# --------------------------------------------------------------------------- #
def years_to_significance(ir: float) -> float:
    """``Y* = (2/IR)²`` - years until a real IR's t-stat would reach 2, under the
    simplified ``SE{IR} ≈ 1/√Y`` this formula itself is built from (the more
    precise :func:`src.analytics.information.ir_standard_error` includes an
    ``IR²/2`` term the closed-form ``Y*`` ignores - both are reported so the
    approximation is visible, not hidden).
    """
    if ir == 0:
        return float("inf")
    return float((2.0 / ir) ** 2)


def prob_positive_over_years(ir: float, years: float) -> float:
    """``P(positive cumulative alpha over` years) = Φ(IR·√years)``."""
    return float(norm_cdf(ir * math.sqrt(max(years, 0.0))))


# --------------------------------------------------------------------------- #
# Honest cumulation (the "cumulation trap")
# --------------------------------------------------------------------------- #
def cumulate_top_down(
    component_series: Dict[str, Sequence[float]],
    r_active_series: Sequence[float],
    r_portfolio_series: Sequence[float],
    r_bench_series: Sequence[float],
) -> Dict[str, object]:
    """Cumulative active return is ``ΠR_P − ΠR_B``, *never* ``Π(1+r_a)`` - and
    per-component cumulation is top-down (17A.9), with the compounding residual
    ``δ_CP`` reported, not silently absorbed.

    ``component_series`` values must sum to ``r_active_series`` **at each
    period** (the per-period regression identity from :func:`attribute_period`).
    Each component is chain-linked by the compounding path up to (not including)
    its own period - ``Σ_t component(t)·Π_{s<t}(1+r_a(s))`` - which telescopes so
    ``Σ_component linked_component == Π_t(1+r_a(t)) − 1`` (``naive_cumulative``)
    exactly. The gap between that naive linked total and the honest
    ``ΠR_P − ΠR_B`` (``honest_car``) is ``delta_cp``; by construction
    ``Σ linked_components + delta_cp == honest_car`` exactly - the identity this
    module tests for.
    """
    r_a = np.asarray(r_active_series, dtype=float)
    r_p = np.asarray(r_portfolio_series, dtype=float)
    r_b = np.asarray(r_bench_series, dtype=float)

    if len(r_a) == 0:
        return {
            "linked_components": {k: 0.0 for k in component_series},
            "naive_cumulative": 0.0,
            "honest_car": 0.0,
            "delta_cp": 0.0,
        }

    growth = np.concatenate([[1.0], np.cumprod(1.0 + r_a)])[:-1]  # Π_{s<t}(1+r_a(s))
    naive_cumulative = float(np.prod(1.0 + r_a) - 1.0)
    honest_car = float(np.prod(1.0 + r_p) - np.prod(1.0 + r_b)) if len(r_p) and len(r_b) else naive_cumulative

    linked_components = {}
    for name, series in component_series.items():
        arr = np.asarray(series, dtype=float)
        linked_components[name] = float(np.sum(arr * growth))

    delta_cp = honest_car - naive_cumulative
    return {
        "linked_components": linked_components,
        "naive_cumulative": naive_cumulative,
        "honest_car": honest_car,
        "delta_cp": delta_cp,
    }
