"""Information analysis - measure skill (IC), scope (breadth), and reconcile the IR.

Specs 005-008 *assume* an information coefficient and *predict* an information
ratio. This module **measures** them and confronts the prediction with reality - the
most honest diagnostic in the whole system. Classical information analysis is
cross-sectional and decompositional: how much skill (IC), across how many
*independent* bets (BR), and does the realized IR match ``IC·√BR``?

Everything here is pure math on already-computed forecasts and realized returns; the
data wiring (sampling rebalances, aligning forward returns) lives in the service
layer. Research-clock only.
"""

import math
from typing import Dict, Sequence

import numpy as np
import pandas as pd

#: Below this many rebalances the IC volatility - and so the t-stat - is meaningless.
MIN_PERIODS = 12


def pearson_ic(forecast: pd.Series, realized: pd.Series) -> float:
    """Cross-sectional Pearson correlation between a forecast and the realized return."""
    aligned = pd.concat([forecast, realized], axis=1).dropna()
    if len(aligned) < 2 or aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def rank_ic(forecast: pd.Series, realized: pd.Series) -> float:
    """Spearman (rank) IC - robust to alpha scale and outliers."""
    aligned = pd.concat([forecast, realized], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    ranked = aligned.rank()
    if ranked.iloc[:, 0].std() == 0 or ranked.iloc[:, 1].std() == 0:
        return float("nan")
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))


def ic_stats(ics: Sequence[float]) -> Dict[str, float]:
    """Summarize an IC time series: mean, volatility, and the skill t-stat.

    ``IC_tstat = mean_IC / (IC_vol / √P)`` is the honesty gate on skill - a fine mean
    IC with a t-stat below ~2 is a few lucky periods, not evidence of skill.
    """
    arr = np.array([x for x in ics if x == x], dtype=float)  # drop NaNs
    p = len(arr)
    mean = float(arr.mean()) if p else 0.0
    vol = float(arr.std(ddof=1)) if p > 1 else 0.0
    tstat = float(mean / (vol / math.sqrt(p))) if vol > 0 and p > 1 else 0.0
    return {"mean_ic": mean, "ic_vol": vol, "ic_tstat": tstat, "periods": p}


def effective_breadth(n_names: int, rebalances_per_year: float, rho_bar: float) -> Dict[str, float]:
    """Effective vs naive breadth - the most common self-deception in quant.

    ``BR_naive = (rebalances/yr)·N`` counts every name as an independent bet.
    ``BR_eff = (rebalances/yr)·N / (1 + (N−1)·ρ̄)`` deflates it by the average pairwise
    correlation ρ̄: perfectly correlated bets (ρ̄→1) collapse to ``rebalances/yr``
    (effectively one bet), independent bets (ρ̄=0) keep the full count.
    """
    br_naive = rebalances_per_year * n_names
    denom = 1.0 + (n_names - 1) * rho_bar
    br_eff = br_naive / denom if denom > 0 else br_naive
    return {"br_eff": float(br_eff), "br_naive": float(br_naive), "rho_bar": float(rho_bar)}


def predicted_ir(mean_ic: float, br_eff: float) -> float:
    """The Fundamental Law: ``predicted_IR = mean_IC · √BR_eff``."""
    return float(mean_ic * math.sqrt(max(br_eff, 0.0)))


def ir_standard_error(ir: float, years: float) -> float:
    """Standard error of a realized IR over ``years``: ``√((1 + IR²/2) / T)``.

    A 3-year backtest has ``SE(IR) ≈ 0.58`` - so an IR of 0.5 is statistically
    indistinguishable from 0. Report IR with this band, not as a point estimate.
    """
    if years <= 0:
        return float("inf")
    return float(math.sqrt((1.0 + ir * ir / 2.0) / years))


def multiple_testing_inflation(n_trials: int) -> float:
    """``P(any |t| > 2 in m trials) = 1 − 0.95^m`` - why one lucky backtest means little.

    With 20 informationless strategies this is ~0.64, not 0.05.
    """
    return float(1.0 - 0.95 ** max(n_trials, 0))
