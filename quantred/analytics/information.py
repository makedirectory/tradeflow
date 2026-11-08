"""Information analysis - measure skill (IC), scope (breadth), and reconcile the IR.

The rest of the stack *assumes* an information coefficient and *predicts* an information
ratio. This module **measures** them and confronts the prediction with reality - the
most honest diagnostic in the whole system. Classical information analysis is
cross-sectional and decompositional: how much skill (IC), across how many
*independent* bets (BR), and does the realized IR match ``IC·√BR``?

Everything here is pure math on already-computed forecasts and realized returns; the
data wiring (sampling rebalances, aligning forward returns) lives in the service
layer. Research-clock only.
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from quantred.utils.numeric import is_flat

#: Below this many rebalances the IC volatility - and so the t-stat - is meaningless.
MIN_PERIODS = 12


def pearson_ic(forecast: pd.Series, realized: pd.Series) -> float:
    """Cross-sectional Pearson correlation between a forecast and the realized return."""
    aligned = pd.concat([forecast, realized], axis=1).dropna()
    if len(aligned) < 2 or is_flat(aligned.iloc[:, 0]) or is_flat(aligned.iloc[:, 1]):
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def rank_ic(forecast: pd.Series, realized: pd.Series) -> float:
    """Spearman (rank) IC - robust to alpha scale and outliers."""
    aligned = pd.concat([forecast, realized], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    ranked = aligned.rank()
    if is_flat(ranked.iloc[:, 0]) or is_flat(ranked.iloc[:, 1]):
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
    tstat = float(mean / (vol / math.sqrt(p))) if p > 1 and not is_flat(arr) else 0.0
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


#: Fewest overlapping bars a beta, or a pair's residual correlation, may rest on. Below it a
#: correlation is mostly its own sampling error (a correlation of two series of ``k`` bars
#: has a standard error near ``1/sqrt(k)``), so the pair is left out rather than counted.
MIN_OVERLAP = 20

#: Fraction of all name pairs that must have a usable correlation before the mean is trusted.
MIN_PAIR_COVERAGE = 0.5


def mean_residual_correlation(
    returns: pd.DataFrame, benchmark_returns: pd.Series, *, min_overlap: int = MIN_OVERLAP
) -> Optional[float]:
    """Mean pairwise correlation of the names' *benchmark-residual* returns, or ``None``.

    ``returns`` is (periods × names), ``benchmark_returns`` the aligned benchmark. Each
    name's residual is ``r − β·r_benchmark`` with ``β = cov(r, r_b)/var(r_b)`` - the same
    residual the information coefficient is measured on. Breadth must be deflated by the
    correlation of the *bets being counted*: total-return correlations include the common
    market move, which the residual IC has already stripped out, so using them deflates
    breadth by a large factor (and moves with whatever risk model estimated them).

    Histories are allowed to differ in length (a recent listing, a halt). Each name's β is
    estimated over *its own* overlap with the benchmark, and each pair's correlation over
    the bars where *both* residuals exist - never over the shortest history shared by all
    names, which would let one two-bar listing erase everything the other names say about
    their dependence. A name, or a pair, with fewer than ``min_overlap`` bars contributes
    nothing; the mean is over the pairs that remain.

    Returns ``None`` - not ``0.0`` - when the correlation cannot be estimated: fewer than
    two names with enough history, a flat benchmark, or fewer than
    :data:`MIN_PAIR_COVERAGE` of all pairs with enough shared bars. ``0.0`` would mean
    "the bets are independent" and leave breadth undeflated, the most optimistic answer, for
    exactly the case where nothing is known.
    """
    names = list(returns.columns)
    n_total = len(names)
    if n_total < 2:
        return None
    bench = benchmark_returns.astype(float)
    floor = max(min_overlap, 3)
    residuals = {}
    for name in names:
        pair = pd.concat([returns[name].astype(float), bench], axis=1, keys=["r", "b"]).dropna()
        if len(pair) < floor:
            continue
        var_b = float(pair["b"].var())
        if not var_b > 0:
            continue
        beta = float(pair["r"].cov(pair["b"])) / var_b
        residuals[name] = pair["r"] - beta * pair["b"]

    usable = []
    kept = list(residuals)
    for i, a_name in enumerate(kept):
        for b_name in kept[i + 1 :]:
            both = pd.concat([residuals[a_name], residuals[b_name]], axis=1, keys=["a", "b"]).dropna()
            if len(both) < floor or is_flat(both["a"]) or is_flat(both["b"]):
                continue
            value = float(both["a"].corr(both["b"]))
            if value == value:
                usable.append(value)
    total_pairs = n_total * (n_total - 1) / 2
    if not usable or len(usable) < MIN_PAIR_COVERAGE * total_pairs:
        return None
    return float(np.mean(usable))


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


def risk_bucket_diagnostic(
    active_weights: pd.Series,
    sigma: np.ndarray,
    symbols: Sequence[str],
    residual_vol: pd.Series,
    *,
    min_per_bucket: int = 8,
) -> Dict:
    """Equal-risk-contribution check by residual-vol bucket.

    Under **correct** alpha scaling every name contributes ~equally to active variance
    (``E{z²}=1``), so a residual-vol bucket's share of ``w_aᵀΣw_a`` should track its
    share of *names*, not its vol. A monotone gradient across vol buckets is the
    fingerprint of mis-scaled alphas — most often a Case mis-choice (see
    :func:`quantred.alphas.refine.case_test`): scaling a Case-2 signal as Case 1 multiplies
    in ``ω_n`` a second time and tilts variance into the high-vol bucket.

    Buckets names by residual-vol quantile and reports each bucket's share of active
    variance and of ``Σ|w_a|``, the expected (name-fraction) share, and the per-bucket
    sampling error ``√(2/n)`` of mean ``z²``. Degrades quintiles → terciles → suppressed
    as the universe thins (tiny buckets are noise, not signal). ``active_weights`` are
    the book's active weights ``w_a = w − w_B``; ``sigma`` the annualized Σ over
    ``symbols`` in that order.
    """
    aligned = (
        pd.concat([active_weights.rename("w"), residual_vol.rename("vol")], axis=1)
        .reindex(list(symbols))
        .dropna()
    )
    n = len(aligned)

    # Bucket count: a bucket mean of z² has sampling error ≈ √(2/n_bucket); below
    # ~min_per_bucket names per bucket the gradient is noise, so widen or suppress.
    if n >= 5 * min_per_bucket:
        n_buckets = 5
    elif n >= 3 * max(min_per_bucket - 3, 4):
        n_buckets = 3
    else:
        return {
            "engaged": False,
            "n_names": int(n),
            "reason": f"universe too thin for reliable buckets (n={n}); "
            f"need ≥ {3 * max(min_per_bucket - 3, 4)} names for terciles",
        }

    idx = list(aligned.index)
    w = aligned["w"].to_numpy()
    vols = aligned["vol"].to_numpy()
    pos = {s: i for i, s in enumerate(symbols)}
    order = [pos[s] for s in idx]
    sub = np.asarray(sigma, dtype=float)[np.ix_(order, order)]

    # Per-name contribution to active variance: w_i·(Σw)_i sums to w_aᵀΣw_a exactly.
    contrib = w * (sub @ w)
    total_var = float(contrib.sum())
    total_abs_w = float(np.abs(w).sum())

    order_by_vol = np.argsort(vols, kind="stable")
    buckets = np.array_split(order_by_vol, n_buckets)
    rows: List[Dict] = []
    var_shares: List[float] = []
    for b, members in enumerate(buckets):
        nb = len(members)
        var_share = float(contrib[members].sum() / total_var) if total_var != 0 else 0.0
        rows.append(
            {
                "bucket": b + 1,  # 1 = lowest residual vol
                "n_names": int(nb),
                "vol_low": float(vols[members].min()),
                "vol_high": float(vols[members].max()),
                "variance_share": var_share,
                "name_fraction": float(nb / n),
                "abs_weight_share": float(np.abs(w[members]).sum() / total_abs_w) if total_abs_w > 0 else 0.0,
                "sampling_error": float(math.sqrt(2.0 / nb)) if nb > 0 else float("inf"),
            }
        )
        var_shares.append(var_share)

    # Monotone gradient beyond the sampling band ⇒ a vol tilt (mis-scaling).
    diffs = np.diff(var_shares)
    monotone = bool(np.all(diffs > 0) or np.all(diffs < 0))
    gradient = float(var_shares[-1] - var_shares[0])
    band = float(math.sqrt(2.0 / (n / n_buckets)) / n_buckets)
    tilt = bool(monotone and abs(gradient) > band)
    direction = "high-vol" if gradient > 0 else "low-vol"

    return {
        "engaged": True,
        "n_names": int(n),
        "n_buckets": n_buckets,
        "buckets": rows,
        "expected_share": float(1.0 / n_buckets),
        "variance_share_gradient": gradient,
        "monotone": monotone,
        "sampling_band": band,
        "tilt_detected": tilt,
        "verdict": (
            f"vol-bucket tilt toward {direction} names (gradient {gradient:+.2f} vs band "
            f"±{band:.2f}) — likely a Case mis-choice; check the per-signal case_test"
            if tilt
            else "no material vol-bucket tilt (contributions ~equal across buckets)"
        ),
    }
