"""Are two recorded results the same result?

A campaign accumulates trials that look different — different parameters, different
kinds, different windows — and some of them are the same trade expressed twice. Two
candidates correlating at 0.97 are one candidate, and promoting both as independent
evidence double-counts a single bet. Answering that meant pulling both return series
out of SQLite by hand and correlating them in a notebook.

Everything here is a pure function over stored series and returns structured data. It
grades nothing. What it is opinionated about is **refusing**, because a correlation is
a claim about a relationship and there is no honest partial version of one:

* **Below the minimum overlap, a pair is refused.** Not computed-and-caveated. A
  correlation over a handful of shared dates is a number with an error bar wider than
  its own range, and printed to two decimals it looks exactly like a measurement.
* **Across accounting versions, a pair is refused by default.** The two series were
  produced by engines that compute different things; their correlation is a fact about
  the instruments as much as the strategies. Forcing it is allowed and the pair is
  labelled incomparable when you do.
* **Every correlation carries its interval.** Reported with a Fisher-z confidence
  interval, so a coefficient resting on a short overlap arrives visibly wide rather
  than merely short of decimals.

The pairwise matrix alone would hide all of that — a refusal and a correlation of zero
occupy the same cell — so the per-pair diagnostics travel beside it and the matrix
holds ``None`` where nothing was computed.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tradeflow.analytics import metrics

#: Shared dates below which a pair is refused rather than reported. Matches the trial
#: store's own panel default: the number that makes a joint resampling meaningful is
#: the same number that makes a pairwise correlation meaningful.
MIN_OVERLAP = 60

#: Confidence level for the interval attached to every correlation.
CONFIDENCE = 0.95

COMPARED = "compared"
REFUSED = "refused"


def series_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    """What one trial's stored series is, before anything is compared to it."""
    trial_id = entry.get("trial_id")
    dates, values = entry.get("dates"), entry.get("values")
    summary: Dict[str, Any] = {
        "trial_id": trial_id,
        "accounting": entry.get("accounting"),
        "strategy": entry.get("strategy"),
        "kind": entry.get("kind"),
        "available": False,
        "reason": "",
        "periods": None,
        "start": None,
        "end": None,
    }
    if dates is None or values is None:
        summary["reason"] = (
            "no return series was recorded for this trial — not every trial kind persists "
            "one, and trials predating the companion table have none"
        )
        return summary
    if not dates:
        # Distinct from the line above on purpose. "Nothing was stored" and "a series
        # was stored and it is empty" are different facts about the record.
        summary["reason"] = "the stored return series is empty"
        summary["periods"] = 0
        return summary

    arr = np.asarray(values, dtype="float64")
    finite = arr[np.isfinite(arr)]
    summary.update(
        {
            "available": True,
            "periods": len(dates),
            "start": str(dates[0]),
            "end": str(dates[-1]),
            "n_finite": int(finite.size),
            "n_unusable": int(arr.size - finite.size),
            "annualized_sharpe": float(metrics.sharpe_ratio(finite)) if finite.size > 1 else None,
            "annualized_volatility": (
                float(metrics.annualized_volatility(finite)) if finite.size > 1 else None
            ),
            "cumulative_return": float(np.prod(1.0 + finite) - 1.0) if finite.size else None,
        }
    )
    return summary


def _correlation_interval(r: float, n: int) -> Optional[Dict[str, float]]:
    """A Fisher-z interval around a correlation coefficient.

    The point of carrying it is that it does not shrink politely: a coefficient
    resting on a short overlap arrives visibly wide, which is the honest rendering of
    "we do not really know". ``None`` under four observations, where the transform has
    no standard error to work with at all.
    """
    if n < 4 or not -1.0 < r < 1.0:
        return None
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    half = metrics.norm_ppf(1.0 - (1.0 - CONFIDENCE) / 2.0) * se
    return {"low": float(math.tanh(z - half)), "high": float(math.tanh(z + half))}


def _pair(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    min_overlap: int,
    across_accounting: bool,
) -> Dict[str, Any]:
    """One pair: the overlap they share, and the correlation over it or the refusal."""
    out: Dict[str, Any] = {
        "a": left.get("trial_id"),
        "b": right.get("trial_id"),
        "status": REFUSED,
        "reason": "",
        "overlap": 0,
        "start": None,
        "end": None,
        "correlation": None,
        "interval": None,
    }

    for side in (left, right):
        if not (side.get("dates") and side.get("values")):
            out["reason"] = f"{side.get('trial_id')} has no recorded return series to compare"
            return out

    acc_left, acc_right = left.get("accounting"), right.get("accounting")
    if acc_left != acc_right and not across_accounting:
        out["reason"] = (
            f"recorded under accounting v{acc_left} and v{acc_right}. The two series were "
            "produced by engines that compute different things, so their correlation is "
            "partly a fact about the instruments. Pass the cross-accounting option to "
            "compare them anyway"
        )
        return out

    a = dict(zip((str(d) for d in left["dates"]), left["values"]))
    b = dict(zip((str(d) for d in right["dates"]), right["values"]))
    shared = sorted(set(a) & set(b))
    x = np.asarray([a[d] for d in shared], dtype="float64")
    y = np.asarray([b[d] for d in shared], dtype="float64")
    usable = np.isfinite(x) & np.isfinite(y)
    x, y = x[usable], y[usable]
    dates = [d for d, keep in zip(shared, usable) if keep]

    out["overlap"] = int(x.size)
    out["start"] = dates[0] if dates else None
    out["end"] = dates[-1] if dates else None

    if x.size < min_overlap:
        out["reason"] = (
            f"{x.size} shared dates, below the {min_overlap} this comparison requires. A "
            "correlation over that little is an error bar, not a measurement"
        )
        return out
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        out["reason"] = "one series does not vary over the shared dates, so nothing can correlate"
        return out

    r = float(np.corrcoef(x, y)[0, 1])
    out.update(
        {
            "status": COMPARED,
            "correlation": r,
            "interval": _correlation_interval(r, int(x.size)),
            "comparable": acc_left == acc_right,
        }
    )
    if acc_left != acc_right:
        out["reason"] = (
            f"INCOMPARABLE as evidence: accounting v{acc_left} against v{acc_right}. "
            "Computed because it was asked for"
        )
    return out


def compare_series(
    entries: Sequence[Dict[str, Any]],
    *,
    min_overlap: int = MIN_OVERLAP,
    across_accounting: bool = False,
) -> Dict[str, Any]:
    """Every pair among the given trials' recorded return series.

    ``entries`` are ``{trial_id, dates, values, accounting, strategy, kind}``. The
    result carries the per-series summaries, one diagnostic per pair, and the matrix —
    in that order deliberately, because the matrix is the part that reads as an answer
    and the refusals are the part that says how much of one it is.
    """
    summaries = [series_summary(entry) for entry in entries]
    ids = [entry.get("trial_id") for entry in entries]

    pairs: List[Dict[str, Any]] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            pairs.append(
                _pair(
                    entries[i],
                    entries[j],
                    min_overlap=min_overlap,
                    across_accounting=across_accounting,
                )
            )

    # ``None`` where nothing was computed, never 0.0. In a correlation matrix a zero
    # is a strong claim - these two results move independently - and it is the exact
    # claim a refusal is unable to make.
    index = {tid: n for n, tid in enumerate(ids)}
    matrix: List[List[Optional[float]]] = [
        [1.0 if i == j else None for j in range(len(ids))] for i in range(len(ids))
    ]
    for pair in pairs:
        if pair["status"] != COMPARED:
            continue
        i, j = index[pair["a"]], index[pair["b"]]
        matrix[i][j] = matrix[j][i] = pair["correlation"]

    compared = [p for p in pairs if p["status"] == COMPARED]
    return {
        "min_overlap": min_overlap,
        "across_accounting": across_accounting,
        "series": summaries,
        "pairs": pairs,
        "matrix": {"trial_ids": ids, "values": matrix},
        "n_pairs": len(pairs),
        "n_compared": len(compared),
        "n_refused": len(pairs) - len(compared),
        "highest": (max(compared, key=lambda p: p["correlation"]) if compared else None),
    }
