"""Reading a parameter sweep without fooling yourself with the best point.

A screen evaluates many configurations and the natural thing to print is the winner.
That is the one number a sweep cannot support: **the best of N is the maximum of N
draws**, and the maximum of N draws from pure noise is a positive number that grows
with N. A leaderboard with no null beside it is a selection-bias machine — the same
error the Deflated Sharpe exists to prevent, one layer up, with no deflation applied.

So the summary leads with the *distribution* — how many points, where the middle sits,
how wide the spread is, how many were positive at all — and any statement about a best
point carries what the best of that many draws is worth under the null.

The second thing worth reading off a sweep is *shape*. A coherent gradient across a
parameter is different evidence from a scattered set of positive points: it says the
result varies with the parameter the way a real effect would, and it can point off the
edge of the searched space, which a leaderboard never shows.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tradeflow.analytics.metrics import expected_max_sharpe

#: Objectives whose null hypothesis is zero and whose estimate is roughly symmetric —
#: the conditions under which "the expected maximum of N draws" is a statement about
#: noise rather than an arbitrary number.
#:
#: Declared here, beside the code that uses it, and deliberately short. A profit factor
#: is null-centred on 1 and heavily skewed; a total return over a fixed window is
#: skewed too. Computing this for them would produce a confident figure that means
#: nothing, which is worse than the honest refusal below — a screen that cannot say
#: what noise looks like for an objective must say *that*, not guess.
NULL_CENTRED_OBJECTIVES = frozenset({"sharpe_ratio", "sortino_ratio", "information_ratio"})


def score_distribution(scores: Sequence[float]) -> Dict[str, Any]:
    """Where a sweep's results actually sit, before anything is ranked.

    ``n_finite`` is reported separately from ``n``: an evaluation that failed or
    produced no trades is not a result of zero, and silently dropping it would make a
    sweep look narrower and more successful than it was.
    """
    values = np.asarray(list(scores), dtype="float64")
    finite = values[np.isfinite(values)]
    summary: Dict[str, Any] = {
        "n": int(values.size),
        "n_finite": int(finite.size),
        "n_dropped": int(values.size - finite.size),
    }
    if finite.size == 0:
        summary.update(
            {
                "median": None,
                "mean": None,
                "std": None,
                "p25": None,
                "p75": None,
                "min": None,
                "max": None,
                "positive_rate": None,
            }
        )
        return summary
    summary.update(
        {
            "median": float(np.median(finite)),
            "mean": float(np.mean(finite)),
            # Sample standard deviation: with one point there is no spread to report,
            # and 0.0 would read as "every result agreed".
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
            "p25": float(np.percentile(finite, 25)),
            "p75": float(np.percentile(finite, 75)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "positive_rate": float(np.mean(finite > 0)),
        }
    )
    return summary


def noise_baseline(scores: Sequence[float], objective: str) -> Dict[str, Any]:
    """What the best of this many draws would be worth if none of them had any edge.

    The dispersion is taken from the screened results themselves, which is what the
    Deflated Sharpe's own correction asks for: the variance of the estimates across the
    configurations that were tried. Two things about that are worth stating plainly,
    and both are carried in the returned payload rather than left to a reader:

    * It assumes the draws are **independent**. Neighbouring grid points are not — they
      share most of their parameters and most of their trades — so the effective number
      of independent trials is smaller than ``n``, and this baseline is correspondingly
      *higher* than a correlation-aware one would be.
    * The dispersion is measured on results that may contain real structure, so it is
      not a pure-noise dispersion either.

    It is a reference point for reading a leaderboard, not a test. ``applicable`` is
    ``False`` — with the reason attached — whenever the objective's null is not zero,
    because a number computed anyway would be quoted as though it meant something.
    """
    values = np.asarray(list(scores), dtype="float64")
    finite = values[np.isfinite(values)]
    payload: Dict[str, Any] = {
        "objective": objective,
        "n_draws": int(finite.size),
        "applicable": False,
        "reason": "",
        "dispersion": None,
        "expected_best_under_null": None,
        "observed_best": float(np.max(finite)) if finite.size else None,
        "assumes_independence": True,
    }
    if objective not in NULL_CENTRED_OBJECTIVES:
        payload["reason"] = (
            f"no null baseline for {objective}: the maximum of N draws is only "
            f"interpretable for an objective whose null is zero and roughly symmetric "
            f"({', '.join(sorted(NULL_CENTRED_OBJECTIVES))}). The best point here is "
            f"still the best of {int(finite.size)} draws — nothing here says what that "
            f"is worth."
        )
        return payload
    if finite.size < 2:
        payload["reason"] = "fewer than two evaluated points: nothing to measure a spread from"
        return payload
    variance = float(np.var(finite, ddof=1))
    if variance <= 0:
        payload["reason"] = "every evaluated point scored identically, so there is no spread to draw from"
        return payload
    payload.update(
        {
            "applicable": True,
            "dispersion": math.sqrt(variance),
            "expected_best_under_null": float(expected_max_sharpe(variance, int(finite.size))),
        }
    )
    return payload


def parameter_gradient(
    rows: Sequence[Dict[str, Any]], parameter: str, objective: str
) -> Optional[List[Dict[str, Any]]]:
    """How the objective behaves across one parameter's values.

    The question a leaderboard cannot answer. A positive rate that falls monotonically
    as a filter tightens is structure; the same number of positive points scattered at
    random across the axis is not, and the two produce identical winners.

    It also shows where the evidence points *off the edge* of what was searched, which
    is the finding a best-point report structurally cannot produce.

    ``None`` when the parameter did not vary — a constant column has no gradient, and
    reporting a single row would invite reading one value as a trend.
    """
    buckets: Dict[Any, List[float]] = {}
    for row in rows:
        if parameter not in row:
            continue
        score = row.get(objective)
        if score is None or not math.isfinite(float(score)):
            continue
        buckets.setdefault(row[parameter], []).append(float(score))
    if len(buckets) < 2:
        return None
    out = []
    for value in sorted(buckets, key=lambda v: (isinstance(v, str), v)):
        scores = buckets[value]
        out.append(
            {
                "value": value,
                "n": len(scores),
                "positive": int(sum(1 for s in scores if s > 0)),
                "positive_rate": float(sum(1 for s in scores if s > 0) / len(scores)),
                "median": float(np.median(scores)),
                "max": float(np.max(scores)),
            }
        )
    return out


def gradients(
    rows: Sequence[Dict[str, Any]], parameters: Sequence[str], objective: str
) -> Dict[str, List[Dict[str, Any]]]:
    """:func:`parameter_gradient` for every parameter that actually varied."""
    out = {}
    for parameter in parameters:
        gradient = parameter_gradient(rows, parameter, objective)
        if gradient is not None:
            out[parameter] = gradient
    return out
