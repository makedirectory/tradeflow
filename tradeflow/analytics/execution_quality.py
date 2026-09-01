"""Roll up what the live path actually did, from the ledger.

Research-clock code: it reads the trade clock's append-only record and summarises it.
Nothing here is a gate, and nothing here changes a run. What "bad" slippage looks like
for a given strategy is not knowable from one session, so this reports the numbers and
declines to grade them.

The reporting rules that matter here:

* **Absent is not zero.** A fill with no recorded price contributes no slippage, and
  is counted as unmeasured rather than averaged in as zero.
* **Say what was not measured.** Every summary carries how many rows it could not
  evaluate, because a clean-looking average over two of twenty fills is worse than no
  average at all.
* **A modelled cost is not an observed one.** The cost model's estimate and the
  venue's fee are reported separately and never summed.
"""

from statistics import median
from typing import Any, Dict, List, Optional


def _measured(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [row[key] for row in rows if row.get(key) is not None]


def slippage_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Slippage across filled orders, in basis points, positive meaning worse."""
    values = _measured(rows, "slippage_bps")
    filled = [row for row in rows if row.get("filled_qty")]
    summary = {
        "n_filled": len(filled),
        "n_measured": len(values),
        "n_unmeasured": len(filled) - len(values),
        "mean_bps": None,
        "median_bps": None,
        "worst_bps": None,
        "worst_symbol": None,
        "best_bps": None,
    }
    if not values:
        return summary
    worst = max(rows, key=lambda r: r.get("slippage_bps") if r.get("slippage_bps") is not None else -1e9)
    summary.update(
        mean_bps=sum(values) / len(values),
        median_bps=median(values),
        worst_bps=max(values),
        worst_symbol=worst.get("symbol"),
        best_bps=min(values),
    )
    return summary


def latency_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How long the venue took, from the decision to the fill it produced.

    Negative elapsed times are counted apart and never averaged in. They mean our
    clock and the venue's disagree, not that a fill arrived before it was requested —
    and a mean quietly dragged negative by skew is worse than an admission that some
    rows could not be timed.
    """
    values = _measured(rows, "decision_to_fill_ms")
    forward = [value for value in values if value >= 0]
    summary = {
        "n_measured": len(forward),
        "n_clock_skew": len(values) - len(forward),
        "median_ms": None,
        "worst_ms": None,
    }
    if forward:
        summary.update(median_ms=median(forward), worst_ms=max(forward))
    return summary


def fill_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Submitted versus filled, in shares and in notional."""
    submitted_notional = sum(
        abs(row["submitted_qty"]) * row["reference_price"]
        for row in rows
        if row.get("submitted_qty") and row.get("reference_price")
    )
    filled_notional = sum(
        abs(row["filled_qty"]) * row["fill_price"]
        for row in rows
        if row.get("filled_qty") and row.get("fill_price")
    )
    unfilled = [row for row in rows if not row.get("filled_qty")]
    partial = [
        row
        for row in rows
        if row.get("filled_qty")
        and row.get("submitted_qty")
        and abs(row["filled_qty"]) < abs(row["submitted_qty"]) - 1e-9
    ]
    return {
        "n_orders": len(rows),
        "n_unfilled": len(unfilled),
        "n_partial": len(partial),
        "submitted_notional": submitted_notional,
        "filled_notional": filled_notional,
        # None rather than 0.0: with nothing submitted there is no ratio to report,
        # and 0.0 would read as "nothing filled".
        "fill_ratio": (filled_notional / submitted_notional) if submitted_notional else None,
    }


def cost_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Modelled cost and observed fees, kept apart on purpose."""
    estimates = _measured(rows, "cost_estimate")
    fees = _measured(rows, "broker_fee")
    return {
        "n_estimated": len(estimates),
        "model_cost_estimate": sum(estimates) if estimates else None,
        "n_fees_reported": len(fees),
        "broker_fees": sum(fees) if fees else None,
        # The distinction the operator has to see: a paper venue reports no fees, so
        # "no fees" must never be presented as "fees were zero".
        "fees_reported": bool(fees),
    }


def decline_summary(declines: List[Dict[str, Any]]) -> Dict[str, int]:
    """How often each reason stopped a signal, worst first."""
    counts: Dict[str, int] = {}
    for record in declines:
        reason = str(record.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def execution_report(
    rows: List[Dict[str, Any]], declines: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Everything the ledger can say about how well this book was executed."""
    return {
        "slippage": slippage_summary(rows),
        "latency": latency_summary(rows),
        "fills": fill_summary(rows),
        "costs": cost_summary(rows),
        "declines": decline_summary(declines or []),
        "orders": rows,
    }
