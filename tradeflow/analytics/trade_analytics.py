"""Reading a recorded run's trade table without opening SQLite.

Exit-reason P&L, win and loss by reason, how long positions were held, how far they
went against the book before they worked — the ordinary questions about a run that
somebody has already paid for. They were ordinary enough to be asked by hand, in SQL,
against a schema nobody should have to read.

Everything here is a **pure function over a stored table** — ``{columns, rows,
total_rows, truncated}`` — and returns structured data. Nothing here formats anything,
grades anything, or decides whether a number is good. The register is the one the
execution report and the P&L-concentration block already set: report the number, say
what it does not cover, leave the judgement to somebody who has seen a few.

Three absences are kept apart, because collapsing any two produces a confident answer
to a question that was never asked:

* **No table.** The run did not opt into keeping one. Nothing is known about its trades
  and no section is computed.
* **No trades.** The run opted in and closed nothing. Every section is computed and
  every count is zero — a real, measured answer.
* **No column.** The table is there but does not carry what a section needs. That
  section says which column it wanted and is silent about the rest, rather than
  reporting a distribution over data it did not have.

**A truncated table does not get aggregated.** A trade table capped at the storage
ceiling holds a prefix of a run's trades, so a total over it is not a smaller number
than the truth, it is a wrong one — and it looks exactly like a right one. Sums refuse
by default; ``allow_partial=True`` computes them anyway and the result says, in the
payload rather than in some caller's formatting, that the numbers cover the stored rows
only. A table whose completeness was never recorded is refused on the same terms: it
did not claim to be whole, and treating "unknown" as "yes" is the failure this whole
path exists to close.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tradeflow.analytics import metrics

#: Aggregate status. ``complete`` — the numbers cover every trade the run made.
#: ``truncated`` — they cover the stored rows only, and the caller asked for that.
#: ``unavailable`` — there are no numbers, and ``reason`` says why.
COMPLETE = "complete"
TRUNCATED = "truncated"
UNAVAILABLE = "unavailable"

#: What the stored table itself is, which is a different question from what the
#: aggregate over it is worth. ``unknown`` is not a synonym for ``whole``.
SOURCE_WHOLE = "whole"
SOURCE_CAPPED = "capped"
SOURCE_UNKNOWN = "unknown"
SOURCE_ABSENT = "not recorded"


def describe_source(table: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the stored table is, before anything is computed from it."""
    if table is None:
        return {
            "recorded": False,
            "completeness": SOURCE_ABSENT,
            "rows_stored": None,
            "total_rows": None,
            "columns": [],
        }
    rows = table.get("rows") or []
    truncated = table.get("truncated")
    completeness = SOURCE_UNKNOWN if truncated is None else (SOURCE_CAPPED if truncated else SOURCE_WHOLE)
    return {
        "recorded": True,
        "completeness": completeness,
        "rows_stored": len(rows),
        "total_rows": table.get("total_rows"),
        "columns": [str(c) for c in (table.get("columns") or [])],
    }


def _columns(table: Dict[str, Any]) -> Dict[str, List[Any]]:
    """The table transposed into ``{column: values}``.

    Rows shorter than the header are padded with ``None`` rather than dropped: a
    malformed row is one trade with fields missing, and losing it silently would move
    every count in this module.
    """
    names = [str(c) for c in (table.get("columns") or [])]
    rows = table.get("rows") or []
    out: Dict[str, List[Any]] = {name: [] for name in names}
    for row in rows:
        for i, name in enumerate(names):
            out[name].append(row[i] if i < len(row) else None)
    return out


def _floats(values: Sequence[Any]) -> "np.ndarray":
    """A column as floats, with anything unparseable as NaN rather than dropped -
    so the count of what could not be measured survives to be reported."""
    out = np.full(len(values), np.nan, dtype="float64")
    for i, value in enumerate(values):
        try:
            if value is None or isinstance(value, bool):
                continue
            out[i] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _as_datetime(value: Any) -> Optional[datetime]:
    """One cell as a datetime, or ``None``. Stored rows carry ISO strings (the journal
    is JSON); a live frame carries timestamps."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _distribution(values: "np.ndarray", n_total: int) -> Dict[str, Any]:
    """Median/quartiles/extremes of a measured column, with the unmeasured counted.

    ``n_measured`` leads. A tidy median over three of two hundred values is not a
    description of the run, and the only thing that says so is the count beside it.
    """
    finite = values[np.isfinite(values)]
    summary: Dict[str, Any] = {
        "n_measured": int(finite.size),
        "n_unmeasured": int(n_total - finite.size),
    }
    if finite.size == 0:
        summary.update({k: None for k in ("median", "mean", "p25", "p75", "p90", "min", "max")})
        return summary
    summary.update(
        {
            "median": float(np.median(finite)),
            "mean": float(np.mean(finite)),
            "p25": float(np.percentile(finite, 25)),
            "p75": float(np.percentile(finite, 75)),
            "p90": float(np.percentile(finite, 90)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    )
    return summary


def _unavailable(requires: Sequence[str], present: Sequence[str]) -> Dict[str, Any]:
    """A section that could not be computed, naming the column it wanted.

    Not an empty distribution. "No trade lasted any time" and "this table does not
    record when trades opened" are different statements, and only one of them is about
    the strategy.
    """
    missing = [c for c in requires if c not in present]
    return {
        "available": False,
        "requires": list(requires),
        "missing_columns": missing,
        "note": f"not computed: the stored table has no {', '.join(missing)} column",
    }


def _win_loss_over(pnl: "np.ndarray") -> Dict[str, Any]:
    """Wins, losses, and the ratios between them for one set of trades.

    Rates are ``None`` with nothing to average rather than ``0.0``: a set with no
    losing trades has no average loss, and zero is a measurement.
    """
    finite = pnl[np.isfinite(pnl)]
    wins, losses = finite[finite > 0], finite[finite < 0]
    return {
        "trades": int(finite.size),
        "unmeasured": int(pnl.size - finite.size),
        "wins": int(wins.size),
        "losses": int(losses.size),
        "flat": int(finite.size - wins.size - losses.size),
        "win_rate": float(metrics.win_rate(finite)) if finite.size else None,
        "net_pnl": float(np.sum(finite)) if finite.size else None,
        "gross_profit": float(np.sum(wins)) if wins.size else 0.0,
        "gross_loss": float(np.sum(losses)) if losses.size else 0.0,
        "avg_win": float(np.mean(wins)) if wins.size else None,
        "avg_loss": float(np.mean(losses)) if losses.size else None,
        "largest_win": float(np.max(wins)) if wins.size else None,
        "largest_loss": float(np.min(losses)) if losses.size else None,
        "expectancy": float(metrics.expectancy(finite)) if finite.size else None,
        "profit_factor": float(metrics.profit_factor(finite)) if finite.size else None,
        "payoff_ratio": float(metrics.payoff_ratio(finite)) if finite.size else None,
    }


def exit_reason_breakdown(columns: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Net P&L, win rate and share of trades, per exit reason.

    A headline return says nothing about where it came from, and a book whose entire
    edge arrives through one exit path is a bet on that path's fill assumption. Nothing
    in the summary metrics distinguishes it from one whose edge is spread across exits,
    so this reports the split and names the concentration when there is one.
    """
    present = list(columns)
    if "exit_reason" not in present or "pnl" not in present:
        return _unavailable(("exit_reason", "pnl"), present)

    reasons = ["" if r is None else str(r) for r in columns["exit_reason"]]
    pnl = _floats(columns["pnl"])
    n_total = len(reasons)

    rows: List[Dict[str, Any]] = []
    for reason in sorted(set(reasons)):
        mask = np.array([r == reason for r in reasons], dtype=bool)
        summary = _win_loss_over(pnl[mask])
        summary["exit_reason"] = reason or "(unlabelled)"
        summary["share_of_trades"] = (int(mask.sum()) / n_total) if n_total else None
        rows.append(summary)
    rows.sort(key=lambda r: (r["net_pnl"] is None, -(r["net_pnl"] or 0.0)))

    # Concentration is measured over the *gains* only. A loss-making exit does not
    # dilute the claim that one path produced the profit; it is a separate fact.
    gains = {r["exit_reason"]: r["net_pnl"] for r in rows if (r["net_pnl"] or 0.0) > 0}
    concentration = None
    if gains:
        total_gain = sum(gains.values())
        top = max(gains, key=lambda k: gains[k])
        concentration = {
            "exit_reason": top,
            "share_of_gain": gains[top] / total_gain if total_gain else None,
        }
    return {
        "available": True,
        "rows": rows,
        "n_trades": n_total,
        "concentration": concentration,
    }


def duration_summary(columns: Dict[str, List[Any]]) -> Dict[str, Any]:
    """How long positions were held, in days.

    Held time is the bridge between a per-trade number and a portfolio one: the same
    per-trade excursion means something different when the average position is open a
    day than when it is open a month.
    """
    present = list(columns)
    if "entry_time" not in present or "exit_time" not in present:
        return _unavailable(("entry_time", "exit_time"), present)

    entries = [_as_datetime(v) for v in columns["entry_time"]]
    exits = [_as_datetime(v) for v in columns["exit_time"]]
    days = np.full(len(entries), np.nan, dtype="float64")
    for i, (opened, closed) in enumerate(zip(entries, exits)):
        if opened is not None and closed is not None:
            days[i] = (closed - opened).total_seconds() / 86_400.0
    summary = _distribution(days, len(entries))
    summary["available"] = True
    summary["unit"] = "days"
    return summary


def excursion_summary(columns: Dict[str, List[Any]]) -> Dict[str, Any]:
    """How far each position went against, and in favour of, the book while open.

    Per *trade*, which is the figure recorded — not the book's aggregate open
    drawdown, which is a different question this does not answer and must not be read
    as answering. A position deep underwater that is a small fraction of the book did
    not put the book that far underwater, and reading the first as the second is the
    mistake this label exists to prevent.
    """
    present = list(columns)
    if "mae_pct" not in present and "mfe_pct" not in present:
        return _unavailable(("mae_pct", "mfe_pct"), present)
    out: Dict[str, Any] = {
        "available": True,
        "unit": "percent of entry price, per trade",
        "note": "per-trade excursion — not the book's aggregate open drawdown",
    }
    for name in ("mae_pct", "mfe_pct"):
        if name in present:
            values = _floats(columns[name])
            out[name] = _distribution(values, len(values))
        else:
            out[name] = _unavailable((name,), present)
    return out


def trade_analytics(table: Optional[Dict[str, Any]], *, allow_partial: bool = False) -> Dict[str, Any]:
    """Everything this module can say about one recorded run's trades.

    ``status`` is the thing to read first. ``complete`` means the sections below cover
    every trade the run made. ``truncated`` means they cover the stored rows only and
    the caller asked for that. ``unavailable`` means there are no sections, and
    ``reason`` says why — no table, or a table whose totals cannot honestly be taken.
    """
    source = describe_source(table)
    result: Dict[str, Any] = {"status": UNAVAILABLE, "reason": "", "source": source}

    if table is None:
        result["reason"] = (
            "no trade table was recorded for this trial — re-run it with --record-trades. "
            "This is not the same as a run that made no trades"
        )
        return result

    completeness = source["completeness"]
    if completeness != SOURCE_WHOLE and not allow_partial:
        stored, total = source["rows_stored"], source["total_rows"]
        result["reason"] = (
            (
                f"the stored table holds {stored:,} of the run's {total:,} trades — it was capped "
                "at the storage ceiling, so a total over it would be short by the rest"
            )
            if completeness == SOURCE_CAPPED
            else (
                "whether the stored table holds all of the run's trades was not recorded, so a "
                "total over it cannot be called complete"
            )
        )
        return result

    columns = _columns(table)
    result.update(
        {
            "status": COMPLETE if completeness == SOURCE_WHOLE else TRUNCATED,
            "n_trades": source["rows_stored"],
            "overall": (
                _win_loss_over(_floats(columns["pnl"]))
                if "pnl" in columns
                else _unavailable(("pnl",), list(columns))
            ),
            "exit_reasons": exit_reason_breakdown(columns),
            "duration": duration_summary(columns),
            "excursion": excursion_summary(columns),
        }
    )
    if result["status"] == TRUNCATED:
        if completeness == SOURCE_CAPPED:
            of_total = f" of the run's {source['total_rows']:,}" if source["total_rows"] else ""
            result["reason"] = (
                f"computed over the {source['rows_stored']:,} stored rows only{of_total} — every "
                "count and total below is a partial"
            )
        else:
            result["reason"] = "computed over the stored rows, which were not recorded as being all of them"
    return result
