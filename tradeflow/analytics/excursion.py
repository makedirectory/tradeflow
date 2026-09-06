"""How bad did the *book* get while it was open?

Per-trade MAE answers a narrower question than it looks like it answers. A position
30% underwater is alarming until you notice it was a small fraction of the book, at
which point the same number says almost nothing about what the portfolio went through.
Establishing that meant pulling a trial's trade table and comparing per-trade
excursions against the equity drawdown by hand — and the hand comparison is the whole
diagnostic, so it belongs in the tool.

The equity curve is marked at each bar's **close**. Everything that happened inside the
bar — the tick where three positions were simultaneously at their worst — is invisible
to it. So the curve can show a shallow drawdown over a period the book actually spent
in considerably more trouble, and nothing in the summary metrics says which of those
two stories is true.

This reports both ends of that, and is careful about what each one is:

* **The closing mark** is what the equity curve saw. It is a real sequence of prices
  the book actually printed at, and it is a *lower* bound on the pain.
* **The simultaneous-extremes bound** marks every open position at its own worst tick
  inside the same bar. That assumes they all got there at once, which they did not.
  It is an *upper* bound and is never a measurement.

The realized worst lies between them, and neither number is the answer on its own.
Reporting only the second would be the pessimistic reading the per-trade figures
already invite; reporting only the first is the complacent one the closing mark already
gives. Diagnostic only — nothing here gates anything, and the point of the pair is that
a reader can see how far apart they are.
"""

from typing import Any, Dict, List, Optional, Sequence

#: Below this, the gap between the two bounds is not worth a reader's attention: a book
#: whose intra-bar worst is within a fraction of a percent of its closing marks was
#: sampled honestly by its own equity curve. Stated as a display threshold, not a gate —
#: nothing changes when it is crossed except that the report says so.
MATERIAL_GAP_PCT = 1.0


def _instant(sample: Dict[str, Any], equity_key: str, peak: float) -> Dict[str, Any]:
    """One sample rendered as the moment it describes, with the book's shape at it."""
    equity = float(sample.get(equity_key, 0.0))
    close = float(sample.get("equity_close", 0.0))
    gross, net = float(sample.get("gross", 0.0)), float(sample.get("net", 0.0))
    return {
        "time": sample.get("time"),
        "equity": equity,
        "equity_at_close": close,
        "peak_equity": peak,
        "open_positions": int(sample.get("open_positions", 0)),
        "gross_exposure": gross,
        "net_exposure": net,
        # As fractions of the book at that instant, because the dollar figure alone
        # cannot say whether it was a large position or a large book.
        "gross_exposure_pct": (gross / close * 100.0) if close else None,
        "net_exposure_pct": (net / close * 100.0) if close else None,
        # What the closing mark showed at the same instant. The gap between this and
        # the excursion is the whole finding.
        "drawdown_at_close_pct": ((peak - close) / peak * 100.0) if peak else None,
    }


def portfolio_excursion(samples: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """The book's aggregate adverse and favourable excursion through time.

    ``samples`` are the engine's per-step records: the book marked at the bar's close,
    at every open position's worst tick within the bar, and at every one's best, plus
    the exposure and position count carried at that step.

    Both excursions are measured against the running peak of the **closing** curve — a
    level the book actually printed, rather than an intra-bar extreme it only touched.
    Measuring a drawdown from a peak that never closed would inflate it with the same
    intra-bar noise the diagnostic exists to isolate.
    """
    report: Dict[str, Any] = {
        "available": False,
        "reason": "",
        "n_steps": 0,
        "basis": (
            "every open position marked at its own worst (and best) tick within the same "
            "bar, which assumes they all got there at once — an upper bound on the pain, "
            "never a measurement of it"
        ),
    }
    if not samples:
        # Distinct from a book that never went underwater. No samples means no run to
        # look at: a strategy that opened nothing, or a result from before this was
        # recorded.
        report["reason"] = (
            "no excursion samples were recorded for this run — it opened no positions, "
            "or predates this diagnostic"
        )
        return report

    peak = float(samples[0].get("equity_close", 0.0))
    worst: Optional[Dict[str, Any]] = None
    best: Optional[Dict[str, Any]] = None
    worst_pct = 0.0
    best_pct = 0.0
    close_worst: Optional[Dict[str, Any]] = None
    close_worst_pct = 0.0

    for sample in samples:
        close = float(sample.get("equity_close", 0.0))
        peak = max(peak, close)
        if peak <= 0:
            # A book with no equity has no meaningful fraction of one.
            continue
        adverse_pct = (peak - float(sample.get("equity_adverse", close))) / peak * 100.0
        favourable_pct = (float(sample.get("equity_favourable", close)) - peak) / peak * 100.0
        closing_pct = (peak - close) / peak * 100.0
        if adverse_pct > worst_pct or worst is None:
            worst_pct, worst = adverse_pct, _instant(sample, "equity_adverse", peak)
        if favourable_pct > best_pct or best is None:
            best_pct, best = favourable_pct, _instant(sample, "equity_favourable", peak)
        if closing_pct > close_worst_pct or close_worst is None:
            close_worst_pct, close_worst = closing_pct, _instant(sample, "equity_close", peak)

    report.update(
        {
            "available": True,
            "n_steps": len(samples),
            "max_adverse_excursion_pct": worst_pct,
            "adverse_at": worst,
            "max_favourable_excursion_pct": best_pct,
            "favourable_at": best,
            # The same book measured the way the equity curve measures it. The pair is
            # the finding; either alone is half of it.
            "closing_mark": {
                "max_drawdown_pct": close_worst_pct,
                "at": close_worst,
            },
        }
    )
    report["understatement_pct"] = worst_pct - close_worst_pct
    report["sampled_the_same_pain"] = report["understatement_pct"] < MATERIAL_GAP_PCT
    return report


def excursion_lines(report: Dict[str, Any]) -> List[str]:
    """The diagnostic as terminal lines. Reports the pair; grades neither."""
    if not report.get("available"):
        return ["", "=== Portfolio excursion ===", f"  {report.get('reason', 'not available')}"]

    worst, close = report.get("adverse_at") or {}, (report.get("closing_mark") or {}).get("at") or {}
    lines = [
        "",
        "=== Portfolio excursion ===",
        f"  Worst the book ever looked : -{report['max_adverse_excursion_pct']:.2f}% from its peak",
        f"    at {str(worst.get('time'))[:19]} — {worst.get('open_positions')} open, "
        f"gross {_pct(worst.get('gross_exposure_pct'))}, net {_pct(worst.get('net_exposure_pct'))} of equity",
        f"    the equity curve showed -{_num(worst.get('drawdown_at_close_pct'))}% at that same instant",
        f"  Deepest closing drawdown   : -{report['closing_mark']['max_drawdown_pct']:.2f}% "
        f"at {str(close.get('time'))[:19]}",
        f"  Best the book ever looked  : +{report['max_favourable_excursion_pct']:.2f}% above its peak",
    ]
    gap = report.get("understatement_pct", 0.0)
    if report.get("sampled_the_same_pain"):
        lines.append(
            f"  The curve sampled the same pain: intra-bar worst is {gap:.2f}pp below the "
            "closing marks,\n  so the drawdown you can see is the drawdown there was."
        )
    else:
        lines.append(
            f"  The curve did NOT sample the same pain: {gap:.2f}pp deeper intra-bar than the\n"
            "  closing marks ever showed."
        )
    # Wrapped rather than one long line: this is the sentence that stops the upper
    # bound being quoted as a measurement, and it has to be readable to do that.
    lines.append("  Basis: every open position marked at its own worst (and best) tick within")
    lines.append("  the same bar, which assumes they all got there at once — an upper bound on")
    lines.append("  the pain, never a measurement of it.")
    lines.append(
        "  The realized worst lies between the two: the closing mark is a lower bound and\n"
        "  the simultaneous-extremes figure an upper one. Neither is the answer alone."
    )
    return lines


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _num(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}"
