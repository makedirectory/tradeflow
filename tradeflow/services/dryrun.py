"""What a validated contract would try to do, without trying it.

The decision path and the execution path answer different questions, and running them
together is what forced a $200-300 position ceiling onto a book that wanted far more:
fills were needed to observe anything, so the caps were shrunk until fills happened, and
the resulting sample was biased toward low-priced names. The strategy under observation
was not the strategy that was validated.

A dry run answers the first question alone. It drives the real decision path - the same
``LiveTrader``, the same guards, the same sizing - against a stated account, with the
caps exactly as configured, and reports what *would* have happened. There are no fills,
no slippage and no fees here, because a broker that cannot trade produces none, and
inventing them would recreate the confusion this separation exists to end.

**Nothing is journaled.** A dry run measures nothing: no fills, no realized P&L, no
observed costs, no return series. Recording it would spend a family's statistical budget
on a rehearsal, which is the same reason a screen journals nothing.

**The vocabulary is deliberate.** ``would_submit``, ``would_skip``, ``would_bind`` -
never *submitted*, *filled* or *blocked*. A report that borrows execution's words will
be read as execution's evidence, and the one thing this mode must never be mistaken for
is the small-real session that produces the real thing.
"""

from typing import Any, Dict, List, Optional, Sequence

from tradeflow.brokers.dryrun import BANNER, DryRunBroker
from tradeflow.execution import decision as decisions

#: Refusal families that mean *a configured cap bound*, as opposed to some other reason
#: an order did not go. Only these are reported as ``would_bind``: calling a market-hours
#: skip a "bind" would attribute to the book a limit that the clock imposed.
_BINDING_CODES = frozenset(
    {
        decisions.BOOK_FULL,
        decisions.GROSS_EXPOSURE,
        decisions.NET_EXPOSURE,
        decisions.RISK_BUDGET,
    }
)


def resolve_capital(
    config_capital: Optional[float] = None, explicit: Optional[float] = None
) -> "tuple[float, str]":
    """The capital a dry run reports against, or a refusal.

    Config first, then an explicit override, and **no third source**. There is
    deliberately no fallback to a broker's equity and no default account size: the caps
    a dry run reports are only meaningful against the capital they bound, so inventing
    one would answer a question about a book nobody chose - which is the exact failure
    this mode exists to remove, arriving by a different route.
    """
    capital = explicit if explicit is not None else config_capital
    source = "--capital" if explicit is not None else "from config"
    if capital is None:
        raise ValueError(
            "a dry run needs a stated capital. Promote a config that records one, or "
            "pass --capital. There is no default: the caps this reports are only "
            "meaningful against the capital they bound"
        )
    if float(capital) <= 0:
        raise ValueError(f"a dry run needs a positive capital, not {capital}")
    # Returned with its source, because a reader judging whether the caps are the right
    # ones needs to know whether the number came from the validated config or was typed
    # at the prompt. The two are different claims about what is being rehearsed.
    return float(capital), source


def classify(decision: decisions.Decision, refusal_marker: str) -> str:
    """Which bucket one decision belongs to.

    ``would_submit`` is a decision that reached the broker and was refused *by the dry
    run itself* - it got all the way to submission, so it is an order that would have
    been sent. The plan it carries is what would have been sent, and it exists because
    ``LiveTrader`` builds the plan before submitting for exactly this reason.

    ``would_bind`` is a configured cap refusing it. ``would_skip`` is everything else:
    no signal, market closed, a position already open. The distinction matters because
    only the middle one is a statement about the book's shape.
    """
    if decision.allowed:
        # Unreachable through a dry-run broker, which refuses before any decision can
        # be allowed. Bucketed anyway rather than dropped: a decision that *was* allowed
        # in a mode that cannot trade is a defect, and losing it would hide one.
        return "would_submit"
    if refusal_marker in (decision.reason or ""):
        return "would_submit"
    if decision.reason_code in _BINDING_CODES:
        return "would_bind"
    return "would_skip"


def dry_run_report(
    decisions_seen: Sequence[decisions.Decision],
    *,
    capital: float,
    capital_source: str,
    positions_source: str,
    universe: Sequence[str],
    unable_to_evaluate: Optional[Sequence[Dict[str, Any]]] = None,
    as_of: Optional[Any] = None,
    refusal_marker: str = "this is a dry run",
) -> Dict[str, Any]:
    """The decision path as a report. No fills, no costs, no verdict.

    Every bucket is reported even when empty, because "no signal bound a cap" and "the
    caps were never reached" are different findings and silence would render them alike.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "would_submit": [],
        "would_bind": [],
        "would_skip": [],
        # A symbol no decision could be made about is not a skip. A skip is an outcome
        # the strategy reached; this is the absence of one, and folding the two together
        # would report a universe as fully evaluated when part of it was never asked.
        "unable_to_evaluate": list(unable_to_evaluate or []),
    }
    for decision in decisions_seen:
        bucket = classify(decision, refusal_marker)
        row = {
            "symbol": decision.symbol,
            "signal": decision.signal,
            # A `would_submit` row reached the broker and was refused *by this mode*,
            # which is the mode working. Reporting "broker refused: this is a dry run"
            # would describe the mechanism as a failure of the one thing that succeeded,
            # so the outcome is stated and the mechanism stays internal.
            "reason": "would submit" if bucket == "would_submit" else decision.reason,
            "reason_code": None if bucket == "would_submit" else decision.reason_code,
            "guards_consulted": list(decision.guards_consulted),
        }
        plan = decision.plan
        if plan is not None:
            # What would have been sent, in full. Named `plan` rather than `order`: an
            # order is a thing a venue has; this is a thing nobody sent.
            row["plan"] = {
                "side": plan.side,
                "qty": plan.qty,
                "reference_price": plan.reference_price,
                "stop_loss": plan.stop_loss,
                "take_profit": plan.take_profit,
                "cost_estimate": plan.cost_estimate,
            }
        buckets[bucket].append(row)

    evaluated = len(decisions_seen)
    return {
        "mode": "dry_run",
        "banner": BANNER,
        # Stated rather than discovered, and named as such, so no reader takes it for a
        # balance read off an account.
        "capital": capital,
        "capital_source": capital_source,
        "starting_positions": positions_source,
        "universe": list(universe),
        "as_of": as_of,
        **buckets,
        "counts": {name: len(rows) for name, rows in buckets.items()},
        "n_decisions": evaluated,
        # Reported beside the universe size so a shortfall is visible in the summary and
        # not only in a section somebody has to scroll to. A symbol that vanished from
        # the detail is a symbol nobody notices was never asked.
        "n_evaluated": evaluated,
        "n_unable_to_evaluate": len(buckets["unable_to_evaluate"]),
        # Said explicitly rather than left to be inferred from an absent section. A
        # reader looking for slippage should find out here that this mode cannot produce
        # it, not conclude the run was clean.
        "not_covered": (
            "fills, slippage, broker fees, queueing, and paper/live account effects. "
            "A dry run submits nothing, so it observes none of them — that is what a "
            "small-real session is for"
        ),
        "journaled": False,
    }


def build_broker(
    *,
    capital: float,
    positions: Optional[List[Any]] = None,
    positions_source: Optional[str] = None,
    tradable: Optional[Dict[str, bool]] = None,
    market_open: bool = True,
) -> DryRunBroker:
    """A dry-run broker, with the source of its starting book recorded.

    ``positions_source`` defaults to describing what was actually supplied, so an empty
    book is never ambiguous between "start flat" and "nobody said".
    """
    source = positions_source or ("adopted" if positions else "flat")
    return DryRunBroker(
        capital=capital,
        positions=positions,
        positions_source=source,
        tradable=tradable,
        market_open=market_open,
    )
