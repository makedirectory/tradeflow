"""Flatten: stop, cancel everything, close everything.

The most drastic thing this system can be asked to do, and the one most likely to be
asked for by someone who does not trust it any more. Two properties follow from that.

**It goes through the broker, not the engine.** Nothing here asks a running
:class:`~tradeflow.engine.live.LiveEngine` to do anything, so it works when the engine
is wedged, mid-restart, or holding state that is exactly what you no longer believe.

**It halts first.** Cancelling and closing while an engine is still streaming bars is
a race the engine can win - it re-enters on the next bar, and the account refills
behind you. Recording the halt before touching anything closes that window, and is
also the step most worth having if a later one fails.

Every step is attempted even if an earlier one failed. A partial flatten is a bad
outcome, but stopping halfway through because the cancel call errored - and leaving
the positions open - is a worse one. What actually happened comes back in the report,
and the exit code reflects it.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tradeflow.brokers.base import Broker
from tradeflow.brokers.errors import BrokerError
from tradeflow.execution.halt import ALL, HaltState

logger = logging.getLogger(__name__)


#: What a broker read afterwards actually established about the positions.
#:
#: Four values, not a boolean, because submitting a close and observing it is not the
#: same fact and the gap between them is where an operator gets hurt. Closes queue
#: outside market hours and fill piecemeal at the open: a real flatten reported every
#: position closed and the account still held twelve of them eight minutes later.
CLOSED = "yes"  # a broker read came back with no positions
PENDING = "pending"  # closes were submitted and positions are still open
NOT_CLOSED = "no"  # the close request itself failed, and positions are still open
UNKNOWN = "unknown"  # the confirming read failed; nothing here knows what is open


@dataclass
class FlattenReport:
    """What each step of the flatten did, and what a broker read afterwards observed.

    The distinction this type exists to keep is between a request that was *accepted*
    and a state that was *observed*. ``close_submitted`` is the first; ``observed`` is
    the second, and only the second can be called flat.
    """

    started_at: str
    halted: bool = False
    orders_cancelled: bool = False
    close_submitted: bool = False
    observed: str = UNKNOWN
    checked_at: Optional[str] = None
    remaining: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Flat, and known to be. Deliberately requires the *observation*.

        This used to be satisfied by the close call returning, so a flatten whose orders
        merely queued reported success and printed a reassuring next step. Submission is
        not a terminal state; nothing may claim one without a read that saw it.
        """
        return self.halted and self.orders_cancelled and self.observed == CLOSED

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "complete": self.complete,
            "halted": self.halted,
            "orders_cancelled": self.orders_cancelled,
            "close_submitted": self.close_submitted,
            "observed": self.observed,
            "checked_at": self.checked_at,
            "remaining": self.remaining,
            "failures": self.failures,
        }

    def summary(self) -> str:
        remaining = (
            f"{len(self.remaining)} ({', '.join(sorted(self.remaining))})"
            if self.remaining
            else ("none" if self.observed == CLOSED else "unknown")
        )
        lines = [
            "FLATTEN",
            f"  halt set                  : {'yes' if self.halted else 'NO'}",
            f"  orders cancelled          : {'yes' if self.orders_cancelled else 'NO'}",
            f"  close orders submitted    : {'yes' if self.close_submitted else 'NO'}",
            f"  positions observed closed : {self.observed}",
            f"  last broker position check: {self.checked_at or 'never'}",
            f"  remaining positions       : {remaining}",
        ]
        for failure in self.failures:
            lines.append(f"  ! {failure}")

        if self.complete:
            lines.append("\nFlat, and confirmed by a broker read at the time above.")
            lines.append("The engine cannot re-enter while the halt stands;")
            lines.append("`tradeflow resume all` when you are ready.")
        elif self.observed == PENDING:
            # The case that used to read as success. Say what is true: the venue has the
            # orders and has not filled them, which outside market hours can last hours.
            lines.append("\nNOT FLAT YET — the close orders are with the broker and these positions")
            lines.append("are still open. Queued closes do not fill outside market hours, and")
            lines.append("fill piecemeal at the open. Re-check before believing you are flat.")
        elif self.observed == UNKNOWN:
            lines.append("\nUNCONFIRMED — the position read failed, so nothing here knows what")
            lines.append("is open. Check the broker directly.")
        else:
            lines.append("\nINCOMPLETE — check the broker directly and finish by hand.")
        return "\n".join(lines)


def flatten(
    broker: Broker,
    *,
    reason: str,
    actor: str = "cli",
    halt_state: Optional[HaltState] = None,
) -> FlattenReport:
    """Halt, cancel all open orders, and close all positions. Returns what happened."""
    report = FlattenReport(started_at=datetime.now(timezone.utc).isoformat())
    halts = halt_state or HaltState()

    try:
        halts.set(f"flatten: {reason}", actor=actor, scope=ALL)
        report.halted = True
    except OSError as exc:
        # Keep going. An un-halted flatten is worth far more than no flatten, and the
        # report says plainly that re-entry is not blocked.
        report.failures.append(f"could not record the halt: {exc}")
        logger.error("Could not record the halt; continuing to cancel and close", exc_info=True)

    try:
        broker.cancel_all_orders()
        report.orders_cancelled = True
    except BrokerError as exc:
        report.failures.append(f"could not cancel open orders: {exc}")
        logger.error("Could not cancel open orders; still attempting to close positions", exc_info=True)

    try:
        # Orders were cancelled above; asking again is harmless and covers the case
        # where that call failed but the close path can still clear them.
        broker.close_all_positions(cancel_orders=True)
        report.close_submitted = True
    except BrokerError as exc:
        report.failures.append(f"could not close positions: {exc}")
        logger.error("Could not close positions", exc_info=True)

    _confirm(broker, report)

    if report.complete:
        logger.warning("Flatten complete and confirmed flat: %s", reason)
    elif report.observed == PENDING:
        logger.error(
            "Flatten submitted but NOT FLAT: %d position(s) still open at %s (%s)",
            len(report.remaining),
            report.checked_at,
            ", ".join(sorted(report.remaining)),
        )
    else:
        logger.error("Flatten INCOMPLETE: %s", report.failures)
    return report


def _confirm(broker: Broker, report: FlattenReport) -> None:
    """Ask the broker what is actually open, and record it.

    The step that makes the difference between "we asked" and "it happened". A close is
    a *request*: outside market hours it queues, and at the open it fills piecemeal, so
    the interval where the account still holds everything is minutes wide at best.

    One read, and no waiting. This is the command someone runs when they have stopped
    trusting the system, so it must answer now rather than block on a fill that may be
    hours away — and it must never poll a broker in a loop. Reporting a truthful
    ``pending`` is worth more than a confident answer arrived at late.

    A read that fails leaves ``UNKNOWN``, never ``CLOSED``: this is exactly the place
    where an absent answer must not be rendered as a clean one.
    """
    report.checked_at = datetime.now(timezone.utc).isoformat()
    try:
        positions = broker.list_positions() or []
    except BrokerError as exc:
        report.failures.append(f"could not confirm positions: {exc}")
        report.observed = UNKNOWN
        logger.error("Could not confirm the flatten against the broker", exc_info=True)
        return

    report.remaining = [p.symbol for p in positions]
    if not positions:
        report.observed = CLOSED
    else:
        report.observed = PENDING if report.close_submitted else NOT_CLOSED
