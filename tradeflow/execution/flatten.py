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
#: position closed and the account still held its whole book minutes later.
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
    #: Resting orders the confirming read still saw. ``None`` means it could not be
    #: asked — absent, not zero, in the field that decides whether the book can refill.
    open_orders: Optional[int] = None
    failures: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Flat, and known to be. Deliberately requires the *observation*.

        This used to be satisfied by the close call returning, so a flatten whose orders
        merely queued reported success and printed a reassuring next step. Submission is
        not a terminal state; nothing may claim one without a read that saw it.

        Both legs are held to that. ``open_orders == 0`` rather than ``orders_cancelled``
        because the cancel is a submitted fact too, and a resting order the cancel missed
        can refill the book after the instant the read was taken — so a verifiably empty
        account with an unknown order book is not a terminal state either. ``None``
        (unreadable) fails this, as it should.
        """
        return self.halted and self.observed == CLOSED and self.open_orders == 0

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
            "open_orders": self.open_orders,
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
            f"  resting orders observed   : {'unknown' if self.open_orders is None else self.open_orders}",
        ]
        for failure in self.failures:
            lines.append(f"  ! {failure}")

        if self.complete:
            lines.append("\nFlat, and confirmed by a broker read at the time above: no")
            lines.append("positions and no resting orders. The engine cannot re-enter while")
            lines.append("the halt stands; `tradeflow resume all` when you are ready.")
        elif self.observed == PENDING:
            # The case that used to read as success. Say what is true and no more. It
            # must NOT promise a fill: a symbol refused for being halted, non-tradable
            # or restricted never will, and this cannot tell that apart from a queued
            # close, because the venue's per-symbol answer is not surfaced here.
            lines.append("\nNOT FLAT — the close orders were accepted and these positions are")
            lines.append("still open. A queued close fills at the next open; a *refused* one")
            lines.append("never will, and this cannot tell them apart. Re-run to re-read, and")
            lines.append("check the symbols above at the broker if they persist.")
        elif self.observed == UNKNOWN:
            lines.append("\nUNCONFIRMED — the position read failed, so nothing here knows what")
            lines.append("is open. Check the broker directly.")
        elif self.observed == CLOSED:
            # Positions verifiably gone, but something else is unfinished. The old
            # catch-all said "finish by hand" while pointing at the positions, which are
            # the one thing that is done.
            lines.append("\nPOSITIONS FLAT, BUT NOT FINISHED:")
            unfinished = []
            if not self.halted:
                unfinished.append("the halt was not recorded, so the engine may re-enter")
            if self.open_orders is None:
                unfinished.append("the resting orders could not be read")
            elif self.open_orders:
                unfinished.append(f"{self.open_orders} order(s) still resting, which can refill the book")
            for item in unfinished or ["something above did not complete"]:
                lines.append(f"  - {item}")
        else:
            lines.append("\nNOT FLAT — the close request was not accepted and these positions")
            lines.append("are still open. Check the broker directly and finish by hand.")
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
        # The return value is consulted, not discarded. A broker that answers False has
        # said it did not take the request, and recording that as "submitted" would be
        # the same lie this change exists to remove, one line above the fix.
        report.close_submitted = bool(broker.close_all_positions(cancel_orders=True))
        if not report.close_submitted:
            report.failures.append("the broker did not accept the close request")
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
        positions = broker.list_positions()
        if positions is None:
            # A broker that answers "I don't know" must not be read as "nothing is
            # open". `or []` did exactly that, in the one function whose whole purpose
            # is refusing to call an unobserved book flat.
            raise BrokerError("broker returned no position list")
        # Inside the guard: a lazy or paged result raises while being consumed, not
        # when it is returned.
        report.remaining = [position.symbol for position in positions]
    except Exception as exc:  # noqa: BLE001 - see below
        # Deliberately bare, matching `PositionLedger.reconcile`. This is a *reporting*
        # step running after the halt, the cancel and the closes have already happened,
        # and a reporting step that raises destroys the report of the order path — the
        # operator loses not just the confirmation but the fact that the halt was set
        # and the closes were sent. Catching only `BrokerError` was a promise about
        # brokers this module does not control: a `TimeoutError` from a socket, or an
        # `AttributeError` from a partial implementation, escaped and left a traceback
        # where the report should be. `UNKNOWN` already means precisely this.
        report.failures.append(f"could not confirm positions: {exc}")
        report.observed = UNKNOWN
        logger.error("Could not confirm the flatten against the broker", exc_info=True)
        return

    if report.remaining:
        report.observed = PENDING if report.close_submitted else NOT_CLOSED
    else:
        report.observed = CLOSED

    # The same question asked of the other leg. `orders_cancelled` is a *submitted*
    # fact, and this change exists because submitted is not observed — applying that to
    # the positions and not to the orders would leave `complete` meaning "one observed
    # fact and one hopeful one". A resting order the cancel missed can refill the book
    # after the instant this read was taken.
    try:
        resting = broker.list_open_orders()
        report.open_orders = len(resting or [])
    except Exception as exc:  # noqa: BLE001 - as above; never break the report
        report.failures.append(f"could not confirm open orders: {exc}")
        report.open_orders = None
        logger.error("Could not confirm resting orders against the broker", exc_info=True)
