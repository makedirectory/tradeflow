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


@dataclass
class FlattenReport:
    """What each step of the flatten actually did."""

    started_at: str
    halted: bool = False
    orders_cancelled: bool = False
    positions_closed: bool = False
    failures: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.halted and self.orders_cancelled and self.positions_closed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "complete": self.complete,
            "halted": self.halted,
            "orders_cancelled": self.orders_cancelled,
            "positions_closed": self.positions_closed,
            "failures": self.failures,
        }

    def summary(self) -> str:
        lines = [
            "FLATTEN",
            f"  halt set          : {'yes' if self.halted else 'NO'}",
            f"  orders cancelled  : {'yes' if self.orders_cancelled else 'NO'}",
            f"  positions closed  : {'yes' if self.positions_closed else 'NO'}",
        ]
        for failure in self.failures:
            lines.append(f"  ! {failure}")
        if self.complete:
            lines.append("\nThe engine cannot re-enter while the halt stands. Verify at the broker,")
            lines.append("then `tradeflow resume all` when you are ready.")
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
        report.positions_closed = True
    except BrokerError as exc:
        report.failures.append(f"could not close positions: {exc}")
        logger.error("Could not close positions", exc_info=True)

    if report.complete:
        logger.warning("Flatten complete: %s", reason)
    else:
        logger.error("Flatten INCOMPLETE: %s", report.failures)
    return report
