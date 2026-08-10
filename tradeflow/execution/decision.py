"""What execution decided, and why.

A signal that produced no order used to leave a log line and nothing else. The engine
saw ``None``, which is the same answer for "the market is closed", "we are halted",
"you already hold this", "the size rounded to zero", and "the broker refused" - so
the one question worth asking afterwards, *why did nothing happen on that bar*, could
only be answered by reading logs, if they still existed.

A :class:`Decision` makes the answer a value instead. It records the outcome, the
reason in words, and - importantly - **which guards were consulted**, not merely which
one fired. A veto list that only ever names the guard that tripped cannot distinguish
a guard that passed from a guard that never ran, which is exactly how a check silently
stops being applied and nobody notices.

Trade-clock code: plain stdlib, no research imports, nothing that can block.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from tradeflow.brokers.base import OrderResult

# The guards, named once so a reason string and a consulted-list entry cannot drift.
HOLD = "hold"
MARKET_HOURS = "market_hours"
HALT = "halt"
EXISTING_POSITION = "existing_position"
PENDING_ORDER = "pending_order"
ACCOUNT = "account"
SIZING = "sizing"
BUYING_POWER = "buying_power"
BROKER = "broker"
POSITION_MATCH = "position_match"


@dataclass(frozen=True)
class Decision:
    """The outcome of handing one signal to execution."""

    symbol: str
    signal: str
    allowed: bool
    reason: str
    guards_consulted: Tuple[str, ...] = field(default_factory=tuple)
    order: Optional[OrderResult] = None

    def __bool__(self) -> bool:
        """Truthy when execution acted, so callers can read as `if decision:`."""
        return self.allowed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "allowed": self.allowed,
            "reason": self.reason,
            "guards_consulted": list(self.guards_consulted),
            "order_id": self.order.id if self.order is not None else None,
        }

    def __str__(self) -> str:
        verdict = "acted" if self.allowed else "declined"
        return f"{self.signal} {self.symbol}: {verdict} — {self.reason}"


def allow(symbol: str, signal: str, reason: str, guards: Tuple[str, ...], order=None) -> Decision:
    return Decision(symbol, signal, True, reason, guards, order)


def decline(symbol: str, signal: str, reason: str, guards: Tuple[str, ...]) -> Decision:
    return Decision(symbol, signal, False, reason, guards, None)
