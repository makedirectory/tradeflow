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

import uuid
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
POSITION_LIMITS = "position_limits"
BROKER = "broker"
POSITION_MATCH = "position_match"

# Why a refusal happened, as a stable family. The reason *message* embeds the numbers
# that caused it - "gross exposure capped: $7,617.12 of $7,200.00" - so counting
# messages turns sixteen refusals of one kind into sixteen rows of one, which hides a
# throttle rather than showing it. The code groups; the message still explains.
BOOK_FULL = "book_full"
GROSS_EXPOSURE = "gross_exposure_capped"
NET_EXPOSURE = "net_exposure_capped"
RISK_BUDGET = "risk_budget_exhausted"
EQUITY_UNREADABLE = "equity_unreadable"

#: Message prefixes written before decisions carried a code, mapped to the family they
#: belong to. A read-path concern only: rows written from here on carry a code and never
#: consult this. It exists because an append-only ledger keeps its history, so a report
#: that groups only new rows shows one throttle as two — some tidy families beside a
#: scatter of one-off messages saying the same thing.
#:
#: Matched by prefix because the numbers follow the colon. A message not listed keeps
#: its own text rather than being forced into a family it may not belong to.
_LEGACY_REASON_PREFIXES = (
    ("book is full", BOOK_FULL),
    ("gross exposure capped", GROSS_EXPOSURE),
    ("net exposure capped", NET_EXPOSURE),
    ("risk budget exhausted", RISK_BUDGET),
    ("cannot check portfolio limits", EQUITY_UNREADABLE),
)


def reason_family(record: Dict[str, Any]) -> str:
    """The family a recorded decision belongs to, whenever it was written.

    Prefers the code the decision carries. Falls back to recognising the message a
    pre-code row was written with, and finally to the message itself — which is the
    right answer for a fixed phrase like "market is closed" that has no numbers in it
    and therefore never fragmented in the first place.
    """
    code = record.get("reason_code")
    reason = str(record.get("reason") or "")
    # A pre-code row has no `reason_code` at all; a coded row whose value equals its own
    # message is one that never had a family, and both are worth normalising.
    if code and code != reason:
        return str(code)
    for prefix, family in _LEGACY_REASON_PREFIXES:
        if reason.startswith(prefix):
            return family
    return reason or "unknown"


@dataclass(frozen=True)
class OrderPlan:
    """What execution intended to send, recorded whether or not the venue took it.

    Separate from the :class:`~tradeflow.brokers.base.OrderResult` the broker returns,
    because the interesting comparison is between the two. A result alone cannot say
    what price the decision was made at, and so cannot say what the fill cost relative
    to it.
    """

    side: str
    qty: float
    #: The price the signal fired at — the bar close execution decided on. Everything
    #: called "slippage" downstream is measured against this and nothing else.
    reference_price: float
    order_type: str = "bracket"
    time_in_force: str = "day"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_order_id: Optional[str] = None
    #: What the cost model expected this to cost. Deliberately not stored in the same
    #: field as a broker fee: a paper venue reports no fees, and collapsing the two
    #: would make a modelled number look like an observed one.
    cost_estimate: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "qty": self.qty,
            "reference_price": self.reference_price,
            "order_type": self.order_type,
            "time_in_force": self.time_in_force,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "client_order_id": self.client_order_id,
            "cost_estimate": self.cost_estimate,
        }


@dataclass(frozen=True)
class Decision:
    """The outcome of handing one signal to execution."""

    symbol: str
    signal: str
    allowed: bool
    reason: str
    guards_consulted: Tuple[str, ...] = field(default_factory=tuple)
    order: Optional[OrderResult] = None
    #: Joins this decision to the intent and the fills that follow it. Generated for
    #: declines too — "why did nothing happen" is the question the audit trail exists
    #: for, and a decline with no id cannot be referred to.
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    plan: Optional[OrderPlan] = None
    #: Stable family for this outcome, for grouping. ``None`` where the reason text is
    #: already a fixed phrase with no numbers in it.
    reason_code: Optional[str] = None

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
            "decision_id": self.decision_id,
            "reason_code": self.reason_code or self.reason,
            "plan": self.plan.as_dict() if self.plan is not None else None,
        }

    def __str__(self) -> str:
        verdict = "acted" if self.allowed else "declined"
        return f"{self.signal} {self.symbol}: {verdict} — {self.reason}"


def allow(symbol, signal, reason, guards, order=None, plan: Optional[OrderPlan] = None) -> Decision:
    return Decision(symbol, signal, True, reason, guards, order, plan=plan)


def decline(symbol, signal, reason, guards, plan=None, code: Optional[str] = None) -> Decision:
    """A refusal. ``plan`` is carried when there was one, so a declined entry still
    records what it would have sent — otherwise the size a limit rejected is lost.

    ``code`` groups refusals of the same kind whose messages differ only in numbers.
    """
    return Decision(symbol, signal, False, reason, guards, None, plan=plan, reason_code=code)
