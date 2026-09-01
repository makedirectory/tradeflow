"""The position reconciliation ledger: what we meant to do, versus what happened.

Orders are submitted and then forgotten. A partial fill, a rejection, or a human
closing a position in the broker's web UI is currently discovered by reading the
P&L and being surprised — which is to say, discovered late, by inference, at the
worst moment.

This is an append-only local record of **intent** (what was submitted) and
**observation** (what the broker reported), reconciled on a schedule against the
broker's actual account state. Its whole job is to make divergence *visible*.

Three rules, and the first two are the ones that keep it safe:

**The broker is authoritative, always.** This ledger never overrides, corrects, or
"repairs" the account. It records what we believed so a difference can be noticed;
when the two disagree, the broker is right and the ledger is a question for a human.

**It reports; it never remediates.** No corrective order is ever placed from here.
An automated system that notices a missed fill and fixes it is an automated system
that can double a position at 3am while nobody is watching. Detection is the
feature; action is a decision.

**Append-only.** Like the research journal, entries are never edited or deleted. A
ledger you can rewrite is a ledger that cannot be trusted about the past, which is
the only thing it exists to be trusted about.

Deliberately trade-clock only: plain stdlib, no pandas, no research imports, and no
unbounded work in the order path.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

#: Divergence classes. Named rather than boolean because the three mean genuinely
#: different things to whoever has to act on them.
#: What a recorded fill quantity measures. ``CUMULATIVE`` is the order's running
#: total (what Alpaca reports); ``INCREMENTAL`` is this event's own shares.
CUMULATIVE = "cumulative"
INCREMENTAL = "incremental"

MISSING = "missing"  # we believe in a position the broker does not have
UNEXPECTED = "unexpected"  # the broker holds something we never ordered
QUANTITY_DRIFT = "quantity_drift"  # both agree it exists, at different sizes

_LOCK = threading.Lock()


def slippage_bps(side: Optional[str], reference_price, fill_price) -> Optional[float]:
    """Adverse price movement between deciding and filling, in basis points.

    Signed so that **positive is always worse**, whichever way the trade went: a buy
    that paid more than the reference price and a sell that received less both come
    out positive. An unsigned measure would let a good sell cancel a bad buy in any
    average taken over it.

    ``None`` when either price is missing — absent is not zero, and a fill with no
    recorded price must not read as one that filled exactly on reference.
    """
    if not reference_price or fill_price is None:
        return None
    direction = 1.0 if str(side).lower() in {"buy", "long"} else -1.0
    return direction * (float(fill_price) - float(reference_price)) / float(reference_price) * 10_000.0


def _elapsed_ms(start: Optional[str], end: Optional[str]) -> Optional[float]:
    """Milliseconds between two ISO timestamps, or None if either is missing."""
    if not start or not end:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() * 1000.0
    except ValueError:
        return None


def default_ledger_path() -> Path:
    """Where the ledger lives — beside the research journal, under the state root."""
    from tradeflow.settings import state_path

    return state_path("logs", "position_ledger.jsonl")


@dataclass
class Divergence:
    """One disagreement between what we intended and what the broker holds."""

    symbol: str
    kind: str
    expected_qty: float
    actual_qty: float
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "kind": self.kind,
            "expected_qty": self.expected_qty,
            "actual_qty": self.actual_qty,
            "detail": self.detail,
        }


@dataclass
class ReconcileReport:
    """The outcome of one sweep."""

    checked_at: str
    divergences: List[Divergence] = field(default_factory=list)
    n_expected: int = 0
    n_actual: int = 0

    @property
    def clean(self) -> bool:
        return not self.divergences

    def as_dict(self) -> Dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "clean": self.clean,
            "n_expected": self.n_expected,
            "n_actual": self.n_actual,
            "divergences": [d.as_dict() for d in self.divergences],
        }

    def summary(self) -> str:
        if self.clean:
            return f"Reconciled {self.n_actual} broker position(s): no divergence."
        lines = [f"RECONCILIATION FOUND {len(self.divergences)} DIVERGENCE(S):"]
        for d in self.divergences:
            lines.append(f"  [{d.kind}] {d.symbol}: {d.detail}")
        lines.append("The broker's state is authoritative. Nothing has been corrected automatically.")
        return "\n".join(lines)


class PositionLedger:
    """Append-only record of intended orders and observed fills."""

    def __init__(self, path: Optional[Any] = None):
        self.path = Path(path) if path else default_ledger_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def record_intent(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> None:
        """An order we submitted. Written at submission, before any fill is known.

        ``decision_id`` and ``order_id`` are what make the lifecycle reconstructable:
        the decision says what the strategy wanted, this says what was sent, and the
        fills that follow carry the same ``order_id``. Without the pair, "what did we
        expect, what did we submit, what did the broker do" is three separate guesses.
        """
        self._append(
            {
                "event": "intent",
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "order_id": order_id,
                "decision_id": decision_id,
                **({"plan": plan} if plan else {}),
            }
        )

    def record_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_id: Optional[str] = None,
        status: str = "filled",
        basis: str = CUMULATIVE,
        fill_price: Optional[float] = None,
        filled_at: Optional[str] = None,
        broker_fee: Optional[float] = None,
    ) -> None:
        """A fill (or partial fill) the broker reported.

        ``basis`` says what ``qty`` measures, and getting it wrong silently inflates
        the book. Alpaca reports the order's *running total* on every partial fill and
        again on the final fill, so summing those events counts the same shares
        repeatedly - an order that filled 8 in three reports arrives as 21.
        """
        self._append(
            {
                "event": "fill",
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "order_id": order_id,
                "status": status,
                "basis": basis,
                "fill_price": fill_price,
                "filled_at": filled_at,
                # Separate from the model's estimate on the intent. A paper venue
                # reports no fees, and ``None`` there means "not reported", which is
                # not the same as zero and must not be averaged as though it were.
                "broker_fee": broker_fee,
            }
        )

    def record_adoption(self, symbol: str, side: str, qty: float, source: str = "broker") -> None:
        """A position found already open and taken over, at start-up or a resync.

        A *baseline*, not a fill: it replaces whatever the ledger believed about the
        symbol rather than adding to it, because the broker's holding at that moment
        is the whole truth about it. Without this the durable record disagrees with
        the very book the process just adopted - the engine resumes six positions and
        reconciliation calls all six unexpected, which is the one situation where an
        operator most needs the two to agree.
        """
        self._append(
            {
                "event": "adopt",
                "symbol": symbol,
                "side": side,
                "qty": abs(float(qty)),
                "source": source,
            }
        )

    def record_close(self, symbol: str) -> None:
        """A position we asked to close. Zeroes our expectation for the symbol."""
        self._append({"event": "close", "symbol": symbol})

    def record_decision(self, decision) -> None:
        """What execution decided about a signal, and which guards it consulted.

        Deliberately recorded whether or not an order resulted: a *declined* signal is
        the case that leaves no other trace, and it is the one you go looking for when
        asking why a strategy did nothing all day. Carries no position meaning, so
        :meth:`expected_positions` ignores it.
        """
        self._append({"event": "decision", **decision.as_dict()})

    def _append(self, record: Dict[str, Any]) -> None:
        """One line, flushed, under a lock.

        Failure here is logged and swallowed: this is a *detection* aid, and a
        ledger that cannot be written must never be the reason an order path
        raises. A gap in the ledger is a visible reconciliation divergence, which
        is exactly the signal it exists to produce.
        """
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        try:
            with _LOCK:
                with self.path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
        except OSError:
            logger.warning("Could not append to the position ledger at %s", self.path, exc_info=True)

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def expected_positions(self) -> Dict[str, float]:
        """Net signed quantity per symbol implied by the ledger.

        A replay, not a running total: the file is the state, so a restarted process
        recovers its expectation exactly. Long is positive, short negative, and a
        symbol netting to zero is dropped rather than recorded as a zero position —
        "flat" and "never traded" should look the same to a reconciliation.
        """
        net, _ = self._replay()
        return net

    def _replay(self) -> tuple:
        """``(net signed quantity per symbol, saw pre-basis fill records)``.

        Cumulative fills are resolved to the **last** report per order rather than
        summed, which makes the replay idempotent: a duplicated event, a stream
        reconnect that repeats history, or a missed intermediate partial all land on
        the same answer, because the broker's running total is already the whole truth
        about that order.

        Bracket legs fall out of this correctly without special-casing - the entry and
        each protective leg carry their own order id, so a stop that fills nets against
        the entry it closes.
        """
        by_order: Dict[str, Dict[str, Any]] = {}
        #: symbol -> (sequence, quantity it was reset to). A close resets to zero; an
        #: adoption resets to what the broker held. Both discard everything before.
        reset: Dict[str, tuple] = {}
        legacy = False
        seq = 0

        for record in self._read():
            symbol = record.get("symbol")
            if not symbol:
                continue
            seq += 1
            event = record.get("event")
            if event == "close":
                # A close zeroes the symbol, so only activity *after* it counts.
                reset[symbol] = (seq, 0.0)
            elif event == "adopt":
                signed = abs(float(record.get("qty") or 0.0))
                if str(record.get("side", "")).lower() in {"sell", "short"}:
                    signed = -signed
                reset[symbol] = (seq, signed)
            elif event == "fill":
                basis = record.get("basis")
                if basis is None:
                    # Written before fill accounting distinguished the two, when the
                    # side was defaulted as well. Counted the old way so a reconcile
                    # still runs, and reported, because silently mixing the two would
                    # produce a number with no meaning.
                    legacy = True
                signed = float(record.get("qty") or 0.0)
                if str(record.get("side", "")).lower() in {"sell", "short"}:
                    signed = -signed
                order_id = record.get("order_id")
                if basis == CUMULATIVE and order_id:
                    by_order[str(order_id)] = {"symbol": symbol, "qty": signed, "seq": seq}
                else:
                    # Incremental, or no order id to collapse on: each event stands
                    # alone, keyed uniquely so none overwrites another.
                    by_order[f"#{seq}"] = {"symbol": symbol, "qty": signed, "seq": seq}

        net: Dict[str, float] = {symbol: qty for symbol, (_, qty) in reset.items()}
        for entry in by_order.values():
            baseline_seq = reset.get(entry["symbol"], (0, 0.0))[0]
            if entry["seq"] < baseline_seq:
                continue  # superseded by a close or an adoption
            net[entry["symbol"]] = net.get(entry["symbol"], 0.0) + entry["qty"]
        return {s: q for s, q in net.items() if abs(q) > 1e-9}, legacy

    # ------------------------------------------------------------------ #
    # Execution quality
    # ------------------------------------------------------------------ #
    def lifecycles(self) -> List[Dict[str, Any]]:
        """One row per submitted order: what was decided, sent, and filled.

        The join the ledger exists to make possible — decision to intent by
        ``decision_id``, intent to fill by ``order_id``. Derived here rather than
        written at fill time so it is correct across a restart: a process that dies
        between submitting and filling still has both halves on disk, and nothing in
        the order path has to carry state to make the arithmetic work.
        """
        decisions: Dict[str, Dict[str, Any]] = {}
        intents: Dict[str, Dict[str, Any]] = {}
        fills: Dict[str, List[Dict[str, Any]]] = {}
        for record in self._read():
            event = record.get("event")
            if event == "decision" and record.get("decision_id"):
                decisions[record["decision_id"]] = record
            elif event == "intent" and record.get("order_id"):
                intents[record["order_id"]] = record
            elif event == "fill" and record.get("order_id"):
                fills.setdefault(record["order_id"], []).append(record)

        rows = []
        for order_id, intent in intents.items():
            plan = intent.get("plan") or {}
            decision = decisions.get(intent.get("decision_id")) or {}
            order_fills = fills.get(order_id, [])
            # Cumulative reporting again: the last fill is the whole truth about the
            # order, so filled quantity is its quantity, not the sum of the reports.
            last = order_fills[-1] if order_fills else None
            reference = plan.get("reference_price")
            fill_price = last.get("fill_price") if last else None
            rows.append(
                {
                    "order_id": order_id,
                    "decision_id": intent.get("decision_id"),
                    "symbol": intent.get("symbol"),
                    "side": intent.get("side"),
                    "submitted_qty": intent.get("qty"),
                    "filled_qty": last.get("qty") if last else 0.0,
                    "reference_price": reference,
                    "fill_price": fill_price,
                    "slippage_bps": slippage_bps(intent.get("side"), reference, fill_price),
                    "submitted_at": intent.get("ts"),
                    "decided_at": decision.get("ts") or intent.get("ts"),
                    "filled_at": last.get("filled_at") or last.get("ts") if last else None,
                    "decision_to_fill_ms": _elapsed_ms(
                        decision.get("ts") or intent.get("ts"),
                        (last.get("filled_at") or last.get("ts")) if last else None,
                    ),
                    "n_fill_events": len(order_fills),
                    "order_type": plan.get("order_type"),
                    "time_in_force": plan.get("time_in_force"),
                    "stop_loss": plan.get("stop_loss"),
                    "take_profit": plan.get("take_profit"),
                    "client_order_id": plan.get("client_order_id"),
                    "cost_estimate": plan.get("cost_estimate"),
                    "broker_fee": last.get("broker_fee") if last else None,
                    "reason": decision.get("reason"),
                }
            )
        return rows

    def declines(self) -> List[Dict[str, Any]]:
        """Decisions that produced no order, with the reason each gave."""
        return [
            record
            for record in self._read()
            if record.get("event") == "decision" and not record.get("allowed")
        ]

    def _read(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        try:
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final line (killed mid-write) must not make the whole
                    # ledger unreadable — that would turn a small gap into total
                    # amnesia at exactly the wrong moment.
                    logger.warning("Skipping malformed ledger line in %s", self.path)
        except OSError:
            logger.warning("Could not read the position ledger at %s", self.path, exc_info=True)
        return records

    # ------------------------------------------------------------------ #
    # Reconciliation
    # ------------------------------------------------------------------ #
    def reconcile(self, broker) -> ReconcileReport:
        """Compare what we believe against what the broker holds.

        One ``list_positions`` call, nothing per symbol — this runs inside the trade
        clock and must not scale its API usage with the universe.
        """
        checked_at = datetime.now(timezone.utc).isoformat()
        expected, legacy = self._replay()
        if legacy:
            logger.error(
                "This ledger contains fill records written before fill accounting "
                "distinguished cumulative from incremental quantities, and those "
                "records also defaulted every side to buy. Divergences below may be "
                "artefacts of that. Archive %s and start a fresh ledger.",
                self.path,
            )
        try:
            positions = broker.list_positions() or []
        except Exception:  # noqa: BLE001 - an unreachable broker is not a divergence
            logger.warning("Reconciliation skipped: could not read broker positions", exc_info=True)
            return ReconcileReport(checked_at=checked_at, n_expected=len(expected), n_actual=0)

        actual = {p.symbol: (p.qty if p.is_long else -p.qty) for p in positions}
        report = ReconcileReport(checked_at=checked_at, n_expected=len(expected), n_actual=len(actual))

        for symbol in sorted(set(expected) | set(actual)):
            want, have = expected.get(symbol, 0.0), actual.get(symbol, 0.0)
            if abs(want - have) <= 1e-6:
                continue
            if symbol not in actual:
                report.divergences.append(
                    Divergence(
                        symbol,
                        MISSING,
                        want,
                        0.0,
                        f"ledger expects {want:+g} but the broker holds nothing — "
                        "a fill may have been missed, or the position closed elsewhere",
                    )
                )
            elif symbol not in expected:
                report.divergences.append(
                    Divergence(
                        symbol,
                        UNEXPECTED,
                        0.0,
                        have,
                        f"broker holds {have:+g} that this ledger never ordered — "
                        "opened manually, or by another process",
                    )
                )
            else:
                report.divergences.append(
                    Divergence(
                        symbol,
                        QUANTITY_DRIFT,
                        want,
                        have,
                        f"ledger expects {want:+g}, broker holds {have:+g} — likely a partial fill",
                    )
                )

        self._append({"event": "reconcile", **report.as_dict()})
        if not report.clean:
            logger.error("%s", report.summary())
        return report
