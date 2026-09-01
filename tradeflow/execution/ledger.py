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
    def record_intent(self, symbol: str, side: str, qty: float, order_id: Optional[str] = None) -> None:
        """An order we submitted. Written at submission, before any fill is known."""
        self._append(
            {"event": "intent", "symbol": symbol, "side": side, "qty": float(qty), "order_id": order_id}
        )

    def record_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_id: Optional[str] = None,
        status: str = "filled",
        basis: str = CUMULATIVE,
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
