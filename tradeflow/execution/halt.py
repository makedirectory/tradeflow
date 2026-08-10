"""The kill switch.

There was previously no way to say "stop trading" and have it stick. Orders could be
cancelled and positions closed, but nothing recorded the *decision*, so a running
engine would open a new position on the next bar and an operator who had just
flattened the account would watch it refill. Stopping needs durable state, not a
one-off action.

That state is a file, deliberately:

**No database on the trade clock.** The order path holds no connection to anything
that can be down. A file under the state root is readable by a hung engine, by the
CLI, and by a human with `cat`, which is exactly the population that needs it during
an incident.

**A halt blocks entries, never exits.** A switch that also blocked closing orders
would trap the book at precisely the moment someone had decided to stop - and would
deadlock any flatten against its own gate. Getting out is always allowed.

**It reports; it does not remediate.** Setting a halt places no orders and closes
nothing. Flattening is a separate, explicit act (:mod:`tradeflow.execution.flatten`).

**Absent is not halted.** A missing or unreadable file means "no halt recorded", so a
corrupt state file cannot silently freeze trading. The reverse default would make an
unrelated disk problem look like a deliberate stop.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: The scope that halts everything, whatever strategy is running.
ALL = "all"

_LOCK = threading.Lock()


def default_halt_path() -> Path:
    """Where halt state lives - beside the ledger, under the state root."""
    from tradeflow.settings import state_path

    return state_path("logs", "halts.json")


@dataclass(frozen=True)
class Halt:
    """One recorded decision to stop trading."""

    scope: str
    reason: str
    actor: str
    set_at: str

    def as_dict(self) -> Dict[str, Any]:
        return {"scope": self.scope, "reason": self.reason, "actor": self.actor, "set_at": self.set_at}

    def __str__(self) -> str:
        return f"[{self.scope}] {self.reason} (set by {self.actor} at {self.set_at})"


class HaltState:
    """Durable halt state, scoped globally or to a single strategy."""

    def __init__(self, path: Optional[Any] = None):
        self.path = Path(path) if path else default_halt_path()

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def active(self, scope: Optional[str] = None) -> Optional[Halt]:
        """The halt in force for ``scope``, if any. A global halt covers every scope.

        Returns the global halt in preference to a scoped one, since that is the more
        severe of the two and the more useful thing to show an operator.
        """
        halts = self._read()
        if ALL in halts:
            return halts[ALL]
        if scope is not None and scope in halts:
            return halts[scope]
        return None

    def is_halted(self, scope: Optional[str] = None) -> bool:
        return self.active(scope) is not None

    def list(self) -> List[Halt]:
        return sorted(self._read().values(), key=lambda h: h.set_at, reverse=True)

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def set(self, reason: str, *, actor: str, scope: str = ALL) -> Halt:
        """Record a halt. Re-halting an already-halted scope replaces the reason."""
        halt = Halt(
            scope=scope,
            reason=reason,
            actor=actor,
            set_at=datetime.now(timezone.utc).isoformat(),
        )
        halts = self._read()
        halts[scope] = halt
        self._write(halts)
        logger.warning("HALT SET %s", halt)
        return halt

    def clear(self, scope: str = ALL) -> bool:
        """Lift a halt. Returns True if one was actually in force."""
        halts = self._read()
        if scope not in halts:
            return False
        removed = halts.pop(scope)
        self._write(halts)
        logger.warning("Halt cleared: %s", removed)
        return True

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    def _read(self) -> Dict[str, Halt]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text() or "{}")
        except (OSError, json.JSONDecodeError):
            # Absent is not halted: an unreadable file must not look like a decision
            # nobody made. Loud, because the switch is now not working.
            logger.error("Halt state at %s is unreadable; treating as NO halt", self.path, exc_info=True)
            return {}
        halts = {}
        for scope, record in raw.items():
            try:
                halts[scope] = Halt(
                    scope=scope,
                    reason=record["reason"],
                    actor=record.get("actor", "unknown"),
                    set_at=record.get("set_at", ""),
                )
            except (TypeError, KeyError):
                logger.error("Skipping malformed halt record for scope %r", scope)
        return halts

    def _write(self, halts: Dict[str, Halt]) -> None:
        payload = json.dumps({scope: halt.as_dict() for scope, halt in halts.items()}, indent=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a process killed mid-write leaves the previous state
        # intact rather than a truncated file that reads as "no halt".
        temporary = self.path.with_suffix(".json.tmp")
        with _LOCK:
            temporary.write_text(payload)
            os.replace(temporary, self.path)
