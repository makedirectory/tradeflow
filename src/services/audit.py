"""Append-only audit logging (Spec 003 §5.2).

Every tool/service call can be logged with its inputs, the exact resolved config,
a ``run_id``, the git SHA, and a server-side timestamp - so any decision an agent
makes is replayable by a human later. Timestamps come from here, never the agent.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.optimization.config_store import current_git_sha

logger = logging.getLogger(__name__)

#: Default audit log location (append-only JSONL).
DEFAULT_AUDIT_PATH = Path("logs") / "mcp_audit.jsonl"

_LOCK = threading.Lock()


def new_run_id() -> str:
    """A short, unique id for one tool invocation (for cross-referencing artifacts)."""
    return uuid.uuid4().hex[:12]


def audit_log(
    tool: str,
    inputs: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    resolved_config: Optional[Dict[str, Any]] = None,
    result_summary: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Append one well-formed JSONL record describing a call; return its ``run_id``."""
    run_id = run_id or new_run_id()
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "pid": os.getpid(),
        "tool": tool,
        "inputs": _safe(inputs),
        "resolved_config": _safe(resolved_config) if resolved_config is not None else None,
        "result_summary": _safe(result_summary) if result_summary is not None else None,
    }
    if extra:
        record.update(_safe(extra))

    target = Path(path) if path else DEFAULT_AUDIT_PATH
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    return run_id


def _safe(value: Any) -> Any:
    """Best-effort JSON-able coercion (datetimes, numpy scalars, nested)."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
