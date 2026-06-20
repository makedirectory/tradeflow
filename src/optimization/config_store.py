"""Shared config persistence with provenance - so when a config makes (or loses)
money, you can reconstruct exactly which decisions to blame.

Walk-forward (and, later, the MCP server) produces a *chosen config*
worth saving. This is the one serialization layer both use: plain JSON, with a
``provenance`` block recording exactly how the config was produced so a human can
audit it before promoting it to live trading.

Saving a config **never** alters live behaviour - it writes a file a human
chooses to use. Configs land in a gitignored ``configs/`` directory by default.
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Default directory for saved configs (gitignored).
DEFAULT_CONFIG_DIR = Path("configs")


@dataclass
class Provenance:
    """How a config was produced - the audit trail."""

    method: str = ""  # grid | random | bayesian
    objective: str = ""
    windows: Dict[str, Any] = field(default_factory=dict)  # start/end/folds/holdout
    oos_metrics: Dict[str, float] = field(default_factory=dict)
    n_trials: int = 1
    seed: Optional[int] = None
    git_sha: Optional[str] = None
    timestamp: Optional[str] = None
    notes: str = ""


def current_git_sha() -> Optional[str]:
    """Best-effort short git SHA of the working tree; ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def build_provenance(
    *,
    method: str,
    objective: str,
    windows: Dict[str, Any],
    oos_metrics: Dict[str, float],
    n_trials: int = 1,
    seed: Optional[int] = None,
    timestamp: Optional[datetime] = None,
    notes: str = "",
) -> Provenance:
    """Assemble a :class:`Provenance` record, stamping git SHA and time."""
    stamp = (timestamp or datetime.now(timezone.utc)).isoformat()
    return Provenance(
        method=method,
        objective=objective,
        windows=_jsonable(windows),
        oos_metrics=_jsonable(oos_metrics),
        n_trials=int(n_trials),
        seed=seed,
        git_sha=current_git_sha(),
        timestamp=stamp,
        notes=notes,
    )


def save_config(
    path,
    *,
    strategy: str,
    params: Dict[str, Any],
    scanner: Optional[str] = None,
    provenance: Optional[Provenance] = None,
) -> Path:
    """Write ``{strategy, scanner, params, provenance}`` as JSON; return the path.

    A relative ``path`` with no directory is placed under :data:`DEFAULT_CONFIG_DIR`.
    """
    path = Path(path)
    if not path.is_absolute() and path.parent == Path("."):
        path = DEFAULT_CONFIG_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "strategy": strategy,
        "scanner": scanner,
        "params": _jsonable(params),
        "provenance": asdict(provenance) if provenance else {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    logger.info("Saved config to %s", path)
    return path


def load_config(path) -> Dict[str, Any]:
    """Load a config JSON written by :func:`save_config`.

    The returned ``params`` flow straight into ``strategy_class(params)``.
    """
    return json.loads(Path(path).read_text())


def _jsonable(value: Any) -> Any:
    """Coerce numpy / datetime / nested values into JSON-native types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    # numpy scalars expose .item(); fall back to the value itself.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return value
    return value
