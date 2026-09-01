"""Shared config persistence with provenance - so when a config makes (or loses)
money, you can reconstruct exactly which decisions to blame.

Walk-forward (and, later, the MCP server) produces a *chosen config*
worth saving. This is the one serialization layer both use: plain JSON, with a
``provenance`` block recording exactly how the config was produced so a human can
audit it before promoting it to live trading.

Saving a config **never** alters live behavior - it writes a file a human
chooses to use. Configs land in a gitignored ``configs/`` directory by default.
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tradeflow.engine.backtest import ACCOUNTING_VERSION
from tradeflow.settings import state_root

logger = logging.getLogger(__name__)

#: Default directory for saved configs (gitignored).
DEFAULT_CONFIG_DIR = state_root() / "configs"


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
    #: Engine accounting model behind ``oos_metrics``. Defaults to 1 so a config
    #: written before this field existed loads as the original accounting model —
    #: which is exactly what it is. :func:`build_provenance` always stamps the
    #: current version.
    accounting: int = 1


def current_git_sha() -> Optional[str]:
    """Best-effort identifier for the code that produced a result; ``None`` if unavailable.

    This is a *working-tree* identifier, not just ``HEAD``. Callers use it to decide
    whether a stored trial may be served instead of re-run, and HEAD alone cannot
    answer that: uncommitted edits leave the SHA untouched, so a strategy edited but
    not committed would match its own pre-edit result and be served as though the
    change had been evaluated. That is the one thing the guard exists to prevent, and
    it would fire exactly when iterating, which is when the tree is dirty.

    So a dirty tree gets a ``-dirty`` suffix, which matches no stored row and forces a
    fresh run. Failing toward a redundant run is always safe; silently skipping one
    never is.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        sha = out.stdout.strip()
        if not sha:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return f"{sha}-dirty" if status.stdout.strip() else sha
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
        accounting=ACCOUNTING_VERSION,
    )


def save_config(
    path,
    *,
    strategy: str,
    params: Dict[str, Any],
    scanner: Optional[str] = None,
    symbols: Optional[Any] = None,
    candidate_symbols: Optional[Any] = None,
    capital: Optional[float] = None,
    cost: Optional[Dict[str, Any]] = None,
    provenance: Optional[Provenance] = None,
) -> Path:
    """Write the run configuration as JSON; return the path.

    ``{strategy, scanner, symbols, candidate_symbols, capital, cost, params,
    provenance}``. All but ``provenance`` are *inputs* - what to run - so one file can
    configure a run whatever its type,
    which is the point of a config a private repository versions alongside its
    strategies. ``provenance`` is the opposite: a record of how the params were
    arrived at, never read back as input.

    ``symbols`` is the universe the scanner **resolved**: the book that was validated,
    and what a replay trades. ``candidate_symbols`` is the list it was resolved *from*,
    kept because they are different decisions - re-running a scanner over the resolved
    61 names is a second filter over an already-filtered set, not the original
    85-candidate scan. Only the candidates make a genuine re-resolution possible.

    The window is deliberately absent. A config carrying its own tuning dates would
    make every later run re-evaluate that period by default, which is the one thing a
    saved config must not quietly do; ``provenance.windows`` records what it was tuned
    on, for reading rather than for replaying.

    Keys whose value is ``None`` are omitted rather than written as null, so a config
    that never had a universe is distinguishable from one that was saved without one.
    A relative ``path`` with no directory is placed under :data:`DEFAULT_CONFIG_DIR`.
    """
    path = Path(path)
    if not path.is_absolute() and path.parent == Path("."):
        path = DEFAULT_CONFIG_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "strategy": strategy,
        "scanner": scanner,
        "params": _jsonable(params),
        "provenance": asdict(provenance) if provenance else {},
    }
    for key, value in (
        ("symbols", symbols),
        ("candidate_symbols", candidate_symbols),
        ("capital", capital),
        ("cost", cost),
    ):
        if value is not None:
            payload[key] = _jsonable(value)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    logger.info("Saved config to %s", path)
    return path


def load_config(path) -> Dict[str, Any]:
    """Load a config JSON written by :func:`save_config`.

    The returned ``params`` flow straight into ``strategy_class(params)``.

    Warns when the recorded metrics predate the current accounting model: the
    params are still valid, but the ``oos_metrics`` beside them were measured a
    different way and must not be compared against a fresh run.
    """
    payload = json.loads(Path(path).read_text())
    stored = (payload.get("provenance") or {}).get("accounting", 1)
    if stored != ACCOUNTING_VERSION:
        logger.warning(
            "%s carries accounting v%s metrics but the engine is v%s — its recorded "
            "oos_metrics are NOT comparable with a current run. Re-run the config to "
            "get metrics on the current model.",
            path,
            stored,
            ACCOUNTING_VERSION,
        )
    return payload


def is_current_accounting(payload: Dict[str, Any]) -> bool:
    """Whether a loaded config's metrics were produced by the current engine.

    Callers that rank or compare stored results should check this rather than
    assume every record on disk measured the same thing.
    """
    return (payload.get("provenance") or {}).get("accounting", 1) == ACCOUNTING_VERSION


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
