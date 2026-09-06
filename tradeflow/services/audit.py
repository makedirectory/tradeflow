"""Append-only audit logging.

Every tool/service call can be logged with its inputs, the exact resolved config,
a ``run_id``, the git SHA, and a server-side timestamp - so any decision an agent
makes is replayable by a human later. Timestamps come from here, never the agent.
"""

import contextlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tradeflow.engine.backtest import ACCOUNTING_VERSION
from tradeflow.optimization.config_store import current_git_sha
from tradeflow.settings import state_root, trial_journal_path

logger = logging.getLogger(__name__)

#: Default audit log location (append-only JSONL).
DEFAULT_AUDIT_PATH = state_root() / "logs" / "mcp_audit.jsonl"


#: The shared research/trial journal: the append-only source of truth a trial store
#: indexes so multiple-testing counts can span a campaign, not one run.
#: The research agent and CLI ``backtest``/``optimize`` all append here, so they must
#: name the *same* file.
#:
#: Resolved on first *access*, not at import — see ``store.trials`` for why: reaching
#: the path creates the directory, and a constant that does so at import time turns a
#: read-only state root into a failure every command inherits before it runs.
def default_trial_journal() -> Path:
    """The journal every writer here appends to, honouring an override of the constant.

    See ``store.trials.default_journal_path`` for why this is a function: a module
    ``__getattr__`` fires only for attribute access on the module, and an override
    binds a name into the globals, which shadows it. Both paths have to keep working -
    the constant is lazy *and* still redirectable.
    """
    override = globals().get("DEFAULT_TRIAL_JOURNAL")
    return Path(override) if override is not None else trial_journal_path()


def __getattr__(name: str) -> Any:
    if name == "DEFAULT_TRIAL_JOURNAL":
        return trial_journal_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@contextlib.contextmanager
def open_trial_store(journal_path: Optional[Any] = None):
    """A trial store against ``journal_path`` (default: this module's journal), or
    ``None`` on any failure to open one.

    One definition, because there were three: the CLI's, the analysis service's, and
    the MCP server's, each opening the same file the same way and each free to drift
    on the thing that matters — *which* journal. The default has to be
    :func:`default_trial_journal`, not ``store.trials.default_journal_path``: those
    two resolve alike in production and differ under a redirected journal, which is
    every test in the suite and any run passing an explicit path.

    The store is derived and passive, so a failure to open one is never the caller's
    problem: every caller treats ``None`` as "skip the store, run normally", never as
    an error to propagate. A broken index must not brick the command attached to it.
    """
    from tradeflow.store.trials import TrialStore, db_path_for_journal

    path = journal_path or default_trial_journal()
    try:
        store = TrialStore(db_path_for_journal(path), journal_path=path)
    except Exception:  # noqa: BLE001 - a passive store never breaks its caller
        logger.warning("Trial store unavailable; continuing without it", exc_info=True)
        yield None
        return
    try:
        yield store
    finally:
        store.close()


#: Metrics denormalized onto a trial record — enough for the gates and the Deflated
#: Sharpe without dumping the full metric block per config.
_TRIAL_METRICS = (
    "sharpe_ratio",
    "total_return",
    "max_drawdown",
    "profit_factor",
    "total_trades",
    "deflated_sharpe_ratio",
)

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
        # Which capital-accounting model produced any metrics in this record, so a
        # journal spanning an engine change stays interpretable on replay.
        "accounting": ACCOUNTING_VERSION,
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


def journal_trial(
    kind: str,
    *,
    strategy: str,
    symbols: Any,
    candidate_symbols: Optional[Any] = None,
    start: Any,
    end: Any,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    objective: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    returns: Optional[Any] = None,
    weights: Optional[Dict[str, Any]] = None,
    trades: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
    dedup_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Record one *evaluated configuration* as a trial in the research journal.

    A trial is one config scored on one ``(universe, window)`` — the unit the
    Deflated Sharpe counts. A grid search of 50 configs is 50 trials, so callers
    over a search loop invoke this once per config, not once per run.

    ``kind`` groups records so a query can include or exclude a class of trial —
    e.g. ``alpha`` runs are read-only forecasts with no Sharpe, so a multiple-testing
    count should skip them while a dedup or IC query would not.

    ``extra`` carries record-level fields the flat ``(params, metrics)`` shape does
    not, such as a walk-forward's internal ``n_trials`` or a promotion verdict.

    ``returns`` (optional) is this trial's own dated per-period OOS
    return series (a ``pandas.Series`` with a ``DatetimeIndex``) — when given, it
    is journaled alongside the summary metrics (so ``rebuild()`` can restore it
    from the journal, the sole source of truth) and dual-written into the trial
    store's ``trial_returns`` table, which is what lets a later Reality Check
    resample this trial jointly with every other trial in its family. Omit it
    (the default) for trial kinds with no genuine OOS series — e.g. ``optimize``
    rows are in-sample search configs, not OOS track records, so they are never
    passed one.

    The universe is normalized (upper-cased, de-duplicated, sorted) so the same set
    of symbols keys identically regardless of how it was typed — a trial store's
    dedup and campaign counts depend on that.

    ``weights`` (optional) is the book this trial proposed - a
    ``{as_of, weights, active_weights, exposures}`` payload from a trial kind that
    constructs a portfolio. Journaled on the record and mirrored into the trial
    store's ``trial_weights`` table on the same journal-first terms as ``returns``,
    so a result's holdings and factor exposures survive without re-running the
    optimizer. Trial kinds that propose no portfolio omit it.

    ``trades`` (optional) is this run's trade table as
    ``{columns: [...], rows: [[...], ...]}``, journaled and mirrored into the trial
    store's ``trial_trades`` table on the same journal-first terms. Opt-in at the
    caller: a search evaluating thousands of candidates, each with hundreds of
    trades, is storage nobody asked for, so only runs meant to be inspected pass it.

    ``dedup_params`` (optional) overrides what the trial store's dedup hash is
    computed from, when a kind's *identity* (what makes a repeat a repeat) isn't
    the same thing as ``params`` (what's useful to display). Persisted alongside
    the record so a later ``trials rebuild`` reconstructs the identical hash from
    the journal alone. Most kinds omit it — ``params`` already serves both roles.
    """
    inputs: Dict[str, Any] = {
        "strategy": strategy,
        "symbols": sorted({str(s).upper() for s in symbols}),
        "window": {"start": _iso(start), "end": _iso(end)},
    }
    if candidate_symbols:
        # The list the universe was resolved *from*, so a config promoted out of this
        # trial can offer a genuine re-resolution rather than a second filter over an
        # already-filtered book. Omitted when absent: a trial that never recorded its
        # candidates should read as incomplete, not as one whose universe was its own
        # candidate list.
        inputs["candidate_symbols"] = sorted({str(s).upper() for s in candidate_symbols})
    if objective:
        inputs["objective"] = objective
    result_summary = {k: metrics[k] for k in _TRIAL_METRICS if k in metrics}
    journal_path = path or default_trial_journal()
    extra_dict = dict(extra or {})
    if dedup_params is not None:
        extra_dict["dedup_params"] = dedup_params
    returns_payload = _serialize_returns(returns)
    if returns_payload is not None:
        extra_dict["returns"] = returns_payload
    if weights:
        extra_dict["weights"] = _safe(weights)
    if trades:
        extra_dict["trades"] = _safe(trades)
    run_id = audit_log(
        f"trial:{kind}",
        inputs,
        resolved_config=dict(params),
        result_summary=result_summary,
        path=journal_path,
        extra={"kind": kind, **extra_dict},
    )
    _index_trial(run_id, kind, inputs, params, result_summary, extra_dict, journal_path)
    return run_id


def _serialize_returns(returns: Optional[Any]) -> Optional[Dict[str, Any]]:
    """A dated return series -> a JSON-safe ``{dates, values}`` payload, or
    ``None`` when there's nothing to persist (never a zero-length record)."""
    if returns is None or len(returns) == 0:
        return None
    return {
        "dates": [_iso(d) for d in returns.index],
        "values": [float(v) for v in returns.to_numpy()],
    }


def _index_trial(
    run_id: str,
    kind: str,
    inputs: Dict[str, Any],
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    extra: Dict[str, Any],
    journal_path: Path,
) -> None:
    """Dual-write this trial into the trial store, alongside the
    journal append. Best-effort: the store is derived, never authoritative, so a
    write failure here must never break the caller - ``trials rebuild`` resyncs
    from the journal, which just succeeded.
    """
    try:
        from tradeflow.store.trials import TrialStore, db_path_for_journal

        window = inputs.get("window") or {}
        with TrialStore(db_path_for_journal(journal_path), journal_path=journal_path) as store:
            store.record(
                id=run_id,
                kind=kind,
                strategy=inputs.get("strategy"),
                symbols=inputs.get("symbols") or [],
                params=params,
                accounting=ACCOUNTING_VERSION,
                ts=datetime.now(timezone.utc).isoformat(),
                window_start=window.get("start"),
                window_end=window.get("end"),
                oos_sharpe=metrics.get("sharpe_ratio"),
                oos_profit_factor=metrics.get("profit_factor"),
                oos_max_dd=metrics.get("max_drawdown"),
                deflated_sharpe=metrics.get("deflated_sharpe_ratio"),
                oos_trades=metrics.get("total_trades"),
                efficiency=extra.get("efficiency"),
                promotable=extra.get("promotable"),
                n_trials_in_session=extra.get("n_trials"),
                git_sha=current_git_sha(),
                metrics_full=metrics,
                hash_params=extra.get("dedup_params"),
            )
            returns_payload = extra.get("returns")
            if returns_payload:
                store.record_returns(
                    run_id, returns_payload.get("dates") or [], returns_payload.get("values") or []
                )
            store.record_weights(run_id, extra.get("weights"))
            store.record_trades(run_id, extra.get("trades"))
    except Exception:  # noqa: BLE001 - the journal append above already succeeded
        logger.warning("Trial store dual-write failed; `trials rebuild` will resync", exc_info=True)


def _iso(when: Any) -> Optional[str]:
    """ISO-format a datetime; pass through strings; ``None`` stays ``None``."""
    if when is None or isinstance(when, str):
        return when
    iso = getattr(when, "isoformat", None)
    return iso() if callable(iso) else str(when)


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


def journal_record_for_trial(trial_id: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The whole journal line a trial was written as, or ``None`` if there is none.

    The store is derived and deliberately keeps only what its queries need — a
    universe *hash* rather than symbols, a params *hash* rather than the recipe that
    was hashed. Everything else a trial recorded is still in the journal, which is the
    source of truth, so reading through to it beats adding columns: a trial written
    before some field existed then reads as *less complete* rather than being
    backfilled into looking more authoritative than it is.

    One scanner, because "which line is this trial's" is one question. A second copy
    of this loop would be free to disagree about what counts as a trial record.
    """
    journal = Path(path or default_trial_journal())
    if not journal.exists():
        return None
    for line in journal.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:  # a torn final line loses one record, never the file
            continue
        if record.get("run_id") == trial_id and str(record.get("tool", "")).startswith("trial:"):
            return record
    return None


def universe_for_trial(trial_id: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The universe a journaled trial ran on, read from the journal itself.

    Returns ``None`` when no journaled trial carries this id - which is different from
    a trial whose journal line has no ``candidate_symbols``, and callers must keep the
    two apart.
    """
    record = journal_record_for_trial(trial_id, path)
    if record is None:
        return None
    inputs = record.get("inputs") or {}
    return {
        "symbols": list(inputs.get("symbols") or []),
        "candidate_symbols": list(inputs.get("candidate_symbols") or []) or None,
        "strategy": inputs.get("strategy"),
        "window": inputs.get("window") or {},
    }
