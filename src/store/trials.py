"""The trial store (spec 026): a queryable SQLite index over the research journal.

The journal (``logs/research_journal.jsonl``) is the source of truth - append-only,
human-readable. This module builds a derived, disposable index over it so a campaign
can be asked "how many configs have I really tried against this strategy+universe,
across every session I've ever run?" JSONL answers that question in O(n) file reads
per query; this answers it with an index lookup.

**Derived, never authoritative.** :meth:`TrialStore.rebuild` reconstructs the whole
table by replaying the journal from scratch - deleting the database file loses
nothing. Two record shapes live in the journal and this module must parse both:

- ``trial:{kind}`` records (``kind`` in backtest/optimize/walkforward/alpha), written
  by :func:`src.services.audit.journal_trial` - one row is one evaluated config.
- ``research:trial`` records, written by :class:`src.research.agent.ResearchAgent` -
  these carry no strategy/universe/window of their own (that lives on the sibling
  ``research:session_start`` record for the same ``session_id``), and carry a
  *cumulative* trial count rather than a per-round one, so replay must track
  per-session state as it walks the journal in order.

v1 is passive: this module only records and answers queries. Nothing here changes
a gate verdict (see spec 026 §3.5 - wiring campaign counts into the DSR is a
deliberately separate, evidence-backed decision for later).

**Return-series retention (spec 023).** v1 denormalized only summary floats
(``oos_sharpe``, ``deflated_sharpe``, ...) onto each row - enough for the DSR, not
enough for White's Reality Check, which needs every trial's actual OOS return
*series* to jointly resample. The ``trial_returns`` companion table
(:meth:`TrialStore.record_returns`, :meth:`TrialStore.returns_panel`) closes that
gap: every trial that has one (not just the survivors - Reality Check over only
the keepers is survivorship bias with extra steps) gets its dated per-period
return series retained, joinable into a common-calendar panel per
``(strategy, universe, accounting)`` family.
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

#: Default trial-store location (gitignored, alongside the journal it indexes).
DEFAULT_DB_PATH = Path("logs") / "trials.db"
#: Default journal location - kept in sync with ``src.services.audit.DEFAULT_TRIAL_JOURNAL``.
DEFAULT_JOURNAL_PATH = Path("logs") / "research_journal.jsonl"

#: Bump when the schema changes shape. On mismatch the store rebuilds from the
#: journal rather than running migration code - the journal is the source of truth,
#: which is exactly what makes that cheap (spec 026 §4.5).
#: v2 (spec 023): adds the ``trial_returns`` companion table - per-trial OOS
#: return-series retention, the precondition Reality Check needs (spec 023 §4
#: hidden factor 1: the trial store must contain the failures, not just summary
#: floats). Trials recorded before v2 simply have no stored series (an honest
#: "not used" in any family panel, not a fabricated one) - rebuilding replays the
#: journal, and only journal lines written after this shipped carry a series.
SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
  id                  TEXT PRIMARY KEY,
  session_id          TEXT,
  ts                  TEXT,
  kind                TEXT NOT NULL,
  strategy            TEXT,
  universe_hash       TEXT NOT NULL,
  window_start        TEXT,
  window_end          TEXT,
  params_hash         TEXT NOT NULL,
  params_json         TEXT NOT NULL,
  is_sharpe           REAL,
  oos_sharpe          REAL,
  oos_profit_factor   REAL,
  oos_max_dd          REAL,
  deflated_sharpe     REAL,
  efficiency          REAL,
  oos_trades          INTEGER,
  promotable          INTEGER,
  n_trials_in_session INTEGER,
  accounting          INTEGER NOT NULL,
  git_sha             TEXT,
  seed                INTEGER,
  metrics_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_family ON trials(strategy, universe_hash, accounting);
CREATE INDEX IF NOT EXISTS idx_trials_dedup  ON trials(params_hash, universe_hash, window_start, window_end);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS trial_returns (
  trial_id     TEXT PRIMARY KEY,
  dates_json   TEXT NOT NULL,
  returns_json TEXT NOT NULL
);
"""

#: Trials excluded from a campaign's multiple-testing count: a forecast has no
#: Sharpe to deflate (spec 026 §4.7).
_EXCLUDED_FROM_FAMILY_COUNT = ("alpha",)
#: Kinds whose row carries an internal trial count to SUM rather than 1 to COUNT -
#: a walk-forward validates many inner configs per row, as does a research round
#: that let the optimizer search internally (spec 026 §4.7).
_SUMMED_KINDS = ("walkforward", "research")


def db_path_for_journal(journal_path) -> Path:
    """The trial-store DB that indexes a given journal file - always its sibling.

    Keeping the two co-located means a test (or an alternate ``--journal`` run) that
    redirects the journal automatically gets an isolated store too.
    """
    return Path(journal_path).with_name("trials.db")


def normalize_universe(symbols: Iterable[Any]) -> List[str]:
    """Upper-case, dedupe, sort - so ``["AAPL","MSFT"]`` and ``["msft","aapl"]`` key alike."""
    return sorted({str(s).strip().upper() for s in symbols if str(s).strip()})


def universe_hash(symbols: Iterable[Any]) -> str:
    canon = ",".join(normalize_universe(symbols))
    return hashlib.sha256(canon.encode()).hexdigest()


def params_hash(params: Optional[Dict[str, Any]]) -> str:
    """A hash stable across key order and int/float spelling of the same value.

    ``{"a": 1}`` and ``{"a": 1.0}`` must hash alike or dedup silently fragments
    (spec 026 §4.4).
    """
    canon = _canon_value(dict(params or {}))
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _canon_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        f = round(float(value), 9)
        return 0.0 if f == 0 else f
    if isinstance(value, dict):
        return {str(k): _canon_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon_value(v) for v in value]
    # numpy scalars (int64/float64/bool_) don't all subclass Python's int/float,
    # so a raw one would otherwise reach json.dumps() and raise TypeError -
    # coerce via .item() so a numpy-typed param hashes like its Python equivalent.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _canon_value(item())
        except (ValueError, TypeError):
            return str(value)
    return value


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe coercion, independent of ``src.services.audit`` to
    avoid a circular import (that module dual-writes into this one)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _iso(value: Any) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


def _is_trial_tool(tool: str) -> bool:
    return tool.startswith("trial:") or tool == "research:trial"


class TrialStore:
    """A SQLite index over the research journal. Derived; safe to delete."""

    def __init__(self, db_path: Optional[Any] = None, journal_path: Optional[Any] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        #: The journal this store indexes - a schema-mismatch rebuild (below) must
        #: replay *this* store's own journal, not some other store's.
        self.journal_path = Path(journal_path) if journal_path else DEFAULT_JOURNAL_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrialStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._conn.commit()
        elif int(row[0]) != SCHEMA_VERSION:
            logger.warning(
                "Trial store schema v%s != code v%s; rebuilding from the journal", row[0], SCHEMA_VERSION
            )
            self.rebuild()

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def _get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def record(
        self,
        *,
        id: str,
        kind: str,
        strategy: Optional[str],
        symbols: Iterable[Any],
        params: Dict[str, Any],
        accounting: int,
        session_id: Optional[str] = None,
        ts: Optional[str] = None,
        window_start: Optional[Any] = None,
        window_end: Optional[Any] = None,
        is_sharpe: Optional[float] = None,
        oos_sharpe: Optional[float] = None,
        oos_profit_factor: Optional[float] = None,
        oos_max_dd: Optional[float] = None,
        deflated_sharpe: Optional[float] = None,
        efficiency: Optional[float] = None,
        oos_trades: Optional[int] = None,
        promotable: Optional[bool] = None,
        n_trials_in_session: Optional[int] = None,
        git_sha: Optional[str] = None,
        seed: Optional[int] = None,
        metrics_full: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert (or idempotently replace) one trial row.

        Keyed on ``id`` - the journal's ``run_id`` - so recording the same trial
        twice (a live write, then a later rebuild replaying it) is a no-op, not a
        duplicate (spec 026 §3.3).
        """
        row = {
            "id": id,
            "session_id": session_id,
            "ts": ts,
            "kind": kind,
            "strategy": strategy,
            "universe_hash": universe_hash(symbols),
            "window_start": _iso(window_start),
            "window_end": _iso(window_end),
            "params_hash": params_hash(params),
            "params_json": json.dumps(_jsonable(params), sort_keys=True),
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "oos_profit_factor": oos_profit_factor,
            "oos_max_dd": oos_max_dd,
            "deflated_sharpe": deflated_sharpe,
            "efficiency": efficiency,
            "oos_trades": oos_trades,
            "promotable": None if promotable is None else int(bool(promotable)),
            "n_trials_in_session": n_trials_in_session,
            "accounting": int(accounting),
            "git_sha": git_sha,
            "seed": seed,
            "metrics_json": json.dumps(_jsonable(metrics_full or {}), sort_keys=True),
        }
        columns = list(row.keys())
        placeholders = ", ".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT OR REPLACE INTO trials ({', '.join(columns)}) VALUES ({placeholders})",
            [row[c] for c in columns],
        )
        self._conn.commit()

    def record_returns(self, trial_id: str, dates: List[Any], values: List[Any]) -> None:
        """Persist one trial's OOS return series - the per-trial input Reality
        Check (spec 023) needs from the *whole* family, not just the survivors.

        A companion table, not a column on ``trials``: the hot family/dedup/gate
        queries (``family_count``, ``seen``, ``query``) scan the small ``trials``
        row only; the (larger) series blob loads only when a Reality Check
        actually runs. Best-effort shape validation only (mismatched/empty
        lengths are silently skipped, matching this module's house style
        elsewhere - :meth:`seen`'s "a lookup failure is not the caller's
        problem") since a trial with no return series is still a valid trial row,
        just one :meth:`returns_panel` will count as unusable.
        """
        if not dates or not values or len(dates) != len(values):
            return
        try:
            values_f = [float(v) for v in values]
        except (TypeError, ValueError):
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO trial_returns (trial_id, dates_json, returns_json) VALUES (?, ?, ?)",
            (trial_id, json.dumps([str(d) for d in dates]), json.dumps(values_f)),
        )
        self._conn.commit()

    def returns_panel(
        self, strategy: str, symbols: Iterable[Any], accounting: int, *, min_overlap: int = 60
    ) -> Dict[str, Any]:
        """The T x K joint panel of stored OOS return series for a trial family,
        on a common (inner-joined) calendar - the input White's Reality Check
        (spec 023) needs.

        Trials with no stored return series, or whose overlap with the reference
        calendar is too small, are excluded and *counted*, never silently dropped
        (spec 023 §4 hidden factors 1/2). The reference calendar is the longest
        stored series (most inclusive), never the shortest - so the family can
        never "improve" by conveniently losing an inconvenient trial's dates.
        Returns a plain-list ``matrix`` (``T`` rows of ``K`` floats, no numpy
        dependency in this module by design); the caller converts.
        """
        uh = universe_hash(symbols)
        rows = self._conn.execute(
            """
            SELECT t.id AS id, r.dates_json AS dates_json, r.returns_json AS returns_json
            FROM trials t JOIN trial_returns r ON r.trial_id = t.id
            WHERE t.strategy = ? AND t.universe_hash = ? AND t.accounting = ?
            """,
            (strategy, uh, int(accounting)),
        ).fetchall()
        n_attempted = self.family_count(strategy, symbols, accounting)

        series: Dict[str, Dict[str, float]] = {}
        for row in rows:
            try:
                dates = json.loads(row["dates_json"])
                values = json.loads(row["returns_json"])
            except json.JSONDecodeError:
                continue
            series[row["id"]] = dict(zip(dates, values))

        empty = {
            "trial_ids": [],
            "dates": [],
            "matrix": [],
            "n_attempted": n_attempted,
            "n_with_returns": len(series),
            "n_used": 0,
            "n_excluded_short": 0,
        }
        if not series:
            return empty

        ref_id = max(series, key=lambda k: len(series[k]))
        ref_dates = set(series[ref_id].keys())
        included: Dict[str, Dict[str, float]] = {}
        for trial_id, s in series.items():
            if len(ref_dates & s.keys()) >= min_overlap:
                included[trial_id] = s
        excluded_short = len(series) - len(included)
        if not included:
            empty["n_excluded_short"] = excluded_short
            return empty

        common_dates = sorted(set.intersection(*(set(s.keys()) for s in included.values())))
        if len(common_dates) < min_overlap:
            empty["n_excluded_short"] = excluded_short
            return empty

        trial_ids = sorted(included)
        matrix = [[included[tid][d] for tid in trial_ids] for d in common_dates]
        return {
            "trial_ids": trial_ids,
            "dates": common_dates,
            "matrix": matrix,
            "n_attempted": n_attempted,
            "n_with_returns": len(series),
            "n_used": len(trial_ids),
            "n_excluded_short": excluded_short,
        }

    # ------------------------------------------------------------------ #
    # Rebuild (the proof that the store is derived, not authoritative)
    # ------------------------------------------------------------------ #
    def rebuild(self, journal_path: Optional[Any] = None) -> Dict[str, int]:
        """Truncate and replay the journal from scratch. Idempotent: rebuilding
        twice in a row produces identical rows (dedup is keyed on ``run_id``)."""
        journal_path = Path(journal_path) if journal_path else self.journal_path
        self._conn.execute("DELETE FROM trials")
        self._conn.execute("DELETE FROM trial_returns")
        self._conn.commit()

        n_lines = 0
        n_rows = 0
        session_ctx: Dict[str, Dict[str, Any]] = {}
        if journal_path.exists():
            with journal_path.open() as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    n_lines += 1
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed journal line %d in %s", n_lines, journal_path)
                        continue
                    if self._ingest_journal_record(record, session_ctx):
                        n_rows += 1

        self._set_meta("schema_version", str(SCHEMA_VERSION))
        self._set_meta("last_rebuild_journal_lines", str(n_lines))
        self._conn.commit()
        logger.info("Rebuilt trial store: %d rows from %d journal lines", n_rows, n_lines)
        return {"rows": n_rows, "journal_lines": n_lines}

    def _ingest_journal_record(self, record: Dict[str, Any], session_ctx: Dict[str, Dict[str, Any]]) -> bool:
        """Parse one journal line; insert a row if it represents a trial. Returns
        whether a row was inserted."""
        tool = record.get("tool", "")
        returns_payload = None
        if tool.startswith("trial:"):
            kwargs = _row_kwargs_from_cli_trial(record)
            returns_payload = record.get("returns")
        elif tool == "research:session_start":
            _update_session_context(record, session_ctx)
            return False
        elif tool == "research:trial":
            sid = (record.get("inputs") or {}).get("session_id")
            ctx = session_ctx.setdefault(sid, {"last_cumulative": 0}) if sid else {"last_cumulative": 0}
            if "strategy" not in ctx:
                # No (or not-yet-seen) research:session_start for this session_id -
                # a truncated/corrupted/out-of-order journal (spec 026 §4.1/§4.2).
                # The row still gets recorded (never silently drop a trial), but
                # with strategy=None it would otherwise vanish from every
                # family_count()/query(strategy=...) with no trace - loud beats
                # quiet here, since that's exactly the undercount this store must
                # never allow unnoticed.
                logger.warning(
                    "research:trial for session_id=%r has no matching session_start; "
                    "recording with strategy=None (run `trials status` to see orphaned rows)",
                    sid,
                )
            kwargs = _row_kwargs_from_research_trial(record, ctx)
            returns_payload = (record.get("inputs") or {}).get("oos_returns")
        else:
            return False
        if not kwargs or not kwargs.get("id"):
            return False
        self.record(**kwargs)
        if returns_payload:
            self.record_returns(kwargs["id"], returns_payload.get("dates") or [], returns_payload.get("values") or [])
        return True

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def family_count(self, strategy: str, symbols: Iterable[Any], accounting: int) -> int:
        """How many trials this ``(strategy, universe, accounting)`` family has seen.

        ``walkforward``/``research`` rows carry an internal count to SUM (each
        validated one config's inner search); every other counted kind is 1 to
        COUNT. ``alpha`` rows never count - a forecast has no Sharpe to deflate.
        """
        uh = universe_hash(symbols)
        placeholders = ", ".join("?" for _ in _SUMMED_KINDS)
        excluded_placeholders = ", ".join("?" for _ in _EXCLUDED_FROM_FAMILY_COUNT)
        cur = self._conn.execute(
            f"""
            SELECT SUM(CASE WHEN kind IN ({placeholders})
                            THEN COALESCE(n_trials_in_session, 1)
                            ELSE 1 END)
            FROM trials
            WHERE strategy = ? AND universe_hash = ? AND accounting = ?
              AND kind NOT IN ({excluded_placeholders})
            """,
            [*_SUMMED_KINDS, strategy, uh, int(accounting), *_EXCLUDED_FROM_FAMILY_COUNT],
        )
        total = cur.fetchone()[0]
        return int(total or 0)

    def seen(
        self,
        *,
        strategy: str,
        params: Dict[str, Any],
        symbols: Iterable[Any],
        window_start: Optional[Any] = None,
        window_end: Optional[Any] = None,
        accounting: Optional[int] = None,
    ) -> bool:
        """Has this exact ``(strategy, params, universe, window[, accounting])``
        already been recorded? Best-effort: a lookup failure is treated as "not
        seen" (a false negative just means one redundant walk-forward, not a
        wrongly-skipped new config) - never let this block the caller.
        """
        try:
            ph = params_hash(params)
            uh = universe_hash(symbols)
            ws, we = _iso(window_start) or "", _iso(window_end) or ""
            clauses = [
                "params_hash = ?",
                "universe_hash = ?",
                "strategy = ?",
                "IFNULL(window_start, '') = ?",
                "IFNULL(window_end, '') = ?",
            ]
            args: List[Any] = [ph, uh, strategy, ws, we]
            if accounting is not None:
                clauses.append("accounting = ?")
                args.append(int(accounting))
            cur = self._conn.execute(f"SELECT 1 FROM trials WHERE {' AND '.join(clauses)} LIMIT 1", args)
            return cur.fetchone() is not None
        except (sqlite3.Error, TypeError, ValueError):
            # TypeError/ValueError: params_hash()/universe_hash() above can raise on
            # a value json.dumps can't serialize. The docstring promises this never
            # blocks the caller, so the catch must cover hashing failures too, not
            # just the query itself.
            logger.warning("Trial store dedup lookup failed; treating as unseen", exc_info=True)
            return False

    def query(
        self,
        *,
        strategy: Optional[str] = None,
        kind: Optional[str] = None,
        accounting: Optional[int] = None,
        all_accounting: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Recent rows matching the given filters, newest first.

        Defaults to the current engine's ``ACCOUNTING_VERSION`` - a listing that
        silently pools rows from different accounting versions invites comparing
        incommensurable numbers (spec 026 §4.6). Pass ``all_accounting=True`` to
        see every version, or ``accounting=N`` for one specific other version.
        """
        if accounting is None and not all_accounting:
            from src.engine.backtest import ACCOUNTING_VERSION

            accounting = ACCOUNTING_VERSION
        clauses, args = [], []
        if strategy:
            clauses.append("strategy = ?")
            args.append(strategy)
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        if accounting is not None:
            clauses.append("accounting = ?")
            args.append(int(accounting))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self._conn.execute(
            f"SELECT * FROM trials {where} ORDER BY ts DESC LIMIT ?", [*args, int(limit)]
        )
        return [dict(row) for row in cur.fetchall()]

    def status(self, journal_path: Optional[Any] = None) -> Dict[str, Any]:
        """Row/journal-line counts and a drift check.

        An undercount (fewer rows than trial-producing journal lines) is the
        dangerous direction - it makes a campaign's Deflated Sharpe *weaker* than
        it should be - so this never silently serves a short count; it flags it.
        """
        journal_path = Path(journal_path) if journal_path else self.journal_path
        n_total_lines = 0
        n_trial_lines = 0
        if journal_path.exists():
            with journal_path.open() as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    n_total_lines += 1
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if _is_trial_tool(record.get("tool", "")):
                        n_trial_lines += 1
        n_rows = self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        # A row with no strategy is a trial that's invisible to every family_count()/
        # query(strategy=...) - a research:trial replayed with no matching
        # session_start (truncated/corrupted/out-of-order journal, spec 026
        # §4.1/§4.2). Row count vs line count alone can't see this (the row still
        # exists), so it needs its own check.
        n_orphaned = self._conn.execute("SELECT COUNT(*) FROM trials WHERE strategy IS NULL").fetchone()[0]
        schema_version = self._get_meta("schema_version")
        return {
            "db_path": str(self.db_path),
            "journal_path": str(journal_path),
            "rows": n_rows,
            "journal_lines": n_total_lines,
            "journal_trial_lines": n_trial_lines,
            "orphaned_rows": n_orphaned,
            "schema_version": int(schema_version) if schema_version is not None else None,
            "drift": n_rows < n_trial_lines or n_orphaned > 0,
        }


# ---------------------------------------------------------------------------- #
# Journal-record parsing (used by rebuild())
# ---------------------------------------------------------------------------- #
def _row_kwargs_from_cli_trial(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a ``trial:{kind}`` record (written by ``journal_trial``) into
    :meth:`TrialStore.record` kwargs."""
    inputs = record.get("inputs") or {}
    window = inputs.get("window") or {}
    metrics = record.get("result_summary") or {}
    tool = record.get("tool", "")
    kind = record.get("kind") or (tool.split(":", 1)[1] if ":" in tool else tool)
    return {
        "id": record.get("run_id"),
        "kind": kind,
        "strategy": inputs.get("strategy"),
        "symbols": inputs.get("symbols") or [],
        "params": record.get("resolved_config") or {},
        "accounting": record.get("accounting", 1),
        "ts": record.get("timestamp"),
        "window_start": window.get("start"),
        "window_end": window.get("end"),
        "oos_sharpe": metrics.get("sharpe_ratio"),
        "oos_profit_factor": metrics.get("profit_factor"),
        "oos_max_dd": metrics.get("max_drawdown"),
        "deflated_sharpe": metrics.get("deflated_sharpe_ratio"),
        "oos_trades": metrics.get("total_trades"),
        "efficiency": record.get("efficiency"),
        "promotable": record.get("promotable"),
        "n_trials_in_session": record.get("n_trials"),
        "git_sha": record.get("git_sha"),
        "metrics_full": metrics,
    }


def _update_session_context(record: Dict[str, Any], session_ctx: Dict[str, Dict[str, Any]]) -> None:
    """Capture ``(strategy, universe, window, seed)`` off a ``research:session_start``
    record - ``research:trial`` records for the same session carry none of it."""
    payload = record.get("inputs") or {}
    sid = payload.get("session_id")
    if not sid:
        return
    window = payload.get("research_window") or {}
    session_ctx[sid] = {
        "strategy": payload.get("strategy"),
        "symbols": payload.get("symbols") or [],
        "window_start": window.get("start"),
        "window_end": window.get("end"),
        "seed": payload.get("seed"),
        "last_cumulative": 0,
    }


def _row_kwargs_from_research_trial(record: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a ``research:trial`` record into :meth:`TrialStore.record` kwargs.

    ``n_trials_cumulative`` in the payload is a running total since the session
    started, not this round's own count - subtract the session's last-seen
    cumulative to recover it (mutates ``ctx`` to track that running total).
    """
    payload = record.get("inputs") or {}
    oos = payload.get("oos_aggregate") or {}
    cumulative = payload.get("n_trials_cumulative")
    n_this_round = None
    if cumulative is not None:
        prior = ctx.get("last_cumulative", 0)
        n_this_round = max(int(cumulative) - int(prior), 1)
        ctx["last_cumulative"] = cumulative
    return {
        "id": record.get("run_id"),
        "kind": "research",
        "strategy": ctx.get("strategy"),
        "symbols": ctx.get("symbols") or [],
        "params": payload.get("params") or {},
        "accounting": record.get("accounting", 1),
        "session_id": payload.get("session_id"),
        "ts": record.get("timestamp"),
        "window_start": ctx.get("window_start"),
        "window_end": ctx.get("window_end"),
        "is_sharpe": payload.get("is_sharpe"),
        "oos_sharpe": payload.get("oos_sharpe", oos.get("sharpe_ratio")),
        "oos_profit_factor": oos.get("profit_factor"),
        "oos_max_dd": payload.get("oos_max_drawdown", oos.get("max_drawdown")),
        "deflated_sharpe": oos.get("deflated_sharpe_ratio"),
        "efficiency": payload.get("efficiency"),
        "oos_trades": oos.get("total_trades"),
        "promotable": payload.get("promotable"),
        "n_trials_in_session": n_this_round,
        "git_sha": record.get("git_sha"),
        "seed": ctx.get("seed"),
        "metrics_full": oos,
    }
