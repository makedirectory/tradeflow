"""The trial store: a queryable SQLite index over the research journal.

The journal (``logs/research_journal.jsonl``) is the source of truth - append-only,
human-readable. This module builds a derived, disposable index over it so a campaign
can be asked "how many configs have I really tried against this strategy+universe,
across every session I've ever run?" JSONL answers that question in O(n) file reads
per query; this answers it with an index lookup.

**Derived, never authoritative.** :meth:`TrialStore.rebuild` reconstructs the whole
table by replaying the journal from scratch - deleting the database file loses
nothing. Two record shapes live in the journal and this module must parse both:

- ``trial:{kind}`` records (``kind`` in backtest/optimize/walkforward/alpha), written
  by :func:`tradeflow.services.audit.journal_trial` - one row is one evaluated config.
- ``research:trial`` records, written by :class:`tradeflow.research.agent.ResearchAgent` -
  these carry no strategy/universe/window of their own (that lives on the sibling
  ``research:session_start`` record for the same ``session_id``), and carry a
  *cumulative* trial count rather than a per-round one, so replay must track
  per-session state as it walks the journal in order.

v1 is passive: this module only records and answers queries. Nothing here changes
a gate verdict - wiring campaign counts into the DSR is a
deliberately separate, evidence-backed decision for later.

**Return-series retention.** v1 denormalized only summary floats
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tradeflow.settings import state_root, trial_journal_path

logger = logging.getLogger(__name__)

#: Default trial-store location (gitignored, alongside the journal it indexes).
DEFAULT_DB_PATH = state_root() / "logs" / "trials.db"


#: Default journal location. One definition in ``settings``, so this and
#: ``services.audit.DEFAULT_TRIAL_JOURNAL`` cannot drift into indexing one file while
#: writing another.
#:
#: Resolved on first *access*, not at import. ``trial_journal_path()`` goes through
#: ``state_path()``, which creates the directory, so evaluating it here made importing
#: this module a filesystem write - and every command imports it. An unwritable or
#: read-only state root then raised at import rather than at first use, which is the
#: "a broken environment must not brick the CLI" case the registry code goes out of
#: its way to protect. Before the journal's location converged onto one definition
#: this was pure path construction and the constant was free.
def default_journal_path() -> Path:
    """The journal this store indexes, honouring an override of the constant.

    Callers inside this module go through here rather than reading the constant
    directly: a module-level ``__getattr__`` is consulted only for attribute access
    *on the module*, and a name bound into the module globals - which is what
    overriding the constant does - shadows it. Reading through ``globals()`` is what
    keeps both true at once, so redirecting the journal still redirects every writer.
    """
    override = globals().get("DEFAULT_JOURNAL_PATH")
    return Path(override) if override is not None else trial_journal_path()


def __getattr__(name: str) -> Any:
    if name == "DEFAULT_JOURNAL_PATH":
        return trial_journal_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Bump when the schema changes shape. On mismatch the store rebuilds from the
#: journal rather than running migration code - the journal is the source of truth,
#: which is exactly what makes that cheap.
#: v2 adds the ``trial_returns`` companion table - per-trial OOS
#: return-series retention, the precondition Reality Check needs (the trial store
#: must contain the failures, not just summary
#: floats). Trials recorded before v2 simply have no stored series (an honest
#: "not used" in any family panel, not a fabricated one) - rebuilding replays the
#: journal, and only journal lines written after this shipped carry a series.
#: v3 adds ``trial_weights`` on the same journal-first pattern - the proposed
#: holdings and factor exposures a portfolio-producing trial arrived at, so a
#: result's book is recoverable without re-running the optimizer. Absent for
#: every row recorded before it, which readers must render as "not recorded"
#: rather than as an empty book.
#: v4 adds ``trial_trades``, the same pattern again for a run's trade table.
#: Opt-in at the point of recording, not here: a long optimization campaign
#: multiplying thousands of candidates by hundreds of trades each is exactly the
#: storage nobody asked for, so only runs you intend to inspect journal one.
#: v5 adds ``contaminated_at``/``contamination_reason``, set by replaying a
#: ``trials:contaminated`` journal event rather than written at record time. Nothing
#: about history is rewritten: the quarantine is itself an appended fact, and a rebuild
#: reproduces it from the journal like every other column here.
SCHEMA_VERSION = 5

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
  metrics_json        TEXT,
  contaminated_at     TEXT,
  contamination_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_family ON trials(strategy, universe_hash, accounting);
CREATE INDEX IF NOT EXISTS idx_trials_dedup  ON trials(params_hash, universe_hash, window_start, window_end);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS trial_returns (
  trial_id     TEXT PRIMARY KEY,
  dates_json   TEXT NOT NULL,
  returns_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trial_weights (
  trial_id             TEXT PRIMARY KEY,
  as_of                TEXT,
  weights_json         TEXT NOT NULL,
  active_weights_json  TEXT,
  exposures_json       TEXT
);
CREATE TABLE IF NOT EXISTS trial_trades (
  trial_id     TEXT PRIMARY KEY,
  columns_json TEXT NOT NULL,
  rows_json    TEXT NOT NULL
);
"""

#: Set while a rebuild is in flight and cleared when it finishes. A replay that dies
#: partway leaves a half-filled index, and without this the next open would find a
#: conforming schema with a current version stamp and no reason to look further - a
#: silently short campaign, which is the one failure this store must never present as
#: a complete one.
_REBUILD_SENTINEL = "rebuild_in_progress"

#: Printed beside a leaderboard, in the payload as well as the terminal. Ranking a
#: campaign's trials and showing the winner is selection bias by construction; the
#: only honest version says so next to the number, every time.
_LEADERBOARD_CAVEAT = {
    "dsr": (
        "Ranked by DEFLATED Sharpe, which already discounts for how many configs the "
        "family tried. Each row's family n_trials is shown: the larger it is, the more "
        "of the leader's edge is selection."
    ),
    "sharpe": (
        "Ranked by RAW Sharpe — the best of N tried configs is a biased estimate of "
        "its own future performance, and this ranking does not correct for it. Compare "
        "each row's deflated Sharpe and its family n_trials before believing the order."
    ),
}

#: Trials excluded from a campaign's multiple-testing count: a forecast has no
#: Sharpe to deflate.
_EXCLUDED_FROM_FAMILY_COUNT = ("alpha",)

#: Kinds a leaderboard must not rank by default. An ``optimize`` row is the *winner
#: of a search* — best-of-N by construction, which is the selection bias the whole
#: evaluation stack exists to correct for, not a result that survived anything. A
#: leaderboard topped by them presents the artifact as the finding, which is exactly
#: what this view is supposed to prevent. ``alpha`` rows are forecasts with no track
#: record at all.
IN_SAMPLE_KINDS = ("optimize", "alpha")
#: The journal event that quarantines a suspect subset of trials. Append-only, like
#: everything else the journal holds: the trials it names stay exactly as they were
#: recorded, and this says what was later learned about them.
CONTAMINATION_TOOL = "trials:contaminated"

#: Kinds whose row carries an internal trial count to SUM rather than 1 to COUNT -
#: a walk-forward validates many inner configs per row, as does a research round
#: that let the optimizer search internally.
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

    ``{"a": 1}`` and ``{"a": 1.0}`` must hash alike or dedup silently fragments.
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
    """Best-effort JSON-safe coercion, independent of ``tradeflow.services.audit`` to
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


#: Memoized: the declared shape never changes within a process, and every store open
#: asks for it.
_DECLARED_SHAPE: Optional[Dict[str, tuple]] = None


def _shape_of(conn: sqlite3.Connection) -> Dict[str, tuple]:
    """Every table in a database and the columns it actually has."""
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {name: tuple(r[1] for r in conn.execute(f"PRAGMA table_info({name})")) for name in names}


def declared_shape() -> Dict[str, tuple]:
    """The shape ``_SCHEMA`` declares, read back from a database built *from it*.

    Deliberately not a hand-written list of tables and columns beside the DDL. A
    second statement of the same thing is a second thing to forget to update, and the
    whole point of the conformance check below is to catch a schema that says one
    thing and a file that says another - a checker with its own idea of the shape
    could disagree with both.
    """
    global _DECLARED_SHAPE
    if _DECLARED_SHAPE is None:
        probe = sqlite3.connect(":memory:")
        try:
            probe.executescript(_SCHEMA)
            _DECLARED_SHAPE = _shape_of(probe)
        finally:
            probe.close()
    return _DECLARED_SHAPE


def schema_drift(conn: sqlite3.Connection) -> List[str]:
    """Where the database on disk disagrees with ``_SCHEMA``, in plain words.

    ``CREATE TABLE IF NOT EXISTS`` adds a *table* that is missing and does nothing at
    all to one that already exists with the wrong columns - so a schema change that
    adds a column reaches a fresh database and no existing one. The version stamp does
    not catch it either: the stamp is written by the rebuild, which used to ``DELETE``
    the rows rather than reshape the tables, leaving a file that declares the current
    version and does not have its columns.

    An *extra* column counts as drift too. It means the file was written by a newer
    schema than this code declares, and an older reader over a newer index is exactly
    as wrong as the reverse - it just fails more quietly.
    """
    declared, actual = declared_shape(), _shape_of(conn)
    problems: List[str] = []
    for table, columns in sorted(declared.items()):
        if table not in actual:
            problems.append(f"table {table!r} is missing")
            continue
        missing = [c for c in columns if c not in actual[table]]
        extra = [c for c in actual[table] if c not in columns]
        if missing:
            problems.append(f"{table} is missing column(s) {', '.join(missing)}")
        if extra:
            problems.append(f"{table} has unexpected column(s) {', '.join(extra)}")
    return problems


class TrialStoreRebuildRefused(RuntimeError):
    """A rebuild would have replaced real rows with an empty index.

    Raised only when the journal this store indexes cannot be read *and* the store
    holds rows. Dropping them then would destroy the only surviving copy of a campaign
    - the store is derived, but derived from a file that has to be there.
    """


class TrialStore:
    """A SQLite index over the research journal. Derived; safe to delete."""

    def __init__(self, db_path: Optional[Any] = None, journal_path: Optional[Any] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        #: The journal this store indexes - a schema-mismatch rebuild (below) must
        #: replay *this* store's own journal, not some other store's.
        self.journal_path = Path(journal_path) if journal_path else default_journal_path()
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
        """Create what is absent, then check that what exists is the right shape.

        The version stamp alone is not enough, and believing it was is what shipped a
        store that reported v5 while its ``trials`` table was still v4 - so
        ``mark-contaminated`` raised ``no such column`` on every store that predated
        the column. Three separate things say the index needs rebuilding, and any one
        of them is sufficient: the stamp disagrees with the code, the file's actual
        columns disagree with ``_SCHEMA``, or a previous rebuild never finished.
        """
        self._conn.executescript(_SCHEMA)
        reasons: List[str] = []

        if self._get_meta(_REBUILD_SENTINEL):
            reasons.append("a previous rebuild did not finish")

        stamped = self._get_meta("schema_version")
        if stamped is not None:
            try:
                if int(stamped) != SCHEMA_VERSION:
                    reasons.append(f"schema v{stamped} != code v{SCHEMA_VERSION}")
            except (TypeError, ValueError):
                reasons.append(f"schema version {stamped!r} is unreadable")

        reasons.extend(schema_drift(self._conn))

        if not reasons:
            if stamped is None:
                self._set_meta("schema_version", str(SCHEMA_VERSION))
                self._conn.commit()
            return

        logger.warning("Trial store needs rebuilding (%s); replaying the journal", "; ".join(reasons))
        try:
            self.rebuild()
        except TrialStoreRebuildRefused as exc:
            # A store must never brick the command it is attached to, and this one is
            # still readable for everything that does not touch the changed columns.
            # Saying so beats both a traceback and silence.
            logger.error("Trial store left as it is: %s", exc)

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
        hash_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert (or idempotently replace) one trial row.

        Keyed on ``id`` - the journal's ``run_id`` - so recording the same trial
        twice (a live write, then a later rebuild replaying it) is a no-op, not a
        duplicate.

        ``hash_params``, when given, is what ``params_hash`` is computed from
        instead of ``params`` - for a trial kind whose *identity* (what makes a
        repeat a repeat) isn't the same thing as what's useful to display. A
        walk-forward's dedup identity is its validation recipe (known before it
        runs); ``params`` stays the chosen tuned config (only known after) so
        ``trials query``/the journal still show what was actually promoted.
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
            "params_hash": params_hash(hash_params if hash_params is not None else params),
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
        Check needs from the *whole* family, not just the survivors.

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

    def record_weights(self, trial_id: str, payload: Optional[Dict[str, Any]]) -> None:
        """Persist one trial's proposed book: the weights it arrived at, its active
        weights against a benchmark portfolio (when it had one), and its factor
        exposures.

        A companion table for the same reason ``trial_returns`` is one - the hot
        family/dedup queries never need a book, and a per-name vector is far
        larger than the summary floats on the row. Same best-effort contract too:
        a payload with no weights is skipped rather than stored as an empty book,
        because "this trial proposed nothing" and "this trial's weights were never
        recorded" must not become the same row.
        """
        if not payload:
            return
        weights = payload.get("weights")
        if not weights:
            return
        try:
            weights_json = json.dumps(_jsonable(weights), sort_keys=True)
            active = payload.get("active_weights")
            exposures = payload.get("exposures")
            active_json = json.dumps(_jsonable(active), sort_keys=True) if active else None
            exposures_json = json.dumps(_jsonable(exposures), sort_keys=True) if exposures else None
        except (TypeError, ValueError):
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO trial_weights "
            "(trial_id, as_of, weights_json, active_weights_json, exposures_json) VALUES (?, ?, ?, ?, ?)",
            (trial_id, _iso(payload.get("as_of")), weights_json, active_json, exposures_json),
        )
        self._conn.commit()

    def record_trades(self, trial_id: str, payload: Optional[Dict[str, Any]]) -> None:
        """Persist one trial's trade table: ``{columns: [...], rows: [[...], ...]}``.

        Opt-in at the caller, and stored as a companion for the same reason the
        return series and the book are: it is the largest thing a trial can carry
        and the hot queries never want it.
        """
        if not payload:
            return
        columns, rows = payload.get("columns"), payload.get("rows")
        if not columns or rows is None:
            return
        try:
            columns_json = json.dumps([str(c) for c in columns])
            rows_json = json.dumps(_jsonable(rows))
        except (TypeError, ValueError):
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO trial_trades (trial_id, columns_json, rows_json) VALUES (?, ?, ?)",
            (trial_id, columns_json, rows_json),
        )
        self._conn.commit()

    def trades_for(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """One trial's stored trade table, or ``None`` when it has none.

        ``None`` means *not recorded* - the run did not opt in, or predates this
        table. It never means "this run made no trades"; that is an empty ``rows``.
        """
        row = self._conn.execute(
            "SELECT columns_json, rows_json FROM trial_trades WHERE trial_id = ?", (trial_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            return {"columns": json.loads(row["columns_json"]), "rows": json.loads(row["rows_json"])}
        except json.JSONDecodeError:
            return None

    def weights_for(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """One trial's stored book, or ``None`` when it has none.

        ``None`` means *not recorded* - every trial predating this table has one,
        as does every trial kind that proposes no portfolio. Callers render that
        distinctly from a book that happens to be empty.
        """
        row = self._conn.execute(
            "SELECT as_of, weights_json, active_weights_json, exposures_json "
            "FROM trial_weights WHERE trial_id = ?",
            (trial_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            return {
                "as_of": row["as_of"],
                "weights": json.loads(row["weights_json"]),
                "active_weights": json.loads(row["active_weights_json"])
                if row["active_weights_json"]
                else None,
                "exposures": json.loads(row["exposures_json"]) if row["exposures_json"] else None,
            }
        except json.JSONDecodeError:
            return None

    def returns_panel(
        self, strategy: str, symbols: Iterable[Any], accounting: int, *, min_overlap: int = 60
    ) -> Dict[str, Any]:
        """The T x K joint panel of stored OOS return series for a trial family,
        on a common (inner-joined) calendar - the input White's Reality Check
        needs.

        Trials with no stored return series, or whose overlap with the reference
        calendar is too small, are excluded and *counted*, never silently dropped.
        The reference calendar is the longest
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
    def mark_contaminated(self, trial_ids: Iterable[str], *, reason: str, at: Optional[str] = None) -> int:
        """Quarantine a suspect subset. Returns how many stored rows it reached.

        The rows are *derived*, so setting a column on them rewrites nothing that
        matters: the trials themselves stay exactly as the journal recorded them, and a
        rebuild reproduces the quarantine by replaying the event that declared it. That
        is the whole shape - never an UPDATE to history, always an appended fact about
        it, with the reason on the record.

        What quarantining does and does not do is a decision, not an implementation
        detail. A contaminated trial is never served as a memo and never ranked, because
        it must not stand in for evidence. It **keeps counting** toward its family's
        multiple-testing total, because the search still happened: you did look at that
        configuration, and dropping it from the count would lower the deflation bar - the
        flattering direction, and the one thing this store must never do quietly.
        """
        ids = [str(i) for i in trial_ids if i]
        if not ids:
            return 0
        stamp = at or datetime.now(timezone.utc).isoformat()
        reached = 0
        for chunk_start in range(0, len(ids), 500):
            chunk = ids[chunk_start : chunk_start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            cur = self._conn.execute(
                f"UPDATE trials SET contaminated_at = ?, contamination_reason = ? "
                f"WHERE id IN ({placeholders}) AND contaminated_at IS NULL",
                [stamp, reason, *chunk],
            )
            reached += cur.rowcount
        self._conn.commit()
        return reached

    def contaminated_count(self) -> Optional[int]:
        """How many rows are quarantined, or ``None`` when the file cannot say.

        ``None`` on a store whose repair was refused: the column is genuinely not
        there, so the honest answer is "unknown", never zero. Zero would read as "we
        checked and nothing is quarantined" on a file that has never been asked.
        """
        try:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM trials WHERE contaminated_at IS NOT NULL"
                ).fetchone()[0]
            )
        except sqlite3.OperationalError:
            return None

    def rebuild(self, journal_path: Optional[Any] = None) -> Dict[str, int]:
        """Drop the derived tables, recreate them from ``_SCHEMA``, and replay the
        journal from scratch. Idempotent: rebuilding twice in a row produces identical
        rows (dedup is keyed on ``run_id``).

        **Drop, not ``DELETE``.** Emptying the tables replays the rows into whatever
        shape the file already had, so a schema change reached a fresh database and no
        existing one. Recreating them is what makes ``SCHEMA_VERSION`` mean something,
        and it is safe for exactly one reason: every column of every table here is
        reconstructed from the journal, the quarantine flags included - they are
        replayed from a ``trials:contaminated`` event like any other fact. Nothing may
        be written into this database that the journal cannot produce again, and
        checking that is the precondition for dropping anything.

        ``meta`` goes too. It holds only the version stamp and the last replay's line
        count, and keeping it across a rebuild is precisely how the stamp came to
        describe a file it did not match.

        Refuses, rather than emptying a populated index, when the journal it indexes
        cannot be read: the store is derived, but derived from a file that has to be
        there.
        """
        journal_path = Path(journal_path) if journal_path else self.journal_path
        if not self._journal_readable(journal_path):
            rows = int(self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
            if rows:
                raise TrialStoreRebuildRefused(
                    f"{rows} recorded trial(s) are indexed here but the journal they came from "
                    f"({journal_path}) cannot be read. Rebuilding would replace them with an "
                    "empty index and the journal is the only other copy. Restore the journal, "
                    "or point --journal at the one this store indexes."
                )

        for table in sorted(declared_shape()):
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        self._conn.executescript(_SCHEMA)
        # Written before the replay and cleared after it, so a rebuild interrupted
        # halfway is detected on the next open instead of passing as a complete index
        # under a freshly written version stamp.
        self._set_meta(_REBUILD_SENTINEL, "1")
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
        self._conn.execute("DELETE FROM meta WHERE key = ?", (_REBUILD_SENTINEL,))
        self._conn.commit()
        logger.info("Rebuilt trial store: %d rows from %d journal lines", n_rows, n_lines)
        return {"rows": n_rows, "journal_lines": n_lines}

    @staticmethod
    def _journal_readable(journal_path: Path) -> bool:
        """Whether the journal can actually be opened and read, not merely named.

        ``exists()`` is not the question. A path that is a directory, or unreadable,
        or on a mount that has gone away answers True to it and then produces zero
        lines - which replays as an empty campaign.
        """
        try:
            with journal_path.open() as fh:
                fh.read(1)
            return True
        except OSError:
            return False

    def _ingest_journal_record(self, record: Dict[str, Any], session_ctx: Dict[str, Dict[str, Any]]) -> bool:
        """Parse one journal line; insert a row if it represents a trial. Returns
        whether a row was inserted."""
        tool = record.get("tool", "")
        returns_payload = None
        weights_payload = None
        trades_payload = None
        if tool.startswith("trial:"):
            kwargs = _row_kwargs_from_cli_trial(record)
            returns_payload = record.get("returns")
            weights_payload = record.get("weights")
            trades_payload = record.get("trades")
        elif tool == CONTAMINATION_TOOL:
            inputs = record.get("inputs") or {}
            self.mark_contaminated(
                inputs.get("trial_ids") or [],
                reason=str(inputs.get("reason") or "unspecified"),
                at=record.get("timestamp"),
            )
            return False
        elif tool == "research:session_start":
            _update_session_context(record, session_ctx)
            return False
        elif tool == "research:trial":
            sid = (record.get("inputs") or {}).get("session_id")
            ctx = session_ctx.setdefault(sid, {"last_cumulative": 0}) if sid else {"last_cumulative": 0}
            if "strategy" not in ctx:
                # No (or not-yet-seen) research:session_start for this session_id -
                # a truncated/corrupted/out-of-order journal.
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
            self.record_returns(
                kwargs["id"], returns_payload.get("dates") or [], returns_payload.get("values") or []
            )
        if weights_payload:
            self.record_weights(kwargs["id"], weights_payload)
        if trades_payload:
            self.record_trades(kwargs["id"], trades_payload)
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
        return self.family_count_by_hash(strategy, universe_hash(symbols), accounting)

    def family_count_by_hash(self, strategy: str, uh: str, accounting: int) -> int:
        """:meth:`family_count` keyed on the stored universe hash directly.

        A stored row carries the hash, not the symbol list it was made from, and a
        hash cannot be inverted - so anything asking "how big is *this row's*
        family" (a leaderboard, a detail view) needs this entry point rather than
        re-hashing symbols it does not have.
        """
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
        git_sha: Optional[str] = None,
    ) -> bool:
        """Has this exact ``(strategy, params, universe, window[, accounting])``
        already been recorded? Best-effort: a lookup failure is treated as "not
        seen" (a false negative just means one redundant walk-forward, not a
        wrongly-skipped new config) - never let this block the caller.
        """
        return (
            self.find(
                strategy=strategy,
                params=params,
                symbols=symbols,
                window_start=window_start,
                window_end=window_end,
                accounting=accounting,
                git_sha=git_sha,
            )
            is not None
        )

    def find(
        self,
        *,
        strategy: str,
        params: Dict[str, Any],
        symbols: Iterable[Any],
        window_start: Optional[Any] = None,
        window_end: Optional[Any] = None,
        accounting: Optional[int] = None,
        git_sha: Optional[str] = None,
        require_trades: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """The most recent stored trial matching this exact
        ``(strategy, params, universe, window[, accounting])``, or ``None``.

        Same filter shape as :meth:`seen`, but returns the full row (denormalized
        metrics + provenance) so a caller can *serve* the prior answer instead of
        just rejecting a repeat. Best-effort, same as :meth:`seen`: a lookup
        failure is treated as "no match" - a caller falling back to a fresh run
        is always safe, silently skipping one never is.

        ``git_sha``, when given, restricts the match to rows whose stored
        ``git_sha`` is either unrecorded (a legacy row, or git unavailable at
        record time) or equal to it - a *known* mismatch means the strategy's
        math may have changed since, so it's treated as no match at all (the
        conservative reading of "a fixed bug looking already tested and fine").
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
            if git_sha is not None:
                clauses.append("(git_sha IS NULL OR git_sha = ?)")
                args.append(git_sha)
            if require_trades:
                # Never serve a trial that evaluated nothing. A run whose data fetch
                # failed produces zero trades and zero everything, and is recorded like
                # any other trial - so the next identical run is answered from it
                # instead of being re-run, which is how one broken run poisons every
                # repeat of itself.
                #
                # Opt-in, because "no trades" is only evidence of failure for a kind
                # that trades. An `alpha` trial is a read-only forecast and never has
                # any, and excluding those would be a second bug wearing the first
                # one's clothes.
                #
                # The journal keeps the record either way - it is append-only and the
                # attempt is a fact. This governs only whether it may stand in for a
                # real answer. Falling back to a fresh run is always safe; serving an
                # empty one never is.
                clauses.append("IFNULL(oos_trades, 0) > 0")
            # A quarantined trial is never served as a memo. Whatever it recorded is
            # under suspicion, and standing in for a fresh run is the one thing a
            # suspect number must not be allowed to do - a fresh run is always safe.
            clauses.append("contaminated_at IS NULL")
            cur = self._conn.execute(
                f"SELECT * FROM trials WHERE {' AND '.join(clauses)} ORDER BY ts DESC LIMIT 1", args
            )
            row = cur.fetchone()
            return dict(row) if row is not None else None
        except (sqlite3.Error, TypeError, ValueError):
            # TypeError/ValueError: params_hash()/universe_hash() above can raise on
            # a value json.dumps can't serialize. The docstring promises this never
            # blocks the caller, so the catch must cover hashing failures too, not
            # just the query itself.
            logger.warning("Trial store dedup lookup failed; treating as unseen", exc_info=True)
            return None

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
        incommensurable numbers. Pass ``all_accounting=True`` to
        see every version, or ``accounting=N`` for one specific other version.
        """
        if accounting is None and not all_accounting:
            from tradeflow.engine.backtest import ACCOUNTING_VERSION

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

    def _filter_sql(
        self,
        *,
        strategy: Optional[str] = None,
        kind: Optional[str] = None,
        symbols: Optional[Iterable[Any]] = None,
        since: Optional[Any] = None,
        until: Optional[Any] = None,
        min_sharpe: Optional[float] = None,
        promotable: Optional[bool] = None,
        accounting: Optional[int] = None,
        all_accounting: bool = False,
    ) -> tuple:
        """The shared ``WHERE`` clause behind listing, counting, and ranking.

        One definition of what a filter *means*, so a listing and its own "n of N"
        count can never disagree about which rows matched.
        """
        if accounting is None and not all_accounting:
            from tradeflow.engine.backtest import ACCOUNTING_VERSION

            accounting = ACCOUNTING_VERSION

        clauses: List[str] = []
        args: List[Any] = []
        if strategy:
            clauses.append("strategy = ?")
            args.append(strategy)
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        if symbols is not None:
            # The normalized universe hash, never string matching: "NVDA,MSFT" and
            # "msft, nvda" are one universe, and this is the definition dedup and
            # the campaign count already use.
            clauses.append("universe_hash = ?")
            args.append(universe_hash(symbols))
        if since is not None:
            clauses.append("ts >= ?")
            args.append(_iso(since))
        if until is not None:
            clauses.append("ts <= ?")
            args.append(_iso(until))
        if min_sharpe is not None:
            clauses.append("oos_sharpe IS NOT NULL AND oos_sharpe >= ?")
            args.append(float(min_sharpe))
        if promotable is not None:
            clauses.append("promotable = ?")
            args.append(int(bool(promotable)))
        if accounting is not None:
            clauses.append("accounting = ?")
            args.append(int(accounting))
        return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), args

    def list_trials(
        self, *, sort: str = "date", limit: int = 20, offset: int = 0, **filters: Any
    ) -> List[Dict[str, Any]]:
        """Filtered, sorted, paginated rows - the query surface a browser needs.

        Paging happens in SQL (``LIMIT``/``OFFSET`` over indexed columns), never by
        loading everything and slicing in Python: a long campaign holds tens of
        thousands of rows, and a browser that reads them all to show twenty is a
        browser that stops working exactly when the store gets interesting.

        Sorting by ``sharpe``/``dsr`` puts NULLs **last** rather than treating an
        unrecorded metric as the worst possible value - a trial kind with no Sharpe
        did not score zero.
        """
        order = {
            "date": "ts DESC",
            "sharpe": "oos_sharpe IS NULL, oos_sharpe DESC",
            "dsr": "deflated_sharpe IS NULL, deflated_sharpe DESC",
        }.get(sort)
        if order is None:
            raise ValueError(f"sort must be one of date/sharpe/dsr, got {sort!r}")
        where, args = self._filter_sql(**filters)
        cur = self._conn.execute(
            f"SELECT * FROM trials {where} ORDER BY {order}, id LIMIT ? OFFSET ?",
            [*args, int(limit), int(offset)],
        )
        return [dict(row) for row in cur.fetchall()]

    def count_trials(self, **filters: Any) -> int:
        """How many rows these filters match in total, ignoring paging - so a
        listing can say "20 of 4,318" rather than implying it showed everything."""
        where, args = self._filter_sql(**filters)
        return int(self._conn.execute(f"SELECT COUNT(*) FROM trials {where}", args).fetchone()[0])

    def get_trial(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Everything the store knows about one trial, or ``None`` if there is no
        such row.

        Assembles the row with its companion records: whether a return series was
        persisted (and how long a window it covers), the proposed book if one was,
        and which later trials were served from this one. Absent companions stay
        ``None`` - a trial recorded before a companion table existed did not *fail*
        to record one.
        """
        row = self._conn.execute("SELECT * FROM trials WHERE id = ?", (trial_id,)).fetchone()
        if row is None:
            return None
        trial = dict(row)
        try:
            trial["params"] = json.loads(trial.get("params_json") or "{}")
        except json.JSONDecodeError:
            trial["params"] = {}
        try:
            trial["metrics"] = json.loads(trial.get("metrics_json") or "{}")
        except json.JSONDecodeError:
            trial["metrics"] = {}
        trial["returns"] = self._returns_summary(trial_id)
        trial["weights"] = self.weights_for(trial_id)
        trial["trades"] = self.trades_for(trial_id)
        trial["reused_by"] = self.reused_by(trial)
        return trial

    def _returns_summary(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Length and date span of a trial's stored return series, without loading
        the values - `show` reports that one exists, not what is in it."""
        row = self._conn.execute(
            "SELECT dates_json FROM trial_returns WHERE trial_id = ?", (trial_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            dates = json.loads(row["dates_json"])
        except json.JSONDecodeError:
            return None
        if not dates:
            return None
        return {"periods": len(dates), "start": dates[0], "end": dates[-1]}

    def reused_by(self, trial: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Later trials that share this one's dedup identity - the reverse of the
        memoization lookup.

        A number served from the store should be traceable in both directions: from
        the reuse back to its origin, and from the origin forward to everything it
        was reused for.
        """
        cur = self._conn.execute(
            """
            SELECT id, ts, kind FROM trials
            WHERE params_hash = ? AND universe_hash = ?
              AND IFNULL(window_start,'') = ? AND IFNULL(window_end,'') = ?
              AND accounting = ? AND id != ? AND IFNULL(ts,'') > IFNULL(?, '')
            ORDER BY ts
            """,
            (
                trial.get("params_hash"),
                trial.get("universe_hash"),
                trial.get("window_start") or "",
                trial.get("window_end") or "",
                int(trial.get("accounting") or 0),
                trial.get("id"),
                trial.get("ts"),
            ),
        )
        return [dict(row) for row in cur.fetchall()]

    def best(
        self,
        *,
        rank_by: str = "dsr",
        limit: int = 10,
        include_in_sample: bool = False,
        **filters: Any,
    ) -> Dict[str, Any]:
        """The store's leaderboard, with the context that makes one honest.

        Ranks by **deflated** Sharpe by default. Ranking a research campaign's
        trials by raw Sharpe and showing the winner is the selection-bias trap this
        project exists to fight; doing it inside our own tooling would be worse
        than doing it by hand, because the tool would lend it authority.

        **In-sample kinds are excluded by default** (see :data:`IN_SAMPLE_KINDS`).
        An ``optimize`` row is the winner of a search — best-of-N by construction —
        so ranking one is ranking the selection bias itself. ``include_in_sample``
        opts back in, and the payload records how many rows were dropped so the
        exclusion is visible rather than silent.

        The return value always carries each row's family ``n_trials`` alongside it,
        so the count travels with the data rather than living in one surface's
        formatting - an agent reading this over a wire sees the same caveat a human
        reads in the terminal.
        """
        if rank_by not in ("dsr", "sharpe"):
            raise ValueError(f"rank_by must be 'dsr' or 'sharpe', got {rank_by!r}")
        # Over-fetch, then drop the in-sample kinds, so excluding them does not
        # silently shorten the board.
        candidates = self.list_trials(
            sort=rank_by, limit=limit * 8 if not include_in_sample else limit, **filters
        )
        excluded = 0
        if not include_in_sample:
            kept = [r for r in candidates if r.get("kind") not in IN_SAMPLE_KINDS]
            excluded = len(candidates) - len(kept)
            candidates = kept
        # Quarantined rows are never ranked, and the count is reported rather than
        # dropped silently - a leaderboard that quietly omits rows is a leaderboard
        # whose length is a claim about the campaign that nobody can check.
        uncontaminated = [r for r in candidates if not r.get("contaminated_at")]
        contaminated_excluded = len(candidates) - len(uncontaminated)
        rows = uncontaminated[:limit]
        for row in rows:
            row["family_n_trials"] = (
                self.family_count_by_hash(row["strategy"], row["universe_hash"], row["accounting"])
                if row.get("strategy")
                else None
            )
        counts = [r["family_n_trials"] for r in rows if r.get("family_n_trials")]
        return {
            "rank_by": rank_by,
            "rows": rows,
            "max_family_n_trials": max(counts) if counts else 0,
            "in_sample_included": include_in_sample,
            "in_sample_excluded": excluded,
            "contaminated_excluded": contaminated_excluded,
            "caveat": _LEADERBOARD_CAVEAT[rank_by],
        }

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
        # session_start (truncated/corrupted/out-of-order journal).
        # Row count vs line count alone can't see this (the row still
        # exists), so it needs its own check.
        n_orphaned = self._conn.execute("SELECT COUNT(*) FROM trials WHERE strategy IS NULL").fetchone()[0]
        schema_version = self._get_meta("schema_version")
        # Normally self-healing: opening the store repairs a mis-shaped file. It
        # survives to here only when the repair was *refused* because the journal
        # could not be read, which is exactly the state worth naming out loud.
        shape_problems = schema_drift(self._conn)
        # The row-vs-line comparison cannot see this one: with no journal there are no
        # lines, and `rows < 0 lines` is False, so an index whose source has gone
        # missing reported itself healthy. It is the opposite of healthy - the record
        # behind these rows is the thing that is gone, and they cannot be rebuilt.
        journal_readable = self._journal_readable(journal_path)
        return {
            "db_path": str(self.db_path),
            "journal_path": str(journal_path),
            "journal_readable": journal_readable,
            "rows": n_rows,
            "journal_lines": n_total_lines,
            "journal_trial_lines": n_trial_lines,
            "orphaned_rows": n_orphaned,
            "contaminated_rows": self.contaminated_count(),
            "schema_version": int(schema_version) if schema_version is not None else None,
            "schema_drift": shape_problems,
            "drift": (
                n_rows < n_trial_lines
                or n_orphaned > 0
                or bool(shape_problems)
                or (n_rows > 0 and not journal_readable)
            ),
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
        "hash_params": record.get("dedup_params"),
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
