"""The trial store: the queryable SQLite index over the research
journal. Exercises rebuild idempotence, crash recovery/drift detection, hash
normalization, dedup, and campaign-level family counting - the property that is
broken today (``n_trials`` resets every session).
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from tradeflow.engine.backtest import ACCOUNTING_VERSION
from tradeflow.services.audit import audit_log, journal_trial
from tradeflow.store.trials import (
    _REBUILD_SENTINEL,
    SCHEMA_VERSION,
    TrialStore,
    TrialStoreRebuildRefused,
    db_path_for_journal,
    params_hash,
    schema_drift,
    universe_hash,
)


# --- hashing / normalization -------------------------------------------------
def test_universe_hash_normalizes_case_order_and_dupes():
    assert universe_hash(["msft", "AAPL", "aapl"]) == universe_hash(["AAPL", "MSFT"])
    assert universe_hash(["AAPL"]) != universe_hash(["MSFT"])


def test_params_hash_normalizes_int_float_and_key_order():
    assert params_hash({"a": 1, "b": 2.0}) == params_hash({"b": 2, "a": 1.0})
    assert params_hash({"a": 1}) != params_hash({"a": 2})
    assert params_hash({"flag": True}) != params_hash({"flag": 1})  # bool stays distinct from int


# --- rebuild idempotence -----------------------------------------------------
def test_rebuild_from_cli_journal_is_idempotent(tmp_path):
    journal = tmp_path / "journal.jsonl"
    for p in (10, 20, 30):
        journal_trial(
            "backtest",
            strategy="demo_trend",
            symbols=["aapl", "MSFT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 1),
            params={"fast_ema_period": p},
            metrics={"sharpe_ratio": 1.0, "total_trades": 5},
            path=journal,
        )

    store = TrialStore(tmp_path / "isolated.db")  # distinct from the journal's own sibling db
    stats = store.rebuild(journal)
    assert stats == {"rows": 3, "journal_lines": 3}
    rows = store.query(strategy="demo_trend", limit=10)
    ids_first = {r["id"] for r in rows}
    assert len(ids_first) == 3

    stats2 = store.rebuild(journal)
    assert stats2 == stats
    rows2 = store.query(strategy="demo_trend", limit=10)
    assert {r["id"] for r in rows2} == ids_first


# --- crash recovery / drift ---------------------------------------------------
def test_crash_recovery_drift_detected_and_fixed_by_rebuild(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal_trial(
        "backtest",
        strategy="s",
        symbols=["A"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1),
        params={"x": 1},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )
    # Simulate a crash between the journal append and the row insert: append a
    # trial-shaped record directly, bypassing journal_trial's dual write.
    crashed = {
        "run_id": "crashed01",
        "timestamp": "2024-01-01T00:00:00",
        "git_sha": None,
        "accounting": ACCOUNTING_VERSION,
        "pid": 1,
        "tool": "trial:backtest",
        "kind": "backtest",
        "inputs": {
            "strategy": "s",
            "symbols": ["A"],
            "window": {"start": "2024-01-01T00:00:00", "end": "2024-02-01T00:00:00"},
        },
        "resolved_config": {"x": 2},
        "result_summary": {"sharpe_ratio": 1.5},
    }
    with journal.open("a") as fh:
        fh.write(json.dumps(crashed) + "\n")

    store = TrialStore(db_path_for_journal(journal))  # the same store journal_trial dual-wrote into
    status = store.status(journal)
    assert status["journal_trial_lines"] == 2
    assert status["rows"] == 1
    assert status["drift"] is True

    stats = store.rebuild(journal)
    assert stats["rows"] == 2
    status2 = store.status(journal)
    assert status2["drift"] is False
    assert status2["rows"] == 2


# --- family count: the property that's broken today (resets every session) --
def test_family_count_sums_walkforward_counts_others_as_one_excludes_alpha(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal_trial(
        "backtest",
        strategy="s",
        symbols=["A"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1),
        params={"x": 1},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )
    journal_trial(
        "backtest",
        strategy="s",
        symbols=["A"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1),
        params={"x": 2},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )
    journal_trial(
        "walkforward",
        strategy="s",
        symbols=["A"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1),
        params={"x": 3},
        metrics={"sharpe_ratio": 1.0},
        extra={"n_trials": 12, "promotable": True},
        path=journal,
    )
    journal_trial(
        "alpha",
        strategy="s",
        symbols=["A"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1),
        params={},
        metrics={},
        path=journal,
    )

    store = TrialStore(db_path_for_journal(journal))
    n = store.family_count("s", ["A"], ACCOUNTING_VERSION)
    assert n == 2 + 12  # two backtests (1 each) + one walk-forward's internal 12; alpha excluded

    # N trials across sessions returns N, not the last session's count: a second
    # "session" (a fresh journal_trial call) accumulates rather than resetting.
    journal_trial(
        "backtest",
        strategy="s",
        symbols=["A"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1),
        params={"x": 4},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )
    assert store.family_count("s", ["A"], ACCOUNTING_VERSION) == 2 + 12 + 1


def test_accounting_isolation(tmp_path):
    store = TrialStore(tmp_path / "trials.db")
    store.record(id="a1", kind="backtest", strategy="s", symbols=["A"], params={"x": 1}, accounting=1)
    store.record(id="a2", kind="backtest", strategy="s", symbols=["A"], params={"x": 2}, accounting=2)
    assert store.family_count("s", ["A"], 1) == 1
    assert store.family_count("s", ["A"], 2) == 1


def test_query_defaults_to_current_accounting_version(tmp_path):
    """A listing that silently pools accounting versions invites comparing
    incommensurable numbers - query() must default-filter."""
    store = TrialStore(tmp_path / "trials.db")
    store.record(id="old", kind="backtest", strategy="s", symbols=["A"], params={"x": 1}, accounting=1)
    store.record(
        id="cur", kind="backtest", strategy="s", symbols=["A"], params={"x": 2}, accounting=ACCOUNTING_VERSION
    )
    assert [r["id"] for r in store.query(strategy="s")] == ["cur"]
    assert [r["id"] for r in store.query(strategy="s", accounting=1)] == ["old"]
    assert {r["id"] for r in store.query(strategy="s", all_accounting=True)} == {"old", "cur"}


# --- dedup --------------------------------------------------------------------
def test_seen_dedup_lookup(tmp_path):
    store = TrialStore(tmp_path / "trials.db")
    store.record(
        id="d1",
        kind="research",
        strategy="s",
        symbols=["A", "B"],
        params={"buy_every": 3},
        accounting=2,
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
    )
    # Same config, differently-cased/ordered universe -> still seen.
    assert store.seen(
        strategy="s",
        params={"buy_every": 3},
        symbols=["b", "a"],
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        accounting=2,
    )
    # A changed param counts as a new trial.
    assert not store.seen(
        strategy="s",
        params={"buy_every": 5},
        symbols=["A", "B"],
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        accounting=2,
    )
    # A different accounting version is not the same trial (the same problem, again).
    assert not store.seen(
        strategy="s",
        params={"buy_every": 3},
        symbols=["A", "B"],
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        accounting=1,
    )


def test_find_returns_the_full_row_or_none(tmp_path):
    store = TrialStore(tmp_path / "trials.db")
    store.record(
        id="d1",
        kind="backtest",
        strategy="s",
        symbols=["A", "B"],
        params={"buy_every": 3},
        accounting=2,
        ts="2024-06-01T00:00:00",
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        oos_sharpe=1.5,
    )
    found = store.find(
        strategy="s",
        params={"buy_every": 3},
        symbols=["A", "B"],
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        accounting=2,
    )
    assert found is not None
    assert found["id"] == "d1"
    assert found["oos_sharpe"] == 1.5
    assert (
        store.find(
            strategy="s",
            params={"buy_every": 999},
            symbols=["A", "B"],
            window_start=datetime(2024, 1, 1),
            window_end=datetime(2024, 6, 1),
            accounting=2,
        )
        is None
    )


def test_find_most_recent_match_when_several_exist(tmp_path):
    """--force appends rather than overwrites; find() should serve the freshest."""
    store = TrialStore(tmp_path / "trials.db")
    for run_id, ts in (("old", "2024-01-01T00:00:00"), ("new", "2024-06-01T00:00:00")):
        store.record(
            id=run_id,
            kind="backtest",
            strategy="s",
            symbols=["A"],
            params={"x": 1},
            accounting=2,
            ts=ts,
            window_start=datetime(2024, 1, 1),
            window_end=datetime(2024, 6, 1),
        )
    found = store.find(
        strategy="s",
        params={"x": 1},
        symbols=["A"],
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        accounting=2,
    )
    assert found["id"] == "new"


def test_find_git_sha_mismatch_is_a_miss_but_unrecorded_sha_still_hits(tmp_path):
    store = TrialStore(tmp_path / "trials.db")
    store.record(
        id="known",
        kind="backtest",
        strategy="s",
        symbols=["A"],
        params={"x": 1},
        accounting=2,
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        git_sha="abc123",
    )
    store.record(
        id="legacy",
        kind="backtest",
        strategy="s2",
        symbols=["A"],
        params={"x": 1},
        accounting=2,
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        git_sha=None,
    )
    kwargs = dict(
        params={"x": 1},
        symbols=["A"],
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        accounting=2,
    )
    # A known, differing SHA invalidates the match - the code may have changed.
    assert store.find(strategy="s", git_sha="def456", **kwargs) is None
    # The matching SHA still hits.
    assert store.find(strategy="s", git_sha="abc123", **kwargs) is not None
    # An unrecorded (legacy) SHA is not a *known* mismatch - still hits.
    assert store.find(strategy="s2", git_sha="def456", **kwargs) is not None


def test_record_hash_params_overrides_dedup_hash_but_not_display(tmp_path):
    """A kind whose identity isn't its displayed params (walk-forward's top-level
    recipe-based dedup) hashes on `hash_params`, stores `params`."""
    store = TrialStore(tmp_path / "trials.db")
    store.record(
        id="wf1",
        kind="walkforward",
        strategy="s",
        symbols=["A"],
        params={"chosen_period": 10},
        hash_params={"mode": "anchored", "seed": 42},
        accounting=2,
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
    )
    # A lookup by the recipe (what identifies a repeat) hits...
    found = store.find(
        strategy="s",
        params={"mode": "anchored", "seed": 42},
        symbols=["A"],
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        accounting=2,
    )
    assert found is not None
    assert found["params_json"] == '{"chosen_period": 10}'  # displayed params unaffected
    # ...but a lookup by the displayed (chosen) params does not.
    assert (
        store.find(
            strategy="s",
            params={"chosen_period": 10},
            symbols=["A"],
            window_start=datetime(2024, 1, 1),
            window_end=datetime(2024, 6, 1),
            accounting=2,
        )
        is None
    )


# --- research-session replay (two journal granularities) ---------------------
def test_rebuild_parses_research_session_context(tmp_path):
    """``research:trial`` records carry no strategy/universe/window of their own -
    that lives on the sibling ``research:session_start`` record - and a
    *cumulative* trial count rather than this round's own. Replay must recover
    both by tracking per-session state as it walks the journal in order."""
    journal = tmp_path / "journal.jsonl"
    session_id = "sess1"
    audit_log(
        "research:session_start",
        {
            "session_id": session_id,
            "goal": "g",
            "strategy": "periodic",
            "symbols": ["bbb", "aaa"],
            "research_window": {"start": "2024-01-01T00:00:00", "end": "2024-06-01T00:00:00"},
            "holdout_window": {"start": "2024-06-01T00:00:00", "end": "2024-08-01T00:00:00"},
            "budgets": {},
            "seed": 42,
        },
        path=journal,
    )
    audit_log("research:reject", {"reason": "bad", "hypothesis": "h"}, path=journal)
    audit_log(
        "research:trial",
        {
            "session_id": session_id,
            "round": 1,
            "kind": "tune",
            "hypothesis": "h1",
            "params": {"buy_every": 3},
            "is_sharpe": 0.5,
            "oos_sharpe": 0.6,
            "efficiency": 1.1,
            "oos_max_drawdown": 0.1,
            "oos_aggregate": {
                "sharpe_ratio": 0.6,
                "profit_factor": 1.4,
                "max_drawdown": 0.1,
                "deflated_sharpe_ratio": 0.3,
                "total_trades": 20,
            },
            "gate_report": {},
            "promotable": True,
            "advanced": True,
            "n_trials_cumulative": 1,
            "tokens_used": 10,
        },
        path=journal,
    )
    audit_log(
        "research:trial",
        {
            "session_id": session_id,
            "round": 2,
            "kind": "code",
            "hypothesis": "h2",
            "params": {"threshold": 0.02},
            "is_sharpe": 0.4,
            "oos_sharpe": 0.55,
            "efficiency": 1.0,
            "oos_max_drawdown": 0.12,
            "oos_aggregate": {
                "sharpe_ratio": 0.55,
                "profit_factor": 1.3,
                "max_drawdown": 0.12,
                "deflated_sharpe_ratio": 0.25,
                "total_trades": 45,
            },
            "gate_report": {},
            "promotable": True,
            "advanced": False,
            "n_trials_cumulative": 9,  # this code round searched 8 inner configs (9 - 1)
            "tokens_used": 20,
        },
        path=journal,
    )
    audit_log("research:holdout_score", {"session_id": session_id, "candidate": "c1"}, path=journal)
    audit_log("research:session_end", {"session_id": session_id, "stopped_reason": "x"}, path=journal)

    store = TrialStore(tmp_path / "trials.db")
    stats = store.rebuild(journal)
    assert stats["rows"] == 2  # only the two research:trial lines produce rows

    rows = store.query(strategy="periodic", kind="research", limit=10)
    assert len(rows) == 2
    r1 = next(r for r in rows if r["n_trials_in_session"] == 1)
    r2 = next(r for r in rows if r["n_trials_in_session"] == 8)
    assert r1["oos_sharpe"] == 0.6
    assert r2["oos_sharpe"] == 0.55
    # Universe/window inherited from session_start, normalized.
    assert r1["universe_hash"] == universe_hash(["aaa", "bbb"])
    assert r1["window_start"] == "2024-01-01T00:00:00"
    assert store.family_count("periodic", ["aaa", "bbb"], ACCOUNTING_VERSION) == 1 + 8


# --- the declared schema vs the tables actually on disk -----------------------
def _v4_shaped_store(path, *, version="5"):
    """A database with the pre-quarantine ``trials`` table, stamped however the
    caller asks.

    Built from the DDL the shipped v4 store used, not from today's ``_SCHEMA`` minus
    a column: a fixture that agrees with the code it is testing proves nothing.
    """
    import sqlite3

    path = Path(path)
    for suffix in ("", "-wal", "-shm"):
        leftover = path.with_name(path.name + suffix)
        if leftover.exists():
            leftover.unlink()
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE trials (
          id TEXT PRIMARY KEY, session_id TEXT, ts TEXT, kind TEXT NOT NULL,
          strategy TEXT, universe_hash TEXT NOT NULL, window_start TEXT, window_end TEXT,
          params_hash TEXT NOT NULL, params_json TEXT NOT NULL, is_sharpe REAL,
          oos_sharpe REAL, oos_profit_factor REAL, oos_max_dd REAL, deflated_sharpe REAL,
          efficiency REAL, oos_trades INTEGER, promotable INTEGER,
          n_trials_in_session INTEGER, accounting INTEGER NOT NULL, git_sha TEXT,
          seed INTEGER, metrics_json TEXT
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (version,))
    conn.commit()
    conn.close()


def _one_journalled_trial(journal, fast=5):
    """One journalled trial, returning the run id the journal gave it."""
    return journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast_ema_period": fast},
        metrics={"sharpe_ratio": 1.0, "total_trades": 5},
        path=journal,
    )


def test_a_store_whose_tables_predate_a_column_is_reshaped_not_just_refilled(tmp_path):
    """The failure this exists to stop: ``mark-contaminated`` raised ``no such column:
    contaminated_at`` on every store written before the column existed.

    Bumping ``SCHEMA_VERSION`` triggered a rebuild, but the rebuild emptied the tables
    instead of recreating them, so the file kept its old columns and was stamped with
    the new version - a store that reported v5 and had a v4 ``trials`` table.
    """
    journal = tmp_path / "journal.jsonl"
    trial_id = _one_journalled_trial(journal)
    db = tmp_path / "trials.db"
    _v4_shaped_store(db, version="4")

    with TrialStore(db, journal_path=journal) as store:
        # The symptom first, so a regression names the defect rather than the check:
        # this raised sqlite3.OperationalError: no such column: contaminated_at.
        assert store.mark_contaminated([trial_id], reason="an accounting bump") == 1
        assert store.contaminated_count() == 1
        assert schema_drift(store._conn) == []
        # Replayed from the journal, not lost with the old table.
        assert len(store.query(strategy="demo_trend", limit=10)) == 1


def test_a_current_version_stamp_does_not_excuse_a_stale_table(tmp_path):
    """The nastier half, and the reason the check cannot be ``version != version``.

    A store already reshaped once by the *old* repair path carries a current stamp on
    old columns. Trusting the stamp is what left it broken; the shape has to be looked
    at every open.
    """
    journal = tmp_path / "journal.jsonl"
    trial_id = _one_journalled_trial(journal)
    db = tmp_path / "trials.db"
    _v4_shaped_store(db, version=str(SCHEMA_VERSION))  # stamp agrees, table does not

    with TrialStore(db, journal_path=journal) as store:
        assert store.mark_contaminated([trial_id], reason="an accounting bump") == 1
        assert schema_drift(store._conn) == []


def test_a_conforming_store_is_left_alone(tmp_path):
    """The other direction. A check that rebuilds a healthy store would replay the
    whole journal on every open, and a guard that rejects everything is
    indistinguishable from one that works."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _one_journalled_trial(journal)
    db = tmp_path / "trials.db"
    with TrialStore(db, journal_path=journal) as store:
        store.rebuild(journal)
        assert store._get_meta("last_rebuild_journal_lines") == "1"
        # Rows written straight to the index, recoverable from nothing. They survive
        # only if opening the store does not rebuild it.
        store.record_returns(trial_id, ["2024-01-02"], [0.01])

    with TrialStore(db, journal_path=journal) as reopened:
        assert schema_drift(reopened._conn) == []
        assert reopened._returns_summary(trial_id) is not None


def test_a_rebuild_refuses_to_replace_real_rows_with_an_empty_index(tmp_path):
    """A store is derived, but derived from a file that has to be there. With the
    journal gone, replaying it would report a campaign of zero trials - and the
    deflation bar that count feeds would drop to nothing with no error anywhere."""
    journal = tmp_path / "journal.jsonl"
    _one_journalled_trial(journal)
    db = tmp_path / "trials.db"
    with TrialStore(db, journal_path=journal) as store:
        store.rebuild(journal)
        assert len(store.query(strategy="demo_trend", limit=10)) == 1

    journal.unlink()
    with TrialStore(db, journal_path=journal) as store:
        with pytest.raises(TrialStoreRebuildRefused):
            store.rebuild(journal)
        assert len(store.query(strategy="demo_trend", limit=10)) == 1


def test_a_missing_journal_still_rebuilds_an_index_with_nothing_to_lose(tmp_path):
    """The boundary the refusal must not swallow: an empty index and no journal is a
    fresh store, not a destroyed campaign."""
    store = TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "absent.jsonl")
    assert store.rebuild() == {"rows": 0, "journal_lines": 0}
    store.close()


def test_a_mis_shaped_store_whose_journal_is_gone_says_so_instead_of_emptying_itself(tmp_path):
    """Repair-on-open must inherit the refusal, not route around it — and the store
    that could not be repaired has to say which columns it is missing, because every
    later `no such column` traceback traces back to here."""
    import sqlite3

    db = tmp_path / "trials.db"
    journal = tmp_path / "absent.jsonl"  # never written
    _v4_shaped_store(db, version="4")
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO trials (id, kind, strategy, universe_hash, params_hash, params_json, accounting) "
        "VALUES ('t1', 'backtest', 'demo_trend', 'uh', 'ph', '{}', ?)",
        (ACCOUNTING_VERSION,),
    )
    conn.commit()
    conn.close()

    with TrialStore(db, journal_path=journal) as store:
        assert len(store.query(strategy="demo_trend", limit=10)) == 1
        info = store.status(journal)
        assert info["schema_drift"], "a store left mis-shaped must say so"
        assert info["drift"] is True
        assert info["contaminated_rows"] is None  # unknown, never zero


def test_an_interrupted_rebuild_is_repaired_on_the_next_open(tmp_path):
    """A replay that dies partway leaves a short index under a fresh version stamp,
    which reads as a complete campaign that happens to be small."""
    journal = tmp_path / "journal.jsonl"
    _one_journalled_trial(journal, fast=5)
    second = _one_journalled_trial(journal, fast=9)
    db = tmp_path / "trials.db"
    with TrialStore(db, journal_path=journal) as store:
        store.rebuild(journal)
        store._conn.execute("DELETE FROM trials WHERE id = ?", (second,))
        store._set_meta(_REBUILD_SENTINEL, "1")
        store._conn.commit()

    with TrialStore(db, journal_path=journal) as store:
        assert len(store.query(strategy="demo_trend", limit=10)) == 2
        assert store._get_meta(_REBUILD_SENTINEL) is None


def test_a_store_whose_journal_has_gone_missing_does_not_report_itself_healthy(tmp_path):
    """`drift` compared rows against journal *lines*, and a journal that is not there
    has none — so `3 rows < 0 lines` was False and the check said OK about the one
    state it most needed to name. These rows cannot be rebuilt from anything."""
    journal = tmp_path / "journal.jsonl"
    _one_journalled_trial(journal)
    db = tmp_path / "trials.db"
    with TrialStore(db, journal_path=journal) as store:
        store.rebuild(journal)
        assert store.status(journal)["drift"] is False  # the boundary: healthy is healthy
        assert store.status(journal)["journal_readable"] is True

    journal.unlink()
    with TrialStore(db, journal_path=journal) as store:
        info = store.status(journal)
        assert info["journal_readable"] is False
        assert info["drift"] is True


def test_an_empty_index_with_no_journal_is_a_fresh_store_not_a_broken_one(tmp_path):
    """The other direction again: nothing indexed and nothing journalled is where
    everyone starts, and flagging it would make the check noise."""
    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "absent.jsonl") as store:
        assert store.status()["drift"] is False
