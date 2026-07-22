"""The trial store: the queryable SQLite index over the research
journal. Exercises rebuild idempotence, crash recovery/drift detection, hash
normalization, dedup, and campaign-level family counting - the property that is
broken today (``n_trials`` resets every session).
"""

import json
from datetime import datetime

from src.engine.backtest import ACCOUNTING_VERSION
from src.services.audit import audit_log, journal_trial
from src.store.trials import TrialStore, db_path_for_journal, params_hash, universe_hash


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
            strategy="ma_crossover",
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
    rows = store.query(strategy="ma_crossover", limit=10)
    ids_first = {r["id"] for r in rows}
    assert len(ids_first) == 3

    stats2 = store.rebuild(journal)
    assert stats2 == stats
    rows2 = store.query(strategy="ma_crossover", limit=10)
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
    # A different accounting version is not the same trial (025's problem, again).
    assert not store.seen(
        strategy="s",
        params={"buy_every": 3},
        symbols=["A", "B"],
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        accounting=1,
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
