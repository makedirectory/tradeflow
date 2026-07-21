"""Service-layer + MCP-server tests.

All offline against FakeMarketData. Covers: discovery, the analysis services
(JSON-serializable, trades not inlined, top-N capped), the metric glossary, the
audit log, and - critically - the safety wall (no live/order tool is exposed and
the server refuses a non-data client).
"""

import json
from datetime import datetime

import pytest

from src.analytics.performance import FLAG_KEYS, METRIC_KEYS
from src.marketdata.client import MarketDataClient
from src.services import analysis, glossary, registry
from src.services.audit import audit_log
from tests.fakes import FakeMarketData

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2025, 6, 1)


def _client():
    return MarketDataClient(FakeMarketData(SYMBOLS, n=600, freq="1D"))


@pytest.fixture(autouse=True)
def _artifacts_in_tmp(tmp_path, monkeypatch):
    """Keep test artifacts out of the repo working tree."""
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")


# --- discovery --------------------------------------------------------------
def test_discovery_lists_and_param_ranges():
    assert any(s["name"] == "volume_spike" for s in registry.list_strategies())
    assert any(s["name"] == "volume" for s in registry.list_scanners())
    pr = registry.get_param_ranges("strategy", "volume_spike")
    assert "short_ema_period" in pr["param_ranges"]
    with pytest.raises(ValueError):
        registry.get_param_ranges("strategy", "nope")
    with pytest.raises(ValueError):
        registry.get_param_ranges("bogus", "volume_spike")


# --- analysis services ------------------------------------------------------
def test_run_backtest_is_json_and_does_not_inline_trades():
    result = analysis.run_backtest(_client(), "volume_spike", SYMBOLS, START, END)
    assert json.loads(json.dumps(result))  # JSON round-trips
    assert "trades" not in result  # trades go to a CSV path, never inlined
    assert set(METRIC_KEYS) <= set(result["metrics"])
    assert "run_id" in result


def test_run_optimization_caps_top_n_and_writes_csv():
    result = analysis.run_optimization(
        _client(), "volume_spike", SYMBOLS, START, END, method="random", max_evals=15, seed=1
    )
    assert len(result["top"]) <= analysis.TOP_N
    assert result["n_trials"] == 15
    assert result["truncated"] == max(15 - len(result["top"]), 0)
    assert "IN-SAMPLE" in result["note"]
    assert json.loads(json.dumps(result))


def test_run_walk_forward_returns_gate_verdict():
    result = analysis.run_walk_forward(
        _client(),
        "volume_spike",
        SYMBOLS,
        START,
        END,
        n_folds=3,
        embargo_days=2,
        holdout_days=30,
        method="grid",
        max_evals=8,
    )
    assert "promotable" in result["gate_report"]
    assert result["folds"] and result["n_trials_total"] > 0
    assert "oos_aggregate" in result and "median_oos_sharpe" in result
    assert json.loads(json.dumps(result))


def test_summarize_bars_is_descriptive():
    result = analysis.summarize_bars(_client(), SYMBOLS, "1Day", 90)
    assert result["symbols"]["AAA"]["available"] is True
    assert "annualized_vol_pct" in result["symbols"]["AAA"]
    assert json.loads(json.dumps(result))


# --- glossary ---------------------------------------------------------------
def test_glossary_covers_every_metric():
    g = glossary.metrics_glossary()
    for key in METRIC_KEYS:
        assert key in g["metrics"], f"glossary missing {key}"
    for key in FLAG_KEYS:
        assert key in g["flags"]
    assert g["global_caveats"]  # the over-trust warnings are present


# --- audit ------------------------------------------------------------------
def test_audit_log_appends_one_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    run_id = audit_log(
        "run_backtest", {"strategy": "volume_spike"}, path=path, result_summary={"total_trades": 3}
    )
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "run_backtest"
    assert record["run_id"] == run_id
    assert "timestamp" in record and "git_sha" in record


def test_journal_trial_records_a_normalized_trial(tmp_path):
    from src.engine.backtest import ACCOUNTING_VERSION
    from src.services.audit import journal_trial

    path = tmp_path / "journal.jsonl"
    journal_trial(
        "backtest",
        strategy="ma_crossover",
        symbols=["msft", "AAPL", "aapl"],  # duplicate + mixed case
        start=datetime(2024, 1, 2),
        end=datetime(2024, 3, 1),
        params={"fast_ema_period": 10},
        metrics={"sharpe_ratio": 1.2, "total_trades": 8, "not_a_headline": 999},
        path=path,
    )
    record = json.loads(path.read_text().splitlines()[0])

    assert record["tool"] == "trial:backtest"
    assert record["kind"] == "backtest"
    assert record["accounting"] == ACCOUNTING_VERSION
    # Universe is normalized so the same set keys identically however it was typed.
    assert record["inputs"]["symbols"] == ["AAPL", "MSFT"]
    assert record["inputs"]["window"] == {"start": "2024-01-02T00:00:00", "end": "2024-03-01T00:00:00"}
    assert record["resolved_config"] == {"fast_ema_period": 10}
    # Only headline metrics are denormalized onto the row.
    assert record["result_summary"] == {"sharpe_ratio": 1.2, "total_trades": 8}


def test_journal_trial_records_the_objective_when_given(tmp_path):
    from src.services.audit import journal_trial

    path = tmp_path / "journal.jsonl"
    journal_trial(
        "optimize",
        strategy="ma_crossover",
        symbols=["NVDA"],
        start=datetime(2024, 1, 2),
        end=datetime(2024, 3, 1),
        params={"fast_ema_period": 10},
        metrics={"sharpe_ratio": 2.0},
        objective="sharpe_ratio",
        path=path,
    )
    record = json.loads(path.read_text().splitlines()[0])
    assert record["inputs"]["objective"] == "sharpe_ratio"


# --- the safety wall -----------------------------------------
def test_no_live_or_order_tool_is_exposed():
    from src.mcp import server

    assert set(server.EXPOSED_TOOLS).isdisjoint(server.FORBIDDEN_TOOLS)
    forbidden_substrings = ("order", "live", "cancel", "close_position", "paper_trade", "account")
    for tool in server.EXPOSED_TOOLS:
        assert not any(bad in tool for bad in forbidden_substrings), f"unsafe tool exposed: {tool}"


def test_server_refuses_non_data_client():
    from src.mcp import server

    class FakeBrokered:
        broker = object()  # smells like a trading-capable client

    with pytest.raises(RuntimeError):
        server._assert_no_trading_client(FakeBrokered())


def test_build_server_smoke():
    pytest.importorskip("mcp")
    from src.mcp import server

    built = server.build_server(data_client=_client())
    assert built is not None


# --- CLI trial journaling (spec 026 precondition) ---------------------------
def _fake_cli_env(monkeypatch, tmp_path, symbols):
    """Point the CLI at fake data and a temp journal; return the journal path."""
    import main
    from src.marketdata.client import MarketDataClient
    from src.services import audit
    from tests.fakes import FakeMarketData

    client = MarketDataClient(FakeMarketData(symbols, n=400, freq="1D"))
    monkeypatch.setattr(main, "build_data_and_broker", lambda: (None, client))
    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal)
    return journal


def _trials(journal):
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text().splitlines()]


def test_cli_backtest_journals_one_trial(monkeypatch, tmp_path):
    import main

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args = main.build_parser().parse_args(
        [
            "backtest",
            "--strategy",
            "ma_crossover",
            "--scanner",
            "none",
            "--symbols",
            "AAA,BBB",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-01",
        ]
    )
    args.func(args)

    trials = _trials(journal)
    assert len(trials) == 1
    assert trials[0]["tool"] == "trial:backtest"
    assert trials[0]["inputs"]["symbols"] == ["AAA", "BBB"]


def test_cli_optimize_journals_one_trial_per_config(monkeypatch, tmp_path):
    import main

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args = main.build_parser().parse_args(
        [
            "optimize",
            "--strategy",
            "ma_crossover",
            "--scanner",
            "none",
            "--symbols",
            "AAA,BBB",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-01",
            "--method",
            "grid",
            "--max-evals",
            "6",
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )
    args.func(args)

    trials = _trials(journal)
    # A search of N configs is N trials — the property the Deflated Sharpe count needs.
    assert 1 < len(trials) <= 6
    assert all(t["tool"] == "trial:optimize" for t in trials)
    assert all(t["inputs"]["objective"] == "sharpe_ratio" for t in trials)


def test_cli_no_journal_flag_records_nothing(monkeypatch, tmp_path):
    import main

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args = main.build_parser().parse_args(
        [
            "backtest",
            "--strategy",
            "ma_crossover",
            "--scanner",
            "none",
            "--symbols",
            "AAA,BBB",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-01",
            "--no-journal",
        ]
    )
    args.func(args)
    assert _trials(journal) == []


def test_cli_walkforward_journals_one_validated_trial(monkeypatch, tmp_path):
    import main

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args = main.build_parser().parse_args(
        [
            "walkforward",
            "--strategy",
            "ma_crossover",
            "--scanner",
            "none",
            "--symbols",
            "AAA,BBB",
            "--start",
            "2024-01-02",
            "--end",
            "2025-06-01",
            "--folds",
            "3",
            "--holdout-days",
            "30",
            "--method",
            "grid",
            "--max-evals",
            "6",
        ]
    )
    args.func(args)

    trials = _trials(journal)
    # One walk-forward is one validated config, not one row per inner search config.
    assert len(trials) == 1
    rec = trials[0]
    assert rec["tool"] == "trial:walkforward"
    # The internal search count rides along so a campaign can sum real configs tried.
    assert rec["n_trials"] > 1
    assert "promotable" in rec


def test_cli_alphas_journals_a_readonly_trial(monkeypatch, tmp_path):
    import main

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB", "CCC"])
    args = main.build_parser().parse_args(
        [
            "alphas",
            "--strategy",
            "ma_crossover",
            "--scanner",
            "none",
            "--symbols",
            "AAA,BBB,CCC",
            "--as-of",
            "2025-03-01",
            "--ic",
            "0.05",
        ]
    )
    args.func(args)

    trials = _trials(journal)
    assert len(trials) == 1
    rec = trials[0]
    assert rec["tool"] == "trial:alpha"
    # A forecast has no Sharpe — the row exists for dedup/IC, not the DSR count.
    assert rec["result_summary"] == {}
    # The window collapses to the single as-of date.
    assert rec["inputs"]["window"]["start"] == rec["inputs"]["window"]["end"]
    assert rec["resolved_config"]["ic"] == 0.05


# --- trial store dual-write (spec 026) ---------------------------------------
def test_cli_backtest_also_populates_the_trial_store(monkeypatch, tmp_path):
    """Every journaled trial is dual-written into the sibling trial store, so a
    campaign's n_trials can be counted from an index instead of replaying JSONL
    by hand on every query."""
    import main
    from src.engine.backtest import ACCOUNTING_VERSION
    from src.store.trials import TrialStore, db_path_for_journal

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args = main.build_parser().parse_args(
        [
            "backtest",
            "--strategy",
            "ma_crossover",
            "--scanner",
            "none",
            "--symbols",
            "AAA,BBB",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-01",
        ]
    )
    args.func(args)

    store = TrialStore(db_path_for_journal(journal))
    rows = store.query(strategy="ma_crossover", kind="backtest")
    assert len(rows) == 1
    assert rows[0]["accounting"] == ACCOUNTING_VERSION
    assert store.family_count("ma_crossover", ["AAA", "BBB"], ACCOUNTING_VERSION) == 1
    status = store.status(journal)
    assert status["drift"] is False


def test_walkforward_gate_report_is_unaffected_by_a_broken_trial_store(monkeypatch, tmp_path, capsys):
    """v1 is passive (spec 026 §2): the store only ever observes, so a walk-forward
    run with a working trial store and one where the store is entirely broken must
    produce the identical printed gate report - the dual-write is one-way and can
    never feed back into a verdict."""
    import main
    from src.marketdata.client import MarketDataClient
    from src.services import audit
    from src.store import trials as trials_mod
    from tests.fakes import FakeMarketData

    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=400, freq="1D"))
    monkeypatch.setattr(main, "build_data_and_broker", lambda: (None, client))

    def _run(journal_path):
        monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal_path)
        args = main.build_parser().parse_args(
            [
                "walkforward",
                "--strategy",
                "ma_crossover",
                "--scanner",
                "none",
                "--symbols",
                "AAA,BBB",
                "--start",
                "2024-01-02",
                "--end",
                "2025-06-01",
                "--folds",
                "3",
                "--holdout-days",
                "30",
                "--method",
                "grid",
                "--max-evals",
                "6",
                "--seed",
                "42",
            ]
        )
        args.func(args)
        return capsys.readouterr().out

    working = _run(tmp_path / "a" / "journal.jsonl")

    monkeypatch.setattr(trials_mod, "TrialStore", lambda *a, **k: (_ for _ in ()).throw(OSError("no store")))
    broken = _run(tmp_path / "b" / "journal.jsonl")

    assert working == broken
