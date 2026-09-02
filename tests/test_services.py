"""Service-layer + MCP-server tests.

All offline against FakeMarketData. Covers: discovery, the analysis services
(JSON-serializable, trades not inlined, top-N capped), the metric glossary, the
audit log, and - critically - the safety wall (no live/order tool is exposed and
the server refuses a non-data client).
"""

import contextlib
import json
import re
from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tests.test_research import _VALID_CODE, _VALID_SCANNER_CODE
from tradeflow.analytics.performance import FLAG_KEYS, METRIC_KEYS
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services import analysis, audit, glossary, registry
from tradeflow.services.audit import audit_log

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2025, 6, 1)


def _client():
    return MarketDataClient(FakeMarketData(SYMBOLS, n=600, freq="1D"))


@pytest.fixture(autouse=True)
def _artifacts_in_tmp(tmp_path, monkeypatch):
    """Keep test artifacts, the research journal, and the trial store it dual-
    writes to out of the repo working tree - `analysis.run_backtest` et al. now
    journal every run, so without this a test run pollutes (and, worse, can
    memoize against) the real `logs/research_journal.jsonl`."""
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")


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
    from tradeflow.engine.backtest import ACCOUNTING_VERSION
    from tradeflow.services.audit import journal_trial

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
    from tradeflow.services.audit import journal_trial

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
    from tradeflow.mcp import server

    assert set(server.EXPOSED_TOOLS).isdisjoint(server.FORBIDDEN_TOOLS)
    forbidden_substrings = ("order", "live", "cancel", "close_position", "paper_trade", "account")
    for tool in server.EXPOSED_TOOLS:
        assert not any(bad in tool for bad in forbidden_substrings), f"unsafe tool exposed: {tool}"


def test_server_refuses_non_data_client():
    from tradeflow.mcp import server

    class FakeBrokered:
        broker = object()  # smells like a trading-capable client

    with pytest.raises(RuntimeError):
        server._assert_no_trading_client(FakeBrokered())


def test_build_server_smoke():
    pytest.importorskip("mcp")
    from tradeflow.mcp import server

    built = server.build_server(data_client=_client())
    assert built is not None


# --- CLI trial journaling -------------------------------------------------------
def _fake_cli_env(monkeypatch, tmp_path, symbols):
    """Point the CLI at fake data and a temp journal; return the journal path."""
    from tests.fakes import FakeMarketData
    from tradeflow import cli as main
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import audit

    client = MarketDataClient(FakeMarketData(symbols, n=400, freq="1D"))
    monkeypatch.setattr(main, "build_data_and_broker", lambda **kwargs: (None, client))
    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal)
    return journal


def _trials(journal):
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text().splitlines()]


def test_cli_backtest_journals_one_trial(monkeypatch, tmp_path):
    from tradeflow import cli as main

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
    from tradeflow import cli as main

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
    from tradeflow import cli as main

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
    from tradeflow import cli as main

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
    from tradeflow import cli as main

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


# --- trial store dual-write -----------------------------------------------------
def test_cli_backtest_also_populates_the_trial_store(monkeypatch, tmp_path):
    """Every journaled trial is dual-written into the sibling trial store, so a
    campaign's n_trials can be counted from an index instead of replaying JSONL
    by hand on every query."""
    from tradeflow import cli as main
    from tradeflow.engine.backtest import ACCOUNTING_VERSION
    from tradeflow.store.trials import TrialStore, db_path_for_journal

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
    """v1 is passive: the store only ever observes, so a walk-forward
    run with a working trial store and one where the store is entirely broken must
    produce the identical printed gate report - the dual-write is one-way and can
    never feed back into a verdict."""
    from tests.fakes import FakeMarketData
    from tradeflow import cli as main
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import audit
    from tradeflow.store import trials as trials_mod

    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=400, freq="1D"))
    monkeypatch.setattr(main, "build_data_and_broker", lambda **kwargs: (None, client))

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


# --- memoization + --config (CLI) --------------------------------------
_BT_ARGV = [
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


def test_cli_backtest_memoizes_identical_run_without_resimulating(monkeypatch, tmp_path, capsys):
    from tradeflow import cli as main
    from tradeflow.engine.backtest import BacktestEngine

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args1 = main.build_parser().parse_args(_BT_ARGV)
    args1.func(args1)
    assert len(_trials(journal)) == 1

    calls = {"n": 0}
    original_run = BacktestEngine.run

    def counting_run(self, *a, **k):
        calls["n"] += 1
        return original_run(self, *a, **k)

    monkeypatch.setattr(BacktestEngine, "run", counting_run)

    args2 = main.build_parser().parse_args(_BT_ARGV)
    args2.func(args2)
    out = capsys.readouterr().out

    assert calls["n"] == 0  # served from the trial store, never re-simulated
    assert "REUSED" in out
    assert re.search(r"from \d{4}-\d{2}-\d{2}T", out)  # the original run's timestamp, unmistakably
    assert len(_trials(journal)) == 1  # the re-display did not double-count


def test_cli_backtest_commission_bps_change_is_a_distinct_trial(monkeypatch, tmp_path):
    """Regression test: two runs differing only in a cost flag must not collide
    as 'the same trial'."""
    from tradeflow import cli as main

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args1 = main.build_parser().parse_args(_BT_ARGV + ["--commission-bps", "1.0"])
    args1.func(args1)
    args2 = main.build_parser().parse_args(_BT_ARGV + ["--commission-bps", "5.0"])
    args2.func(args2)
    assert len(_trials(journal)) == 2


def test_cli_backtest_force_reruns_and_appends_rather_than_overwrites(monkeypatch, tmp_path):
    from tradeflow import cli as main

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args1 = main.build_parser().parse_args(_BT_ARGV)
    args1.func(args1)
    args2 = main.build_parser().parse_args(_BT_ARGV + ["--force"])
    args2.func(args2)

    trials = _trials(journal)
    assert len(trials) == 2
    assert trials[0]["run_id"] != trials[1]["run_id"]


# --- bar cache (--cache/--offline) + data-vintage dedup -----------------------
pytest.importorskip("pyarrow")


def _fake_cli_env_with_cache(monkeypatch, tmp_path, symbols, cache_dir=None):
    """Like _fake_cli_env, but the injected data client is cache-backed
    (CachedMarketData wrapping FakeMarketData), so --cache/vintage wiring can be
    exercised without needing real Alpaca settings."""
    from tradeflow import cli as main
    from tradeflow.services import audit
    from tradeflow.store.bars import CachedMarketData

    cache_dir = cache_dir or (tmp_path / "cache" / "bars")
    provider = CachedMarketData(FakeMarketData(symbols, n=400, freq="1D"), cache_dir=cache_dir)
    client = MarketDataClient(provider)
    monkeypatch.setattr(main, "build_data_and_broker", lambda **kwargs: (None, client))
    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal)
    return journal, provider


def test_cli_backtest_with_cache_records_vintage_and_reuse_is_vintage_safe(monkeypatch, tmp_path, capsys):
    from tradeflow import cli as main

    journal, _ = _fake_cli_env_with_cache(monkeypatch, tmp_path, ["AAA", "BBB"])
    args1 = main.build_parser().parse_args(_BT_ARGV + ["--cache"])
    args1.func(args1)
    trials = _trials(journal)
    assert len(trials) == 1
    assert trials[0]["resolved_config"]["_cost"]["_vintage"]  # present and non-empty

    args2 = main.build_parser().parse_args(_BT_ARGV + ["--cache"])
    args2.func(args2)
    out = capsys.readouterr().out
    assert "REUSED" in out
    assert "data vintage confirmed" in out
    assert "no data-vintage stamp" not in out
    assert len(_trials(journal)) == 1  # still memoized, not double-counted


def test_cli_backtest_refresh_between_cache_runs_is_a_new_trial(monkeypatch, tmp_path):
    from tradeflow import cli as main

    journal, provider = _fake_cli_env_with_cache(monkeypatch, tmp_path, ["AAA", "BBB"])
    args1 = main.build_parser().parse_args(_BT_ARGV + ["--cache"])
    args1.func(args1)
    assert len(_trials(journal)) == 1

    provider.refresh(["AAA", "BBB"], "1Day", datetime(2024, 1, 2), datetime(2024, 6, 1))

    args2 = main.build_parser().parse_args(_BT_ARGV + ["--cache"])
    args2.func(args2)
    trials = _trials(journal)
    assert len(trials) == 2  # the underlying data vintage moved - not the same trial


def test_cli_backtest_without_cache_keeps_the_no_vintage_caveat(monkeypatch, tmp_path, capsys):
    from tradeflow import cli as main

    _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    args1 = main.build_parser().parse_args(_BT_ARGV)
    args1.func(args1)
    args2 = main.build_parser().parse_args(_BT_ARGV)
    args2.func(args2)
    out = capsys.readouterr().out
    assert "REUSED" in out
    assert "no data-vintage stamp" in out


def test_build_data_client_cache_flag_wraps_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("APCA_API_KEY_ID", "fake")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fake")
    from tradeflow.services.data import build_data_client
    from tradeflow.store.bars import CachedMarketData

    cached = build_data_client(cache=True, cache_dir=tmp_path / "bars")
    assert isinstance(cached.provider, CachedMarketData)

    plain = build_data_client()
    assert not isinstance(plain.provider, CachedMarketData)


def test_build_data_and_broker_forwards_cache_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("APCA_API_KEY_ID", "fake")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fake")
    from tradeflow import cli as main
    from tradeflow.store.bars import CachedMarketData

    _, client = main.build_data_and_broker(cache=True, cache_dir=tmp_path / "bars")
    assert isinstance(client.provider, CachedMarketData)

    _, plain_client = main.build_data_and_broker()
    assert not isinstance(plain_client.provider, CachedMarketData)


# --- `cache` CLI subcommand ------------------------------------------------
def test_cache_status_on_empty_dir_prints_a_sane_empty_state(tmp_path, capsys):
    from tradeflow import cli as main

    args = main.build_parser().parse_args(["cache", "status", "--cache-dir", str(tmp_path / "bars")])
    args.func(args)
    out = capsys.readouterr().out
    assert "No cached symbols." in out
    assert "OK — no drift detected." in out


def _mock_cached_build_data_client(monkeypatch, cache_dir, symbols=("AAA",)):
    from tradeflow.store.bars import CachedMarketData

    def _build(cache=False, offline=False, cache_dir=None):
        return MarketDataClient(
            CachedMarketData(FakeMarketData(list(symbols), n=200, freq="1D"), cache_dir=cache_dir)
        )

    monkeypatch.setattr("tradeflow.services.data.build_data_client", _build)


def test_cache_warm_then_status_round_trips(monkeypatch, tmp_path, capsys):
    from tradeflow import cli as main

    cache_dir = tmp_path / "bars"
    _mock_cached_build_data_client(monkeypatch, cache_dir)

    warm_args = main.build_parser().parse_args(
        [
            "cache",
            "warm",
            "--symbols",
            "AAA",
            "--cache-dir",
            str(cache_dir),
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-01",
        ]
    )
    warm_args.func(warm_args)
    assert "AAA" in capsys.readouterr().out

    status_args = main.build_parser().parse_args(["cache", "status", "--cache-dir", str(cache_dir)])
    status_args.func(status_args)
    out = capsys.readouterr().out
    assert "AAA" in out
    assert "OK — no drift detected." in out


def test_cache_refresh_reports_per_symbol(monkeypatch, tmp_path, capsys):
    from tradeflow import cli as main

    cache_dir = tmp_path / "bars"
    _mock_cached_build_data_client(monkeypatch, cache_dir)

    refresh_args = main.build_parser().parse_args(
        [
            "cache",
            "refresh",
            "--symbols",
            "AAA",
            "--cache-dir",
            str(cache_dir),
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-01",
        ]
    )
    refresh_args.func(refresh_args)
    assert "refreshed" in capsys.readouterr().out


def test_walkforward_memoizes_identical_recipe_without_rerunning(monkeypatch, tmp_path, capsys):
    from tradeflow import cli as main
    from tradeflow.optimization.walk_forward import WalkForwardValidator

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    wf_argv = [
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
    args1 = main.build_parser().parse_args(wf_argv)
    args1.func(args1)
    assert len(_trials(journal)) == 1

    calls = {"n": 0}
    original_run = WalkForwardValidator.run

    def counting_run(self, *a, **k):
        calls["n"] += 1
        return original_run(self, *a, **k)

    monkeypatch.setattr(WalkForwardValidator, "run", counting_run)

    args2 = main.build_parser().parse_args(wf_argv)
    args2.func(args2)
    out = capsys.readouterr().out

    assert calls["n"] == 0
    assert "REUSED" in out
    assert len(_trials(journal)) == 1


def test_walkforward_save_config_then_backtest_config_round_trips(monkeypatch, tmp_path):
    from tradeflow import cli as main

    journal = _fake_cli_env(monkeypatch, tmp_path, ["AAA", "BBB"])
    config_path = tmp_path / "cfg.json"
    wf_args = main.build_parser().parse_args(
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
            "--save-config",
            str(config_path),
        ]
    )
    wf_args.func(wf_args)
    assert config_path.exists()
    saved = json.loads(config_path.read_text())

    bt_args = main.build_parser().parse_args(
        [
            "backtest",
            "--config",
            str(config_path),
            "--scanner",
            "none",
            "--symbols",
            "AAA,BBB",
            "--start",
            "2024-06-01",
            "--end",
            "2024-12-01",
        ]
    )
    bt_args.func(bt_args)

    backtest_trials = [t for t in _trials(journal) if t["tool"] == "trial:backtest"]
    assert len(backtest_trials) == 1
    # Underscore-prefixed keys are the reserved assumption folds (`_cost`, `_limits`)
    # that make up a run's dedup identity alongside its params. They are deliberately
    # part of the recorded config and deliberately not part of the params round-trip.
    resolved = {k: v for k, v in backtest_trials[0]["resolved_config"].items() if not k.startswith("_")}
    assert resolved == saved["params"]


def test_backtest_config_out_of_range_param_fails_loudly(tmp_path):
    from tradeflow import cli as main
    from tradeflow.services.registry import resolve_strategy_class

    cls = resolve_strategy_class("ma_crossover")
    params = {name: spec["default"] for name, spec in cls.PARAM_RANGES.items()}
    params["fast_ema_period"] = 9999  # out of PARAM_RANGES bounds
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"strategy": "ma_crossover", "scanner": None, "params": params}))

    args = main.build_parser().parse_args(
        [
            "backtest",
            "--config",
            str(config_path),
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
    with pytest.raises(ValueError):
        args.func(args)


# --- MCP parity ---------------------------------------------------------
def test_analysis_run_backtest_journals_a_trial_and_memoizes_a_repeat():
    """Regression test for a gap found while wiring memoization: MCP's
    analysis.run_backtest never journaled a trial at all, so nothing an agent ran
    over MCP ever counted toward the campaign's multiple-testing total."""
    result1 = analysis.run_backtest(_client(), "volume_spike", SYMBOLS, START, END)
    assert not result1.get("memoized")
    trials = _trials(audit.DEFAULT_TRIAL_JOURNAL)
    assert len(trials) == 1
    assert trials[0]["tool"] == "trial:backtest"

    result2 = analysis.run_backtest(_client(), "volume_spike", SYMBOLS, START, END)
    assert result2["memoized"] is True
    assert result2["trial_id"] == trials[0]["run_id"]
    assert len(_trials(audit.DEFAULT_TRIAL_JOURNAL)) == 1  # not double-counted

    result3 = analysis.run_backtest(_client(), "volume_spike", SYMBOLS, START, END, force=True)
    assert not result3.get("memoized")
    assert len(_trials(audit.DEFAULT_TRIAL_JOURNAL)) == 2


# --- draft walk-forward journaling ------------------------------------------
@contextlib.contextmanager
def _store_that_will_not_open():
    """What ``_open_trial_store`` yields when the memo DB cannot be opened."""
    yield None


def test_a_draft_run_is_journaled_even_when_the_trial_store_will_not_open(monkeypatch, tmp_path):
    """The store is a derived memo cache; the journal is the campaign's own record.

    Journaling used to be gated on the store, so a store that failed to open spent an
    out-of-sample test without recording it - and the trial count the deflated Sharpe
    rests on then under-counts, silently, which is the one number that must not.
    """
    monkeypatch.setattr(analysis, "_open_trial_store", _store_that_will_not_open)

    payload = analysis.run_draft_walk_forward(
        _client(), _VALID_CODE, SYMBOLS, START, END, n_folds=2, method="grid", max_evals=1
    )

    assert payload["strategy"].startswith("draft:GenStrat:")
    assert payload["draft"]["journaled"] is True
    assert "not_journaled_reason" not in payload["draft"]
    recorded = [
        t for t in _trials(tmp_path / "journal.jsonl") if t["inputs"]["strategy"] == payload["strategy"]
    ]
    assert len(recorded) == 1


# --- draft entry points answer, never raise ---------------------------------
_MALFORMED_STRATEGY = _VALID_CODE.replace(
    '{"type": "float", "min": 0.0, "max": 0.05, "step": 0.01, "default": 0.01}', "0.01"
)
_MALFORMED_SCANNER = _VALID_SCANNER_CODE.replace(
    '{"type": "float", "min": 0.0, "max": 5.0, "step": 0.5, "default": 0.5}', "0.5"
)


def test_draft_validators_return_a_verdict_for_a_malformed_param_spec():
    """A bare spec value used to raise `TypeError: argument of type 'int' is not
    iterable` out of tools whose only job is to say whether the code is usable."""
    strategy = analysis.validate_draft_strategy_code(_MALFORMED_STRATEGY)
    scanner = analysis.validate_draft_scanner_code(_MALFORMED_SCANNER)

    for verdict, param in ((strategy, "threshold"), (scanner, "min_move")):
        assert verdict["valid"] is False
        assert verdict["error_kind"] == "invalid_draft"
        assert param in verdict["error"]  # names what to fix, not just that it failed


def test_a_draft_walk_forward_reports_a_rejection_instead_of_raising(tmp_path):
    """This entry point guarded nothing at all, so even the anticipated rejection -
    source that simply fails hygiene - came back as an exception rather than an
    answer, from the one of the three draft tools that costs a trial to call.
    """
    payload = analysis.run_draft_walk_forward(
        _client(), _MALFORMED_STRATEGY, SYMBOLS, START, END, n_folds=2, max_evals=1
    )

    assert payload["valid"] is False
    assert payload["error_kind"] == "invalid_draft"
    assert "threshold" in payload["error"]
    # Nothing ran, so nothing may have been charged against the campaign's budget.
    assert _trials(tmp_path / "journal.jsonl") == []


def test_a_validator_that_breaks_does_not_blame_the_draft(monkeypatch):
    """The two failures need opposite responses, so they cannot share one label.

    Reporting a defect in the validator as `invalid_draft` sends an agent rewriting
    source that was never the problem - and it would keep rewriting it.
    """
    from tradeflow.research import sandbox

    def _explode(*args, **kwargs):
        raise RuntimeError("validator fell over")

    monkeypatch.setattr(sandbox, "load_strategy_from_code", _explode)
    verdict = analysis.validate_draft_strategy_code(_VALID_CODE)

    assert verdict["valid"] is False
    assert verdict["error_kind"] == "validator_error"
    assert "validator fell over" in verdict["error"]


def test_valid_draft_source_is_still_accepted():
    """The other direction: the guards must not reject everything."""
    assert analysis.validate_draft_strategy_code(_VALID_CODE)["valid"] is True
    assert analysis.validate_draft_scanner_code(_VALID_SCANNER_CODE)["valid"] is True


# --- cost stress --------------------------------------------------------------
def _stress(strategy, n_symbols, **kwargs):
    symbols = [f"S{i}" for i in range(n_symbols)]
    client = MarketDataClient(FakeMarketData([*symbols, "SPY"], n=500, freq="1D"))
    return analysis.run_cost_stress(
        client, strategy, symbols, datetime(2024, 1, 2), datetime(2025, 3, 1), capital=100_000.0, **kwargs
    )


def test_the_stress_curve_separates_robust_from_barely_profitable():
    """The whole point: both of these are "profitable at 1bp" and are not the same.

    One survives five times its assumed cost. The other clears by a hair at its own
    assumptions and is negative at twice them - which a single cost assumption reports
    as a pass, with no way to tell how much of the result was the assumption.
    """
    robust = _stress("ma_crossover", 6)
    fragile = _stress("ma_crossover", 12)

    assert robust["survives_to_multiple"] > fragile["survives_to_multiple"]
    assert fragile["points"][0]["total_return"] > 0  # profitable at its own cost
    assert fragile["points"][1]["total_return"] < 0  # and not at twice it


def test_cost_rises_with_the_multiple():
    """The curve has to actually scale what it says it scales."""
    report = _stress("ma_crossover", 6)
    costs = [point["total_cost"] for point in report["points"]]

    assert costs == sorted(costs)
    assert costs[-1] > costs[0]


def test_the_borrow_axis_isolates_borrow():
    """Worth asking separately because it is carry on inventory rather than a toll on
    turnover — so a long-only book, which borrows nothing, must be flat under it while
    the combined axis still bites."""
    borrow_only = _stress("ma_crossover", 6, axis="borrow")
    everything = _stress("ma_crossover", 6, axis="all")

    assert len({point["total_cost"] for point in borrow_only["points"]}) == 1
    assert len({point["total_cost"] for point in everything["points"]}) > 1


def test_a_stress_curve_journals_nothing(tmp_path):
    """Each point is one candidate under a stated assumption, not a new candidate.

    Journaling them would inflate the multiple-testing total that the deflated Sharpe
    deflates against — the campaign would punish a user for asking how robust their
    strategy was.
    """
    _stress("ma_crossover", 6)

    assert _trials(tmp_path / "journal.jsonl") == []


def test_journaling_is_on_unless_a_caller_opts_out():
    """A run that quietly does not count is how a campaign loses track of what it
    tried, so the opt-out must never be the default."""
    import inspect

    assert inspect.signature(analysis.run_backtest).parameters["journal"].default is True
