"""Service-layer + MCP-server tests (Spec 003).

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
    assert "volume_threshold" in pr["param_ranges"]
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
        _client(), "volume_spike", SYMBOLS, START, END, n_folds=3, embargo_days=2,
        holdout_days=30, method="grid", max_evals=8,
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
    run_id = audit_log("run_backtest", {"strategy": "volume_spike"}, path=path,
                       result_summary={"total_trades": 3})
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "run_backtest"
    assert record["run_id"] == run_id
    assert "timestamp" in record and "git_sha" in record


# --- the safety wall (Spec 003 §4) -----------------------------------------
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
