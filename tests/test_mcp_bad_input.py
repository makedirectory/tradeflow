"""Bad input to an MCP tool is a tool error, and a saved config cannot leave its directory.

Some trial tools answered an unknown id with a normal result carrying an `error` field
while others raised, so an agent had to know which kind of failure each tool used. The
config tools took a name straight from the agent and joined it onto a path. All offline,
driven through the real tool call.
"""

import asyncio
import json
from datetime import datetime

import pandas as pd
import pytest

from quantred.marketdata.client import MarketDataClient
from quantred.mcp import server as mcp_server
from quantred.optimization import config_store
from quantred.services import analysis, audit
from quantred.services.audit import journal_trial
from tests.fakes import FakeMarketData

pytest.importorskip("mcp", reason="the MCP surface needs the 'mcp' extra")


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(audit, "DEFAULT_AUDIT_PATH", tmp_path / "mcp_audit.jsonl")
    monkeypatch.setattr(config_store, "DEFAULT_CONFIG_DIR", tmp_path / "configs")


@pytest.fixture
def built():
    return mcp_server.build_server(data_client=MarketDataClient(FakeMarketData(["AAA"], n=50, freq="1D")))


def _call(built, tool, **kwargs):
    content = asyncio.run(built.call_tool(tool, kwargs))
    if isinstance(content, tuple):
        content = content[0]
    return json.loads(content[0].text)


def _refused(built, tool, **kwargs) -> str:
    with pytest.raises(Exception) as raised:
        _call(built, tool, **kwargs)
    return str(raised.value)


def _trial(kind="backtest"):
    dates = pd.date_range("2024-01-02", periods=80, freq="D")
    return journal_trial(
        kind,
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 4, 30),
        params={"fast_ema_period": 5},
        metrics={"sharpe_ratio": 0.4},
        returns=pd.Series([0.001, -0.001] * 40, index=dates),
    )


# --- B: bad input raises --------------------------------------------------------------
@pytest.mark.parametrize("tool", ["get_trial", "analyze_trial", "get_campaign_material"])
def test_an_unknown_trial_id_is_a_tool_error_that_names_list_trials(built, tool):
    """These returned a normal result holding `{"error": ...}`, so a client saw success."""
    assert "list_trials" in _refused(built, tool, trial_id="nope")


def test_compare_trials_with_fewer_than_two_known_ids_is_a_tool_error(built):
    known = _trial()
    assert "list_trials" in _refused(built, "compare_trials", trial_ids=[known])
    message = _refused(built, "compare_trials", trial_ids=[known, "nope"])
    assert "nope" in message and "list_trials" in message


def test_compare_trials_with_two_known_ids_and_one_unknown_is_still_a_result(built):
    """Both directions: a genuinely partial answer keeps its payload field."""
    first, second = _trial(), _trial_other()
    report = _call(built, "compare_trials", trial_ids=[first, second, "nope"])
    assert report["unknown_trial_ids"] == ["nope"]


def _trial_other():
    dates = pd.date_range("2024-01-02", periods=80, freq="D")
    return journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 4, 30),
        params={"fast_ema_period": 9},
        metrics={"sharpe_ratio": 0.2},
        returns=pd.Series([0.002, -0.001] * 40, index=dates),
    )


def test_loading_an_unknown_config_names_list_configs_and_leaks_no_path(built, tmp_path):
    message = _refused(built, "load_config", name="nope")
    assert "no saved config" in message.lower() and "list_configs" in message
    assert str(tmp_path) not in message and "Errno" not in message


# --- D: the verdict trial's trade note -----------------------------------------------
def test_a_verdict_trial_is_not_told_to_re_run_it_with_trades_recorded(built):
    """`run_verdict` has no `record_trades`, so the shared advice named a parameter the
    tool does not have. The agent is told what it can actually do."""
    verdict = _trial("verdict")
    report = _call(built, "analyze_trial", trial_id=verdict)
    assert report["status"] == "unavailable"
    assert "run_backtest" in report["reason"] and "record_trades" in report["reason"]
    assert "re-run it with trades recorded" not in report["reason"]

    backtest = _call(built, "analyze_trial", trial_id=_trial())
    assert "re-run it with trades recorded" in backtest["reason"]  # other kinds unchanged
