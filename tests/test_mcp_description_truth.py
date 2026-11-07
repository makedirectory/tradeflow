"""MCP tool descriptions against behaviour, driven through the real tool call.

A description is an interface an agent acts on, so each test here pairs one claim a
description makes with what the tool does when it is called.
"""

import asyncio
import json
from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.mcp import server as mcp_server
from tradeflow.services import analysis, audit

pytest.importorskip("mcp", reason="the MCP surface needs the 'mcp' extra")

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
START, END = "2024-01-02", "2025-06-01"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(audit, "DEFAULT_AUDIT_PATH", tmp_path / "mcp_audit.jsonl")
    return tmp_path


@pytest.fixture
def built():
    return mcp_server.build_server(
        data_client=MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=600, freq="1D"))
    )


def _tools(built):
    return {t.name: t for t in asyncio.run(built.list_tools())}


def _call(built, name, **kwargs):
    content = asyncio.run(built.call_tool(name, kwargs))
    if isinstance(content, tuple):
        content = content[0]
    return json.loads(content[0].text)


def _journal(tmp_path):
    path = tmp_path / "journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- 1. a candidate with no bars is reported, not silently dropped ------------------
def test_verdict_reports_candidates_dropped_for_having_no_bars(built):
    result = _call(
        built,
        "run_verdict",
        strategy="demo_trend",
        symbols=["AAA", "BBB", "CCC", "ZZZ"],
        start=START,
        end=END,
        scanner="none",
    )
    assert result["dropped_symbols"] == ["ZZZ"]


def test_a_verdict_with_nothing_dropped_says_so_with_an_empty_list(built):
    result = _call(
        built, "run_verdict", strategy="demo_trend", symbols=SYMBOLS, start=START, end=END, scanner="none"
    )
    assert result["dropped_symbols"] == []


def test_the_terminal_report_names_the_dropped_candidates(built):
    from tradeflow.analytics.reporting import format_verdict_report

    result = _call(
        built,
        "run_verdict",
        strategy="demo_trend",
        symbols=["AAA", "BBB", "CCC", "ZZZ"],
        start=START,
        end=END,
        scanner="none",
    )
    assert "no bars: ZZZ" in format_verdict_report(result)


def test_verdict_description_states_which_infeasible_runs_journal_none(built):
    text = _tools(built)["run_verdict"].description
    assert " ".join(text.split()).count("refused by the allocation preflight") == 1
    assert "dropped_symbols" in text


# --- 2. a memoized hit says nothing was journaled ------------------------------------
def _confirm(built, **extra):
    return _call(
        built,
        "confirm_screen_point",
        strategy="demo_trend",
        symbols=SYMBOLS,
        start=START,
        end=END,
        params={},
        **extra,
    )


def test_a_memoized_confirm_says_journaled_false(built, tmp_path):
    first = _confirm(built)
    second = _confirm(built)
    assert first["journaled"] is True
    assert second["memoized"] is True
    assert second["journaled"] is False
    assert len(_journal(tmp_path)) == 1


def test_a_memoized_backtest_says_journaled_false_too(built):
    args = dict(strategy="demo_trend", symbols=SYMBOLS, start=START, end=END)
    _call(built, "run_backtest", **args)
    again = _call(built, "run_backtest", **args)
    assert again["memoized"] is True
    assert again["journaled"] is False


def test_a_memoized_draft_walk_forward_says_journaled_false(built, tmp_path):
    from tests.test_research import _VALID_CODE

    args = dict(code=_VALID_CODE, symbols=SYMBOLS, start=START, end=END, n_folds=2, max_evals=2)
    first = _call(built, "run_draft_walk_forward", **args)
    second = _call(built, "run_draft_walk_forward", **args)
    assert first["draft"]["journaled"] is True
    assert second["memoized"] is True
    assert second["draft"]["journaled"] is False
    assert len(_journal(tmp_path)) == 1


# --- 3. the shared journaling sentence is only where it is true -----------------------
def test_only_optimization_claims_one_trial_per_evaluated_config(built):
    for name, spec in _tools(built).items():
        claims = "one trial per evaluated config" in spec.description
        assert claims == (name == "run_optimization"), name


def test_the_single_trial_tools_say_one_trial(built):
    tools = _tools(built)
    assert "ONE trial" in tools["confirm_screen_point"].description
    assert "one validated trial" in tools["run_walk_forward"].description
    assert "one validated trial" in tools["run_draft_walk_forward"].description
