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
    assert "refused by the allocation preflight" in text
    assert "dropped_symbols" in text
