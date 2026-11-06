"""A walk-forward's prerequisites and evidence completeness reach both surfaces.

The CLI printed "Promotion prerequisites", "Evidence completeness" and the excess
return by fold; the MCP tool returned only `gate_report`, so an agent saw `promotable`
with no sign that prerequisites existed, let alone failed. One service function now
computes the block; this runs one request over each surface on one set of bars and
compares what each produced rather than asserting each alone.
"""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from tests.fakes import DictMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.utils.timeutils import NEW_YORK

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
START, END = "2024-01-02", "2025-08-01"


def _bars(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-06-01", periods=600, freq="B", tz=NEW_YORK)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(index))))
    open_ = close * (1 + rng.normal(0, 0.003, len(index)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.004,
            "low": np.minimum(open_, close) * 0.996,
            "close": close,
            "volume": rng.integers(100_000, 500_000, len(index)).astype(float),
        },
        index=index,
    )


def _client():
    return MarketDataClient(DictMarketData({s: _bars(i) for i, s in enumerate([*SYMBOLS, "SPY"])}))


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from tradeflow.services import analysis, audit

    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")


def _over_mcp(**extra):
    from tradeflow.mcp import server

    built = server.build_server(data_client=_client())
    arguments = {
        "strategy": "demo_trend",
        "symbols": SYMBOLS,
        "start": START,
        "end": END,
        "n_folds": 3,
        "max_evals": 3,
        "benchmark": "SPY",
        "force": True,
        **extra,
    }
    content = asyncio.run(built.call_tool("run_walk_forward", arguments))
    if isinstance(content, tuple):
        content = content[0]
    return json.loads(content[0].text)


def _over_cli(monkeypatch, capsys, html=None):
    from tradeflow import cli
    from tradeflow.services import analysis

    seen = {}
    real = analysis.walk_forward_evidence

    def record(*args, **kwargs):
        seen["evidence"] = real(*args, **kwargs)
        return seen["evidence"]

    monkeypatch.setattr(analysis, "walk_forward_evidence", record)
    monkeypatch.setattr(cli, "build_research_data", lambda **kw: _client())
    argv = [
        "walkforward",
        "--strategy",
        "demo_trend",
        "--symbols",
        ",".join(SYMBOLS),
        "--start",
        START,
        "--end",
        END,
        "--folds",
        "3",
        "--max-evals",
        "3",
        "--benchmark",
        "SPY",
        "--force",
        "--no-journal",
    ]
    if html:
        argv += ["--html", str(html)]
    cli.cmd_walkforward(cli.build_parser().parse_args(argv))
    return seen["evidence"], capsys.readouterr().out


def test_both_surfaces_report_the_same_prerequisites_and_evidence(monkeypatch, capsys):
    cli_evidence, printed = _over_cli(monkeypatch, capsys)
    mcp_evidence = _over_mcp()["evidence"]
    cli_evidence = json.loads(json.dumps(cli_evidence))

    assert mcp_evidence is not None
    for key in (
        "prerequisites",
        "excess_return_by_fold",
        "evidence_completeness",
        "statistical_verdict",
        "clear",
    ):
        assert mcp_evidence[key] == cli_evidence[key], key
    assert mcp_evidence["benchmark"] == "SPY"
    assert mcp_evidence["benchmark_scored_folds"] == 3
    # The CLI really rendered that block: each evaluated check's verdict word is in it.
    for name, check in cli_evidence["prerequisites"]["checks"].items():
        if check["evaluated"]:
            assert f"[{'PASS' if check['passed'] else 'FAIL'}] {name}" in printed
    assert mcp_evidence["evidence_completeness"] in printed


def test_promotable_never_reads_as_clear_while_a_prerequisite_is_unmet():
    evidence = _over_mcp()["evidence"]
    checks = evidence["prerequisites"]["checks"]
    # The bootstrap input does not exist over MCP: unevaluated, never passed.
    assert checks["family_bootstrap"]["evaluated"] is False
    assert evidence["prerequisites"]["evaluated"] < evidence["prerequisites"]["total"]
    assert evidence["clear"] is False
    assert "does not mean" in evidence["note"]


def test_without_a_benchmark_the_benchmark_checks_are_unevaluated_not_failed():
    payload = _over_mcp(benchmark=None, cost_stress=None)
    checks = payload["evidence"]["prerequisites"]["checks"]
    assert checks["benchmark_relative"]["evaluated"] is False
    assert checks["benchmark_excess"]["evaluated"] is False
    assert checks["cost_stress"]["evaluated"] is False
    assert payload["evidence"]["cost_stress"] is None
    assert payload["benchmark"] is None
