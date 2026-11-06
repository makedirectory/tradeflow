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


def _header(html: str, label: str) -> str:
    import re

    match = re.search(rf"<dt>{label}</dt><dd>(.*?)</dd>", html, re.S)
    assert match, f"no {label} in the header"
    return match.group(1)


def test_the_walk_forward_report_names_what_it_ran_and_carries_the_evidence(monkeypatch, capsys, tmp_path):
    """The HTML's only verdict used to be the six gates, with a dash for the benchmark
    and timeframe, so a run whose gates passed and prerequisites failed looked clear."""
    from tradeflow.analytics.htmlreport import render_html

    cli_evidence, _ = _over_cli(monkeypatch, capsys, html=tmp_path / "wf.html")
    cli_html = (tmp_path / "wf.html").read_text(encoding="utf-8")
    mcp_html = render_html(_over_mcp(), "walkforward")

    for html in (cli_html, mcp_html):
        assert _header(html, "Benchmark") == "SPY"
        assert _header(html, "Timeframe") == "1Day"
        assert "Promotion prerequisites" in html
        assert cli_evidence["evidence_completeness"] in html
        assert _header(html, "Evidence completeness") == cli_evidence["evidence_completeness"]
    # Same evidence, same words, whichever surface produced the file.
    assert _header(cli_html, "Universe") == _header(mcp_html, "Universe")


def test_a_promotable_walk_forward_with_unmet_prerequisites_is_not_rendered_as_clear():
    from tradeflow.analytics.htmlreport import render_html

    payload = {
        "folds": [],
        "strategy": "s",
        "gate_report": {"promotable": True, "checks": {}},
        "evidence": {
            "clear": False,
            "evidence_completeness": "1 of 4 evaluated - NOT clear; 3 unknown",
            "statistical_verdict": "PASS",
            "prerequisites": {"checks": {}},
        },
    }
    html = render_html(payload, "walkforward")
    assert "evidence NOT clear" in html and 'class="verdict pass"' not in html
    # And the other direction: cleared evidence is allowed to read as clear.
    payload["evidence"]["clear"] = True
    assert 'class="verdict pass"' in render_html(payload, "walkforward")


def test_the_information_report_header_names_the_run_it_describes():
    from tests.fakes import FakeMarketData
    from tradeflow.analytics.htmlreport import render_html
    from tradeflow.services import analysis

    names = ["S0", "S1", "S2", "S3", "S4", "S5"]
    client = MarketDataClient(FakeMarketData([*names, "SPY"], n=300, freq="1D"))
    start, end = pd.Timestamp("2023-06-01").to_pydatetime(), pd.Timestamp("2024-03-01").to_pydatetime()
    result = analysis.compute_information(client, "demo_trend", names, start, end)
    html = render_html(result, "info")
    assert _header(html, "Universe").startswith("6 names")
    assert _header(html, "Benchmark") == "SPY"
