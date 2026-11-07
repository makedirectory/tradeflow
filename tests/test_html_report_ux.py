"""What a rendered HTML report must say that the terminal says.

Every test here reads the rendered document, not the payload: a payload can carry a
caveat the page then drops, which is exactly how the gaps these pin went unnoticed.
"""

import asyncio
import json
import re

import numpy as np
import pandas as pd
import pytest

from tests.fakes import DictMarketData
from tradeflow import cli
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.utils.timeutils import NEW_YORK

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
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
    client = _client()
    monkeypatch.setattr(cli, "build_research_data", lambda **kw: client)


def _cli(argv, html):
    args = cli.build_parser().parse_args([*argv, "--html", str(html)])
    args.func(args)
    return html.read_text() if html.exists() else None


def _mcp(tool, arguments):
    from tradeflow.mcp import server

    built = server.build_server(data_client=_client())
    content = asyncio.run(built.call_tool(tool, arguments))
    if isinstance(content, tuple):
        content = content[0]
    return json.loads(content[0].text)


def _text(document: str) -> str:
    """The visible text of a document, tags dropped, so assertions read what a person reads."""
    import html

    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", document)))


WF = [
    "walkforward",
    "--strategy",
    "demo_trend",
    "--scanner",
    "none",
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
]
WF_ARGS = {
    "strategy": "demo_trend",
    "symbols": SYMBOLS,
    "start": START,
    "end": END,
    "n_folds": 3,
    "max_evals": 3,
    "benchmark": "SPY",
}


# ---- item 1: a reused walk-forward renders on both surfaces ---------------------------
def test_a_reused_walkforward_renders_over_the_cli(tmp_path, capsys):
    _cli(WF, tmp_path / "first.html")
    capsys.readouterr()
    document = _cli(WF, tmp_path / "second.html")
    out = capsys.readouterr().out

    assert document is not None, "a reused walk-forward wrote no report and said nothing"
    assert "HTML report written" in out
    text = _text(document)
    assert "REUSED" in text
    assert "per-fold detail" in text and "holdout" in text
    assert "Out-of-sample aggregate" in text


def test_a_reused_walkforward_renders_over_mcp():
    first = _mcp("run_walk_forward", WF_ARGS)
    assert not first.get("memoized")
    reused = _mcp("run_walk_forward", WF_ARGS)
    assert reused["memoized"] is True
    rendered = _mcp("render_report", {"kind": "walkforward", "result": reused})
    text = _text(rendered["html"])
    assert "REUSED" in text and reused["trial_id"] in text
    assert "per-fold detail" in text


# ---- item 2: the verdict report carries the alphas caveats the terminal prints --------
VERDICT = [
    "verdict",
    "--strategy",
    "demo_trend",
    "--scanner",
    "none",
    "--symbols",
    ",".join(SYMBOLS),
    "--start",
    START,
    "--end",
    END,
]


def test_the_verdict_report_states_the_assumed_ic_and_scaling_the_terminal_prints(tmp_path, capsys):
    document = _cli(VERDICT, tmp_path / "v.html")
    printed = capsys.readouterr().out
    heading = re.search(r"Alphas \(.*?\): \d+ names[^\n]*", printed).group(0).strip()
    heading = re.sub(r"\s+", " ", heading)
    assert heading in _text(document)
    assert "assumed IC" in _text(document)


def test_the_reason_a_universe_is_low_confidence_reaches_the_verdict_report():
    from tradeflow.analytics.htmlreport import render_html
    from tradeflow.services import analysis

    result = analysis.run_verdict(
        _client(), "demo_trend", SYMBOLS, cli._date(START), cli._date(END), journal=False
    )
    assert result["alphas"]["low_confidence"], "fixture should be a thin universe"
    reason = result["alphas"]["low_confidence_reason"]
    assert reason and reason in _text(render_html(result, "verdict"))


# ---- item 3: every metrics table uses the same unit formatter -------------------------
def _metric_rows(document: str, section: str) -> dict:
    """Metric name -> displayed value for the table inside the card titled ``section``."""
    card = re.search(rf"<h2>{section}</h2>(.*?)</section>", document, re.S).group(1)
    return {
        _text(name).strip(): _text(value).strip()
        for name, value in re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>", card, re.S)
    }


def _unit_marker(shown: str):
    return ("%" in shown, "$" in shown)


def test_the_holdout_table_formats_metrics_as_the_aggregate_table_does(tmp_path):
    document = _cli([*WF, "--holdout-days", "60", "--force"], tmp_path / "wf.html")
    aggregate = _metric_rows(document, "Out-of-sample aggregate")
    holdout = _metric_rows(document, "Holdout")
    shared = [k for k in ("cagr", "max_drawdown", "avg_loss") if k in aggregate and k in holdout]
    assert shared, "fixture should share unit-declared metrics between the two tables"
    for key in shared:
        # A raw 0.9594 beside an aggregate 4.44% is the defect: the unit marker must agree.
        assert _unit_marker(holdout[key]) == _unit_marker(aggregate[key]), (
            key,
            holdout[key],
            aggregate[key],
        )


# ---- item 4: the backtest report carries the labels the terminal prints ---------------
BT = [
    "backtest",
    "--strategy",
    "demo_trend",
    "--scanner",
    "none",
    "--symbols",
    ",".join(SYMBOLS),
    "--start",
    START,
    "--end",
    END,
]


def test_the_backtest_report_says_it_is_not_a_validation_and_names_survivorship(tmp_path, capsys):
    document = _cli([*BT, "--benchmark", "SPY"], tmp_path / "bt.html")
    printed = capsys.readouterr().out
    text = _text(document)
    assert re.search(r"Statistical validation\s+not assessed here", printed)
    assert re.search(r"Statistical validation\s+not assessed here", text)
    assert "in-sample" in text
    assert "survivorship" in printed and "survivorship" in text
    assert "membership was not point-in-time" in text


def test_the_backtest_report_carries_the_pnl_split_when_the_terminal_prints_it(tmp_path, capsys):
    document = _cli([*BT, "--benchmark", "SPY"], tmp_path / "bt.html")
    printed = capsys.readouterr().out
    assert "Where the P&L came from" in printed, "fixture should produce exit reasons"
    text = _text(document)
    assert "Where the P&L came from" in text
    after = printed.split("Where the P&L came from")[1]
    for row in re.findall(r"^  (\w+)\s+\d+\s+[\d.]+%", after, re.M):
        assert row in text
    if "Nearly all of the gain" in printed:
        assert "Nearly all of the gain" in text


def test_a_ratio_with_no_benchmark_is_not_measured_rather_than_zero():
    from tradeflow.analytics.htmlreport import render_html

    metrics = {"treynor_ratio": 0.0, "treynor_available": False, "benchmark_available": True, "beta": 0.01}
    result = {"metrics": metrics, "strategy": "s", "symbols": ["AAA"], "window": {}}
    assert re.search(r"treynor_ratio\s+not measured", _text(render_html(result, "backtest")))
    bare = {**metrics, "benchmark_available": False, "alpha": 0.0, "information_ratio": 0.0}
    text = _text(render_html({**result, "metrics": bare}, "backtest"))
    for name in ("alpha", "information_ratio", "treynor_ratio", "beta"):
        assert re.search(rf"{name}\s+not measured \(no benchmark\)", text), name


# ---- item 5: a reused backtest's header is filled from the request --------------------
def test_a_reused_backtest_header_names_the_benchmark_and_timeframe():
    arguments = {"strategy": "demo_trend", "symbols": SYMBOLS, "start": START, "end": END, "benchmark": "SPY"}
    _mcp("run_backtest", arguments)
    reused = _mcp("run_backtest", arguments)
    assert reused["memoized"] is True
    text = _text(_mcp("render_report", {"kind": "backtest", "result": reused})["html"])
    assert re.search(r"Benchmark\s+SPY \(as requested\)", text)
    assert not re.search(r"Timeframe\s+—", text) and not re.search(r"Benchmark\s+—", text)
    assert re.search(r"Timeframe\s+\S+ \(as requested\)", text)


# ---- item 6: ids and timestamps -------------------------------------------------------
def _journaled_ids(tmp_path):
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines() if line.strip()]
    return [r["run_id"] for r in rows if r.get("tool") == "trial:walkforward"]


def test_a_journaled_walkforward_report_names_its_trial_and_campaign_count(tmp_path):
    document = _cli(WF, tmp_path / "wf.html")
    ids = _journaled_ids(tmp_path)
    assert len(ids) == 1, "the run should have been journaled once"
    text = _text(document)
    assert re.search(rf"Trial id\s+{re.escape(ids[0])}", text)
    assert not re.search(r"Campaign trials\s+—", text)
    assert re.search(r"Campaign trials\s+\d+", text)


def test_the_mcp_walkforward_report_names_its_trial_too(tmp_path):
    fresh = _mcp("run_walk_forward", WF_ARGS)
    ids = _journaled_ids(tmp_path)
    assert fresh["trial_id"] == ids[0]
    text = _text(_mcp("render_report", {"kind": "walkforward", "result": fresh})["html"])
    assert re.search(rf"Trial id\s+{re.escape(ids[0])}", text)


def test_no_prerequisite_label_is_glued_to_its_value(tmp_path, capsys):
    """`benchmark_relative` is as wide as the old fixed column, so its value ran into it."""
    argv = [a for a in WF if a not in ("--benchmark", "SPY")]
    _cli(argv, tmp_path / "wf.html")
    block = capsys.readouterr().out.split("Promotion prerequisites")[1].split("Prerequisites:")[0]
    rows = re.findall(r"^  \[(?:PASS|FAIL| -- )\] (.*)$", block, re.M)
    assert rows and any(r.startswith("benchmark_relative") for r in rows)
    for row in rows:
        assert re.match(r"\w+ {2,}\S", row), row


def test_a_verdict_report_is_stamped_in_utc_like_the_others(tmp_path):
    text = _text(_cli(VERDICT, tmp_path / "v.html"))
    stamp = re.search(r"Generated\s+(\S+)", text).group(1)
    assert stamp.endswith("+00:00"), stamp
