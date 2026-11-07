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
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", document))


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

