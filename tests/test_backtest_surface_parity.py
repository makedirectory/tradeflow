"""One backtest request over the CLI and over MCP gives one set of numbers.

Identically written dates once produced different benchmark and buy-and-hold figures
on the two surfaces while the final capital matched: MCP read a bare date as a naive
(UTC) instant and the CLI as New York midnight, and daily bars are stamped at New York
midnight, so the window's last bar was in one run and not the other. The date parsers
now share one definition; this compares the *results* end to end, on bars stamped the
way a daily feed stamps them, so a second reading of a date cannot creep back in.
"""

import asyncio
import contextlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from quantred.marketdata.client import MarketDataClient
from quantred.utils.timeutils import NEW_YORK
from tests.fakes import DictMarketData

SYMBOLS = ["AAA", "BBB", "CCC"]
REQUEST = {"start": "2024-01-02", "end": "2024-12-31", "capital": 4000.0}


def _daily_at_exchange_midnight(seed: int) -> pd.DataFrame:
    """Daily bars stamped at New York midnight, as a daily feed stamps them."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-10-02", "2025-02-28", freq="B", tz=NEW_YORK)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, len(index))))
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


class _VendorWindowed(DictMarketData):
    """Serves the requested window the way a vendor API does: a naive instant is UTC.

    The plain fake ignores the window, which is exactly what hid this: two readings of
    one date only produce different bars once something actually slices by the instant.
    """

    def get_bars(self, symbols, timeframe, start, end):
        lo, hi = (pd.Timestamp(t) for t in (start, end))
        lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo
        hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi
        frames = super().get_bars(symbols, timeframe, start, end)
        return {s: f.loc[(f.index >= lo) & (f.index <= hi)] for s, f in frames.items()}


def _client():
    frames = {s: _daily_at_exchange_midnight(i) for i, s in enumerate([*SYMBOLS, "SPY"])}
    return MarketDataClient(_VendorWindowed(frames))


def _over_mcp():
    from quantred.mcp import server

    built = server.build_server(data_client=_client())
    arguments = {
        "strategy": "demo_trend",
        "symbols": SYMBOLS,
        "start": REQUEST["start"],
        "end": REQUEST["end"],
        "capital": REQUEST["capital"],
        "force": True,
    }
    content = asyncio.run(built.call_tool("run_backtest", arguments))
    if isinstance(content, tuple):
        content = content[0]
    return json.loads(content[0].text)["metrics"]


def _over_cli(monkeypatch):
    from quantred import cli
    from quantred.analytics import reporting

    captured = {}
    render = reporting.format_backtest_report

    def capture(metrics, *args, **kwargs):
        captured.update(metrics)
        return render(metrics, *args, **kwargs)

    monkeypatch.setattr(reporting, "format_backtest_report", capture)
    monkeypatch.setattr(cli, "build_research_data", lambda **kw: _client())
    argv = [
        "backtest",
        "--strategy",
        "demo_trend",
        "--symbols",
        ",".join(SYMBOLS),
        "--start",
        REQUEST["start"],
        "--end",
        REQUEST["end"],
        "--capital",
        str(REQUEST["capital"]),
        "--force",
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        cli.cmd_backtest(cli.build_parser().parse_args(argv))
    return captured


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Both runs journal a trial; keep them out of the session's shared state root."""
    from quantred.services import analysis, audit

    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")


def _differences(a, b):
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def test_one_backtest_request_gives_one_set_of_numbers_over_either_surface(monkeypatch):
    mcp = _over_mcp()
    cli = _over_cli(monkeypatch)
    assert mcp and cli
    assert "benchmark_buy_hold_return" in mcp
    assert _differences(mcp, cli) == {}


def test_the_comparison_notices_a_second_reading_of_a_date(monkeypatch):
    """The comparison above is only worth having if it fails when the old naive reading
    returns: with MCP reading a bare date as naive again, the figures must differ."""
    from datetime import datetime

    from quantred.mcp import server

    monkeypatch.setattr(server, "_parse_date", lambda value: datetime.fromisoformat(value))
    differences = _differences(_over_mcp(), _over_cli(monkeypatch))
    if not differences:
        pytest.fail("a naive date reading went unnoticed")
    assert "benchmark_buy_hold_return" in differences or "buy_hold_return" in differences
