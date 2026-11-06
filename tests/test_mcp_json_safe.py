"""Every MCP result is valid JSON: non-finite numbers are spelled, never emitted bare.

`compute_horizon` returned a bare `Infinity` and `combine_alphas` a bare `NaN` beside
`"nan"` strings in the same payload; a strict client refuses to parse the first two.
"""

import asyncio
import json

import numpy as np
import pytest

from tests.fakes import FakeMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.mcp import server as mcp_server
from tradeflow.services import analysis, audit

pytest.importorskip("mcp", reason="the MCP surface needs the 'mcp' extra")


def _strict(text):
    def refuse(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    return json.loads(text, parse_constant=refuse)


def test_a_tool_result_with_non_finite_numbers_parses_as_strict_json(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "DEFAULT_AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(
        analysis,
        "compute_horizon",
        lambda *a, **k: {
            "half_life_upper": float("inf"),
            "floor": float("-inf"),
            "weights": {"a": float("nan"), "b": np.float64("nan")},
            "series": [1.0, float("nan")],
            "ok": 0.5,
        },
    )
    built = mcp_server.build_server(data_client=MarketDataClient(FakeMarketData(["AAA"], n=50, freq="1D")))
    content = asyncio.run(
        built.call_tool(
            "compute_horizon",
            {"strategy": "demo_trend", "symbols": ["AAA"], "start": "2024-01-02", "end": "2024-03-01"},
        )
    )
    content = content[0] if isinstance(content, tuple) else content
    payload = _strict(content[0].text)
    assert payload["half_life_upper"] == "inf" and payload["floor"] == "-inf"
    assert payload["weights"] == {"a": "nan", "b": "nan"}
    assert payload["series"] == [1.0, "nan"] and payload["ok"] == 0.5
