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


def test_a_numpy_array_becomes_a_list_not_an_error():
    """Found in review: `.item()` on a multi-element array raised, so a tool returning an
    ndarray errored where FastMCP had stringified it."""
    assert mcp_server._json_safe({"a": np.array([1.0, np.nan]), "b": np.float32(2.5)}) == {
        "a": [1.0, "nan"],
        "b": 2.5,
    }


def test_an_unknown_cost_stress_axis_is_refused_before_anything_is_journaled(tmp_path, monkeypatch):
    """Found in review: any other string scaled nothing, so every "stress" point was the
    baseline and the cost prerequisite passed at the top multiple."""
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=500, freq="1D"))
    from tradeflow.utils.timeutils import parse_market_date as day

    with pytest.raises(analysis.RunRefused, match="cost_stress must be one of"):
        analysis.run_walk_forward(
            client,
            "demo_trend",
            ["AAA", "BBB"],
            day("2023-01-02"),
            day("2024-06-01"),
            n_folds=2,
            max_evals=2,
            cost_stress="none",
        )
    with pytest.raises(analysis.RunRefused):
        analysis.run_cost_stress(
            client, "demo_trend", ["AAA"], day("2023-01-02"), day("2024-06-01"), axis="bogus"
        )
    assert not (tmp_path / "journal.jsonl").exists()


def test_a_reused_or_draft_walk_forward_says_why_it_has_no_evidence(tmp_path, monkeypatch):
    """Found in review: a reused answer had no `evidence` key at all though the docs said
    null, and a draft's null evidence read as "reused" in the HTML banner."""
    from tradeflow.analytics.htmlreport import render_html
    from tradeflow.utils.timeutils import parse_market_date as day

    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=500, freq="1D"))
    kw = dict(n_folds=2, max_evals=2, cost_stress=None)
    fresh = analysis.run_walk_forward(
        client, "demo_trend", ["AAA", "BBB"], day("2023-01-02"), day("2024-06-01"), **kw
    )
    reused = analysis.run_walk_forward(
        client, "demo_trend", ["AAA", "BBB"], day("2023-01-02"), day("2024-06-01"), **kw
    )
    assert reused["memoized"] is True
    assert "evidence" in reused and reused["evidence"] is None
    assert "reused result" in reused["evidence_unavailable"]
    draft = {
        **fresh,
        "evidence": None,
        "evidence_unavailable": "a draft walk-forward does not run the promotion prerequisites",
    }
    page = render_html(draft, "walkforward")
    assert "a draft walk-forward does not run" in page and "reused result keeps" not in page
