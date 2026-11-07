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


# --- 4. what compare_trials can actually compare --------------------------------------
def test_two_backtests_recorded_over_mcp_can_be_compared(built):
    """A backtest journaled over MCP stored no return series (the CLI stored one), so
    the tool whose description says it reads recorded series found nothing."""
    a = _call(built, "run_backtest", strategy="demo_trend", symbols=SYMBOLS[:3], start=START, end=END)
    b = _call(built, "run_backtest", strategy="demo_trend", symbols=SYMBOLS[3:], start=START, end=END)
    report = _call(built, "compare_trials", trial_ids=[a["trial_id"], b["trial_id"]])
    pair = report["pairs"][0]
    assert pair.get("correlation") is not None, pair


def test_descriptions_do_not_send_compare_trials_to_the_trade_table(built):
    tools = _tools(built)
    flat = " ".join(tools["run_backtest"].description.split())
    assert "`analyze_trial` and `compare_trials` read" not in flat
    assert "one in-sample pass" not in tools["compare_trials"].description


# --- 5. the cost stress trades the book the folds were validated at ------------------
def _stress_configs(monkeypatch):
    seen = []
    real = analysis.run_cost_stress

    def spy(*args, **kwargs):
        seen.append(kwargs.get("config"))
        return real(*args, **kwargs)

    monkeypatch.setattr(analysis, "run_cost_stress", spy)
    return seen


def test_walk_forward_cost_stress_runs_at_the_requested_book(monkeypatch):
    from tests.test_walk_forward_evidence_parity import _over_mcp

    seen = _stress_configs(monkeypatch)
    result = _over_mcp(position_limits={"max_positions": 3}, cost_stress="turnover")
    assert seen and seen[0]["position_limits"] == {"max_positions": 3}
    assert result["evidence"]["prerequisites"]["checks"]["cost_stress"]["evaluated"] is True


def test_cost_stress_without_a_book_override_carries_none(monkeypatch):
    from tests.test_walk_forward_evidence_parity import _over_mcp

    seen = _stress_configs(monkeypatch)
    _over_mcp(cost_stress="turnover")
    assert seen and "position_limits" not in seen[0]


def test_the_cli_hands_the_configs_book_to_the_evidence(monkeypatch, tmp_path, capsys):
    """Both surfaces call one evidence function; the CLI's call is its own line."""
    from tests.test_walk_forward_evidence_parity import SYMBOLS as WF_SYMBOLS
    from tests.test_walk_forward_evidence_parity import _client
    from tradeflow import cli

    seen = {}
    real = analysis.walk_forward_evidence

    def spy(*args, **kwargs):
        seen["limits"] = kwargs.get("position_limits")
        return real(*args, **kwargs)

    monkeypatch.setattr(analysis, "walk_forward_evidence", spy)
    monkeypatch.setattr(cli, "build_research_data", lambda **kw: _client())
    path = tmp_path / "book.json"
    path.write_text(
        json.dumps(
            {
                "strategy": "demo_trend",
                "symbols": WF_SYMBOLS,
                "params": {
                    "fast_ema_period": 18,
                    "slow_ema_period": 24,
                    "risk_per_trade": 0.04,
                    "stop_loss": 0.02,
                    "take_profit": 0.03,
                },
                "position_limits": {"max_positions": 3},
                "provenance": {},
            }
        )
    )
    argv = ["walkforward", "--config", str(path), "--start", "2024-01-02", "--end", "2025-08-01"]
    argv += ["--folds", "3", "--max-evals", "3", "--benchmark", "SPY", "--force", "--no-journal"]
    cli.cmd_walkforward(cli.parse_cli(argv))
    assert seen["limits"] == {"max_positions": 3}


# --- 6. the bootstrap note agrees with the description -------------------------------
def test_the_family_bootstrap_note_says_this_surface_supplies_no_bootstrap():
    from tests.test_walk_forward_evidence_parity import _over_mcp

    check = _over_mcp()["evidence"]["prerequisites"]["checks"]["family_bootstrap"]
    assert check["evaluated"] is False
    assert "does not take" in check["note"]
    assert "available" not in check["note"]


# --- 7 and 11. confirm takes declared parameters only; a screen range only narrows ----
def _screen(built, **extra):
    return _call(
        built,
        "run_screen",
        strategy="demo_trend",
        symbols=SYMBOLS,
        start=START,
        end=END,
        max_evals=4,
        **extra,
    )


def _confirm_with(built, params):
    return _call(
        built,
        "confirm_screen_point",
        strategy="demo_trend",
        symbols=SYMBOLS,
        start=START,
        end=END,
        params=params,
    )


def test_confirm_refuses_a_key_the_strategy_does_not_declare(built, tmp_path):
    with pytest.raises(Exception) as exc:
        _confirm_with(built, {"bogus": 1})
    assert "bogus" in str(exc.value)
    assert _journal(tmp_path) == []


def test_a_whole_best_point_is_refused_but_its_params_are_confirmable(built, tmp_path):
    best = _screen(built)["best_point"]
    assert set(best["params"]) < set(best)
    with pytest.raises(Exception):
        _confirm_with(built, {k: v for k, v in best.items() if k != "params"})
    assert _journal(tmp_path) == []

    confirmed = _confirm_with(built, best["params"])
    assert confirmed["journaled"] is True
    config = _journal(tmp_path)[0]["resolved_config"]
    assert set(config).isdisjoint({"sharpe_ratio", "total_return"})


def test_a_screen_range_cannot_widen_past_the_declared_bounds(built):
    from tradeflow.services.registry import STRATEGIES

    declared = STRATEGIES["demo_trend"].PARAM_RANGES["fast_ema_period"]
    wider = {"fast_ema_period": {"min": declared["min"], "max": declared["max"] + 80}}
    with pytest.raises(Exception) as exc:
        _screen(built, param_ranges=wider)
    assert "fast_ema_period" in str(exc.value)
    with pytest.raises(ValueError):
        analysis.screen_ranges(STRATEGIES["demo_trend"], wider)
    inside = {"fast_ema_period": {"min": declared["min"] + 1, "max": declared["max"]}}
    assert _screen(built, param_ranges=inside)["searched"]["requested"] >= 1


# --- 8. recovery advice names a tool that can do it -----------------------------------
def _analyze_reason(built, trial_id):
    report = _call(built, "analyze_trial", trial_id=trial_id)
    assert report["status"] == "unavailable"
    return report["reason"]


def _first_id(built, kind):
    rows = _call(built, "list_trials", kind=kind)
    rows = rows["rows"]
    return rows[0]["id"]


def test_an_optimization_trial_is_not_told_to_rerun_with_record_trades(built):
    _call(
        built,
        "run_optimization",
        strategy="demo_trend",
        symbols=SYMBOLS,
        start=START,
        end=END,
        max_evals=2,
    )
    reason = _analyze_reason(built, _first_id(built, "optimize"))
    assert "`run_optimization`" in reason
    assert "`run_backtest` with `record_trades=true`" in reason


def test_a_draft_walk_forward_trial_is_not_told_to_rerun_with_record_trades(built):
    from tests.test_research import _VALID_CODE

    _call(
        built,
        "run_draft_walk_forward",
        code=_VALID_CODE,
        symbols=SYMBOLS,
        start=START,
        end=END,
        n_folds=2,
        max_evals=2,
    )
    reason = _analyze_reason(built, _first_id(built, "walkforward"))
    assert "`run_draft_walk_forward` cannot record" in reason


def test_a_backtest_without_a_table_names_the_tool_that_can_record_one(built):
    result = _call(built, "run_backtest", strategy="demo_trend", symbols=SYMBOLS, start=START, end=END)
    assert "`run_backtest` with `record_trades=true`" in _analyze_reason(built, result["trial_id"])


def test_an_unknown_campaign_material_id_does_not_name_a_cli_command(built):
    with pytest.raises(Exception) as exc:
        _call(built, "get_campaign_material", trial_id="deadbeef")
    message = str(exc.value)
    assert "trials rebuild" not in message
    assert "operator action" in message


# --- 9. compute_alphas: an unknown scaling is refused; a thin universe says what it skipped
def _alphas(built, **extra):
    return _call(
        built,
        "compute_alphas",
        strategy="demo_trend",
        symbols=SYMBOLS,
        as_of="2025-05-30",
        source="strategy",
        **extra,
    )


def test_an_unknown_scaling_is_refused_not_treated_as_case1(built):
    with pytest.raises(Exception) as exc:
        _alphas(built, scaling="bogus")
    assert "case1" in str(exc.value)
    assert _alphas(built, scaling="case1")["scaling"] == "case1"


def test_a_thin_universe_says_it_skipped_neutralization_and_the_case_choice(built):
    result = _alphas(built, neutralize_factors=["market", "size"], scaling="case2")
    assert result["low_confidence"] is True
    assert result["neutralize_factors"] == ["market", "size"]
    assert result["neutralized_against"] == []
    reason = result["low_confidence_reason"]
    assert "neutralization against market, size" in reason
    assert "the case choice" in reason


def test_a_thin_universe_that_asked_for_nothing_extra_adds_no_skip_text(built):
    reason = _alphas(built)["low_confidence_reason"]
    assert "thin universe" in reason
    assert "skipped" not in reason


# --- 10. compute_horizon does not point an agent at a CLI flag it cannot use ----------
def test_compute_horizon_payload_does_not_recommend_a_cli_policy_flag(built):
    result = _call(
        built, "compute_horizon", strategy="demo_trend", symbols=SYMBOLS, start=START, end=END
    )
    assert "recommended_cadence" in result
    assert "--policy" not in json.dumps(result)
    assert "blend_superseded_by" not in result


def test_the_service_report_the_cli_prints_keeps_its_pointer():
    direct = analysis.compute_horizon(
        MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=600, freq="1D")),
        "demo_trend",
        SYMBOLS,
        datetime(2024, 1, 2),
        datetime(2025, 6, 1),
    )
    assert "--policy aim" in direct["blend_superseded_by"]
