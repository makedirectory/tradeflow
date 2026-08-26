"""The MCP research surface: the new tools, the descriptions, and the wall.

All offline. Two things are being defended here. The first is parity — a tool must
return exactly what the service returns, or the CLI and an agent are looking at two
different systems. The second is the descriptions themselves: an agent cannot notice
that a description is stale, it can only act on one, and every action it takes costs
a journaled trial. So the description audit is pinned by tests, crude as string
assertions are, rather than left to good intentions.
"""

import asyncio
import json
from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tests.test_research import _VALID_CODE, _VALID_SCANNER_CODE
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.mcp import server as mcp_server
from tradeflow.mcp.server import EVIDENCE_GATED, EXPOSED_TOOLS, FORBIDDEN_TOOLS, JOURNALING_TOOLS
from tradeflow.services import analysis, audit
from tradeflow.store.trials import TrialStore

pytest.importorskip("mcp", reason="the MCP surface needs the 'mcp' extra")

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
START, END = datetime(2024, 1, 2), datetime(2025, 6, 1)


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
    """Invoke a registered tool and decode the payload a client actually receives.

    Deliberately reads the serialized text content rather than the server's
    in-process return value: what matters is what survives the wire, which is the
    whole point of asserting the leaderboard's caveat lives in the data.
    """
    content = asyncio.run(built.call_tool(name, kwargs))
    if isinstance(content, tuple):
        content = content[0]
    return json.loads(content[0].text)


# --- the surface ------------------------------------------------------------
def test_the_registered_surface_is_exactly_the_declared_one(built):
    assert set(_tools(built)) == set(EXPOSED_TOOLS)


def test_the_read_only_wall_still_holds(built):
    registered = set(_tools(built))
    assert not (registered & FORBIDDEN_TOOLS)
    with pytest.raises(Exception):
        mcp_server.build_server(data_client=object())


def test_the_server_module_never_imports_a_trading_path():
    """A layering violation here is a safety violation: the wall is the *absence*
    of the capability, so an import that could reach one is review-blocking."""
    source = __import__("pathlib").Path(mcp_server.__file__).read_text()
    for forbidden in ("tradeflow.execution", "engine.live", "build_broker", "TradingClient"):
        assert forbidden not in source


# --- the description audit --------------------------------------------------
def test_every_tool_has_a_real_description(built):
    for name, spec in _tools(built).items():
        assert spec.description, f"{name} has no description"
        assert len(spec.description) > 60, f"{name}'s description is too thin to act on"


def test_journaling_tools_say_that_they_journal(built):
    """An agent that does not know a call costs a trial will burn the campaign's
    multiple-testing budget at machine speed."""
    tools = _tools(built)
    for name in JOURNALING_TOOLS:
        description = tools[name].description.lower()
        assert "journal" in description, f"{name} does not mention journaling"
        assert "memoized" in description, f"{name} does not mention memoization"
        assert "multiple-testing" in description


def test_non_journaling_tools_do_not_claim_to_journal(built):
    tools = _tools(built)
    for name in ("list_trials", "get_trial", "best_trials", "render_report", "compute_alphas"):
        assert "Journals one trial" not in tools[name].description


def test_tools_exposing_an_evidence_gated_feature_name_the_gate(built):
    tools = _tools(built)
    for name in ("construct_portfolio", "compute_risk", "run_verdict"):
        description = tools[name].description
        assert "Evidence-gated" in description, f"{name} presents a gated flag as neutral"
        assert "does not clear" in description or "none of them clears" in description
    # And the gated feature names themselves are stated, not alluded to.
    assert all(term in tools["construct_portfolio"].description.lower() for term in EVIDENCE_GATED[:1])


def test_metric_vocabulary_comes_from_the_glossary_not_a_restatement(built):
    """One definition with two readers cannot drift; two descriptions will."""
    from tradeflow.services.glossary import definitions_for

    canonical = definitions_for(["deflated_sharpe_ratio"])["deflated_sharpe_ratio"]
    description = _tools(built)["run_backtest"].description
    assert canonical in description
    assert "get_metrics_glossary" in description


def test_the_leaderboard_tool_warns_in_its_own_description(built):
    description = _tools(built)["best_trials"].description
    assert "selection bias" in description
    assert "deflated" in description.lower()
    assert "n_trials" in description


def test_run_scan_accepts_a_historical_as_of(built):
    payload = _call(
        built,
        "run_scan",
        scanner="volume",
        symbols=SYMBOLS,
        as_of="2024-06-01T16:00:00-04:00",
    )

    assert payload["as_of"] == "2024-06-01T16:00:00-04:00"
    assert payload["candidates"] == SYMBOLS


# --- draft strategy/scanner workflow ---------------------------------------
def test_draft_validation_tools_return_contract_metadata(built):
    strategy = _call(built, "validate_draft_strategy_code", code=_VALID_CODE)
    scanner = _call(built, "validate_draft_scanner_code", code=_VALID_SCANNER_CODE)

    assert strategy["valid"] is True
    assert strategy["class_name"] == "GenStrat"
    assert strategy["code_hash"]
    assert scanner["valid"] is True
    assert scanner["class_name"] == "GenScanner"


def test_draft_walk_forward_runs_without_registering_source(built):
    payload = _call(
        built,
        "run_draft_walk_forward",
        code=_VALID_CODE,
        symbols=SYMBOLS,
        start=START.strftime("%Y-%m-%d"),
        end=END.strftime("%Y-%m-%d"),
        n_folds=2,
        method="grid",
        max_evals=1,
        journal=False,
    )

    assert payload["strategy"].startswith("draft:GenStrat:")
    assert payload["draft"]["journaled"] is False
    assert payload["n_trials_total"] == 2
    assert "gate_report" in payload
    assert "GenStrat" not in mcp_server.registry.STRATEGIES


# --- run_verdict parity -----------------------------------------------------
def test_run_verdict_returns_what_the_service_returns(built):
    """Parity by construction: the tool is a passthrough, not a second pipeline."""
    direct = analysis.run_verdict(
        MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=600, freq="1D")),
        "volume_spike",
        SYMBOLS,
        START,
        END,
    )
    over_mcp = _call(
        built,
        "run_verdict",
        strategy="volume_spike",
        symbols=SYMBOLS,
        start=START.strftime("%Y-%m-%d"),
        end=END.strftime("%Y-%m-%d"),
    )
    # The second run memoizes off the first — which is itself the parity proof.
    assert over_mcp["schema"] == direct["schema"]
    assert over_mcp["verdict"]["verdict"] == direct["verdict"]["verdict"]
    assert over_mcp["verdict"]["checks"].keys() == direct["verdict"]["checks"].keys()


# --- render_report ----------------------------------------------------------
def test_render_report_returns_a_self_contained_document(built):
    result = analysis.run_verdict(
        MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=600, freq="1D")),
        "volume_spike",
        SYMBOLS,
        START,
        END,
    )
    payload = _call(built, "render_report", kind="verdict", result=result)
    document = payload["html"]

    assert document.startswith("<!doctype html>")
    assert "http://" not in document and "https://" not in document
    assert payload["bytes"] == len(document.encode("utf-8"))


def test_render_report_shares_the_cli_renderer_rather_than_a_second_route(built):
    """030's escaping and self-containment rules apply here because it is literally
    the same function — there is no unescaped MCP render path to audit separately."""
    from tradeflow.analytics.htmlreport import render_html

    result = analysis.run_verdict(
        MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=600, freq="1D")),
        "volume_spike",
        SYMBOLS,
        START,
        END,
    )
    assert _call(built, "render_report", kind="verdict", result=result)["html"] == render_html(
        result, "verdict"
    )


def test_a_schema_mismatched_payload_fails_rather_than_half_rendering(built):
    with pytest.raises(Exception) as exc:
        _call(built, "render_report", kind="verdict", result={"schema": "verdict/0"})
    assert "verdict/1" in str(exc.value)


def test_hostile_content_stays_inert_through_the_mcp_path(built):
    result = analysis.run_verdict(
        MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=600, freq="1D")),
        "volume_spike",
        SYMBOLS,
        START,
        END,
    )
    result["inputs"]["universe"] = ["<script>alert(1)</script>"]
    document = _call(built, "render_report", kind="verdict", result=result)["html"]
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;" in document


# --- the trial-store tools --------------------------------------------------
def _seed_store(tmp_path):
    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl") as store:
        for i, (sharpe, dsr) in enumerate([(0.4, 0.1), (2.1, 0.2), (1.5, 0.9)], start=1):
            store.record(
                id=f"t{i}",
                kind="backtest",
                strategy="volume_spike",
                symbols=SYMBOLS,
                params={"fast": i},
                accounting=3,
                ts=f"2025-0{i}-01T00:00:00",
                oos_sharpe=sharpe,
                deflated_sharpe=dsr,
            )


def test_list_trials_returns_rows_and_the_full_match_count(built, _isolated_state):
    _seed_store(_isolated_state)
    payload = _call(built, "list_trials", strategy="volume_spike", all_accounting=True)
    assert payload["total"] == 3
    assert {r["id"] for r in payload["rows"]} == {"t1", "t2", "t3"}


def test_get_trial_reports_companions_as_null_not_zero(built, _isolated_state):
    _seed_store(_isolated_state)
    trial = _call(built, "get_trial", trial_id="t1")
    assert trial["id"] == "t1"
    assert trial["returns"] is None and trial["weights"] is None and trial["trades"] is None


def test_get_trial_on_an_unknown_id_explains_rather_than_crashes(built, _isolated_state):
    _seed_store(_isolated_state)
    assert "No trial with id" in _call(built, "get_trial", trial_id="nope")["error"]


def test_the_leaderboard_rules_survive_serialization(built, _isolated_state):
    """031's honesty requirements must live in the payload — an agent never sees the
    CLI's caveat line unless the data carries it."""
    _seed_store(_isolated_state)
    board = _call(built, "best_trials", strategy="volume_spike", all_accounting=True)

    assert board["rank_by"] == "dsr"
    assert board["rows"][0]["id"] == "t3"  # deflated winner, not the raw winner
    assert all("family_n_trials" in row for row in board["rows"])
    assert "DEFLATED" in board["caveat"]

    raw = _call(built, "best_trials", strategy="volume_spike", rank_by="sharpe", all_accounting=True)
    assert raw["rows"][0]["id"] == "t2"
    assert "RAW Sharpe" in raw["caveat"]


# --- the audit trail --------------------------------------------------------
def test_the_new_tools_are_audited_like_the_old_ones(built, _isolated_state):
    _seed_store(_isolated_state)
    _call(built, "list_trials")
    _call(built, "best_trials")

    lines = [json.loads(line) for line in (_isolated_state / "mcp_audit.jsonl").read_text().splitlines()]
    assert {"list_trials", "best_trials"} <= {r["tool"] for r in lines}


# --- the dependency contract ------------------------------------------------
def test_the_declared_mcp_constraint_excludes_the_incompatible_major():
    """The SDK's 2.x line removed `mcp.server.fastmcp.FastMCP`, which this server is
    built on. An unconstrained `mcp>=1.0` therefore resolved to a version that could
    not be imported, and `tradeflow-engine[mcp]` shipped broken — invisible locally,
    because the lockfile held a 1.x that worked."""
    from pathlib import Path

    import tomllib

    manifest = tomllib.loads(Path("pyproject.toml").read_text())
    constraint = manifest["project"]["optional-dependencies"]["mcp"]
    assert constraint == ["mcp>=1.0,<2"], constraint


def test_the_installed_sdk_satisfies_that_constraint():
    """Asserts the environment actually running these tests matches what a user
    would get, rather than whatever a lockfile pinned years ago."""
    from importlib.metadata import version

    major = int(version("mcp").split(".")[0])
    assert major == 1, f"mcp {version('mcp')} is outside the declared constraint"


def test_the_server_entry_point_the_sdk_provides_still_exists():
    """The specific import that broke. Cheap, and it fails loudly the day the SDK
    moves it again."""
    from mcp.server.fastmcp import FastMCP

    assert callable(FastMCP)
