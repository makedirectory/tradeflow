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
        scanner="demo_volume",
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
        "demo_trend",
        SYMBOLS,
        START,
        END,
    )
    over_mcp = _call(
        built,
        "run_verdict",
        strategy="demo_trend",
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
        "demo_trend",
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
        "demo_trend",
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
        "demo_trend",
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
                strategy="demo_trend",
                symbols=SYMBOLS,
                params={"fast": i},
                accounting=3,
                ts=f"2025-0{i}-01T00:00:00",
                oos_sharpe=sharpe,
                deflated_sharpe=dsr,
            )


def test_list_trials_returns_rows_and_the_full_match_count(built, _isolated_state):
    _seed_store(_isolated_state)
    payload = _call(built, "list_trials", strategy="demo_trend", all_accounting=True)
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
    board = _call(built, "best_trials", strategy="demo_trend", all_accounting=True)

    assert board["rank_by"] == "dsr"
    assert board["rows"][0]["id"] == "t3"  # deflated winner, not the raw winner
    assert all("family_n_trials" in row for row in board["rows"])
    assert "DEFLATED" in board["caveat"]

    raw = _call(built, "best_trials", strategy="demo_trend", rank_by="sharpe", all_accounting=True)
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


def test_the_mcp_log_and_the_journal_name_a_draft_the_same_way(built):
    """One hash, not two byte-identical ones.

    The MCP layer logged a `code_hash` from its own copy of the digest while the
    service journalled the trial under `draft:<ClassName>:<hash>` from another. They
    agreed only for as long as nobody touched either, and what they name is how a
    draft's calls are tied to the trials it spent.
    """
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

    assert payload["strategy"] == f"draft:GenStrat:{analysis.draft_code_hash(_VALID_CODE)}"
    assert payload["draft"]["code_hash"] == analysis.draft_code_hash(_VALID_CODE)


def test_trial_store_maintenance_is_deliberately_not_an_mcp_tool():
    """Not an oversight, and worth pinning so it does not become one. Quarantining
    evidence and retiring an era are operator decisions about a campaign's *record*,
    not run configuration: one changes what every later leaderboard and memo reports,
    the other moves a user's files. An agent that believes a trial is contaminated
    should say so and let a person act.

    Distinct from the trading wall — these are not dangerous capabilities, they are
    somebody else's decision.
    """
    from tradeflow.mcp import server as mcp_server

    assert mcp_server.OPERATOR_ONLY
    for name in ("archive", "mark_contaminated", "mark-contaminated"):
        assert name not in mcp_server.EXPOSED_TOOLS


def test_the_mode_that_can_place_real_orders_is_unreachable_over_mcp(built):
    """Small-real reaches a broker that really can trade. The wall is structural — the
    server builds only a data client — and this pins the decision as well as the
    mechanism, in every spelling, because a list of forbidden capabilities that omits
    the newest one reads as a list somebody checked.

    Deciding to spend real money is not a research step an agent takes on somebody's
    behalf. An agent that thinks execution telemetry is worth gathering should say so
    and let a person start the run.
    """
    from tradeflow.mcp import server as mcp_server

    registered = set(_tools(built))
    for name in ("small_real", "run_small_real", "start_small_real"):
        assert name in mcp_server.FORBIDDEN_TOOLS
        assert name not in registered
    assert "small-real" in mcp_server.OPERATOR_ONLY
    # And nothing registered merely mentions it under another name.
    assert not [name for name in registered if "small" in name and "real" in name]


# --- CLI/service surface parity, enumerated rather than remembered ------------------
#: Parameters a tool legitimately does not take because they are not knobs: the injected
#: data client, and the CLI's own config plumbing.
#:
#: Deliberately only two. It briefly held `risk_model` too, excused as "the handle the
#: server owns" — which was false: `run_verdict` exposes it and the CLI has
#: `--risk-model`, so the exception was hiding a real knob behind a rationale its own
#: sibling tool contradicted. It also held `current_weights`, which is exposed and so
#: could never have been flagged anyway. An escape hatch is the loophole, so it stays
#: small enough to check by eye.
_NOT_KNOBS = {"data_client", "config"}


def _tool_params(built, name):
    import asyncio

    tools = {t.name: t for t in asyncio.run(built.list_tools())}
    return set((tools[name].inputSchema.get("properties") or {}).keys())


def _service_params(fn):
    import inspect

    return {
        p
        for p, spec in inspect.signature(fn).parameters.items()
        if spec.kind not in (spec.VAR_POSITIONAL, spec.VAR_KEYWORD)
    } - _NOT_KNOBS


@pytest.mark.parametrize(
    "tool_name, service_name",
    [
        ("construct_portfolio", "construct_portfolio"),
        ("compute_alphas", "compute_alphas"),
        ("compute_attribution", "compute_attribution"),
    ],
)
def test_every_service_knob_is_exposed_or_deliberately_deferred(built, tool_name, service_name):
    """The parity point, enumerated from the service signature.

    An MCP surface that lags the CLI is not merely less convenient — a knob that cannot
    be set on a surface is a setting that silently never applies there, and nothing says
    so. `construct_portfolio` exposed nine of the service's parameters while the CLI
    carried about thirty.

    Every service parameter must now be either reachable as a tool argument or listed in
    `DEFERRED_PARAMS` with a reason. Read from the signature rather than a hand-written
    list, so a parameter added to the service later fails here instead of quietly
    becoming unreachable — which is exactly how the gap this closes opened.
    """
    from tradeflow.services import analysis

    exposed = _tool_params(built, tool_name)
    service = _service_params(getattr(analysis, service_name))
    unreachable = service - exposed - set(mcp_server.DEFERRED_PARAMS)

    assert not unreachable, (
        f"{tool_name} cannot set {sorted(unreachable)}.\n"
        "If the knob is ungated, expose it. If it belongs to an evidence-gated family "
        "(conditional risk, the Black-Litterman posterior, the aim policy), it must NOT "
        "be exposed — add it to DEFERRED_PARAMS with its reason. Exposing a gated knob "
        "makes this surface the easy way to switch on a feature nothing has validated."
    )


def test_no_evidence_gated_knob_is_reachable_from_an_agent():
    """The rule that wins where parity and evidence gating disagree.

    A feature whose own adoption gate does not clear must not find in the agent surface
    an easier way to be switched on than the one a human reads a warning before using.
    Being reachable is not the same as being validated, and an agent acts on a
    description at machine speed with every call costing a journaled trial.
    """
    # Every gated name, not a subset. The first version pinned six of the ten, and the
    # other four could each be deleted from DEFERRED_PARAMS *and* exposed with the suite
    # still green — because the absence check below iterates the dict, so removing an
    # entry removes its own guard. A list that shrinks silently is not a guard.
    assert set(mcp_server.DEFERRED_PARAMS) == {
        "conditional",
        "conditional_lambda",
        "conditional_method",
        "posterior",
        "posterior_ic",
        "posterior_t_eff",
        "posterior_tau",
        "policy",
        "trade_rate",
        "decay_lookback_days",
    }
    for param, reason in mcp_server.DEFERRED_PARAMS.items():
        assert reason.strip(), f"{param} is deferred with no reason"
        assert "evidence-gated" in reason, f"{param}'s reason does not say why it is withheld"


def test_the_deferred_knobs_are_absent_from_every_registered_tool(built):
    """Both directions: the list is only worth keeping if it describes reality."""
    import asyncio

    for tool in asyncio.run(built.list_tools()):
        params = set((tool.inputSchema.get("properties") or {}).keys())
        leaked = params & set(mcp_server.DEFERRED_PARAMS)
        assert not leaked, f"{tool.name} exposes deferred knob(s) {sorted(leaked)}"


def test_the_gated_ab_tools_are_not_exposed(built):
    """`run_conditional_risk_ab`, `run_policy_ab` and `evaluate_conditional_risk` exist
    as services and stay off this surface: each one exists to evaluate a feature whose
    gate has not cleared, so exposing it makes the agent the judge of its own gate."""
    import asyncio

    registered = {t.name for t in asyncio.run(built.list_tools())}
    for name in ("run_conditional_risk_ab", "run_policy_ab", "evaluate_conditional_risk"):
        assert name not in registered
        assert name not in mcp_server.EXPOSED_TOOLS


def test_attribution_is_exposed_and_journals_nothing(built):
    """The one genuinely missing read-only diagnostic. It answers where realized return
    came from, which is a different question from whether the signal ranks names."""
    import asyncio

    registered = {t.name for t in asyncio.run(built.list_tools())}
    assert "compute_attribution" in registered
    assert "compute_attribution" in mcp_server.EXPOSED_TOOLS
    assert "compute_attribution" not in mcp_server.JOURNALING_TOOLS


def test_an_exposed_knob_actually_reaches_the_service(monkeypatch):
    """Presence in the schema is not reach, and the original defect was exactly that gap.

    `compute_alphas` advertised a `neutralized_against` field and could never populate
    it, because the tool had no parameter to pass. Wiring one is only half the fix: a
    parameter that exists in the signature and is dropped from the forwarding dict looks
    identical from the schema, and a mutation doing precisely that passed every other
    test here.

    So this calls the tools for real and asserts the service was handed the values.
    """
    import asyncio

    from tradeflow.services import analysis

    seen = {}

    def _spy(name, real):
        def recorder(*args, **kwargs):
            seen[name] = kwargs
            return {"ok": True}

        return recorder

    monkeypatch.setattr(analysis, "compute_alphas", _spy("compute_alphas", None))
    monkeypatch.setattr(analysis, "construct_portfolio", _spy("construct_portfolio", None))
    monkeypatch.setattr(analysis, "compute_attribution", _spy("compute_attribution", None))

    built = mcp_server.build_server(
        data_client=MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=200, freq="1D"))
    )

    asyncio.run(
        built.call_tool(
            "compute_alphas",
            {
                "strategy": "demo_trend",
                "symbols": SYMBOLS,
                "as_of": "2025-06-01",
                "neutralize_factors": ["market", "size"],
            },
        )
    )
    assert seen["compute_alphas"].get("neutralize_factors") == ["market", "size"]

    asyncio.run(
        built.call_tool(
            "construct_portfolio",
            {
                "strategy": "demo_trend",
                "symbols": SYMBOLS,
                "as_of": "2025-06-01",
                "book": "market_neutral",
                "gross_leverage": 1.6,
                "short_max_weight": 0.1,
                "neutralize_factors": ["momentum"],
                "min_weight": 0.01,
                "commission_bps": 2.5,
            },
        )
    )
    forwarded = seen["construct_portfolio"]
    assert forwarded.get("book") == "market_neutral"
    assert forwarded.get("gross_leverage") == 1.6
    assert forwarded.get("short_max_weight") == 0.1
    assert forwarded.get("neutralize_factors") == ["momentum"]
    assert forwarded.get("min_weight") == 0.01
    assert forwarded.get("commission_bps") == 2.5

    asyncio.run(
        built.call_tool(
            "compute_attribution",
            {
                "strategy": "demo_trend",
                "symbols": SYMBOLS,
                "start": "2024-01-02",
                "end": "2025-06-01",
                "neutralize_factors": ["volatility"],
                "n_trials": 12,
            },
        )
    )
    assert seen["compute_attribution"].get("neutralize_factors") == ["volatility"]
    assert seen["compute_attribution"].get("n_trials") == 12


def test_an_omitted_knob_is_not_forwarded_at_all(monkeypatch):
    """Both directions. Omitted must mean "the service keeps its own default", not "this
    surface restates one" — a second copy of a default is a second thing to keep in
    step, and the two would drift silently."""
    import asyncio

    from tradeflow.services import analysis

    seen = {}
    monkeypatch.setattr(analysis, "construct_portfolio", lambda *a, **k: seen.update(k) or {"ok": True})
    built = mcp_server.build_server(
        data_client=MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=200, freq="1D"))
    )

    asyncio.run(
        built.call_tool(
            "construct_portfolio",
            {"strategy": "demo_trend", "symbols": SYMBOLS, "as_of": "2025-06-01"},
        )
    )

    for absent in ("book", "gross_leverage", "neutralize_factors", "commission_bps", "min_weight"):
        assert absent not in seen, f"{absent} was forwarded despite not being set"


# --- connection and discovery ------------------------------------------------------
def test_every_documented_connection_recipe_covers_both_kinds_of_copy():
    """The installed-copy-versus-checkout point, applied to connection instructions.

    An installed reader has no `main.py` and no checkout to `cwd` into, so a page that
    offers only `uv run python main.py mcp` sends them to a file that was never there —
    the failure this project has already fixed in several other messages. Every page
    that shows a client config must show both forms.

    The requirement is deliberately one-directional, and narrow. A page may show *only*
    the installed form — `getting-started.md` does, and that is right for a reader who
    arrived via `uv tool install` and has no checkout to point at. A container recipe is
    a third thing again, with its own prerequisites. What no page may do is offer the
    checkout form *alone*, because that is the reader who cannot follow it.

    Checked against the pages themselves rather than a remembered list of them, so a new
    page carrying a recipe is covered the day it is added.
    """
    import pathlib
    import re

    roots = [pathlib.Path("README.md"), *pathlib.Path("docs/content").rglob("*.md")]
    carriers = [p for p in roots if "mcpServers" in p.read_text()]
    assert carriers, "no page documents an MCP client config any more"

    checked = 0
    for page in carriers:
        text = page.read_text()
        # Only pages offering a *checkout-anchored local* recipe are in scope. A
        # container recipe (`command: docker`) is a third deployment with its own
        # prerequisites and no local path to get wrong.
        if not re.search(r'"command":\s*"uv"', text):
            continue
        checked += 1
        assert re.search(r'"command":\s*"tradeflow"', text), (
            f"{page} offers a checkout recipe with no installed-copy one beside it. "
            "`uv run python main.py` sends an installed reader to a file that was "
            "never there."
        )
    assert checked, "no page offers a checkout recipe any more — is this check stale?"


def test_the_prerequisites_are_stated_where_a_client_is_registered():
    """Both prerequisites fail the same way from a client — the server exits before
    answering, and the client shows only that it would not start. Verified by
    handshaking an installed copy with no credentials: the pipe closes before the first
    response. So the pages that tell someone to register it have to say so."""
    import pathlib

    guide = pathlib.Path("docs/content/engineering/mcp-server.md").read_text().lower()
    assert "credentials" in guide and "extra" in guide
    assert "would not start" in guide or "never appears" in guide


def test_the_audit_log_records_the_knobs_a_proposal_was_made_with(tmp_path, monkeypatch):
    """The audit log exists so a human can replay what an agent did.

    `construct_portfolio` grew from nine parameters to twenty-seven while its audit
    record stayed at four, so a market-neutral, leveraged, cost-adjusted proposal audited
    identically to a default one — the record present but not saying what was decided.
    """
    import asyncio
    import json

    from tradeflow.services import audit

    monkeypatch.setattr(audit, "DEFAULT_AUDIT_PATH", tmp_path / "audit.jsonl")
    built = mcp_server.build_server(
        data_client=MarketDataClient(FakeMarketData([*SYMBOLS, "SPY"], n=300, freq="1D"))
    )

    asyncio.run(
        built.call_tool(
            "construct_portfolio",
            {
                "strategy": "demo_trend",
                "symbols": SYMBOLS,
                "as_of": "2025-06-01",
                "book": "market_neutral",
                "gross_leverage": 2.0,
                "short_max_weight": 0.1,
                "commission_bps": 25.0,
            },
        )
    )

    line = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])
    recorded = line.get("inputs", {})
    assert recorded.get("book") == "market_neutral"
    assert recorded.get("gross_leverage") == 2.0
    assert recorded.get("commission_bps") == 25.0
    # And an unset knob is not recorded as though it had been chosen.
    assert "posterior" not in recorded and "min_weight" not in recorded


# --- an argument this surface does not accept is an error, not a silence ------------
def _dispatch(built, tool, args):
    """Straight through `call_tool`, which is where the guard lives."""
    import asyncio

    return asyncio.run(built.call_tool(tool, args))


def test_a_withheld_knob_is_refused_and_says_why(built):
    """The gap the parity work left open. The framework validates against the schema and
    then *drops* undeclared keys, so an agent told to "turn on conditional risk" passed
    `conditional="ewma"`, got a portfolio back, and reported that it had done so. The
    wall held; the agent's account of what it did did not.

    The reason is quoted at the moment it is needed rather than referred to — this is
    where an agent actually has to learn that the knob is withheld and not broken.
    """
    with pytest.raises(Exception) as exc:
        _dispatch(
            built,
            "construct_portfolio",
            {
                "strategy": "demo_trend",
                "symbols": SYMBOLS,
                "as_of": "2025-06-01",
                "conditional": "ewma",
            },
        )

    message = str(exc.value)
    assert "does not accept: conditional" in message
    assert "withheld from this surface" in message
    assert "adoption gate does not clear" in message


def test_a_mistyped_argument_is_refused_and_the_near_miss_named(built):
    """A typo has the identical shape: the default silently stays in place. A human
    would see it in the output and wonder; an agent has nothing to wonder at."""
    with pytest.raises(Exception) as exc:
        _dispatch(
            built,
            "construct_portfolio",
            {"strategy": "demo_trend", "symbols": SYMBOLS, "as_of": "2025-06-01", "targt_te": 0.04},
        )

    message = str(exc.value)
    assert "targt_te" in message
    assert "Did you mean target_te" in message


def test_every_declared_argument_is_still_accepted(built):
    """Both directions, and the one that matters for a guard at dispatch: it must not
    reject the calls it exists to permit. Driven from each tool's own schema, so a knob
    added later is covered without being remembered here."""
    import asyncio

    tools = {t.name: t for t in asyncio.run(built.list_tools())}
    declared = set((tools["construct_portfolio"].inputSchema.get("properties") or {}).keys())

    # Every ungated knob, passed at once, must get through the guard.
    args = {
        "strategy": "demo_trend",
        "symbols": SYMBOLS,
        "as_of": "2025-06-01",
        "book": "market_neutral",
        "gross_leverage": 1.6,
        "short_max_weight": 0.1,
        "neutralize_factors": ["market"],
        "min_weight": 0.01,
        "commission_bps": 2.0,
        "risk_model": "shrinkage",
    }
    assert set(args) <= declared, "the test is passing something the tool never declared"
    _dispatch(built, "construct_portfolio", args)  # must not raise


def test_a_call_with_no_arguments_is_untouched(built):
    _dispatch(built, "get_metrics_glossary", {})


def test_a_tool_that_declares_no_arguments_still_refuses_one(built):
    """A no-argument tool declares `properties: {}`, which is a real answer — it accepts
    nothing — and must be distinguished from a schema that could not be read at all.
    Passing an argument to such a tool is exactly as silent a mistake as passing an
    unknown one anywhere else."""
    with pytest.raises(Exception) as exc:
        _dispatch(built, "get_metrics_glossary", {"bogus": 1})

    assert "does not accept: bogus" in str(exc.value)


def test_an_unknown_tool_still_fails_the_frameworks_way(built):
    """The guard must not invent a second way for a call to fail. A tool nobody
    registered is the framework's business."""
    with pytest.raises(Exception) as exc:
        _dispatch(built, "no_such_tool", {"x": 1})

    assert "no_such_tool" in str(exc.value)


def test_an_unreadable_schema_does_not_turn_into_a_call_failure(built, monkeypatch):
    """Refusing a call on a guess about what a tool accepts would be worse than the
    silence this replaces, so any doubt passes through."""
    monkeypatch.setattr(
        built._tool_manager, "get_tool", lambda name: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    # A *legitimate* argument, so the assertion can tell "passed through" from
    # "refused". With `{}` this test could not see the difference: nothing is unknown in
    # an empty mapping, so a guard that had decided the tool accepts nothing would look
    # identical to one that stood aside.
    with pytest.raises(Exception) as exc:
        _dispatch(built, "get_trial", {"trial_id": "whatever"})

    assert "does not accept" not in str(exc.value)
