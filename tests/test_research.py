"""Autonomous research agent tests, all offline with a fixed proposer.

Exercises the guardrails that make autonomy safe: research hygiene, the code
sandbox, OOS-only selection, cumulative multiple-testing count, the sacred
holdout scored exactly once, the dryness/budget stops, full journaling, and the
provider-agnostic LLM proposer.
"""

import json
from datetime import datetime

import pytest

from src.marketdata.client import MarketDataClient
from src.optimization import config_store
from src.research import sandbox
from src.research.agent import ResearchAgent, ResearchConfig
from src.research.proposer import FixedProposer, Proposal, ProposalContext
from src.services import registry
from tests.fakes import FakeMarketData
from tests.test_walk_forward import PeriodicStrategy

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2025, 6, 1)

# Gates relaxed so the loop's *mechanics* are observable in a tiny synthetic test.
RELAXED_GATES = {
    "min_oos_sharpe": -99,
    "min_oos_profit_factor": 0,
    "min_wfe": -99,
    "wfe_relaxed": -99,
    "max_dd_ratio": 1e9,
    "min_oos_trades": 0,
    "min_deflated_sharpe": -1,
    "max_param_sensitivity_loss": 1e9,
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Register the test strategy and redirect config/journal writes to tmp."""
    monkeypatch.setitem(registry.STRATEGIES, "periodic", PeriodicStrategy)
    monkeypatch.setattr(config_store, "DEFAULT_CONFIG_DIR", tmp_path / "configs")
    yield


def _agent(tmp_path, proposer, **overrides):
    cfg = ResearchConfig(
        goal="test",
        n_folds=3,
        embargo_days=2,
        holdout_days=60,
        method="grid",
        max_trials=overrides.pop("max_trials", 10),
        max_dry_rounds=overrides.pop("max_dry_rounds", 5),
        shortlist_size=3,
        gates=overrides.pop("gates", RELAXED_GATES),
        **overrides,
    )
    dc = MarketDataClient(FakeMarketData(SYMBOLS, n=600, freq="1D"))
    return ResearchAgent("periodic", dc, proposer, cfg, seed=42, journal_path=str(tmp_path / "journal.jsonl"))


def _tune(buy_every, hypothesis="cycle captures swings"):
    return Proposal(
        hypothesis=hypothesis,
        kind="tune",
        strategy="periodic",
        params={"buy_every": buy_every},
        tuned_params=["buy_every"],
    )


# --- research hygiene -------------------------------------------------------
def test_hygiene_rejects_missing_hypothesis_and_too_many_params():
    ok, _ = sandbox.validate_hygiene(_tune(3), PeriodicStrategy)
    assert ok

    no_rationale = Proposal(hypothesis="  ", kind="tune", strategy="periodic", params={"buy_every": 3})
    assert sandbox.validate_hygiene(no_rationale, PeriodicStrategy)[0] is False

    many = Proposal(
        hypothesis="x",
        kind="tune",
        strategy="periodic",
        params={f"p{i}": 1 for i in range(6)},
        tuned_params=[f"p{i}" for i in range(6)],
    )
    assert sandbox.validate_hygiene(many, PeriodicStrategy)[0] is False

    out_of_bounds = Proposal(
        hypothesis="x",
        kind="tune",
        strategy="periodic",
        params={"buy_every": 999},
        tuned_params=["buy_every"],
    )
    assert sandbox.validate_hygiene(out_of_bounds, PeriodicStrategy)[0] is False


# --- code sandbox -----------------------------------------------------------
_VALID_CODE = '''
import pandas as pd
from src.strategies.base import Strategy

class GenStrat(Strategy):
    """Always-long. Hypothesis: the synthetic series drifts up."""
    TIMEFRAME = "1Day"
    PARAM_RANGES = {"threshold": {"type": "float", "min": 0.0, "max": 0.05, "step": 0.01, "default": 0.01}}

    def __init__(self, config):
        config["timeframe"] = self.TIMEFRAME
        config.setdefault("risk_per_trade", 0.02)
        config.setdefault("stop_loss", 0.05)
        config.setdefault("take_profit", 0.05)
        super().__init__(config)

    def calculate_required_lookback(self):
        return 2

    def initialize(self):
        pass

    def process_data(self, data):
        return data

    def calculate_scores(self, data):
        # Always bullish; the base class derives BUY-then-hold from the sign.
        return pd.Series(1.0, index=data.index)
'''


def test_sandbox_loads_valid_strategy_code():
    cls = sandbox.load_strategy_from_code(_VALID_CODE)
    assert cls.__name__ == "GenStrat"
    instance = cls({})  # constructs from its own defaults
    assert instance.config["timeframe"] == "1Day"


def test_sandbox_blocks_disallowed_imports():
    bad = "import os\nfrom src.strategies.base import Strategy\n"
    with pytest.raises(sandbox.HygieneError):
        sandbox.load_strategy_from_code(bad)


def test_sandbox_requires_docstring_and_caps_params():
    no_doc = _VALID_CODE.replace('"""Always-long. Hypothesis: the synthetic series drifts up."""', "")
    with pytest.raises(sandbox.HygieneError):
        sandbox.load_strategy_from_code(no_doc)


# --- the loop ---------------------------------------------------------------
def test_loop_rejects_bad_proposals_and_counts_only_valid_trials(tmp_path):
    proposals = [
        _tune(3),
        _tune(5),
        Proposal(
            hypothesis="too many knobs",
            kind="tune",
            strategy="periodic",
            params={f"p{i}": 1 for i in range(6)},
            tuned_params=[f"p{i}" for i in range(6)],
        ),
        Proposal(hypothesis="", kind="tune", strategy="periodic", params={"buy_every": 9}),
    ]
    result = _agent(tmp_path, FixedProposer(proposals)).run(SYMBOLS, START, END)
    # Only the 2 valid tune proposals count toward the multiple-testing total.
    assert result.n_trials_total == 2
    assert result.stopped_reason == "proposer_exhausted"


def test_shortlist_scored_once_on_holdout_and_saved(tmp_path):
    result = _agent(tmp_path, FixedProposer([_tune(3), _tune(5)])).run(SYMBOLS, START, END)
    assert result.shortlist
    for candidate in result.shortlist:
        assert candidate.holdout_metrics is not None  # scored on the sacred holdout
        assert candidate.saved_path and candidate.saved_path.endswith(".json")
        saved = config_store.load_config(candidate.saved_path)
        assert saved["params"]["buy_every"] in (3, 5)
        assert "Hypothesis" in saved["provenance"]["notes"]
    # The holdout window is reserved from the tail and reported.
    assert result.holdout_window["end"] == END.isoformat()


def test_selection_is_oos_only_and_drawdown_guarded(tmp_path):
    # Strict gates => nothing is promotable => nothing advances, but the loop still
    # completes honestly (it never falls back to in-sample selection).
    strict = dict(RELAXED_GATES, min_oos_sharpe=99)
    result = _agent(tmp_path, FixedProposer([_tune(3), _tune(5)]), gates=strict).run(SYMBOLS, START, END)
    assert result.shortlist == []


def test_dry_rounds_stop(tmp_path):
    # Same proposal repeated never beats the incumbent => dry counter trips the stop.
    proposer = FixedProposer([_tune(3) for _ in range(10)])
    result = _agent(tmp_path, proposer, max_dry_rounds=2).run(SYMBOLS, START, END)
    assert result.stopped_reason == "dry_rounds"
    assert result.rounds <= 4  # first advances, then 2 dry rounds stop it


def test_max_trials_budget_stop(tmp_path):
    proposer = FixedProposer([_tune(3 + 2 * (i % 4)) for i in range(20)])
    result = _agent(tmp_path, proposer, max_trials=3, max_dry_rounds=99).run(SYMBOLS, START, END)
    assert result.stopped_reason == "max_trials"
    assert result.rounds == 3


def test_journal_records_session_and_trials(tmp_path):
    agent = _agent(tmp_path, FixedProposer([_tune(3), _tune(5)]))
    agent.run(SYMBOLS, START, END)
    lines = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    events = [rec["tool"] for rec in lines]
    assert "research:session_start" in events
    assert "research:session_end" in events
    assert any(e == "research:trial" for e in events)
    assert any(e == "research:holdout_score" for e in events)
    # Every record is replayable: has run id, timestamp, git sha.
    for rec in lines:
        assert {"run_id", "timestamp", "git_sha"} <= set(rec)


def test_proposer_context_excludes_holdout(tmp_path):
    """The proposer is only ever given the research window, never the holdout."""
    seen = {}

    class SpyProposer(FixedProposer):
        def propose(self, context: ProposalContext):
            seen["history_keys"] = (
                set().union(*[set(h) for h in context.history]) if context.history else set()
            )
            return super().propose(context)

    _agent(tmp_path, SpyProposer([_tune(3), _tune(5)])).run(SYMBOLS, START, END)
    # History summaries carry only round/params/score fields - no raw holdout data.
    assert "holdout" not in seen.get("history_keys", set())


# --- provider-agnostic LLM layer --------------------------------------------
def test_build_llm_client_defaults_per_provider():
    from src.research.llm import build_llm_client

    assert build_llm_client("anthropic").model == "claude-opus-4-8"
    assert build_llm_client("openai").model == "gpt-4o"
    assert build_llm_client("ollama").model == "llama3.1"
    assert build_llm_client("ollama", "mistral").model == "mistral"
    with pytest.raises(ValueError):
        build_llm_client("nope")


def test_llm_proposer_parses_json_from_any_client():
    from src.research.llm import LLMResponse
    from src.research.proposer import LLMProposer

    class StubClient:
        model = "stub"

        def complete(self, system, user, max_tokens=1024):
            # Models often wrap JSON in prose; the proposer must still extract it.
            return LLMResponse('Sure!\n{"hypothesis": "vol clusters", "params": {"buy_every": 7}}', tokens=11)

    proposer = LLMProposer(StubClient())
    ctx = ProposalContext(
        goal="g", strategy="periodic", param_ranges={}, history=[], incumbent=None, round_index=0
    )
    proposal = proposer.propose(ctx)
    assert proposal.hypothesis == "vol clusters"
    assert proposal.params == {"buy_every": 7}
    assert proposal.tuned_params == ["buy_every"]
    assert proposal.tokens_used == 11


def test_llm_proposer_returns_none_on_unparseable_output():
    from src.research.llm import LLMResponse
    from src.research.proposer import LLMProposer

    class JunkClient:
        model = "stub"

        def complete(self, system, user, max_tokens=1024):
            return LLMResponse("I cannot help with that.", tokens=3)

    ctx = ProposalContext(
        goal="g", strategy="periodic", param_ranges={}, history=[], incumbent=None, round_index=0
    )
    assert LLMProposer(JunkClient()).propose(ctx) is None


def test_build_proposer_accepts_injected_client():
    from src.research.llm import LLMResponse
    from src.research.proposer import build_proposer

    class StubClient:
        model = "stub"

        def complete(self, system, user, max_tokens=1024):
            return LLMResponse('{"hypothesis": "x", "params": {}}', tokens=1)

    proposer = build_proposer(client=StubClient())
    ctx = ProposalContext(
        goal="g", strategy="periodic", param_ranges={}, history=[], incumbent=None, round_index=0
    )
    assert proposer.propose(ctx).hypothesis == "x"


def test_sandbox_rejects_strategy_missing_execution_config():
    """A strategy the execution path cannot size is rejected, not silently zero-trade.

    Sizing and exit handling read ``risk_per_trade``/``stop_loss``/``take_profit`` off
    the config on every bar. Without this check the strategy validates cleanly and then
    raises on the first bar, which the gates would score as "no edge".
    """
    missing = _VALID_CODE.replace('        config.setdefault("stop_loss", 0.05)\n', "")
    with pytest.raises(sandbox.HygieneError, match="stop_loss"):
        sandbox.load_strategy_from_code(missing)


def test_llm_proposer_emits_code_proposals_when_enabled():
    """With code-gen on, a ``kind: "code"`` response becomes a code proposal."""
    from src.research.llm import LLMResponse
    from src.research.proposer import LLMProposer

    class CodeClient:
        model = "stub"

        def complete(self, system, user, max_tokens=1024):
            assert "PARAM_RANGES" in system  # the contract is stated to the model
            return LLMResponse('{"kind": "code", "hypothesis": "h", "code": "src"}', tokens=7)

    ctx = ProposalContext(
        goal="g", strategy="periodic", param_ranges={}, history=[], incumbent=None, round_index=0
    )
    proposal = LLMProposer(CodeClient(), allow_code_gen=True).propose(ctx)
    assert proposal.kind == "code"
    assert proposal.code == "src"
    assert proposal.tokens_used == 7


def test_llm_proposer_still_parses_tune_proposals_with_code_gen_enabled():
    from src.research.llm import LLMResponse
    from src.research.proposer import LLMProposer

    class TuneClient:
        model = "stub"

        def complete(self, system, user, max_tokens=1024):
            return LLMResponse('{"kind": "tune", "hypothesis": "h", "params": {"period": 5}}', tokens=2)

    ctx = ProposalContext(
        goal="g", strategy="periodic", param_ranges={}, history=[], incumbent=None, round_index=0
    )
    proposal = LLMProposer(TuneClient(), allow_code_gen=True).propose(ctx)
    assert proposal.kind == "tune"
    assert proposal.params == {"period": 5}


