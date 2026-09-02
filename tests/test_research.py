"""Autonomous research agent tests, all offline with a fixed proposer.

Exercises the guardrails that make autonomy safe: research hygiene, the code
sandbox, OOS-only selection, cumulative multiple-testing count, the sacred
holdout scored exactly once, the dryness/budget stops, full journaling, and the
provider-agnostic LLM proposer.
"""

import json
from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tests.test_walk_forward import PeriodicStrategy
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.optimization import config_store
from tradeflow.research import sandbox
from tradeflow.research.agent import ResearchAgent, ResearchConfig
from tradeflow.research.proposer import FixedProposer, Proposal, ProposalContext
from tradeflow.services import registry

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
from tradeflow.strategies.base import Strategy

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


_VALID_SCANNER_CODE = '''
import pandas as pd
from tradeflow.scanners.base import SCANNER_BUY, SCANNER_HOLD, ScannerStrategy

class GenScanner(ScannerStrategy):
    """Flags up bars. Hypothesis: attention follows simple upward pressure."""
    TIMEFRAME = "1Day"
    PARAM_RANGES = {"min_move": {"type": "float", "min": 0.0, "max": 5.0, "step": 0.5, "default": 0.5}}

    def initialize(self):
        pass

    def process_data(self, data):
        enriched = data.copy()
        enriched["move"] = (enriched["close"] - enriched["open"]) / enriched["open"] * 100
        return enriched

    def generate_signals_df(self, data):
        out = pd.DataFrame(index=data.index)
        out["signal"] = SCANNER_HOLD
        out["signal_strength"] = data["move"].abs()
        out.loc[data["move"] > self.config["min_move"], "signal"] = SCANNER_BUY
        return out
'''


def test_sandbox_loads_valid_scanner_code():
    cls = sandbox.load_scanner_from_code(_VALID_SCANNER_CODE)
    assert cls.__name__ == "GenScanner"
    assert cls({}).config["min_move"] == 0.5


def test_sandbox_requires_scanner_signal_strength():
    bad = _VALID_SCANNER_CODE.replace('        out["signal_strength"] = data["move"].abs()\n', "")
    with pytest.raises(sandbox.HygieneError):
        sandbox.load_scanner_from_code(bad)


def test_sandbox_blocks_disallowed_imports():
    # `os` is denied outright; the broker layer is a real module that is simply not
    # on the allowlist — generated code may reach the strategy base, never a venue.
    bad = "import os\nfrom tradeflow.brokers.base import Broker\n"
    with pytest.raises(sandbox.HygieneError):
        sandbox.load_strategy_from_code(bad)


def test_a_malformed_strategy_param_spec_is_a_rejection_not_a_crash():
    """`PARAM_RANGES = {"threshold": 0.01}` is what generated code actually writes.

    Reading it raised a bare `TypeError` out of `ParameterSpace` - straight through
    validators whose entire contract is to answer "is this valid?" with a verdict.
    Every rejection has to leave the sandbox as a HygieneError or callers are left
    enumerating the exception types its internals happen to use.
    """
    bad = _VALID_CODE.replace(
        '{"type": "float", "min": 0.0, "max": 0.05, "step": 0.01, "default": 0.01}', "0.01"
    )
    with pytest.raises(sandbox.HygieneError, match="threshold"):
        sandbox.load_strategy_from_code(bad)


def test_a_malformed_scanner_param_spec_is_a_rejection_not_a_crash():
    """Same defect on the scanner path, which reaches it twice - once in the spec
    scan and again in `_defaults()` reading `spec["default"]`."""
    bad = _VALID_SCANNER_CODE.replace(
        '{"type": "float", "min": 0.0, "max": 5.0, "step": 0.5, "default": 0.5}', "0.5"
    )
    with pytest.raises(sandbox.HygieneError, match="min_move"):
        sandbox.load_scanner_from_code(bad)


def test_a_well_formed_param_spec_is_still_accepted():
    """The other direction. A shape check that rejects every spec is not a check."""
    assert sandbox.load_strategy_from_code(_VALID_CODE).__name__ == "GenStrat"
    assert sandbox.load_scanner_from_code(_VALID_SCANNER_CODE).__name__ == "GenScanner"


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


def test_duplicate_tune_proposal_is_deduped_via_the_trial_store(tmp_path):
    """A repeated exact config is the same lottery ticket checked twice - reject
    it before it burns a second walk-forward, rather than letting it
    count toward the campaign's multiple-testing total again."""
    result = _agent(tmp_path, FixedProposer([_tune(3), _tune(3), _tune(5)]), max_dry_rounds=99).run(
        SYMBOLS, START, END
    )
    # Only the two *distinct* configs count as trials — the repeat is a dedup hit.
    assert result.n_trials_total == 2

    lines = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    rejects = [rec["inputs"] for rec in lines if rec["tool"] == "research:reject"]
    assert any("duplicate" in (r.get("reason") or "") for r in rejects)


def test_research_trials_are_recorded_in_the_trial_store(tmp_path):
    from tradeflow.engine.backtest import ACCOUNTING_VERSION

    agent = _agent(tmp_path, FixedProposer([_tune(3), _tune(5)]))
    agent.run(SYMBOLS, START, END)

    rows = agent.trial_store.query(strategy="periodic", kind="research")
    assert len(rows) == 2
    assert agent.trial_store.family_count("periodic", SYMBOLS, ACCOUNTING_VERSION) == 2


# --- bootstrap skill ------------------------------------------------------------
def test_research_trials_persist_oos_return_series_in_the_store(tmp_path):
    """The trial store must contain a genuine OOS
    return series per trial (not just summary floats) for Reality Check to
    resample - every recorded research trial gets one."""
    agent = _agent(tmp_path, FixedProposer([_tune(3), _tune(5)]))
    agent.run(SYMBOLS, START, END)

    rows = agent.trial_store.query(strategy="periodic", kind="research")
    assert len(rows) == 2
    panel = agent.trial_store.returns_panel("periodic", SYMBOLS, rows[0]["accounting"], min_overlap=5)
    assert panel["n_with_returns"] == 2


def test_bootstrap_skill_annotation_is_advisory_and_off_by_default(tmp_path):
    """Off by default: no candidate gets a bootstrap_skill annotation unless the
    config explicitly asks for it - and turning it on never changes which
    candidates advance (guardrail 1, OOS-only fitness, stays untouched)."""
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off = _agent(tmp_path / "off", FixedProposer([_tune(3), _tune(5)])).run(SYMBOLS, START, END)
    assert off.shortlist
    assert all(c.bootstrap_skill is None for c in off.shortlist)

    on = _agent(tmp_path / "on", FixedProposer([_tune(3), _tune(5)]), bootstrap_skill=True).run(
        SYMBOLS, START, END
    )
    assert on.shortlist
    assert all(c.bootstrap_skill is not None for c in on.shortlist)
    for c in on.shortlist:
        assert "own" in c.bootstrap_skill and "family" in c.bootstrap_skill
    assert [c.params for c in on.shortlist] == [c.params for c in off.shortlist]


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
    from tradeflow.research.llm import build_llm_client

    assert build_llm_client("anthropic").model == "claude-opus-4-8"
    assert build_llm_client("openai").model == "gpt-4o"
    assert build_llm_client("ollama").model == "llama3.1"
    assert build_llm_client("ollama", "mistral").model == "mistral"
    with pytest.raises(ValueError):
        build_llm_client("nope")


def test_llm_proposer_parses_json_from_any_client():
    from tradeflow.research.llm import LLMResponse
    from tradeflow.research.proposer import LLMProposer

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
    from tradeflow.research.llm import LLMResponse
    from tradeflow.research.proposer import LLMProposer

    class JunkClient:
        model = "stub"

        def complete(self, system, user, max_tokens=1024):
            return LLMResponse("I cannot help with that.", tokens=3)

    ctx = ProposalContext(
        goal="g", strategy="periodic", param_ranges={}, history=[], incumbent=None, round_index=0
    )
    assert LLMProposer(JunkClient()).propose(ctx) is None


def test_build_proposer_accepts_injected_client():
    from tradeflow.research.llm import LLMResponse
    from tradeflow.research.proposer import build_proposer

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
    from tradeflow.research.llm import LLMResponse
    from tradeflow.research.proposer import LLMProposer

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
    from tradeflow.research.llm import LLMResponse
    from tradeflow.research.proposer import LLMProposer

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


def test_observer_sees_events_and_cannot_break_the_loop(tmp_path):
    """The narration hook is observational: it receives events and its errors are contained."""
    seen = []

    def boom(event, payload):
        seen.append(event)
        raise RuntimeError("observer exploded")

    agent = ResearchAgent(
        "periodic",
        MarketDataClient(FakeMarketData(SYMBOLS, n=600, freq="1D")),
        FixedProposer([_tune(3)]),
        ResearchConfig(goal="g", n_folds=3, max_trials=1, gates=RELAXED_GATES),
        seed=42,
        journal_path=str(tmp_path / "journal.jsonl"),
        observer=boom,
    )
    result = agent.run(SYMBOLS, START, END)  # must not raise
    assert "session_start" in seen and "session_end" in seen
    assert result.rounds == 1


# --- the validators must accept what this package already ships ---------------
def _shipped_source(cls):
    import inspect

    return inspect.getsource(inspect.getmodule(cls))


#: The one rejection a curated shipped strategy is allowed to draw. The tunable-param
#: cap is deliberately stricter for an unreviewed draft than for code that went
#: through review, and two shipped strategies declare six searchable params. Any
#: *other* rejection means the validator is measuring itself.
_DRAFT_ONLY_RULE = "searchable params"


def test_every_shipped_strategy_satisfies_the_draft_contract():
    """A fixture written to pass a validator proves nothing about the validator.

    All three shipped strategies were rejected outright - they annotate their
    PARAM_RANGES, and `typing` was not on the import allowlist, which grants no
    capability and was never what the list is for. The suite stayed green because
    every fixture had been hand-written to avoid the import.
    """
    from tradeflow.services.registry import BUILTIN_STRATEGIES

    accepted = []
    for name, cls in BUILTIN_STRATEGIES.items():
        try:
            sandbox.load_strategy_from_code(_shipped_source(cls), class_name=cls.__name__)
            accepted.append(name)
        except sandbox.HygieneError as exc:
            assert _DRAFT_ONLY_RULE in str(exc), f"{name} rejected by the draft validator: {exc}"
    assert accepted, "no shipped strategy passed at all — the contract itself is broken"


def test_every_shipped_scanner_satisfies_the_draft_contract():
    """Same rule for scanners, which is where the sample-size defect also lived."""
    from tradeflow.services.registry import BUILTIN_SCANNERS

    for name, cls in BUILTIN_SCANNERS.items():
        try:
            sandbox.load_scanner_from_code(_shipped_source(cls), class_name=cls.__name__)
        except sandbox.HygieneError as exc:
            raise AssertionError(f"shipped scanner {name} fails its own validator: {exc}") from exc


def test_the_scanner_sample_is_long_enough_for_the_scanner_it_validates():
    """It was a fixed five bars against a scanner needing eleven, so the frame the
    contract was checked on was entirely warm-up."""
    from tradeflow.services.registry import BUILTIN_SCANNERS

    cls = BUILTIN_SCANNERS["volume"]
    instance = cls({p: s["default"] for p, s in cls.PARAM_RANGES.items()})
    instance.initialize()

    sample = sandbox._scanner_sample(cls, instance)
    assert len(sample) > instance.required_data_points()


def test_the_signal_vocabulary_check_sees_actionable_labels():
    """The check could not fail. On five bars a real scanner emits only HOLD, and
    `{HOLD}` is a subset of the vocabulary — so a scanner emitting a bogus label once
    its indicators warmed up would have passed."""
    from tradeflow.services.registry import BUILTIN_SCANNERS

    cls = BUILTIN_SCANNERS["volume"]
    instance = cls({p: s["default"] for p, s in cls.PARAM_RANGES.items()})
    instance.initialize()

    sample = sandbox._scanner_sample(cls, instance)
    emitted = set(instance.generate_signals_df(instance.process_data(sample))["signal"].dropna())
    assert emitted - {"SCANNER_HOLD"}, f"sample only ever produced {emitted}"


def test_a_scanner_emitting_an_unknown_label_is_rejected():
    """The other direction: the check must be able to fail, not merely to pass."""
    bad = _VALID_SCANNER_CODE.replace('out["signal"] = SCANNER_HOLD', 'out["signal"] = "MAYBE"')
    with pytest.raises(sandbox.HygieneError, match="unknown scanner signals"):
        sandbox.load_scanner_from_code(bad)
