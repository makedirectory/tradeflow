"""What the research loop does with a round it cannot evaluate, and what it keeps.

All offline: a scripted model client drives the real proposer and agent over fake bars.
"""

import json

import pytest

from tests.test_research import (
    _VALID_CODE,
    END,
    RELAXED_GATES,
    START,
    SYMBOLS,
    _agent,
    _isolate,  # noqa: F401 - the autouse fixture registers the test strategy
    _tune,
)
from tradeflow.costs import ParametricCostModel
from tradeflow.optimization import config_store
from tradeflow.research.agent import Candidate, prior_holdout_looks
from tradeflow.research.llm import LLMResponse
from tradeflow.research.proposer import FixedProposer, LLMProposer, Proposal
from tradeflow.research.provider_errors import ModelProviderError
from tradeflow.utils.errors import GeneratedCodeRefused, RunRefused

#: Always-long code never closes a trade, and a run with no OOS trades cannot be promotable.
#: This alternates conviction so the generated strategy actually trades.
_TRADING_CODE = _VALID_CODE.replace(
    "return pd.Series(1.0, index=data.index)",
    "return pd.Series([1.0 if (i // 5) % 2 == 0 else -1.0 for i in range(len(data))], index=data.index)",
)


class ScriptedClient:
    """Answers each call with the next canned reply - no network, no key."""

    model = "scripted"

    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, system, user, max_tokens=1024):
        return LLMResponse(self._replies.pop(0), tokens=5)


class ScriptedProposer(LLMProposer):
    """The real reply parser over a script; a provider with nothing left says so with None."""

    def propose(self, context):
        if not self.client._replies:
            return None
        return super().propose(context)


def _reply(params, hypothesis="cycle captures swings"):
    return json.dumps({"hypothesis": hypothesis, "params": params})


def _records(tmp_path):
    return [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]


def _rejects(tmp_path):
    return [r["inputs"] for r in _records(tmp_path) if r["tool"] == "research:reject"]


def _run_llm(tmp_path, replies, **overrides):
    agent = _agent(tmp_path, ScriptedProposer(ScriptedClient(replies)), **overrides)
    return agent, agent.run(SYMBOLS, START, END)


# --- a reply that is not a proposal is a spent round, not the end of the session ---
def test_an_unparseable_reply_is_a_journaled_rejection_and_the_session_continues(tmp_path):
    _, result = _run_llm(
        tmp_path, ["I think you should try a faster EMA!", _reply({"buy_every": 5}), _reply({"buy_every": 7})]
    )
    assert result.n_evaluated == 2
    assert result.n_rejected == 1
    assert [r["reason"] for r in _rejects(tmp_path)] == ["unparseable reply"]


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        ([1, 2], "params must be an object"),
        (None, "params must be an object"),
        ({"buy_every": "ten"}, "is not a number"),
        ({"buy_every": 5.5}, "is not an integer"),
        ({"buy_every": True}, "is a boolean"),
    ],
)
def test_well_formed_json_with_the_wrong_types_is_rejected_with_its_reason(tmp_path, params, fragment):
    _, result = _run_llm(tmp_path, [_reply(params), _reply({"buy_every": 5})])
    assert result.n_evaluated == 1
    assert fragment in _rejects(tmp_path)[0]["reason"]
    # The session still ends with its own record.
    assert _records(tmp_path)[-1]["tool"] == "research:session_end"


def test_an_integral_float_for_an_int_parameter_is_recorded_as_an_int(tmp_path):
    _, result = _run_llm(tmp_path, [_reply({"buy_every": 5.0})])
    assert result.n_evaluated == 1
    saved = config_store.load_config(result.shortlist[0].saved_path)
    assert saved["params"]["buy_every"] == 5 and isinstance(saved["params"]["buy_every"], int)


# --- the budget counts what its name says ---
def test_max_trials_counts_evaluated_trials_not_proposals(tmp_path):
    bad = Proposal(hypothesis="", kind="tune", strategy="periodic", params={"buy_every": 9})
    proposer = FixedProposer([bad, bad, _tune(3), _tune(5), _tune(7), _tune(9)])
    result = _agent(tmp_path, proposer, max_trials=2, max_dry_rounds=99).run(SYMBOLS, START, END)
    assert result.n_evaluated == 2 and result.n_rejected == 2
    assert result.rounds == 4
    assert result.stopped_reason == "max_trials"


def test_a_rejected_round_names_its_round_and_reason_in_the_journal(tmp_path):
    bad = Proposal(hypothesis="", kind="tune", strategy="periodic", params={"buy_every": 9})
    _agent(tmp_path, FixedProposer([_tune(3), bad])).run(SYMBOLS, START, END)
    (rejected,) = _rejects(tmp_path)
    assert rejected["round"] == 2 and "missing hypothesis" in rejected["reason"]


# --- one measure, for the comparison and for the ranking ---
def test_a_round_is_compared_with_the_incumbents_median_not_its_aggregate(tmp_path):
    agent = _agent(tmp_path, FixedProposer([]))
    incumbent = Candidate(
        id="i",
        lineage="root",
        hypothesis="h",
        kind="tune",
        strategy="periodic",
        params={},
        # The stitched aggregate is far above the median the gate uses.
        oos_metrics={"sharpe_ratio": 5.0, "max_drawdown": 0.1},
        gate_report={},
        n_trials_at_selection=1,
        oos_sharpe=1.0,
    )
    assert agent._beats_incumbent(1.5, 0.1, incumbent) is True
    assert agent._beats_incumbent(0.5, 0.1, incumbent) is False


# --- a session that was started always ends ---
def test_a_provider_failure_mid_session_still_writes_a_session_end(tmp_path):
    class Dies(FixedProposer):
        def propose(self, context):
            if context.round_index >= 1:
                raise ModelProviderError("stub", "quota", "credit exhausted")
            return super().propose(context)

    agent = _agent(tmp_path, Dies([_tune(3), _tune(5)]))
    with pytest.raises(ModelProviderError):
        agent.run(SYMBOLS, START, END)
    end = _records(tmp_path)[-1]
    assert end["tool"] == "research:session_end"
    assert end["inputs"]["stopped_reason"] == "provider_error:quota"
    assert end["inputs"]["rounds"] == 1 and end["inputs"]["holdout_scored"] is False
    assert agent.progress["evaluated"] == 1


# --- costs are part of what a round measured ---
def test_the_cost_basis_is_part_of_a_rounds_recorded_identity(tmp_path):
    gross = _agent(tmp_path, FixedProposer([_tune(3)]))
    gross.run(SYMBOLS, START, END)
    net = _agent(tmp_path, FixedProposer([_tune(3)]), cost_model=ParametricCostModel())
    result = net.run(SYMBOLS, START, END)
    # Same config, same window - but measured net, so not a repeat of the gross one.
    assert result.n_evaluated == 1
    trials = [r["inputs"] for r in _records(tmp_path) if r["tool"] == "research:trial"]
    assert [t["params"]["_cost"]["gross"] for t in trials] == [True, False]
    start = next(r["inputs"] for r in _records(tmp_path) if r["tool"] == "research:session_start")
    assert start["cost"] == {"gross": True}


def test_a_saved_research_config_carries_the_cost_it_was_validated_under(tmp_path):
    agent = _agent(tmp_path, FixedProposer([_tune(3)]), cost_model=ParametricCostModel())
    result = agent.run(SYMBOLS, START, END)
    saved = config_store.load_config(result.shortlist[0].saved_path)
    assert saved["cost"]["gross"] is False and saved["cost"]["commission_bps"] == 1.0


# --- generated code is recorded, and never executed by loading a config ---
def test_a_code_candidate_is_saved_under_its_draft_identity_with_its_source(tmp_path):
    proposal = Proposal(hypothesis="the series drifts up", kind="code", code=_TRADING_CODE)
    result = _agent(tmp_path, FixedProposer([proposal]), allow_code_gen=True).run(SYMBOLS, START, END)
    (candidate,) = result.shortlist
    saved = config_store.load_config(candidate.saved_path)
    assert saved["strategy"].startswith("draft:GenStrat:")
    assert saved["provenance"]["generated_code"]["code"] == _TRADING_CODE
    assert saved["provenance"]["generated_code"]["code_hash"] == saved["strategy"].rsplit(":", 1)[1]
    trial = next(r["inputs"] for r in _records(tmp_path) if r["tool"] == "research:trial")
    assert (
        trial["code"] == _TRADING_CODE
        and trial["code_hash"] == saved["provenance"]["generated_code"]["code_hash"]
    )


def test_loading_a_generated_code_config_is_refused_not_run(tmp_path):
    from tradeflow import cli

    proposal = Proposal(hypothesis="the series drifts up", kind="code", code=_TRADING_CODE)
    result = _agent(tmp_path, FixedProposer([proposal]), allow_code_gen=True).run(SYMBOLS, START, END)
    path = result.shortlist[0].saved_path
    with pytest.raises(GeneratedCodeRefused):
        cli._load_strategy_from_config(path)
    args = cli.build_parser().parse_args(["backtest", "--config", path, "--start", "2024-01-02"])
    with pytest.raises(SystemExit) as refused:
        cli.apply_run_config(args)
    assert "generated code" in str(refused.value) and "strategy pack" in str(refused.value)


def test_a_repeated_code_proposal_is_a_duplicate_like_a_repeated_tune(tmp_path):
    proposal = Proposal(hypothesis="the series drifts up", kind="code", code=_TRADING_CODE)
    result = _agent(tmp_path, FixedProposer([proposal, proposal]), allow_code_gen=True, max_dry_rounds=9).run(
        SYMBOLS, START, END
    )
    assert result.n_evaluated == 1 and result.n_rejected == 1
    assert _rejects(tmp_path)[0]["reason"].startswith("duplicate")


# --- holdout ---
def test_an_empty_holdout_is_refused_before_anything_is_journaled(tmp_path):
    agent = _agent(tmp_path, FixedProposer([_tune(3)]), holdout_days=0)
    with pytest.raises(RunRefused):
        agent.run(SYMBOLS, START, END)
    assert not (tmp_path / "journal.jsonl").exists()


def test_a_holdout_larger_than_the_range_is_a_refusal_not_a_raw_error(tmp_path):
    with pytest.raises(RunRefused, match="leaves no research window"):
        _agent(tmp_path, FixedProposer([_tune(3)]), holdout_days=5000).run(SYMBOLS, START, END)


def test_a_session_counts_earlier_sessions_that_scored_the_same_holdout(tmp_path):
    first = _agent(tmp_path, FixedProposer([_tune(3)])).run(SYMBOLS, START, END)
    assert first.prior_holdout_looks == 0 and first.holdout_scored
    second = _agent(tmp_path, FixedProposer([_tune(5)])).run(SYMBOLS, START, END)
    assert second.prior_holdout_looks == 1
    # A different universe is a different holdout.
    assert prior_holdout_looks(str(tmp_path / "journal.jsonl"), second.holdout_window, ["ZZZ"]) == 0


def test_no_shortlist_means_no_holdout_was_scored(tmp_path):
    strict = dict(RELAXED_GATES, min_oos_sharpe=99)
    result = _agent(tmp_path, FixedProposer([_tune(3)]), gates=strict).run(SYMBOLS, START, END)
    assert result.holdout_scored is False
    assert _records(tmp_path)[-1]["inputs"]["holdout_scored"] is False


# --- non-finite metrics in a saved config ---
def test_non_finite_provenance_metrics_are_written_as_strict_json(tmp_path):
    provenance = config_store.build_provenance(
        method="grid",
        objective="sharpe_ratio",
        windows={},
        oos_metrics={"tail_ratio": float("inf"), "x": float("nan"), "y": float("-inf"), "ok": 1.5},
    )
    path = config_store.save_config(
        tmp_path / "c.json", strategy="periodic", params={}, provenance=provenance
    )
    text = path.read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert config_store.load_config(path)["provenance"]["oos_metrics"] == {
        "tail_ratio": "inf",
        "x": "nan",
        "y": "-inf",
        "ok": 1.5,
    }


def test_generated_code_that_raises_is_one_rejection_line_not_a_traceback_per_symbol(tmp_path, caplog):
    import logging

    crashing = _TRADING_CODE.replace("        return data\n", '        raise RuntimeError("boom")\n', 1)
    proposal = Proposal(hypothesis="x", kind="code", code=crashing)
    with caplog.at_level(logging.INFO):
        result = _agent(tmp_path, FixedProposer([proposal]), allow_code_gen=True).run(SYMBOLS, START, END)
    assert result.n_rejected == 1 and result.n_evaluated == 0
    # The code's own search swallows each failed evaluation; what it must not do is print
    # a stack trace for every symbol of every evaluation.
    assert not [r for r in caplog.records if r.exc_info]
