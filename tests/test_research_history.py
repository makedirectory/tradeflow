"""What the trial history says about research-agent rounds.

A generated-code round and a base-strategy round are different mechanisms; recording both
under the base strategy's name made `--strategy <base>` mix them. All offline, through the
real agent with a fixed proposer.
"""

from quantred.research.proposer import FixedProposer, Proposal
from quantred.store.trials import TrialStore, db_path_for_journal
from tests.test_research import (
    _VALID_CODE,
    END,
    START,
    SYMBOLS,
    _agent,
    _isolate,  # noqa: F401 - the autouse fixture registers the test strategy
    _tune,
)


def _run(tmp_path, proposals):
    agent = _agent(tmp_path, FixedProposer(proposals), allow_code_gen=True)
    agent.run(SYMBOLS, START, END)
    journal = tmp_path / "journal.jsonl"
    return journal


def _strategies(journal):
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        return {r["id"]: r["strategy"] for r in store.list_trials(kind="research", limit=50)}


def test_a_generated_code_round_is_not_recorded_as_the_base_strategy(tmp_path):
    journal = _run(
        tmp_path,
        [
            _tune(3),
            Proposal(hypothesis="the series drifts up", kind="code", code=_VALID_CODE, tuned_params=[]),
        ],
    )
    recorded = list(_strategies(journal).values())
    assert recorded.count("periodic") == 1
    generated = [name for name in recorded if name != "periodic"]
    assert len(generated) == 1
    assert generated[0].startswith("draft:GenStrat:")  # the identity run_draft_walk_forward uses


def test_a_rebuild_reproduces_the_generated_strategy_identity(tmp_path):
    """The store is derived: the identity has to come back from the journal line, or a
    rebuild silently re-files the round under the session's base strategy."""
    journal = _run(tmp_path, [Proposal(hypothesis="drifts", kind="code", code=_VALID_CODE)])
    before = _strategies(journal)
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        store.rebuild(journal)
    assert _strategies(journal) == before
    assert next(iter(before.values())).startswith("draft:GenStrat:")


def test_a_research_rounds_campaign_does_not_claim_its_journal_line_is_unreadable(tmp_path):
    """The round is journaled as `research:trial`, which the trial-line lookup never
    matched, so the campaign said the line "could not be read" about a line that was in
    the file. What it actually lacks is a per-trial recipe."""
    from quantred.analytics.reporting import format_campaign_material
    from quantred.services.campaign import campaign_material

    journal = _run(tmp_path, [_tune(3)])
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        trial_id = store.list_trials(kind="research", limit=1)[0]["id"]
        material = campaign_material(store, trial_id, journal_path=journal)
    assert material["metadata"]["journal_line_found"] is True
    reason = material["recipe"]["reason"]
    assert "could not be read" not in reason
    assert "research:session_start" in reason
    text = format_campaign_material(material)
    assert "could not be read" not in text
    # And the two Sharpes a research row can show are told apart.
    assert "median fold" in text


def _material(metrics):
    return {
        "available": True,
        "trial_id": "x",
        "kind": "walkforward",
        "recipe": {},
        "evidence": {"accounting": 6, "metrics": metrics},
        "metadata": {},
    }


def test_an_unmeasured_benchmark_metric_is_not_printed_as_a_measured_zero():
    from quantred.analytics.reporting import format_campaign_material

    metrics = {"alpha": 0.0, "beta": 0.0, "benchmark_buy_hold_return": 0.0, "benchmark_available": False}
    text = format_campaign_material(_material(metrics))
    for name in ("alpha", "beta", "benchmark_buy_hold_return"):
        line = next(ln for ln in text.splitlines() if ln.strip().startswith(name))
        assert "not measured" in line
    # The other direction: with a benchmark, a zero is a zero.
    measured = format_campaign_material(_material({**metrics, "benchmark_available": True}))
    assert "not measured" not in measured


def test_a_line_written_before_the_field_existed_still_reads_as_the_session_strategy():
    """Absent stays absent: an old line carries no strategy, and the reader falls back to
    the session's, which is what it was recorded under."""
    from quantred.store.trials import _row_kwargs_from_research_trial

    record = {"run_id": "r1", "inputs": {"params": {}, "oos_aggregate": {}}}
    assert _row_kwargs_from_research_trial(record, {"strategy": "periodic"})["strategy"] == "periodic"
    record["inputs"]["strategy"] = "draft:X:abc"
    assert _row_kwargs_from_research_trial(record, {"strategy": "periodic"})["strategy"] == "draft:X:abc"


def test_campaign_evidence_and_the_html_report_read_unmeasured_metrics_one_way():
    """Found in review: two parallel copies of the "unmeasured metric" rule had diverged -
    the campaign view called a Treynor ratio suppressed for a near-zero beta "no
    benchmark" with a benchmark present. One reader of the flags now serves both."""
    from quantred.analytics.reporting import format_campaign_material

    metrics = {"treynor_ratio": 0.0, "treynor_available": False, "benchmark_available": True, "alpha": 0.4}
    text = format_campaign_material(
        {
            "available": True,
            "trial_id": "t",
            "kind": "backtest",
            "recipe": {},
            "evidence": {"metrics": metrics},
            "metadata": {},
        }
    )
    assert "not measured (beta too close to zero)" in text
    assert "not measured (no benchmark)" not in text
