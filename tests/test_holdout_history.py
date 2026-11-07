"""A walk-forward's holdout is visible from the trial history, and its absence is stated.

The holdout is scored once and was printed once: the journal kept nothing of it, so
`trials show`/`campaign` never mentioned it and a reused result dropped it silently. The
journal field is new, so four states must stay distinguishable - recorded, never
requested, requested but written before the journal kept it, and unreadable.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from tests.fakes import FakeMarketData
from tests.test_saved_config_book import _chosen_params
from tradeflow import cli
from tradeflow.analytics.reporting import format_campaign_material, format_holdout
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services import analysis, audit
from tradeflow.services.analysis import walk_forward_recipe
from tradeflow.services.audit import holdout_for_trial, holdout_from_record, journal_trial
from tradeflow.services.campaign import campaign_material
from tradeflow.services.registry import STRATEGIES
from tradeflow.store.trials import TrialStore, db_path_for_journal

SYMBOLS = [f"S{i}" for i in range(6)]
BASE = ["--symbols", ",".join(SYMBOLS), "--scanner", "none", "--start", "2023-01-02", "--end", "2024-09-30"]


class _Result:
    holdout = {"sharpe_ratio": -0.7, "cagr": -0.5, "total_trades": 2}
    holdout_params = {"fast_ema_period": 9}


def _trial(journal, holdout_days, with_field):
    return journal_trial(
        "walkforward",
        strategy="demo_trend",
        symbols=["AAA", "BBB"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 29),
        params=_chosen_params(),
        metrics={"sharpe_ratio": 0.4},
        extra={"promotable": False, **(audit.holdout_extra(_Result, holdout_days) if with_field else {})},
        returns=pd.Series([0.001] * 30, index=pd.date_range("2024-01-02", periods=30)),
        dedup_params=walk_forward_recipe(
            mode="anchored",
            n_folds=None,
            train_days=252,
            test_days=63,
            embargo_days=5,
            holdout_days=holdout_days,
            method="grid",
            objective="sharpe_ratio",
            max_evals=5,
            seed=42,
            cost_key={},
            strategy_class=STRATEGIES["demo_trend"],
            capital=100_000.0,
            benchmark=None,
        ),
        path=journal,
    )


def test_the_four_states_are_kept_distinct(tmp_path):
    journal = tmp_path / "j.jsonl"
    recorded = holdout_for_trial(_trial(journal, 60, with_field=True), journal)
    assert recorded["state"] == "recorded" and recorded["metrics"]["sharpe_ratio"] == -0.7

    # Written before the field existed: a holdout was asked for, and its result is gone.
    old = holdout_for_trial(_trial(journal, 60, with_field=False), journal)
    assert old["state"] == "not_retained" and old["metrics"] is None and old["days"] == 60

    none = holdout_for_trial(_trial(journal, 0, with_field=True), journal)
    assert none["state"] == "none_requested"

    assert holdout_from_record(None)["state"] == "unknown"


def test_show_and_campaign_say_what_became_of_the_holdout(tmp_path):
    journal = tmp_path / "j.jsonl"
    kept, lost = _trial(journal, 60, True), _trial(journal, 60, False)
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        kept_text = format_campaign_material(campaign_material(store, kept, journal_path=journal))
        lost_text = format_campaign_material(campaign_material(store, lost, journal_path=journal))
    assert "Holdout : 60 days, scored once" in kept_text and "Sharpe -0.70" in kept_text
    assert "its result was not retained" in lost_text
    assert "scored once" not in lost_text  # absent is not rendered as a result


def test_an_unfilled_holdout_is_not_a_missing_one():
    lines = format_holdout({"state": "recorded", "days": 60, "metrics": None})
    assert "produced no result" in lines[0]


@pytest.fixture
def _cli(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    client = MarketDataClient(FakeMarketData(SYMBOLS + ["SPY"], n=700, freq="1D"))
    monkeypatch.setattr(cli, "build_research_data", lambda **kw: client)


def _run(argv):
    args = cli.build_parser().parse_args(argv)
    args.func(args)


def test_a_real_walkforward_and_its_reuse_both_show_the_holdout(_cli, capsys, tmp_path):
    """Through the commands a user runs: the run prints it, the reuse repeats it from the
    journal, and `show`/`campaign` read it back. Reuse used to drop it without a word."""
    wf = ["walkforward", "--strategy", "demo_trend", "--max-evals", "3", "--holdout-days", "60", *BASE]
    _run(wf)
    first = capsys.readouterr().out
    assert "Holdout (scored once)" in first

    _run(wf)
    reused = capsys.readouterr().out
    assert "Not re-run" in reused
    assert "Holdout : 60 days, scored once" in reused

    db = str(db_path_for_journal(tmp_path / "journal.jsonl"))
    journal = str(tmp_path / "journal.jsonl")
    _run(["trials", "list", "--kind", "walkforward", "--json", "--db", db])
    trial_id = json.loads(capsys.readouterr().out)["rows"][0]["id"]
    _run(["trials", "show", trial_id, "--db", db])
    assert "Holdout : 60 days, scored once" in capsys.readouterr().out
    _run(["trials", "campaign", trial_id, "--db", db, "--journal", journal])
    assert "Holdout : 60 days, scored once" in capsys.readouterr().out


def test_a_reused_result_over_the_service_keeps_the_holdout_too(_cli):
    """The MCP tool's path: the memoized payload used to carry no holdout key at all."""
    client = MarketDataClient(FakeMarketData(SYMBOLS + ["SPY"], n=700, freq="1D"))
    kwargs = {"max_evals": 3, "holdout_days": 60}
    start, end = cli._date("2023-01-02"), cli._date("2024-09-30")
    fresh = analysis.run_walk_forward(client, "demo_trend", SYMBOLS, start, end, **kwargs)
    assert not fresh.get("memoized")
    reused = analysis.run_walk_forward(client, "demo_trend", SYMBOLS, start, end, **kwargs)
    assert reused["memoized"] is True
    assert reused["holdout"]["state"] == "recorded"
    assert reused["holdout"]["metrics"] == pytest.approx(fresh["holdout"], rel=1e-9)


def test_a_reused_result_that_predates_the_field_says_so(_cli, capsys, tmp_path):
    row = {"id": _trial(tmp_path / "journal.jsonl", 60, False), "oos_sharpe": 0.4}
    cli._print_cached_walkforward(row)
    assert "was not retained" in capsys.readouterr().out
