"""What validated a config, materialised from what was already recorded.

Promoting the winning row of a walk-forward captured the parameters and lost the
recipe, so the config could not say what validated it. These tests pin the recipe
coming back out of the journal, the three sections staying labelled, and the evidence
being scoped to the accounting era that produced it.

The fixtures journal through `journal_trial` rather than hand-writing a record. The
first version of this code read `record["extra"]["dedup_params"]` — `audit_log`
flattens those onto the record — and a hand-built fixture would have agreed with the
mistake and reported every recipe as present.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from tradeflow.analytics.reporting import format_campaign_material
from tradeflow.engine.backtest import ACCOUNTING_VERSION
from tradeflow.services.analysis import walk_forward_recipe
from tradeflow.services.audit import journal_trial
from tradeflow.services.campaign import EVIDENCE, METADATA, RECIPE, campaign_material
from tradeflow.store.trials import TrialStore, db_path_for_journal

RECIPE_ARGS = dict(
    mode="anchored",
    n_folds=None,
    train_days=252,
    test_days=63,
    embargo_days=5,
    holdout_days=60,
    method="grid",
    objective="sharpe_ratio",
    max_evals=50,
    seed=42,
    cost_key={"commission_bps": 1.0},
    limits={"max_positions": 8},
)


def _journal_walkforward(journal, **overrides):
    dates = pd.date_range("2024-01-02", periods=120, freq="D")
    return journal_trial(
        "walkforward",
        strategy="demo_trend",
        symbols=["AAA", "BBB"],
        candidate_symbols=["AAA", "BBB", "CCC"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 4, 30),
        params={"fast_ema_period": 9},
        metrics={"sharpe_ratio": 1.12, "total_trades": 88},
        objective="sharpe_ratio",
        extra={"n_trials": 50, "promotable": True},
        returns=pd.Series([0.001] * 120, index=dates),
        dedup_params=walk_forward_recipe(**{**RECIPE_ARGS, **overrides}),
        path=journal,
    )


def _journal_backtest(journal):
    return journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 4, 30),
        params={"fast_ema_period": 5},
        metrics={"sharpe_ratio": 0.4, "total_trades": 12},
        path=journal,
    )


def _store(journal):
    return TrialStore(db_path_for_journal(journal), journal_path=journal)


# --- the recipe comes back ----------------------------------------------------------
def test_a_walk_forward_s_validation_recipe_survives_promotion(tmp_path):
    """The gap this closes. A walk-forward is a recipe plus a chosen parameter set plus
    a resolved universe, and promoting the winning row kept only the middle one."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        material = campaign_material(store, trial_id, journal_path=journal)

    recipe = material[RECIPE]
    assert recipe["available"] is True
    assert recipe["validation"]["train_days"] == 252
    assert recipe["validation"]["embargo_days"] == 5
    assert recipe["validation"]["method"] == "grid"
    assert recipe["validation"]["mode"] == "anchored"


def test_the_cost_model_and_book_are_named_not_dropped(tmp_path):
    """They are folded into the recipe's identity because they change what a validation
    *means* — two runs with identical folds at different book sizes are different
    validations. Reading them as search settings would be misleading, so they are
    labelled rather than either hidden or mixed in."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        recipe = campaign_material(store, trial_id, journal_path=journal)[RECIPE]

    assert recipe["folded_into_identity"]["_limits"] == {"max_positions": 8}
    assert recipe["folded_into_identity"]["_cost"] == {"commission_bps": 1.0}
    assert "_limits" not in recipe["validation"]


def test_a_kind_with_no_separate_recipe_says_so_rather_than_reporting_an_empty_one(tmp_path):
    """A backtest's identity *is* its parameters, so it records no separate recipe. An
    empty recipe block would read as "validated with no settings", which is a different
    and much worse claim."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_backtest(journal)

    with _store(journal) as store:
        recipe = campaign_material(store, trial_id, journal_path=journal)[RECIPE]

    assert recipe["available"] is False
    assert "no separate validation recipe" in recipe["reason"]
    assert "validation" not in recipe


def test_a_missing_journal_line_names_which_half_is_gone(tmp_path):
    """The store keeps a hash of the recipe, not the recipe. With the journal
    unreadable the trial is still indexed and the recipe is simply not recoverable —
    which is worth saying precisely."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)
    with _store(journal) as store:
        material = campaign_material(store, trial_id, journal_path=tmp_path / "absent.jsonl")

    assert material["available"] is True  # the trial is indexed
    assert material[RECIPE]["available"] is False
    assert "the store keeps a hash of it" in material[RECIPE]["reason"]
    assert material[METADATA]["journal_line_found"] is False


def test_an_unknown_trial_is_unavailable_with_a_reason(tmp_path):
    journal = tmp_path / "journal.jsonl"
    with _store(journal) as store:
        material = campaign_material(store, "nope", journal_path=journal)

    assert material["available"] is False
    assert "trials rebuild" in material["reason"]
    assert RECIPE not in material


# --- the sections are labelled ------------------------------------------------------
def test_every_section_declares_what_kind_of_thing_it_is(tmp_path):
    """The labels are the feature. A recipe survives an accounting bump and a
    measurement does not, and a reader who cannot tell them apart carries a stale
    number forward beside a recipe that is still good."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        material = campaign_material(store, trial_id, journal_path=journal)

    assert material[RECIPE]["kind"] == RECIPE
    assert material[EVIDENCE]["kind"] == EVIDENCE
    assert material[METADATA]["kind"] == METADATA


# --- evidence is scoped to its era --------------------------------------------------
def test_evidence_recorded_under_an_older_engine_is_marked_incomparable(tmp_path):
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)
    with _store(journal) as store:
        store._conn.execute("UPDATE trials SET accounting = 3 WHERE id = ?", (trial_id,))
        store._conn.commit()
        material = campaign_material(store, trial_id, journal_path=journal)

    evidence = material[EVIDENCE]
    assert evidence["comparable_with_current_engine"] is False
    assert "v3" in evidence["staleness"] and str(ACCOUNTING_VERSION) in evidence["staleness"]
    # The recipe is untouched by the bump, which is the entire reason for splitting them.
    assert material[RECIPE]["available"] is True


def test_evidence_from_the_current_engine_is_not_flagged(tmp_path):
    """Both directions: a warning that always fires teaches people to skip it."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        evidence = campaign_material(store, trial_id, journal_path=journal)[EVIDENCE]

    assert evidence["comparable_with_current_engine"] is True
    assert "staleness" not in evidence


def test_metrics_are_parsed_and_promotable_is_a_boolean(tmp_path):
    """The light row carries `metrics_json` as a string and `promotable` as SQLite's
    0/1. Passing them straight through wrote `"metrics": null` into every materialised
    config, for a trial that had measured plenty."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        evidence = campaign_material(store, trial_id, journal_path=journal)[EVIDENCE]

    assert evidence["metrics"]["sharpe_ratio"] == pytest.approx(1.12)
    assert evidence["promotable"] is True


def test_a_quarantined_trial_is_named_as_such(tmp_path):
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)
    with _store(journal) as store:
        store.mark_contaminated([trial_id], reason="a data correction")
        evidence = campaign_material(store, trial_id, journal_path=journal)[EVIDENCE]

    assert evidence["quarantined"] is True
    assert evidence["quarantine_reason"] == "a data correction"


# --- metadata points somewhere the reader can actually go ---------------------------
def test_stored_artifacts_are_named_by_the_command_that_reads_them(tmp_path):
    """A config is portable and its reader may be an installed copy whose state root is
    somewhere else, so a filesystem path is the one form guaranteed wrong for
    somebody."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        artifacts = campaign_material(store, trial_id, journal_path=journal)[METADATA]["artifacts"]

    by_name = {a["artifact"]: a for a in artifacts}
    assert by_name["return series"]["recorded"] is True
    assert "trials compare" in by_name["return series"]["read_with"]
    # Absent artifacts get no command, because there is nothing to read.
    assert by_name["trade table"]["recorded"] is False
    assert by_name["trade table"]["read_with"] is None


def test_the_artifact_command_matches_how_this_copy_is_actually_run(tmp_path, monkeypatch):
    """`python main.py` for a checkout, `tradeflow` for an installed copy — the same
    rule every other printed instruction follows, reached through the same helper."""
    from tradeflow.services import setup

    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    monkeypatch.setattr("tradeflow.settings.running_from_checkout", lambda: False)
    with _store(journal) as store:
        installed = campaign_material(store, trial_id, journal_path=journal)
    assert installed[METADATA]["artifacts"][0]["read_with"].startswith("tradeflow ")
    assert setup.invocation("trials show x") == "tradeflow trials show x"

    monkeypatch.setattr("tradeflow.settings.running_from_checkout", lambda: True)
    with _store(journal) as store:
        checkout = campaign_material(store, trial_id, journal_path=journal)
    assert checkout[METADATA]["artifacts"][0]["read_with"].startswith("python main.py ")


# --- it lands in the one config format ----------------------------------------------
def test_promotion_writes_the_campaign_into_the_config_s_own_provenance(tmp_path, monkeypatch):
    """Not a second artifact beside the config. `save_config` is the portability
    format, and a campaign export living somewhere else would be the second provenance
    schema this project keeps getting bitten by."""
    from tradeflow.cli import main as cli_main

    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)
    out = tmp_path / "promoted.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tradeflow",
            "trials",
            "promote",
            trial_id,
            "--save-config",
            str(out),
            "--db",
            str(db_path_for_journal(journal)),
        ],
    )
    monkeypatch.setattr("tradeflow.services.audit.DEFAULT_TRIAL_JOURNAL", journal, raising=False)
    cli_main()

    payload = json.loads(out.read_text())
    campaign = payload["provenance"]["campaign"]
    assert campaign[RECIPE]["validation"]["train_days"] == 252
    assert campaign[EVIDENCE]["accounting"] == ACCOUNTING_VERSION
    # The recipe also fills in the fields the format already had, rather than being
    # written twice in two shapes.
    assert payload["provenance"]["method"] == "grid"
    assert payload["provenance"]["objective"] == "sharpe_ratio"


def test_a_config_written_before_campaigns_existed_still_loads(tmp_path):
    """Older configs load unchanged and read as *not materialised from a campaign*,
    which is what they are."""
    from tradeflow.optimization.config_store import load_config

    old = tmp_path / "old.json"
    old.write_text(
        json.dumps(
            {
                "strategy": "demo_trend",
                "params": {"fast_ema_period": 5},
                "provenance": {"objective": "sharpe_ratio", "accounting": 5},
            }
        )
    )

    payload = load_config(old)

    assert payload["params"] == {"fast_ema_period": 5}
    assert payload["provenance"].get("campaign", {}) == {}


def test_the_mcp_surface_writes_the_same_provenance_schema(tmp_path):
    """One format, both surfaces. `configs.save_config` constructs a `Provenance` from
    the dict an agent passes, so a campaign block that the dataclass does not accept
    would fail there while `trials promote` succeeded — two schemas by accident."""
    from tradeflow.services import configs

    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)
    with _store(journal) as store:
        material = campaign_material(store, trial_id, journal_path=journal)

    saved = configs.save_config(
        str(tmp_path / "agent.json"),
        strategy="demo_trend",
        params={"fast_ema_period": 9},
        provenance={"objective": "sharpe_ratio", "campaign": material},
    )
    written = json.loads(open(saved["path"]).read())

    assert written["provenance"]["campaign"][RECIPE]["validation"]["train_days"] == 252


# --- the renderer -------------------------------------------------------------------
def test_the_report_labels_each_section_and_counts_are_not_decimals(tmp_path):
    """`total_trades` printed as `88.000` is the flat-float defect this project has now
    found in three separate renderers."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)
    with _store(journal) as store:
        printed = format_campaign_material(campaign_material(store, trial_id, journal_path=journal))

    assert "RECIPE" in printed and "EVIDENCE" in printed and "METADATA" in printed
    assert "total_trades    88" in printed
    assert "88.00" not in printed


def test_an_unavailable_campaign_renders_as_absent_not_as_an_empty_report(tmp_path):
    journal = tmp_path / "journal.jsonl"
    with _store(journal) as store:
        printed = format_campaign_material(campaign_material(store, "nope", journal_path=journal))

    assert "trials rebuild" in printed
    assert "RECIPE" not in printed
