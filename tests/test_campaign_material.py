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
from tradeflow.services.registry import STRATEGIES
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
    strategy_class=STRATEGIES["demo_trend"],
    limit_overrides={"max_positions": 8},
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

    book = recipe["folded_into_identity"]["_limits"]
    # The *resolved* book: the override won where it spoke, and the class defaults came
    # through where it did not. Recording only the override left a run that overrode
    # nothing recording no book at all, while still having one.
    assert book["max_positions"] == 8
    assert book["max_total_risk"] == pytest.approx(0.05)
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
    # And the runnable half matches the provenance. This call passes no
    # `position_limits`, which used to write a config recording an eight-position
    # validation that would run at the class default of one.
    validated = written["provenance"]["campaign"][RECIPE]["folded_into_identity"]["_limits"]
    assert written["position_limits"] == validated


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


# --- the export carries what it claims, on the surface a reader sees ----------------
def test_the_universe_reaches_the_rendered_block_not_just_the_payload(tmp_path):
    """The block carried the universe in JSON and the renderer printed none of it —
    "which names does this evidence cover" being exactly what an export is for.

    Asserted against the rendered text, because that is where this defect lives: the
    same function already shipped once with a whole vocabulary present in the payload
    and absent from the output, under tests that checked only the payload.
    """
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        material = campaign_material(store, trial_id, journal_path=journal)

    assert material[RECIPE]["universe"]["symbols"] == ["AAA", "BBB"]
    text = format_campaign_material(material)
    assert "AAA" in text and "BBB" in text
    assert "2 symbol(s)" in text
    # And how it was resolved: the pre-scan set is larger than what survived it.
    assert "3 candidate(s) before the scan" in text


def test_a_long_universe_says_how_many_it_elided(tmp_path):
    """A silent truncation would be worse than the wall of tickers it avoids: a reader
    counting names would be counting the renderer's limit, not the book's."""
    journal = tmp_path / "journal.jsonl"
    wide = [f"S{i:02d}" for i in range(18)]
    trial_id = journal_trial(
        "walkforward",
        strategy="demo_trend",
        symbols=wide,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 4, 30),
        params={},
        metrics={"sharpe_ratio": 1.0},
        dedup_params=walk_forward_recipe(**RECIPE_ARGS),
        path=journal,
    )

    with _store(journal) as store:
        text = format_campaign_material(campaign_material(store, trial_id, journal_path=journal))

    assert "18 symbol(s)" in text
    assert "(+6 more)" in text


def test_a_trial_with_no_recorded_universe_says_so_rather_than_rendering_empty(tmp_path):
    """Absent is not empty. A record that never captured its universe must not read as
    a run over no symbols."""
    from tradeflow.analytics.reporting import _universe_lines

    (line,) = _universe_lines({})

    assert "not recorded" in line


def test_the_seed_is_read_from_where_it_actually_lives(tmp_path):
    """The block contradicted itself: `metadata.seed` was `None` while
    `recipe.validation.seed` held the real value two sections above.

    `journal_trial` has no seed parameter, so the `seed` column is only ever populated
    by the research agent's own session records — every walk-forward driven from the CLI
    or MCP reported its seed as unrecorded while carrying it.
    """
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        material = campaign_material(store, trial_id, journal_path=journal)

    seed = material[METADATA]["seed"]
    assert seed["recorded"] is True
    assert seed["value"] == RECIPE_ARGS["seed"]
    assert seed["from"] == "validation recipe"
    # The two halves of the block now agree about it.
    assert seed["value"] == material[RECIPE]["validation"]["seed"]
    assert "(from the validation recipe)" in format_campaign_material(material)


def test_a_run_whose_seed_nobody_recorded_still_says_so(tmp_path):
    """Both directions, and the rule every other fact here follows: a backtest carries
    no validation recipe, so its seed is genuinely absent — and an omitted line would
    read as nothing to say rather than as nothing recorded."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_backtest(journal)

    with _store(journal) as store:
        material = campaign_material(store, trial_id, journal_path=journal)

    seed = material[METADATA]["seed"]
    assert seed["recorded"] is False and seed["value"] is None
    assert "seed" in format_campaign_material(material)
    assert "not recorded" in format_campaign_material(material)


# --- every shape this field has ever had, because configs on disk keep theirs -------
@pytest.mark.parametrize(
    "stored, expected",
    [
        ({}, "not recorded"),  # the key was never written
        ({"seed": None}, "not recorded"),  # explicitly null
        ({"seed": 7}, "7"),  # a bare scalar — what older configs hold
        ({"seed": {"recorded": True, "value": 42, "from": "validation recipe"}}, "42"),
        ({"seed": {"recorded": False, "value": None, "from": None}}, "not recorded"),
    ],
)
def test_the_seed_renders_for_every_shape_this_field_has_had(stored, expected):
    """This block is embedded in saved config files, and a config written last month is
    not going to be rewritten — so the reader has to handle every shape the writer ever
    produced.

    The scalar case is the one that bit: an `isinstance(dict)` check dropped a seed the
    record genuinely held, producing no line at all. That is the same false absence this
    field was changed to fix, arriving from the other direction.
    """
    from tradeflow.analytics.reporting import _seed_cell

    assert expected in _seed_cell(stored)


def test_a_scalar_seed_is_not_silently_dropped():
    """Both directions of the case above: an old scalar must *appear*, not merely avoid
    crashing."""
    from tradeflow.analytics.reporting import _seed_cell

    rendered = _seed_cell({"seed": 7})

    assert "7" in rendered
    assert "not recorded" not in rendered


def test_two_records_of_the_seed_that_disagree_are_both_reported(tmp_path):
    """The recipe defines the run, so it wins — but reporting only the winner would hide
    that the record is internally inconsistent, and this block exists to make provenance
    checkable. Nothing here can tell which is the mistake."""
    from tradeflow.analytics.reporting import _seed_cell
    from tradeflow.services.campaign import _seed

    seed = _seed({"seed": 99}, {"dedup_params": {"seed": 42}})

    assert seed["value"] == 42  # the recipe defines what was validated
    assert seed["from"] == "validation recipe"
    assert seed["disagrees_with"] == {"value": 99, "from": "session record"}
    assert "DISAGREES" in _seed_cell({"seed": seed})


def test_the_session_record_seed_is_still_used_when_there_is_no_recipe(tmp_path):
    """The research agent populates the column and may carry no recipe. Preferring the
    recipe must not mean ignoring the only record that exists."""
    from tradeflow.services.campaign import _seed

    seed = _seed({"seed": 99}, {})

    assert seed == {"recorded": True, "value": 99, "from": "session record"}


@pytest.mark.parametrize(
    "count, expect_elision, expected_more",
    [(11, False, None), (12, False, None), (13, True, 1), (18, True, 6)],
)
def test_the_universe_elision_counts_correctly_at_the_boundary(count, expect_elision, expected_more):
    """Boundary bugs live in polite list renderers. At exactly the sample size there is
    nothing hidden, so a `+0 more` would be a lie about a complete list."""
    from tradeflow.analytics.reporting import _universe_lines

    (line,) = _universe_lines({"symbols": [f"S{i:02d}" for i in range(count)]})

    assert f"{count} symbol(s)" in line
    if expect_elision:
        assert f"(+{expected_more} more)" in line
    else:
        assert "more)" not in line


def test_the_recipe_puts_its_seed_where_the_reader_looks_for_it():
    """`_seed` deliberately reads only the top level of `dedup_params` — no recursive
    "seed anywhere" search, which could just as easily grab the wrong one.

    That makes the top-level placement an assumption, so it is pinned here rather than
    left implicit. A future recipe-bearing kind that nests its seed would report "not
    recorded" while holding it — the same defect this change fixed, one layer down — and
    this test is what turns that into a visible failure instead of a quiet one.
    """
    recipe = walk_forward_recipe(**RECIPE_ARGS)

    assert recipe["seed"] == RECIPE_ARGS["seed"]
    assert "seed" in recipe  # top level, not nested under a sub-dict


def test_the_universe_renders_for_a_kind_with_no_validation_recipe(tmp_path):
    """A surviving mutation: moving the universe render inside the `recipe.available`
    branch kept every test green while every backtest, verdict, optimize and research
    block silently lost its universe.

    Backtests are the most common promotable kind — `confirm_screen_point` delegates
    straight to one — and they have no validation recipe by construction. The universe
    is not part of the recipe; it is part of what the evidence covers.
    """
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_backtest(journal)

    with _store(journal) as store:
        material = campaign_material(store, trial_id, journal_path=journal)

    assert material[RECIPE]["available"] is False  # no recipe, by construction
    text = format_campaign_material(material)
    assert "1 symbol(s): AAA" in text  # and the universe is there anyway


def test_an_unreadable_journal_line_does_not_claim_the_universe_was_unrecorded(tmp_path):
    """ "Nobody wrote it down" and "the record holding it could not be read" are the
    exact distinction this block exists to preserve, and the universe line collapsed
    them — asserting the first while the recipe reason one line above honestly reported
    the second."""
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    with _store(journal) as store:
        # The store row survives; the journal line it was built from does not.
        material = campaign_material(store, trial_id, journal_path=tmp_path / "gone.jsonl")

    assert material[METADATA]["journal_line_found"] is False
    text = format_campaign_material(material)
    assert "could not be read" in text
    assert "universe        — not recorded for this trial" not in text


def test_a_promoted_config_does_not_contradict_itself_about_the_seed(tmp_path):
    """Found one layer out from where this change started. `provenance.seed` is the
    older and more widely read field and was still reading the store column, so a
    promoted config said `seed: null` at the top level and `seed: 42` inside its own
    campaign block — one file, two answers, the null one first.
    """
    journal = tmp_path / "journal.jsonl"
    trial_id = _journal_walkforward(journal)

    # Driven through the real command and read back off disk. The first version of this
    # test compared `material[METADATA]["seed"]["value"]` with an expression that
    # recomputed the same thing — a tautology that passed against the defect it was
    # written for, which is the failure this project's testing rule names.
    import json
    from argparse import Namespace

    from tradeflow.cli import _promote_trial

    out = tmp_path / "promoted.json"
    with _store(journal) as store:
        _promote_trial(
            store,
            Namespace(trial_id=trial_id, save_config=str(out), force=True, journal=str(journal)),
        )

    written = json.loads(out.read_text())
    provenance = written["provenance"]
    assert provenance["seed"] == RECIPE_ARGS["seed"]
    embedded = provenance["campaign"]["metadata"]["seed"]["value"]
    assert provenance["seed"] == embedded, "the file disagrees with its own campaign block"
