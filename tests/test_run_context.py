"""What a run was set up with, recorded beside what it measured.

The capital, the scanner and as-of clock, the cache policy, the probes that ran and the
note somebody left are facts about how a trial was produced. None of them is part of its
identity, so recording them changes no dedup key and invalidates no memo — and every
trial written before them reads as *not recorded* rather than being backfilled with a
default nobody chose.

The last part is the whole point. A config that never recorded its capital is not a
config that ran at zero, and the day one is defaulted to 100,000 is the day a promoted
config claims a capital nobody picked.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from tests.fakes import DictMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services import analysis
from tradeflow.services.audit import (
    cache_policy,
    journal_record_for_trial,
    journal_trial,
    probe_verdicts,
    run_context,
)
from tradeflow.services.campaign import campaign_material
from tradeflow.store.trials import TrialStore, db_path_for_journal, params_hash
from tradeflow.utils.timeutils import NEW_YORK


@pytest.fixture
def _isolated_state(tmp_path, monkeypatch):
    """Journal and artifacts under tmp_path, so a test never memoizes against whatever
    somebody ran yesterday."""
    from tradeflow.services import audit

    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    return tmp_path


def _client():
    idx = pd.date_range("2024-01-02", periods=90, freq="D", tz=NEW_YORK)
    bars = {
        s: pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000},
            index=idx,
        )
        for s in ("AAA", "BBB")
    }
    return MarketDataClient(DictMarketData(bars))


def _one_backtest(capital=250_000.0):
    return analysis.run_backtest(
        _client(),
        "demo_trend",
        ["AAA", "BBB"],
        datetime(2024, 1, 2),
        datetime(2024, 3, 20),
        capital=capital,
    )


# --- the builders -------------------------------------------------------------------
def test_only_what_was_known_is_written():
    """A key absent means the run did not record that fact. Writing every key with a
    null would make "unknown" and "explicitly nothing" the same record."""
    assert run_context() == {}
    assert run_context(capital=100.0) == {"capital": 100.0}
    assert run_context(scanner=None, notes="") == {}
    assert cache_policy() == {}
    assert cache_policy(offline=False) == {"offline": False}  # False is a recorded fact


def test_a_probe_that_did_not_run_has_no_entry():
    """Not the same as one that ran and passed. A gate report that never exercised the
    leakage probe and one that exercised it and cleared look identical everywhere else."""
    assert probe_verdicts({"checks": {"min_wfe": {"value": 0.5, "passed": True}}}) == {}
    assert probe_verdicts(None) == {}

    ran = probe_verdicts({"checks": {"leakage_probe": {"value": True, "passed": True}}})
    assert ran["leakage_probe"] == {"value": True, "passed": True, "ran": True}

    # A probe that ran and *failed* is still a probe that ran.
    failed = probe_verdicts({"checks": {"leakage_probe": {"value": False, "passed": False}}})
    assert failed["leakage_probe"]["ran"] is True
    assert failed["leakage_probe"]["passed"] is False


# --- it changes no identity ----------------------------------------------------------
def test_the_context_lands_under_inputs_and_leaves_the_identity_alone(_isolated_state):
    """Recording context must not move a single dedup hash, or every existing trial
    stops finding itself for the sake of a field nothing keys on."""
    _one_backtest()
    journal = _isolated_state / "journal.jsonl"

    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        row = store.query(limit=5)[0]
        record = journal_record_for_trial(row["id"], journal)

    assert "context" in record["inputs"]
    assert "context" not in record["resolved_config"]
    # The hash still comes from the params alone.
    assert params_hash(json.loads(row["params_json"])) == row["params_hash"]


def test_two_runs_differing_only_in_context_share_one_identity(_isolated_state):
    """The other half of "no identity change": context is deliberately *not* part of
    what makes a repeat a repeat, so two runs differing only in capital still key alike.

    Asserted on the identity rather than on whether the second was served from the
    first — memoization additionally requires a stored trade table, so a run without
    `--record-trades` misses the cache for reasons that have nothing to do with this."""
    _one_backtest(capital=250_000.0)
    _one_backtest(capital=999_000.0)
    journal = _isolated_state / "journal.jsonl"

    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        rows = store.query(limit=5)
        capitals = {journal_record_for_trial(r["id"], journal)["inputs"]["context"]["capital"] for r in rows}

    assert capitals == {250_000.0, 999_000.0}  # the context genuinely differs
    assert len({r["params_hash"] for r in rows}) == 1  # and the identity does not


# --- absent stays absent -------------------------------------------------------------
def test_a_recorded_fact_and_an_unrecorded_one_never_look_alike(_isolated_state):
    _one_backtest(capital=250_000.0)
    journal = _isolated_state / "journal.jsonl"

    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        material = campaign_material(store, store.query(limit=5)[0]["id"], journal_path=journal)

    context = material["recipe"]["context"]
    assert context["capital"] == {"recorded": True, "value": 250_000.0}
    # The service path knows no scanner and no cache policy, and says so rather than
    # reporting a default.
    assert context["scanner"] == {"recorded": False, "value": None}
    assert context["cache"] == {"recorded": False, "value": None}
    assert material["evidence"]["probes"] == {"recorded": False, "value": None}
    assert material["metadata"]["notes"] == {"recorded": False, "value": None}


def test_a_trial_written_before_context_existed_reads_as_not_recorded(tmp_path):
    """Every trial in every existing journal. It must read as "nobody wrote this down",
    never as a capital of zero or a cache policy of false."""
    journal = tmp_path / "journal.jsonl"
    trial_id = journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast_ema_period": 5},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )

    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        material = campaign_material(store, trial_id, journal_path=journal)

    for fact in ("capital", "scanner", "scan_as_of", "cache"):
        assert material["recipe"]["context"][fact] == {"recorded": False, "value": None}


def test_an_old_journal_line_replays_cleanly(tmp_path):
    """The durable-record requirement: a rebuild reads every shape the writer has ever
    produced, including the ones written before this field existed."""
    journal = tmp_path / "journal.jsonl"
    old = journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast_ema_period": 5},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )
    new = journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast_ema_period": 9},
        metrics={"sharpe_ratio": 1.0},
        context=run_context(capital=50_000.0, notes="paper run, week two"),
        path=journal,
    )

    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        stats = store.rebuild(journal)
        assert stats["rows"] == 2
        before = campaign_material(store, old, journal_path=journal)
        after = campaign_material(store, new, journal_path=journal)

    assert before["recipe"]["context"]["capital"]["recorded"] is False
    assert after["recipe"]["context"]["capital"] == {"recorded": True, "value": 50_000.0}
    assert after["metadata"]["notes"]["value"] == "paper run, week two"


# --- one vocabulary across surfaces ---------------------------------------------------
def test_the_cli_adapter_and_the_service_builder_produce_one_shape():
    """A CLI trial saying `scan_as_of` and an MCP one saying `as_of` would be two
    schemas for one fact, and the campaign material reading them would need both."""
    from argparse import Namespace

    from tradeflow.cli import _run_context

    args = Namespace(
        capital=250_000.0,
        scanner="volume_spike",
        scan_as_of=datetime(2024, 3, 1),
        cache=True,
        offline=False,
        cache_dir="/tmp/bars",
        note="paper run",
    )

    from_cli = _run_context(args)
    from_service = run_context(
        capital=250_000.0,
        scanner="volume_spike",
        scan_as_of=datetime(2024, 3, 1),
        cache=cache_policy(cache=True, offline=False, cache_dir="/tmp/bars"),
        notes="paper run",
    )

    assert from_cli == from_service


def test_the_cli_adapter_records_nothing_a_command_does_not_have():
    """The adapter reads a namespace, and a command without these flags must produce an
    empty context rather than an error or a row of Nones."""
    from argparse import Namespace

    from tradeflow.cli import _run_context

    assert _run_context(Namespace()) == {}


# --- the round trip -------------------------------------------------------------------
def test_capital_survives_from_the_run_to_a_promoted_config(_isolated_state, monkeypatch):
    """The payoff. The journal never held a run's capital, so a promoted config could
    not state what it was validated at — it inherited whatever the next run passed."""
    from tradeflow.cli import build_parser

    _one_backtest(capital=250_000.0)
    journal = _isolated_state / "journal.jsonl"
    out = _isolated_state / "promoted.json"
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        trial_id = store.query(limit=5)[0]["id"]

    args = build_parser().parse_args(
        [
            "trials",
            "promote",
            trial_id,
            "--save-config",
            str(out),
            "--force",
            "--db",
            str(db_path_for_journal(journal)),
            "--journal",
            str(journal),
        ]
    )
    args.func(args)

    config = json.loads(out.read_text())
    assert config["capital"] == pytest.approx(250_000.0)
    # And the book from the previous fix still arrives, so the two survive together.
    assert config["position_limits"]["max_positions"] == 1


def test_a_trial_with_no_recorded_capital_promotes_without_one(tmp_path):
    """Absent stays absent all the way to the file. A promoted config with no capital is
    honest; one defaulted to 100,000 states a number nobody chose."""
    from tradeflow.cli import build_parser

    journal = tmp_path / "journal.jsonl"
    trial_id = journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast_ema_period": 5},
        metrics={"sharpe_ratio": 1.0},
        extra={"promotable": True},
        path=journal,
    )
    out = tmp_path / "promoted.json"
    args = build_parser().parse_args(
        [
            "trials",
            "promote",
            trial_id,
            "--save-config",
            str(out),
            "--db",
            str(db_path_for_journal(journal)),
            "--journal",
            str(journal),
        ]
    )
    args.func(args)

    assert "capital" not in json.loads(out.read_text())
