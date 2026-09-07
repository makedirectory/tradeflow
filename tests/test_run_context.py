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
        end=datetime(2024, 6, 1),
        cache=True,
        offline=False,
        cache_dir="/tmp/bars",
        workers=None,
        note="paper run",
    )

    from_cli = _run_context(args)
    from_service = run_context(
        capital=250_000.0,
        scanner="volume_spike",
        scan_as_of=datetime(2024, 3, 1),
        scan_as_of_explicit=True,
        cache=cache_policy(cache=True, offline=False, cache_dir="/tmp/bars", workers=None),
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


# --- findings from an independent Codex review of this branch -----------------------
# One test per finding, named for the requirement it pins rather than the mechanics, so
# a failure says which guarantee broke.


def test_the_rendered_campaign_shows_every_context_fact_and_its_status():
    """R1. `campaign_material` returned `recipe.context`, `evidence.probes` and
    `metadata.notes`; the formatter rendered none of them — and the usage docs showed a
    sample of output the code never produced. The tests asserted the JSON and never the
    rendered text, which is exactly how a fabricated example survives."""
    from tradeflow.analytics.reporting import format_campaign_material

    material = {
        "available": True,
        "trial_id": "t1",
        "kind": "walkforward",
        "recipe": {
            "available": True,
            "window": {"start": "2024-01-01", "end": "2024-06-01"},
            "validation": {"train_days": 252},
            "folded_into_identity": {},
            "context": {
                "capital": {"recorded": True, "value": 250_000.0},
                "scanner": {"recorded": False, "value": None},
            },
        },
        "evidence": {
            "accounting": 5,
            "metrics": {},
            "trial_ids": ["t1"],
            "probes": {"recorded": True, "value": {"leakage_probe": {"ran": True}}},
        },
        "metadata": {"artifacts": [], "notes": {"recorded": True, "value": "paper week two"}},
    }

    printed = format_campaign_material(material)

    assert "SET UP WITH" in printed
    assert "250,000.00" in printed  # a recorded fact shows its value
    assert "not recorded" in printed  # an unrecorded one says so, rather than blank
    assert "leakage_probe" in printed  # probes reach the page
    assert "paper week two" in printed  # so do notes


def test_capital_cannot_contradict_the_campaign_it_was_given():
    """R2. `_reconcile_book` guarded the book and nothing else, so a caller could hand
    over campaign material recording 250,000 and write a config saying 1. Guarding one
    runnable field and not the next leaves the same contradiction one column over."""
    import tempfile

    from tradeflow.services import configs

    material = {"recipe": {"context": {"capital": {"recorded": True, "value": 250_000.0}}}}
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(ValueError, match="disagrees"):
        configs.save_config(
            f"{tmp}/bad.json",
            strategy="demo_trend",
            params={"fast_ema_period": 9},
            capital=1.0,
            provenance={"campaign": material},
        )


def test_an_omitted_capital_is_filled_from_the_campaign_and_an_agreeing_one_passes():
    """R2, both directions. The guard must fill what the caller left out and accept the
    value it was drawn to protect — a guard that only refuses is indistinguishable from
    one that refuses everything."""
    import tempfile

    from tradeflow.services import configs

    material = {"recipe": {"context": {"capital": {"recorded": True, "value": 250_000.0}}}}
    with tempfile.TemporaryDirectory() as tmp:
        filled = configs.save_config(
            f"{tmp}/filled.json",
            strategy="demo_trend",
            params={"fast_ema_period": 9},
            provenance={"campaign": material},
        )
        assert json.loads(open(filled["path"]).read())["capital"] == pytest.approx(250_000.0)

        agreeing = configs.save_config(
            f"{tmp}/ok.json",
            strategy="demo_trend",
            params={"fast_ema_period": 9},
            capital=250_000.0,
            provenance={"campaign": material},
        )
        assert json.loads(open(agreeing["path"]).read())["capital"] == pytest.approx(250_000.0)


def test_a_campaign_with_no_recorded_capital_leaves_the_callers_value_alone():
    """R2, the absent case. Nothing to reconcile against means nothing is invented and
    nothing is refused."""
    import tempfile

    from tradeflow.services import configs

    with tempfile.TemporaryDirectory() as tmp:
        saved = configs.save_config(
            f"{tmp}/plain.json",
            strategy="demo_trend",
            params={"fast_ema_period": 9},
            capital=7.0,
            provenance={"campaign": {"recipe": {"context": {}}}},
        )
        assert json.loads(open(saved["path"]).read())["capital"] == pytest.approx(7.0)


def test_campaign_material_reads_back_the_universe_it_claims_to(tmp_path):
    """R3. The module docstring said it reads the universe back out of the journal, and
    it contained no reference to symbols at all — so a config's symbols could contradict
    the campaign with nothing able to notice, and the claim was simply false."""
    journal = tmp_path / "journal.jsonl"
    trial_id = journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA", "BBB"],
        candidate_symbols=["AAA", "BBB", "CCC"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast_ema_period": 5},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )

    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        universe = campaign_material(store, trial_id, journal_path=journal)["recipe"]["universe"]

    assert universe["symbols"] == ["AAA", "BBB"]
    assert universe["candidate_symbols"] == ["AAA", "BBB", "CCC"]


def test_a_trial_with_no_candidate_list_reports_none_rather_than_its_own_universe(tmp_path):
    """R3, both directions. The resolved book and the list it was resolved *from* are
    different decisions, and a trial that recorded only the first must not appear to
    have recorded both."""
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
        universe = campaign_material(store, trial_id, journal_path=journal)["recipe"]["universe"]

    assert universe["symbols"] == ["AAA"]
    assert universe["candidate_symbols"] is None


def test_a_workers_run_records_the_cache_it_actually_read_through():
    """R4. Parallel execution is cache-backed by construction — `_worker_data_spec`'s own
    docstring says asking for workers implies the cache whether or not `--cache` was
    passed. Recording the flag alone had a `--workers 4` run write `cache: false` while
    every bar it read came through the cache: a record less true than the run."""
    implied = cache_policy(cache=None, workers=4)

    assert implied["cache"] is True
    assert implied["cache_implied_by"] == "workers"
    assert implied["workers"] == 4


def test_a_sequential_run_records_only_the_cache_it_was_asked_for():
    """R4, both directions. The implication must not fire where it does not hold, or
    every run would claim a cache it never used."""
    assert cache_policy(cache=False, workers=1) == {"cache": False, "workers": 1}
    assert "cache_implied_by" not in cache_policy(cache=True, workers=1)
    assert cache_policy() == {}


def test_the_effective_scanner_clock_is_recorded_and_marked_as_defaulted():
    """R5. The universe is resolved at `args.scan_as_of or args.end`, but only the flag
    was recorded — so a defaulted run recorded no clock at all, and the clock is what
    decides whether the universe could have seen the future."""
    from argparse import Namespace

    from tradeflow.cli import _run_context

    defaulted = _run_context(Namespace(scanner="volume_spike", scan_as_of=None, end=datetime(2024, 6, 1)))
    assert defaulted["scan_as_of"].startswith("2024-06-01")
    assert defaulted["scan_as_of_explicit"] is False

    chosen = _run_context(
        Namespace(scanner="volume_spike", scan_as_of=datetime(2024, 3, 1), end=datetime(2024, 6, 1))
    )
    assert chosen["scan_as_of"].startswith("2024-03-01")
    assert chosen["scan_as_of_explicit"] is True


def test_a_run_with_no_scanner_records_no_scanner_clock():
    """R5, the absent case. With symbols given directly there is no scanner clock, and
    recording the window end as one would invent a resolution that never happened."""
    from argparse import Namespace

    from tradeflow.cli import _run_context

    context = _run_context(Namespace(scanner=None, scan_as_of=None, end=datetime(2024, 6, 1)))

    assert "scan_as_of" not in context
    assert "scan_as_of_explicit" not in context


def test_every_journalling_path_that_can_record_context_does(_isolated_state):
    """R6. The draft walk-forward and the service optimize path passed no `context=` at
    all, so the commit's claim that run context is recorded was untrue on both — a field
    landing on one surface and not another, which is the failure this whole branch is
    about."""
    import ast

    import tradeflow.cli
    import tradeflow.services.analysis

    missing = []
    for module in (tradeflow.cli, tradeflow.services.analysis):
        tree = ast.parse(open(module.__file__).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", getattr(node.func, "id", "")) != "journal_trial":
                continue
            if "context" not in {kw.arg for kw in node.keywords}:
                missing.append(f"{module.__name__}:{node.lineno}")

    assert missing == [], f"journal_trial call sites recording no run context: {missing}"
