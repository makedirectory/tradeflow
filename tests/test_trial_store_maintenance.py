"""Retiring an era and quarantining a subset are different jobs, and neither rewrites.

The journal is append-only and has nothing behind it to rebuild from, so a record
written in March is still there in September and no migration can reach it. Everything
here obeys that: a quarantine is an appended *fact about* trials, never an edit to them,
and an archive moves files rather than deleting them.

The failure these exist to prevent is the one a user hit by hand: moving `trials.db`
without `research_journal.jsonl`, or the reverse, leaves a store indexing a file that is
not there — or an era's rows sitting beside a fresh journal, reporting a
multiple-testing count for evidence that no longer exists, with nothing erroring because
both files are individually valid.
"""

from datetime import datetime

import pytest

from tradeflow.services import audit, maintenance
from tradeflow.store.trials import TrialStore, db_path_for_journal

_SYMBOLS = ["AAA", "BBB"]
_START, _END = datetime(2024, 1, 2), datetime(2024, 6, 28)


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """A journal, its sibling store, and a state root of this test's own."""
    from tradeflow.store import trials as trials_module

    path = tmp_path / "logs" / "research_journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", path)
    monkeypatch.setattr(trials_module, "DEFAULT_JOURNAL_PATH", path)
    return path


def _record(journal, *, params, kind="backtest", strategy="demo_trend"):
    return audit.journal_trial(
        kind,
        strategy=strategy,
        symbols=_SYMBOLS,
        start=_START,
        end=_END,
        params=params,
        metrics={"sharpe_ratio": 1.4, "total_trades": 40, "deflated_sharpe_ratio": 0.7},
        path=journal,
    )


def _store(journal):
    return TrialStore(db_path_for_journal(journal), journal_path=journal)


def _seed(journal, n=2):
    ids = [_record(journal, params={"fast_ema_period": 5 + i}) for i in range(n)]
    with _store(journal) as store:
        store.rebuild(journal)
    return ids


def _lines(path):
    return [line for line in path.read_text().splitlines() if line.strip()]


# --- quarantine appends, never edits ----------------------------------------------
def test_a_quarantine_leaves_every_recorded_trial_byte_for_byte_unchanged(journal):
    """The append-only contract. A record written badly in March is still there in
    September and no migration can reach it — so what was learned later is a new line,
    not an edit to an old one."""
    _seed(journal)
    before = _lines(journal)

    maintenance.mark_contaminated(reason="vendor split correction", journal_path=journal)

    after = _lines(journal)
    assert after[: len(before)] == before
    assert len(after) == len(before) + 1


def test_the_quarantine_survives_the_store_being_thrown_away(journal):
    """The store is derived and gets rebuilt routinely — every schema bump does it. A
    quarantine written only into the database would silently disappear at the next
    rebuild, and the rows would quietly become evidence again."""
    _seed(journal)
    maintenance.mark_contaminated(reason="vendor split correction", journal_path=journal)

    with _store(journal) as store:
        store.rebuild(journal)
        assert store.contaminated_count() == 2


def test_the_reason_is_on_the_record_not_just_the_fact(journal):
    _seed(journal, n=1)
    maintenance.mark_contaminated(reason="scanner used a stale universe", journal_path=journal)

    with _store(journal) as store:
        row = store.list_trials(all_accounting=True, limit=5)[0]

    assert row["contamination_reason"] == "scanner used a stale universe"
    assert row["contaminated_at"]


def test_a_quarantine_without_a_reason_is_refused(journal):
    """Rows excluded from every leaderboard with nothing saying why cannot be judged
    later, which is worse than not excluding them."""
    _seed(journal, n=1)

    with pytest.raises(ValueError, match="reason"):
        maintenance.mark_contaminated(reason="   ", journal_path=journal)


def test_a_dry_run_writes_nothing_at_all(journal):
    _seed(journal)
    before = _lines(journal)

    report = maintenance.mark_contaminated(reason="checking", journal_path=journal, dry_run=True)

    assert report["to_mark"] == 2
    assert report["applied"] == 0
    assert _lines(journal) == before
    with _store(journal) as store:
        assert store.contaminated_count() == 0


def test_marking_the_same_trials_twice_does_not_re_mark_them(journal):
    """Both directions of idempotence: the second call reports them as already marked
    rather than appending a second quarantine of the same rows."""
    _seed(journal)
    maintenance.mark_contaminated(reason="first", journal_path=journal)

    again = maintenance.mark_contaminated(reason="second", journal_path=journal)

    assert again["already_contaminated"] == 2
    assert again["to_mark"] == 0


def test_only_the_selected_subset_is_quarantined(journal):
    """It is for a *suspect subset*. A filter that quietly took everything would be an
    archive wearing a quarantine's clothes."""
    ids = _seed(journal, n=3)

    maintenance.mark_contaminated(reason="one bad run", journal_path=journal, trial_ids=[ids[1]])

    with _store(journal) as store:
        marked = {r["id"] for r in store.list_trials(all_accounting=True, limit=10) if r["contaminated_at"]}

    assert marked == {ids[1]}


# --- what a quarantine actually changes for readers -------------------------------
def test_a_quarantined_trial_is_never_served_as_a_memo(journal):
    """The behaviour that matters most. Whatever a suspect trial recorded, standing in
    for a fresh run is the one thing it must not do — falling back to a real run is
    always safe, serving a suspect number never is."""
    params = {"fast_ema_period": 5}
    _record(journal, params=params)
    with _store(journal) as store:
        store.rebuild(journal)
        found = store.find(
            strategy="demo_trend", params=params, symbols=_SYMBOLS, window_start=_START, window_end=_END
        )
    assert found is not None, "the fixture must be findable before the quarantine, or this proves nothing"

    maintenance.mark_contaminated(reason="suspect", journal_path=journal)

    with _store(journal) as store:
        assert (
            store.find(
                strategy="demo_trend",
                params=params,
                symbols=_SYMBOLS,
                window_start=_START,
                window_end=_END,
            )
            is None
        )


def test_a_quarantined_trial_still_counts_toward_the_multiple_testing_total(journal):
    """The decision that could most easily have gone the flattering way. The search
    happened — you did look at that configuration — so dropping it from the count would
    *lower* the deflation bar, which is the one direction this store must never move on
    its own."""
    _seed(journal, n=3)
    with _store(journal) as store:
        before = store.family_count("demo_trend", _SYMBOLS, 4)

    maintenance.mark_contaminated(reason="suspect", journal_path=journal)

    with _store(journal) as store:
        assert store.family_count("demo_trend", _SYMBOLS, 4) == before == 3


def test_a_quarantined_trial_is_not_ranked_and_the_exclusion_is_reported(journal):
    """Silence about what was dropped reads as "everything ran". A leaderboard that
    quietly omits rows makes its own length a claim nobody can check."""
    _seed(journal, n=3)
    maintenance.mark_contaminated(reason="suspect", journal_path=journal)

    with _store(journal) as store:
        board = store.best(all_accounting=True, limit=10)

    assert board["rows"] == []
    assert board["contaminated_excluded"] == 3


def test_an_unquarantined_trial_still_ranks(journal):
    """Both directions, or the exclusion is indistinguishable from a broken leaderboard."""
    ids = _seed(journal, n=3)
    maintenance.mark_contaminated(reason="suspect", journal_path=journal, trial_ids=ids[:2])

    with _store(journal) as store:
        board = store.best(all_accounting=True, limit=10)

    assert [r["id"] for r in board["rows"]] == [ids[2]]
    assert board["contaminated_excluded"] == 2


# --- archive moves both, together -------------------------------------------------
def test_archiving_moves_the_journal_and_its_index_together(journal):
    """The whole point, and the exact thing that was done by hand. Moving one without
    the other leaves a store indexing a file that is not there, or an era's rows beside
    a fresh journal reporting a trial count for evidence that is gone — and nothing
    errors, because both files are individually valid."""
    _seed(journal)
    db = db_path_for_journal(journal)
    assert journal.exists() and db.exists()

    report = maintenance.archive(reason="accounting bump", journal_path=journal, label="pre-v5")

    assert sorted(report["moved"]) == sorted([journal.name, db.name])
    assert not db.exists()
    destination = journal.parent.parent / "archive" / report["destination"].rsplit("/", 1)[-1]
    assert (destination / journal.name).exists()
    assert (destination / db.name).exists()


def test_the_archive_records_why_and_under_which_accounting_version(journal):
    """ "Why" is the part nobody reconstructs a year later, and the accounting version is
    what makes the retired numbers incommensurable in the first place."""
    _seed(journal)

    report = maintenance.archive(reason="accounting v5 invalidated v4 metrics", journal_path=journal)

    entries = maintenance.list_archives()
    assert len(entries) == 1
    assert entries[0]["reason"] == "accounting v5 invalidated v4 metrics"
    assert entries[0]["accounting_version_at_archive"] == report["accounting_version_at_archive"]
    assert entries[0]["rows"] == 2


def test_the_new_journal_says_why_it_is_empty(journal):
    """A fresh journal that simply starts empty loses the fact that an era existed, and
    the emptiness then reads as "nothing has ever been run here"."""
    _seed(journal)

    maintenance.archive(reason="accounting bump", journal_path=journal)

    first = _lines(journal)[0]
    assert "trials:archived" in first
    assert "accounting bump" in first


def test_an_archived_era_leaves_no_trials_behind_it(journal):
    _seed(journal, n=3)

    maintenance.archive(reason="accounting bump", journal_path=journal)

    with _store(journal) as store:
        store.rebuild(journal)
        assert store.family_count("demo_trend", _SYMBOLS, 4) == 0


def test_the_archived_era_is_still_readable_where_it_was_put(journal):
    """Moved, not deleted. The destructive version of this is exactly what the
    append-only rule forbids — a record removed is a record gone."""
    ids = _seed(journal, n=2)

    report = maintenance.archive(reason="accounting bump", journal_path=journal)

    retired = journal.parent.parent / "archive" / report["destination"].rsplit("/", 1)[-1] / journal.name
    text = retired.read_text()
    assert all(trial_id in text for trial_id in ids)


def test_an_archive_without_a_reason_is_refused(journal):
    _seed(journal, n=1)

    with pytest.raises(ValueError, match="reason"):
        maintenance.archive(reason="", journal_path=journal)


def test_a_dry_run_archive_moves_nothing(journal):
    _seed(journal)
    db = db_path_for_journal(journal)

    report = maintenance.archive(reason="checking", journal_path=journal, dry_run=True)

    assert report["moved"] == []
    assert journal.exists() and db.exists()
    assert maintenance.list_archives() == []


def test_archiving_nothing_says_so_rather_than_creating_an_empty_era(journal):
    """There is no era here to retire. Creating a directory anyway would leave a
    listing full of entries that record nothing having happened."""
    report = maintenance.archive(reason="tidy up", journal_path=journal)

    assert report["moved"] == []
    assert "Nothing to archive" in report["note"]
    assert maintenance.list_archives() == []


def test_an_archive_directory_with_no_manifest_is_reported_not_guessed(journal):
    """A directory somebody created by hand is not evidence about what it holds."""
    _seed(journal, n=1)
    maintenance.archive(reason="real one", journal_path=journal)
    (journal.parent.parent / "archive" / "hand-made").mkdir(parents=True)

    entries = {e["name"]: e for e in maintenance.list_archives()}

    assert entries["hand-made"]["manifest_error"] == "missing"
    assert "reason" not in entries["hand-made"]
