"""The trial-store browser: filters, detail, and the honest leaderboard.

All offline, against a fixture store built through the real ``record()`` path so
the rows have the shape the journal actually produces. The properties under test
are the ones a browser can get subtly wrong: a filter that means something other
than what the store means, an absent field rendered as a zero, a leaderboard that
hides how many configs were tried, and paging that quietly truncates.
"""

import json
from datetime import datetime

import pytest

from quantred.analytics.reporting import (
    NOT_RECORDED,
    format_leaderboard,
    format_trial_detail,
    format_trials_table,
)
from quantred.engine.backtest import ACCOUNTING_VERSION
from quantred.services.audit import journal_trial
from quantred.store.trials import TrialStore, db_path_for_journal

# Read from the engine, never copied: a hardcoded number here is a fixture that agrees
# with a *past* accounting version, and it goes quietly stale the moment one is bumped.
ACCOUNTING = ACCOUNTING_VERSION


@pytest.fixture
def store(tmp_path):
    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl") as s:
        yield s


def _record(store, trial_id, **kwargs):
    defaults = {
        "id": trial_id,
        # A kind the leaderboard ranks by default; a plain backtest is not one.
        "kind": "walkforward",
        "strategy": "demo_trend",
        "symbols": ["AAA", "BBB"],
        "params": {"fast": 5},
        "accounting": ACCOUNTING,
        "ts": "2025-01-01T15:00:00+00:00",  # as the journal stamps: UTC, with its offset
        "window_start": "2024-01-02T00:00:00",
        "window_end": "2024-12-31T00:00:00",
    }
    defaults.update(kwargs)
    store.record(**defaults)
    return defaults


def _populate(store):
    _record(
        store, "t1", ts="2025-01-01T15:00:00+00:00", oos_sharpe=0.4, deflated_sharpe=0.1, promotable=False
    )
    _record(store, "t2", ts="2025-02-01T15:00:00+00:00", oos_sharpe=2.1, deflated_sharpe=0.2, promotable=True)
    _record(store, "t3", ts="2025-03-01T15:00:00+00:00", oos_sharpe=1.5, deflated_sharpe=0.9, promotable=True)
    # A different family, and a kind with no Sharpe at all.
    _record(
        store,
        "t4",
        ts="2025-04-01T15:00:00+00:00",
        strategy="another_strategy",
        symbols=["CCC"],
        oos_sharpe=3.0,
        deflated_sharpe=0.05,
    )
    _record(
        store,
        "t5",
        ts="2025-05-01T15:00:00+00:00",
        kind="alpha",
        params={"fast": 9},
        oos_sharpe=None,
        deflated_sharpe=None,
    )


# --- filtering --------------------------------------------------------------
def test_filters_narrow_to_what_they_say(store):
    _populate(store)
    assert {r["id"] for r in store.list_trials(accounting=ACCOUNTING)} == {"t1", "t2", "t3", "t4", "t5"}
    assert {r["id"] for r in store.list_trials(strategy="another_strategy", accounting=ACCOUNTING)} == {"t4"}
    assert {r["id"] for r in store.list_trials(kind="alpha", accounting=ACCOUNTING)} == {"t5"}
    assert {r["id"] for r in store.list_trials(min_sharpe=1.5, accounting=ACCOUNTING)} == {"t2", "t3", "t4"}
    assert {r["id"] for r in store.list_trials(promotable=True, accounting=ACCOUNTING)} == {"t2", "t3"}
    assert {r["id"] for r in store.list_trials(since=datetime(2025, 2, 1), accounting=ACCOUNTING)} >= {
        "t2",
        "t3",
    }
    assert {r["id"] for r in store.list_trials(until=datetime(2025, 1, 15), accounting=ACCOUNTING)} == {"t1"}


def test_symbol_filtering_ignores_order_and_case(store):
    """The store's definition of "the same universe" is the normalized hash, and the
    browser must use it — not string matching over however the user typed it."""
    _populate(store)
    canonical = store.list_trials(symbols=["AAA", "BBB"], accounting=ACCOUNTING)
    reordered = store.list_trials(symbols=["bbb", " aaa "], accounting=ACCOUNTING)
    assert canonical and [r["id"] for r in canonical] == [r["id"] for r in reordered]


def test_min_sharpe_never_admits_a_trial_that_has_no_sharpe(store):
    """A kind with no Sharpe did not score zero; it must not sneak past a floor."""
    _populate(store)
    assert "t5" not in {r["id"] for r in store.list_trials(min_sharpe=-99.0, accounting=ACCOUNTING)}


# --- sorting and paging -----------------------------------------------------
def test_sorting_puts_unrecorded_metrics_last_not_first(store):
    _populate(store)
    by_sharpe = [r["id"] for r in store.list_trials(sort="sharpe", accounting=ACCOUNTING)]
    assert by_sharpe[0] == "t4"  # 3.0
    assert by_sharpe[-1] == "t5"  # no Sharpe recorded — last, not "worst"

    by_dsr = [r["id"] for r in store.list_trials(sort="dsr", accounting=ACCOUNTING)]
    assert by_dsr[0] == "t3"  # 0.9 — the deflated winner is not the raw winner
    assert by_dsr[-1] == "t5"


def test_an_unknown_sort_is_refused(store):
    with pytest.raises(ValueError, match="sort must be"):
        store.list_trials(sort="profit")


def test_paging_happens_in_sql_and_reports_the_full_count(store):
    _populate(store)
    page1 = store.list_trials(sort="date", limit=2, offset=0, accounting=ACCOUNTING)
    page2 = store.list_trials(sort="date", limit=2, offset=2, accounting=ACCOUNTING)
    assert len(page1) == 2 and len(page2) == 2
    assert not ({r["id"] for r in page1} & {r["id"] for r in page2})
    assert store.count_trials(accounting=ACCOUNTING) == 5


def test_the_listing_says_how_many_it_did_not_show(store):
    _populate(store)
    rows = store.list_trials(limit=2, accounting=ACCOUNTING)
    rendered = format_trials_table(rows, total=5)
    assert "Showing 2 of 5" in rendered


# --- absent is not zero -----------------------------------------------------
def test_absent_metrics_render_as_absent(store):
    """A trial recorded before a field existed did not fail to record it."""
    _record(store, "old", oos_sharpe=None, deflated_sharpe=None, promotable=None)
    rendered = format_trials_table(store.list_trials(accounting=ACCOUNTING), total=1)
    assert NOT_RECORDED in rendered
    assert "0.000" not in rendered


def test_a_pre_companion_trial_reports_its_companions_as_not_recorded(store):
    _record(store, "old")
    trial = store.get_trial("old")
    assert trial["returns"] is None
    assert trial["weights"] is None
    assert trial["trades"] is None

    rendered = format_trial_detail(trial)
    assert rendered.count("not recorded") >= 3
    assert "0 periods" not in rendered
    assert "0 names" not in rendered


# --- the detail view --------------------------------------------------------
def test_detail_gathers_the_row_and_its_companions(store):
    _record(store, "t1", oos_sharpe=1.2, deflated_sharpe=0.6, promotable=True)
    store.record_returns("t1", ["2024-01-02", "2024-01-03"], [0.01, -0.02])
    store.record_weights("t1", {"as_of": "2024-12-31", "weights": {"AAA": 0.6}, "exposures": {"market": 0.1}})
    store.record_trades("t1", {"columns": ["symbol", "pnl"], "rows": [["AAA", 12.5]]})

    trial = store.get_trial("t1")
    assert trial["params"] == {"fast": 5}
    assert trial["returns"] == {"periods": 2, "start": "2024-01-02", "end": "2024-01-03"}
    assert trial["weights"]["weights"] == {"AAA": 0.6}
    assert trial["trades"]["rows"] == [["AAA", 12.5]]

    rendered = format_trial_detail(trial)
    assert "2 periods" in rendered
    assert "1 name" in rendered and "factor exposures" in rendered


def test_detail_on_an_unknown_id_is_none(store):
    assert store.get_trial("nope") is None


def test_show_lists_the_later_trials_that_reused_this_one(store):
    """A reused number should be traceable in both directions."""
    _record(store, "origin", ts="2025-01-01T00:00:00")
    _record(store, "reuse", ts="2025-06-01T00:00:00")  # same params/universe/window
    _record(store, "other", ts="2025-07-01T00:00:00", params={"fast": 99})

    reused = store.get_trial("origin")["reused_by"]
    assert [r["id"] for r in reused] == ["reuse"]
    # And the later one does not claim the earlier one reused it.
    assert store.get_trial("reuse")["reused_by"] == []
    assert "reuse" in format_trial_detail(store.get_trial("origin"))


# --- the leaderboard --------------------------------------------------------
def test_best_ranks_by_deflated_sharpe_by_default(store):
    _populate(store)
    board = store.best(accounting=ACCOUNTING, limit=3)
    assert board["rank_by"] == "dsr"
    assert board["rows"][0]["id"] == "t3"  # highest DSR, not highest raw Sharpe


def test_every_leaderboard_row_carries_its_family_trial_count(store):
    """The count lives in the payload, not only in the terminal formatting — an
    agent reading this over a wire sees the same caveat a human does."""
    _populate(store)
    board = store.best(accounting=ACCOUNTING)
    assert all("family_n_trials" in row for row in board["rows"])
    demo = next(r for r in board["rows"] if r["strategy"] == "demo_trend")
    # t1/t2/t3 count; the alpha row never does (a forecast has no Sharpe to deflate).
    assert demo["family_n_trials"] == 3
    assert board["caveat"]


def test_raw_sharpe_ranking_carries_its_own_caveat(store):
    _populate(store)
    board = store.best(rank_by="sharpe", accounting=ACCOUNTING)
    assert board["rows"][0]["id"] == "t4"
    assert "RAW Sharpe" in board["caveat"]
    assert "does not correct" in board["caveat"]

    rendered = format_leaderboard(board)
    assert "RAW Sharpe" in rendered
    assert "FAMILY n_trials" in rendered


def test_the_rendered_leaderboard_always_shows_family_counts(store):
    _populate(store)
    rendered = format_leaderboard(store.best(accounting=ACCOUNTING))
    assert "FAMILY n_trials" in rendered
    assert "deflated Sharpe" in rendered


def test_a_large_family_gets_an_extra_warning(store):
    for i in range(60):
        _record(store, f"x{i}", oos_sharpe=0.1 * i, deflated_sharpe=0.01 * i)
    rendered = format_leaderboard(store.best(accounting=ACCOUNTING, limit=3))
    assert "largely selection" in rendered


def test_an_unknown_ranking_is_refused(store):
    with pytest.raises(ValueError, match="rank_by must be"):
        store.best(rank_by="total_return")


# --- trade-table persistence ------------------------------------------------
def test_trades_round_trip_through_the_journal_alone(tmp_path):
    """The journal is the source of truth: a rebuild must reconstruct the table."""
    from quantred.services import audit

    journal = tmp_path / "journal.jsonl"
    trial_id = audit.journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 2),
        end=datetime(2024, 12, 31),
        params={"fast": 5},
        metrics={"sharpe_ratio": 1.0},
        trades={"columns": ["symbol", "pnl"], "rows": [["AAA", 1.5], ["AAA", -0.5]]},
        path=journal,
    )
    from quantred.store.trials import db_path_for_journal

    with TrialStore(db_path_for_journal(journal), journal_path=journal) as s:
        assert s.trades_for(trial_id)["rows"] == [["AAA", 1.5], ["AAA", -0.5]]
        s.rebuild()
        assert s.trades_for(trial_id)["rows"] == [["AAA", 1.5], ["AAA", -0.5]]


def test_without_the_flag_a_trial_stores_no_trade_table(tmp_path):
    from quantred.services import audit
    from quantred.store.trials import db_path_for_journal

    journal = tmp_path / "journal.jsonl"
    trial_id = audit.journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 2),
        end=datetime(2024, 12, 31),
        params={"fast": 5},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as s:
        assert s.trades_for(trial_id) is None


def test_a_truncated_trade_table_says_so(tmp_path):
    """A ceiling on stored rows is fine; a silent one is not."""
    import pandas as pd

    from quantred.services.analysis import trades_payload

    frame = pd.DataFrame({"symbol": ["AAA"] * 10, "pnl": range(10)})
    payload = trades_payload(frame, max_rows=4)
    assert len(payload["rows"]) == 4
    assert payload["truncated"] is True
    assert payload["total_rows"] == 10

    assert trades_payload(None) is None

    # A complete table says so rather than leaving the key out. Absence has to keep
    # meaning "written before this was recorded"; a payload that simply omits the flag
    # when all is well makes that unreadable one layer down.
    full = trades_payload(frame)
    assert full["truncated"] is False and full["total_rows"] == 10

    # No ceiling at all, for an in-memory result that is the whole frame by definition.
    uncapped = trades_payload(frame, max_rows=None)
    assert uncapped["truncated"] is False and len(uncapped["rows"]) == 10


def test_a_truncated_trade_table_is_still_truncated_after_the_store_writes_it(tmp_path):
    """The payload said so and the store dropped it on the floor.

    ``trades_payload`` has always recorded ``total_rows``/``truncated``; the
    ``trial_trades`` table stored only the columns and rows, so a table capped at the
    storage ceiling read back indistinguishable from a complete one — and every total
    taken over it was short by whatever was cut, with nothing saying so.
    """
    import pandas as pd

    from quantred.services.analysis import trades_payload
    from quantred.store.trials import TrialStore

    frame = pd.DataFrame({"symbol": ["AAA"] * 10, "pnl": range(10)})
    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "j.jsonl") as store:
        store.record_trades("capped", trades_payload(frame, max_rows=4))
        store.record_trades("whole", trades_payload(frame))

        capped = store.trades_for("capped")
        assert capped["truncated"] is True
        assert capped["total_rows"] == 10 and len(capped["rows"]) == 4

        # Both directions: a complete table must not be labelled as capped.
        whole = store.trades_for("whole")
        assert whole["truncated"] is False
        assert whole["total_rows"] == 10 and len(whole["rows"]) == 10


def test_a_trade_table_stored_before_the_count_existed_is_unknown_not_complete(tmp_path):
    """A row written before the store kept a total did not *prove* it held every
    trade, and rendering it as complete would put a confident claim on the one thing
    the record cannot support. Absent is not false."""
    from quantred.store.trials import TrialStore

    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "j.jsonl") as store:
        store.record_trades("old", {"columns": ["symbol", "pnl"], "rows": [["AAA", 1.0]]})
        stored = store.trades_for("old")

    assert stored["truncated"] is None
    assert stored["total_rows"] is None


def test_the_completeness_of_a_stored_table_survives_a_rebuild(tmp_path):
    """The journal carries the whole payload, so nothing here may be a fact only the
    index holds — that is the rule a derived store lives by."""
    import pandas as pd

    from quantred.services.analysis import trades_payload
    from quantred.services.audit import journal_trial
    from quantred.store.trials import TrialStore, db_path_for_journal

    journal = tmp_path / "journal.jsonl"
    frame = pd.DataFrame({"symbol": ["AAA"] * 10, "pnl": range(10)})
    trial_id = journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast": 5},
        metrics={"sharpe_ratio": 1.0},
        trades=trades_payload(frame, max_rows=4),
        path=journal,
    )
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        store.rebuild(journal)
        rebuilt = store.trades_for(trial_id)

    assert rebuilt["truncated"] is True
    assert rebuilt["total_rows"] == 10


def test_a_capped_table_is_described_as_capped_wherever_it_is_described(tmp_path):
    """The user-facing half. `trials show` printed "4 trades" for a run that made ten."""
    from quantred.analytics.reporting import describe_trade_table

    capped = {"columns": [], "rows": [[]] * 4, "total_rows": 10, "truncated": True}
    whole = {"columns": [], "rows": [[]] * 4, "total_rows": 4, "truncated": False}
    unknown = {"columns": [], "rows": [[]] * 4, "total_rows": None, "truncated": None}

    assert "TRUNCATED" in describe_trade_table(capped)
    assert "4 of 10" in describe_trade_table(capped)
    assert describe_trade_table(whole) == "4 trades"
    assert "not recorded" in describe_trade_table(unknown)
    # Absent, zero, and capped are three different sentences.
    assert "not recorded" in describe_trade_table(None)
    assert describe_trade_table({"columns": [], "rows": [], "total_rows": 0, "truncated": False}) == (
        "0 trades"
    )


# --- the CLI surface --------------------------------------------------------
def _cli(monkeypatch, tmp_path, *argv):
    from quantred import cli as main

    args = main.build_parser().parse_args(["trials", *argv, "--db", str(tmp_path / "trials.db")])
    args.func(args)


def test_cli_list_show_and_best_round_trip_as_json(monkeypatch, tmp_path, capsys):
    from quantred import cli as main

    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl") as s:
        _populate(s)

    for argv in (["list", "--json"], ["best", "--json"]):
        args = main.build_parser().parse_args(["trials", *argv, "--db", str(tmp_path / "trials.db")])
        args.func(args)
        payload = json.loads(capsys.readouterr().out)
        assert payload

    args = main.build_parser().parse_args(
        ["trials", "show", "t3", "--json", "--db", str(tmp_path / "trials.db")]
    )
    args.func(args)
    assert json.loads(capsys.readouterr().out)["id"] == "t3"


def test_cli_show_on_an_unknown_id_exits_non_zero(tmp_path):
    from quantred import cli as main

    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl"):
        pass
    args = main.build_parser().parse_args(["trials", "show", "missing", "--db", str(tmp_path / "trials.db")])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert "missing" in str(exc.value)


def test_query_remains_an_alias_for_list(tmp_path, capsys):
    from quantred import cli as main

    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl") as s:
        _populate(s)
    for verb in ("list", "query"):
        args = main.build_parser().parse_args(["trials", verb, "--db", str(tmp_path / "trials.db")])
        args.func(args)
        assert "t3" in capsys.readouterr().out


# --- the leaderboard must not rank search artifacts -------------------------
def test_in_sample_kinds_are_excluded_from_the_leaderboard_by_default(store):
    """An `optimize` row is the winner of a search — best-of-N by construction — so
    ranking one ranks the selection bias itself. Found by using the tool: four of the
    top five rows were in-sample, and nothing in the output said so."""
    _record(store, "opt", kind="optimize", oos_sharpe=9.9, deflated_sharpe=1.0)
    _record(store, "fc", kind="alpha", oos_sharpe=8.8, deflated_sharpe=1.0)
    _record(store, "real", kind="walkforward", oos_sharpe=1.1, deflated_sharpe=0.6)

    board = store.best(accounting=ACCOUNTING)
    assert [r["id"] for r in board["rows"]] == ["real"]
    assert board["in_sample_excluded"] == 2
    assert board["in_sample_included"] is False


def test_a_plain_backtest_is_not_ranked_as_validated_evidence(store):
    """A backtest stored its full-window Sharpe as `oos_sharpe` and was ranked beside
    walk-forwards under "validated runs only", so a novice read one pass over a window
    as out-of-sample evidence. It is shown only when asked for, and labelled then."""
    _record(store, "bt", kind="backtest", oos_sharpe=2.0, deflated_sharpe=0.9)
    _record(store, "wf", kind="walkforward", oos_sharpe=1.1, deflated_sharpe=0.6)

    board = store.best(accounting=ACCOUNTING)
    assert [r["id"] for r in board["rows"]] == ["wf"]
    assert board["in_sample_excluded"] == 1

    opted_in = store.best(accounting=ACCOUNTING, include_in_sample=True)
    assert [r["id"] for r in opted_in["rows"]] == ["bt", "wf"]
    assert "never validated out of sample" in format_leaderboard(opted_in)


def test_a_store_full_of_backtests_still_ranks_its_validated_row(store):
    """Found in review: the exclusion was applied to an over-fetched page, so with plain
    backtests now excluded, twenty of them and one walk-forward ranked by raw Sharpe gave
    an empty board and reported 16 exclusions instead of 20."""
    for i in range(20):
        _record(store, f"bt{i}", kind="backtest", oos_sharpe=5.0 + i, deflated_sharpe=None)
    _record(store, "wf", kind="walkforward", oos_sharpe=1.1, deflated_sharpe=0.6)

    board = store.best(accounting=ACCOUNTING, rank_by="sharpe", limit=2)
    assert [r["id"] for r in board["rows"]] == ["wf"]
    assert board["in_sample_excluded"] == 20


def test_the_header_names_the_population_not_just_the_sort_key(store):
    """ "Top 3 by deflated Sharpe" still reads as "the best of everything I have run"
    unless it says what it ranked over."""
    _record(store, "real", kind="walkforward", oos_sharpe=1.1, deflated_sharpe=0.6)
    assert "validated runs only" in format_leaderboard(store.best(accounting=ACCOUNTING))
    assert "validated runs only" not in format_leaderboard(
        store.best(accounting=ACCOUNTING, include_in_sample=True)
    )


def test_the_exclusion_is_reported_rather_than_silent(store):
    _record(store, "opt", kind="optimize", oos_sharpe=9.9, deflated_sharpe=1.0)
    _record(store, "real", kind="walkforward", oos_sharpe=1.1, deflated_sharpe=0.6)
    rendered = format_leaderboard(store.best(accounting=ACCOUNTING))
    assert "1 in-sample row(s) excluded" in rendered


def test_opting_in_ranks_them_but_says_what_that_means(store):
    _record(store, "opt", kind="optimize", oos_sharpe=9.9, deflated_sharpe=1.0)
    _record(store, "real", kind="walkforward", oos_sharpe=1.1, deflated_sharpe=0.6)

    board = store.best(accounting=ACCOUNTING, include_in_sample=True)
    assert board["rows"][0]["id"] == "opt"
    assert board["in_sample_included"] is True
    assert "IN-SAMPLE rows included" in format_leaderboard(board)


def test_excluding_in_sample_rows_does_not_shorten_the_board(store):
    """Dropping rows after a LIMIT would silently return fewer than asked for."""
    for i in range(12):
        _record(store, f"opt{i}", kind="optimize", oos_sharpe=9.0, deflated_sharpe=0.99)
    for i in range(5):
        _record(store, f"real{i}", kind="walkforward", oos_sharpe=1.0 + i, deflated_sharpe=0.5 + i / 100)

    board = store.best(accounting=ACCOUNTING, limit=5)
    assert len(board["rows"]) == 5
    assert all(r["kind"] == "walkforward" for r in board["rows"])


def test_the_rendered_leaderboard_names_each_row_s_kind(store):
    """Without it, a validated walk-forward and a search's winning candidate look
    identical — and they mean opposite things about whether a number is evidence."""
    _record(store, "real", kind="walkforward", oos_sharpe=1.1, deflated_sharpe=0.6)
    rendered = format_leaderboard(store.best(accounting=ACCOUNTING))
    assert "KIND" in rendered
    assert "walkforward" in rendered


# --- review findings: the exit code must not depend on the output format ------------
@pytest.mark.parametrize(
    ("argv", "why"),
    [
        (("analyze", "no-such-trial-id"), "a trial with no recorded trade table"),
        (("compare", "no-such-a", "no-such-b"), "two trials with nothing to correlate"),
        (("campaign", "no-such-trial-id"), "a trial that is not indexed"),
    ],
)
def test_json_output_still_exits_non_zero_when_there_is_no_answer(monkeypatch, tmp_path, capsys, argv, why):
    """Review finding. All three commands returned from the `--json` branch *before*
    the exit-code check, so `tradeflow trials analyze $ID --json || alert` saw success
    on a trial that was never recorded with --record-trades. `--json` is the form a
    script reads, so an exit code only the human-readable path sets is set for nobody.
    """
    import json as json_mod

    from quantred import cli as main

    journal = tmp_path / "journal.jsonl"
    journal.write_text("")
    monkeypatch.setattr("quantred.services.audit.DEFAULT_TRIAL_JOURNAL", journal, raising=False)

    args = main.build_parser().parse_args(["trials", *argv, "--json", "--db", str(tmp_path / "trials.db")])
    if argv[0] == "analyze":
        # An unknown id exits with a message on this one; use a real trial with no
        # trade table so the path under test is "nothing to total", not "no such id".
        trial_id = journal_trial(
            "backtest",
            strategy="s",
            symbols=["AAA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 1),
            params={"fast": 5},
            metrics={"sharpe_ratio": 1.0},
            path=journal,
        )
        args.trial_id = trial_id

    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1, why

    # And it is still valid JSON, so the caller can read why.
    printed = capsys.readouterr().out
    assert json_mod.loads(printed)


def test_json_output_exits_zero_when_there_is_an_answer(monkeypatch, tmp_path, capsys):
    """Both directions: the exit code must not fire on a successful report."""
    import pandas as pd

    from quantred import cli as main
    from quantred.services.analysis import trades_payload

    journal = tmp_path / "journal.jsonl"
    trial_id = journal_trial(
        "backtest",
        strategy="s",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast": 5},
        metrics={"sharpe_ratio": 1.0},
        trades=trades_payload(pd.DataFrame({"pnl": [1.0, -2.0], "exit_reason": ["TP", "SL"]})),
        path=journal,
    )
    monkeypatch.setattr("quantred.services.audit.DEFAULT_TRIAL_JOURNAL", journal, raising=False)

    args = main.build_parser().parse_args(
        ["trials", "analyze", trial_id, "--json", "--db", str(db_path_for_journal(journal))]
    )
    args.func(args)  # no SystemExit
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


# --- what the trials commands say when an id or a journal is not there -------------
def _run_trials(monkeypatch, tmp_path, *argv):
    from quantred import cli as main

    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr("quantred.services.audit.DEFAULT_TRIAL_JOURNAL", journal, raising=False)
    args = main.build_parser().parse_args(["trials", *argv, "--db", str(tmp_path / "trials.db")])
    args.func(args)


def test_a_fresh_state_root_is_not_reported_as_a_lost_journal(monkeypatch, tmp_path, capsys):
    """`trials status` on a root where nothing had been run printed `Journal : UNREADABLE
    - ... restore it`, then `OK`. A journal never written beside an empty index is where
    everyone starts, not data loss."""
    journal = tmp_path / "journal.jsonl"
    _run_trials(monkeypatch, tmp_path, "status", "--journal", str(journal))
    out = capsys.readouterr().out
    assert "UNREADABLE" not in out
    assert "none yet" in out
    assert "OK" in out


def test_a_missing_journal_under_indexed_rows_still_raises_the_alarm(monkeypatch, tmp_path, capsys):
    """The other direction: rows whose source is gone are the real loss, and the fresh
    root exemption must not swallow them."""
    journal = tmp_path / "journal.jsonl"
    with TrialStore(tmp_path / "trials.db", journal_path=journal) as s:
        _record(s, "t1")
    _run_trials(monkeypatch, tmp_path, "status", "--journal", str(journal))
    out = capsys.readouterr().out
    assert "UNREADABLE" in out
    assert "DRIFT DETECTED" in out


def test_analyze_names_the_cli_command_and_the_service_names_neither(monkeypatch, tmp_path):
    """The shared service said `Use list_trials` - an MCP tool name - to a CLI user.
    The service now names no surface; each surface appends its own pointer."""
    from quantred.services.analysis import analyze_trial

    with pytest.raises(SystemExit) as exc:
        _run_trials(monkeypatch, tmp_path, "analyze", "abc123")
    message = str(exc.value.code)
    assert "abc123" in message and "trials list" in message and "list_trials" not in message

    with TrialStore(tmp_path / "t2.db", journal_path=tmp_path / "j2.jsonl") as s:
        assert "list_trials" not in analyze_trial(s, "abc123")["error"]


@pytest.mark.parametrize("ids", [("abc",), ("abc", "def")])
def test_comparing_unknown_or_too_few_ids_gives_a_refusal_not_an_empty_report(monkeypatch, tmp_path, ids):
    """`trials compare abc def` printed the whole `=== Return-series comparison ===`
    scaffold ending `0 of 0 pair(s) compared`. One id is a usage error; unknown ids are
    named with a pointer to the listing."""
    with pytest.raises(SystemExit) as exc:
        _run_trials(monkeypatch, tmp_path, "compare", *ids)
    message = str(exc.value.code)
    assert "Return-series comparison" not in message
    if len(ids) == 1:
        assert "at least two" in message
    else:
        assert "'abc'" in message and "'def'" in message and "trials list" in message


def test_campaign_for_an_unknown_id_points_at_the_listing(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_trials(monkeypatch, tmp_path, "campaign", "abc123")
    assert exc.value.code == 1
    assert "trials list" in capsys.readouterr().out


def test_quarantining_an_id_that_matches_no_trial_fails_and_writes_nothing(monkeypatch, tmp_path, capsys):
    """`mark-contaminated --id <typo>` printed `Selected: 0 ... Nothing selected.` and
    exited 0. An explicitly named id is a claim the trial exists, so it is a typo signal;
    a filter matching nothing stays an ordinary empty answer."""
    from quantred import cli as main

    journal = tmp_path / "journal.jsonl"
    real = journal_trial(
        "backtest",
        strategy="s",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast": 5},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    before = journal.read_text()

    def run(*extra):
        args = main.build_parser().parse_args(
            ["trials", "mark-contaminated", "--reason", "t", "--journal", str(journal), *extra]
        )
        args.func(args)

    with pytest.raises(SystemExit) as exc:
        run("--id", "abc", "--id", real)
    assert "'abc'" in str(exc.value.code) and "trials list" in str(exc.value.code)
    assert journal.read_text() == before  # the good id beside the typo was not applied either

    run("--strategy", "no_such_strategy")  # a filter matching nothing: exit 0
    assert "Selected" in capsys.readouterr().out


def test_ids_and_filters_together_are_refused_rather_than_the_filters_ignored(tmp_path):
    """Found in review: beside `--id`, `--strategy`/`--kind`/`--accounting`/`--before`
    were silently ignored, so `--id X --strategy other` quarantined X anyway."""
    from quantred import cli as main

    journal = tmp_path / "journal.jsonl"
    real = journal_trial(
        "backtest",
        strategy="s",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast": 5},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )
    before = journal.read_text()
    args = main.build_parser().parse_args(
        ["trials", "mark-contaminated", "--reason", "t", "--journal", str(journal), "--id", real]
        + ["--strategy", "other"]
    )
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert "either ids or filters" in str(exc.value.code)
    assert journal.read_text() == before

    # The other direction: an id alone still quarantines.
    args = main.build_parser().parse_args(
        ["trials", "mark-contaminated", "--reason", "t", "--journal", str(journal), "--id", real]
    )
    args.func(args)
    assert journal.read_text() != before


def test_ids_that_are_all_typos_select_nothing_rather_than_the_whole_store(tmp_path):
    """Found in review: when no typed id resolved, the empty id list read as "no ids, no
    filters", so the report said every trial in the store would be quarantined. Only
    the unmatched-id guard stood between that and a whole-store quarantine."""
    from quantred.services.maintenance import mark_contaminated

    journal = tmp_path / "journal.jsonl"
    for fast in (5, 6, 7):
        journal_trial(
            "backtest",
            strategy="s",
            symbols=["AAA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 1),
            params={"fast": fast},
            metrics={"sharpe_ratio": 1.0},
            path=journal,
        )
    report = mark_contaminated(reason="t", journal_path=journal, trial_ids=["zzzz9999"])
    assert report["selected"] == 0 and report["trial_ids"] == []
    assert report["unmatched_ids"] == ["zzzz9999"]
