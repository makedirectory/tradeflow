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

from src.analytics.reporting import (
    NOT_RECORDED,
    format_leaderboard,
    format_trial_detail,
    format_trials_table,
)
from src.store.trials import TrialStore

ACCOUNTING = 3


@pytest.fixture
def store(tmp_path):
    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl") as s:
        yield s


def _record(store, trial_id, **kwargs):
    defaults = {
        "id": trial_id,
        "kind": "backtest",
        "strategy": "volume_spike",
        "symbols": ["AAA", "BBB"],
        "params": {"fast": 5},
        "accounting": ACCOUNTING,
        "ts": "2025-01-01T00:00:00",
        "window_start": "2024-01-02T00:00:00",
        "window_end": "2024-12-31T00:00:00",
    }
    defaults.update(kwargs)
    store.record(**defaults)
    return defaults


def _populate(store):
    _record(store, "t1", ts="2025-01-01T00:00:00", oos_sharpe=0.4, deflated_sharpe=0.1, promotable=False)
    _record(store, "t2", ts="2025-02-01T00:00:00", oos_sharpe=2.1, deflated_sharpe=0.2, promotable=True)
    _record(store, "t3", ts="2025-03-01T00:00:00", oos_sharpe=1.5, deflated_sharpe=0.9, promotable=True)
    # A different family, and a kind with no Sharpe at all.
    _record(
        store,
        "t4",
        ts="2025-04-01T00:00:00",
        strategy="ma_crossover",
        symbols=["CCC"],
        oos_sharpe=3.0,
        deflated_sharpe=0.05,
    )
    _record(
        store,
        "t5",
        ts="2025-05-01T00:00:00",
        kind="alpha",
        params={"fast": 9},
        oos_sharpe=None,
        deflated_sharpe=None,
    )


# --- filtering --------------------------------------------------------------
def test_filters_narrow_to_what_they_say(store):
    _populate(store)
    assert {r["id"] for r in store.list_trials(accounting=ACCOUNTING)} == {"t1", "t2", "t3", "t4", "t5"}
    assert {r["id"] for r in store.list_trials(strategy="ma_crossover", accounting=ACCOUNTING)} == {"t4"}
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
    volume_spike = next(r for r in board["rows"] if r["strategy"] == "volume_spike")
    # t1/t2/t3 count; the alpha row never does (a forecast has no Sharpe to deflate).
    assert volume_spike["family_n_trials"] == 3
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
    from src.services import audit

    journal = tmp_path / "journal.jsonl"
    trial_id = audit.journal_trial(
        "backtest",
        strategy="volume_spike",
        symbols=["AAA"],
        start=datetime(2024, 1, 2),
        end=datetime(2024, 12, 31),
        params={"fast": 5},
        metrics={"sharpe_ratio": 1.0},
        trades={"columns": ["symbol", "pnl"], "rows": [["AAA", 1.5], ["AAA", -0.5]]},
        path=journal,
    )
    from src.store.trials import db_path_for_journal

    with TrialStore(db_path_for_journal(journal), journal_path=journal) as s:
        assert s.trades_for(trial_id)["rows"] == [["AAA", 1.5], ["AAA", -0.5]]
        s.rebuild()
        assert s.trades_for(trial_id)["rows"] == [["AAA", 1.5], ["AAA", -0.5]]


def test_without_the_flag_a_trial_stores_no_trade_table(tmp_path):
    from src.services import audit
    from src.store.trials import db_path_for_journal

    journal = tmp_path / "journal.jsonl"
    trial_id = audit.journal_trial(
        "backtest",
        strategy="volume_spike",
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

    from src.services.analysis import trades_payload

    frame = pd.DataFrame({"symbol": ["AAA"] * 10, "pnl": range(10)})
    payload = trades_payload(frame, max_rows=4)
    assert len(payload["rows"]) == 4
    assert payload["truncated"] is True
    assert payload["total_rows"] == 10

    assert trades_payload(None) is None
    full = trades_payload(frame)
    assert "truncated" not in full and full["total_rows"] == 10


# --- the CLI surface --------------------------------------------------------
def _cli(monkeypatch, tmp_path, *argv):
    import main

    args = main.build_parser().parse_args(["trials", *argv, "--db", str(tmp_path / "trials.db")])
    args.func(args)


def test_cli_list_show_and_best_round_trip_as_json(monkeypatch, tmp_path, capsys):
    import main

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
    import main

    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl"):
        pass
    args = main.build_parser().parse_args(["trials", "show", "missing", "--db", str(tmp_path / "trials.db")])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert "missing" in str(exc.value)


def test_query_remains_an_alias_for_list(tmp_path, capsys):
    import main

    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl") as s:
        _populate(s)
    for verb in ("list", "query"):
        args = main.build_parser().parse_args(["trials", verb, "--db", str(tmp_path / "trials.db")])
        args.func(args)
        assert "t3" in capsys.readouterr().out
