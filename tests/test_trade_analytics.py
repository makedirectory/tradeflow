"""Trade analytics over a recorded run's stored table.

The questions here — exit-reason P&L, win and loss by reason, holding period, per-trade
excursion — were all answered by hand in SQL before this existed. What the tests are
mostly about is the three absences staying apart, and a capped table never being
summed as though it were whole.
"""

import pytest

from tradeflow.analytics.reporting import format_exit_concentration
from tradeflow.analytics.trade_analytics import (
    COMPLETE,
    TRUNCATED,
    UNAVAILABLE,
    trade_analytics,
)

COLUMNS = ["symbol", "entry_time", "exit_time", "pnl", "exit_reason", "mae_pct", "mfe_pct"]


def _table(rows, *, total_rows=None, truncated=False, columns=None):
    """A stored table in the shape ``TrialStore.trades_for`` returns."""
    return {
        "columns": columns or COLUMNS,
        "rows": rows,
        "total_rows": len(rows) if total_rows is None else total_rows,
        "truncated": truncated,
    }


def _row(pnl, reason, *, entry="2024-01-02T00:00:00", exit_="2024-01-09T00:00:00", mae=2.0, mfe=6.0):
    return ["AAA", entry, exit_, pnl, reason, mae, mfe]


WINNERS_AND_LOSERS = [
    _row(500.0, "TAKE_PROFIT"),
    _row(300.0, "TAKE_PROFIT"),
    _row(-120.0, "STOP_LOSS"),
    _row(-40.0, "SIGNAL"),
]


# --- the three absences ------------------------------------------------------------
def test_no_recorded_table_is_not_a_run_that_made_no_trades():
    """The distinction the whole module turns on. `None` means the run did not opt into
    keeping its trades; it says nothing at all about how many there were."""
    report = trade_analytics(None)

    assert report["status"] == UNAVAILABLE
    assert "--record-trades" in report["reason"]
    assert report["source"]["recorded"] is False
    assert report["source"]["rows_stored"] is None  # not 0
    assert "exit_reasons" not in report


def test_a_run_that_closed_nothing_is_a_measured_answer():
    """The other side of the same distinction: this is a real result, and every count
    in it is a real zero."""
    report = trade_analytics(_table([]))

    assert report["status"] == COMPLETE
    assert report["n_trades"] == 0
    assert report["overall"]["trades"] == 0
    assert report["overall"]["win_rate"] is None  # no trades to have won
    assert report["exit_reasons"]["rows"] == []


def test_a_missing_column_names_itself_rather_than_reporting_a_distribution():
    """A table with no entry/exit times has an unknown holding period, which is not the
    same as trades that lasted no time."""
    columns = ["symbol", "pnl", "exit_reason"]
    report = trade_analytics(_table([["AAA", 100.0, "SIGNAL"]], columns=columns))

    duration = report["duration"]
    assert duration["available"] is False
    assert duration["missing_columns"] == ["entry_time", "exit_time"]
    assert "median" not in duration
    # The sections that could be computed still were.
    assert report["exit_reasons"]["available"] is True


def test_one_excursion_column_present_and_one_absent_are_reported_separately():
    columns = ["symbol", "pnl", "exit_reason", "mae_pct"]
    report = trade_analytics(_table([["AAA", 100.0, "SIGNAL", 3.0]], columns=columns))

    assert report["excursion"]["mae_pct"]["median"] == pytest.approx(3.0)
    assert report["excursion"]["mfe_pct"]["available"] is False


# --- a capped table is not summed ---------------------------------------------------
def test_a_capped_table_refuses_to_be_aggregated():
    """A total over a prefix of a run's trades is not a smaller number than the truth,
    it is a wrong one — and it looks exactly like a right one."""
    report = trade_analytics(_table(WINNERS_AND_LOSERS, total_rows=1200, truncated=True))

    assert report["status"] == UNAVAILABLE
    assert "capped" in report["reason"]
    assert "1,200" in report["reason"] and "4" in report["reason"]
    assert "overall" not in report


def test_a_table_of_unrecorded_completeness_is_refused_on_the_same_terms():
    """It never claimed to be whole. Treating unknown as yes is the failure the
    completeness column was added to end."""
    table = _table(WINNERS_AND_LOSERS)
    table["truncated"] = None
    table["total_rows"] = None

    report = trade_analytics(table)

    assert report["status"] == UNAVAILABLE
    assert "not recorded" in report["reason"]


def test_a_caller_that_asks_for_a_partial_gets_one_that_says_it_is_partial():
    report = trade_analytics(_table(WINNERS_AND_LOSERS, total_rows=1200, truncated=True), allow_partial=True)

    assert report["status"] == TRUNCATED
    assert "partial" in report["reason"]
    assert report["overall"]["trades"] == 4


def test_a_whole_table_aggregates_without_being_asked_twice():
    """Both directions: the guard must accept the case it exists to let through, or it
    is indistinguishable from one that refuses everything."""
    report = trade_analytics(_table(WINNERS_AND_LOSERS))

    assert report["status"] == COMPLETE
    assert report["reason"] == ""
    assert report["overall"]["net_pnl"] == pytest.approx(640.0)


# --- what the sections actually say -------------------------------------------------
def test_exit_reasons_split_the_p_and_l_and_the_win_rate():
    report = trade_analytics(_table(WINNERS_AND_LOSERS))
    by_reason = {r["exit_reason"]: r for r in report["exit_reasons"]["rows"]}

    assert by_reason["TAKE_PROFIT"]["net_pnl"] == pytest.approx(800.0)
    assert by_reason["TAKE_PROFIT"]["win_rate"] == pytest.approx(1.0)
    assert by_reason["STOP_LOSS"]["net_pnl"] == pytest.approx(-120.0)
    assert by_reason["STOP_LOSS"]["avg_win"] is None  # no wins to average
    assert by_reason["SIGNAL"]["share_of_trades"] == pytest.approx(0.25)


def test_concentration_is_measured_over_the_gains_not_the_net():
    """A loss-making exit does not dilute the claim that one path produced the profit.
    Netting them would hide exactly the book this check exists to find."""
    rows = [_row(1000.0, "TAKE_PROFIT"), _row(20.0, "SIGNAL"), _row(-900.0, "STOP_LOSS")]

    concentration = trade_analytics(_table(rows))["exit_reasons"]["concentration"]

    assert concentration["exit_reason"] == "TAKE_PROFIT"
    assert concentration["share_of_gain"] == pytest.approx(1000.0 / 1020.0)


def test_duration_is_measured_in_days_from_the_stored_iso_strings():
    """Stored rows carry ISO strings because the journal is JSON; a live frame carries
    timestamps. Both have to arrive at the same number."""
    report = trade_analytics(_table(WINNERS_AND_LOSERS))

    assert report["duration"]["median"] == pytest.approx(7.0)
    assert report["duration"]["unit"] == "days"
    assert report["duration"]["n_unmeasured"] == 0


def test_an_unparseable_cell_is_counted_as_unmeasured_never_dropped():
    """Silently dropping it would move every count in the report, and nothing would
    say the denominator had changed."""
    rows = [*WINNERS_AND_LOSERS, _row(50.0, "SIGNAL", exit_="not-a-date")]

    report = trade_analytics(_table(rows))

    assert report["duration"]["n_measured"] == 4
    assert report["duration"]["n_unmeasured"] == 1
    assert report["n_trades"] == 5  # the trade itself is still a trade


def test_a_row_missing_its_last_fields_is_unmeasured_not_zero():
    """A short row is one trade with fields missing. Dropping it moves every count;
    padding it with 0.0 makes it a flat trade that closed for nothing and a per-trade
    excursion of zero — both are measurements the record does not support."""
    short = ["AAA", "2024-01-02T00:00:00", "2024-01-09T00:00:00"]  # no pnl/reason/excursions

    report = trade_analytics(_table([*WINNERS_AND_LOSERS, short]))

    assert report["n_trades"] == 5
    assert report["overall"]["trades"] == 4  # four have a measurable P&L
    assert report["overall"]["unmeasured"] == 1
    assert report["overall"]["flat"] == 0  # it did not close flat; it is unrecorded
    assert report["excursion"]["mae_pct"]["n_unmeasured"] == 1


def test_excursion_says_it_is_per_trade_not_the_book_s_drawdown():
    """The two get conflated, and the conflation is why a shallow equity drawdown gets
    read as hiding open-position pain. A position deep underwater that is a small
    fraction of the book does not put the book that far underwater, and only the label
    keeps the per-trade figure from being read as the portfolio one."""
    report = trade_analytics(_table(WINNERS_AND_LOSERS))

    assert "not the book" in report["excursion"]["note"]
    assert report["excursion"]["mae_pct"]["median"] == pytest.approx(2.0)


# --- the renderer over it -----------------------------------------------------------
def test_the_backtest_block_is_one_computation_with_the_recorded_one():
    """`_print_exit_concentration` used to group a live DataFrame itself. Two
    implementations of one idea — one over a live frame, one over a stored table — is
    the shape that drifts while both look right."""
    printed = format_exit_concentration(trade_analytics(_table(WINNERS_AND_LOSERS)))

    assert "TAKE_PROFIT" in printed and "$800" in printed
    assert "-$120" in printed  # the sign goes outside the currency symbol


def test_a_partial_table_prints_its_numbers_under_a_label_and_no_verdict():
    """Partial numbers can be shown; a claim about which exit carried the run cannot
    be made from a prefix of it."""
    report = trade_analytics(
        _table([_row(500.0, "TAKE_PROFIT")] * 9 + [_row(-5.0, "SIGNAL")], total_rows=900, truncated=True),
        allow_partial=True,
    )

    printed = format_exit_concentration(report)

    assert "SHOWN ROWS ONLY" in printed
    assert "No concentration verdict" in printed
    assert "Nearly all of the gain" not in printed


def test_a_run_with_no_trades_gets_no_block_rather_than_a_table_of_zeros():
    assert format_exit_concentration(trade_analytics(_table([]))) == ""
    assert format_exit_concentration(trade_analytics(None)) == ""


# --- review findings ----------------------------------------------------------------
def test_a_partial_report_does_not_claim_to_have_no_totals(monkeypatch):
    """Review finding. The "No totals:" guard fired on any status that was not
    `complete`, so a truncated report printed it directly above a full table of
    (partial) totals — contradicting both the table and its own reason string."""
    from tradeflow.analytics.reporting import format_trial_analysis

    report = trade_analytics(_table(WINNERS_AND_LOSERS, total_rows=1200, truncated=True), allow_partial=True)
    report.update(trial_id="x", kind="backtest", strategy="s", accounting=5)

    printed = format_trial_analysis(report)

    assert "No totals" not in printed
    assert "SHOWN ROWS ONLY" in printed
    assert "Net P&L" in printed  # the totals it does have are still shown


def test_an_unavailable_report_still_says_it_has_no_totals():
    """Both directions: the label has to survive where it is true."""
    from tradeflow.analytics.reporting import format_trial_analysis

    report = trade_analytics(None)
    report.update(trial_id="x", kind="backtest", strategy="s", accounting=5)

    printed = format_trial_analysis(report)

    assert "No totals" in printed
    assert "Net P&L" not in printed


def test_the_trades_column_and_the_share_column_agree_about_their_denominator():
    """Review finding. `trades` counted only rows with a measurable P&L while `share`
    counted every row with that reason, so on a table with any unreadable P&L the
    trades column did not sum to the total while the shares summed to 100% — with
    nothing on screen explaining the gap."""
    from tradeflow.analytics.reporting import format_exit_concentration

    rows = [_row(100.0, "TP"), _row(None, "TP"), _row(-50.0, "SL")]
    report = trade_analytics(_table(rows))
    by_reason = {r["exit_reason"]: r for r in report["exit_reasons"]["rows"]}

    assert by_reason["TP"]["rows"] == 2  # both exited this way
    assert by_reason["TP"]["trades"] == 1  # one had a measurable P&L
    assert by_reason["TP"]["unmeasured"] == 1
    assert sum(r["rows"] for r in report["exit_reasons"]["rows"]) == report["n_trades"]
    assert sum(r["share_of_trades"] for r in report["exit_reasons"]["rows"]) == pytest.approx(1.0)

    # And the gap is explained on screen rather than left as two columns disagreeing.
    assert "no P&L recorded" in format_exit_concentration(report)


def test_a_table_with_every_p_and_l_measured_prints_no_gap_line():
    """Both directions: the explanation must not appear where there is nothing to
    explain."""
    from tradeflow.analytics.reporting import format_exit_concentration

    assert "no P&L recorded" not in format_exit_concentration(trade_analytics(_table(WINNERS_AND_LOSERS)))


def test_the_live_block_does_not_pay_to_serialize_a_frame_it_only_prints(monkeypatch):
    """Review finding. `_print_exit_concentration` routed a live frame through the
    storage payload, whose per-cell JSON coercion is most of the cost of converting a
    large table — paid on every backtest to print five rows, and paid even when there
    was no exit_reason column to group by."""
    import pandas as pd

    from tradeflow.services.analysis import trades_payload

    frame = pd.DataFrame(
        {
            "exit_reason": ["TP"] * 4,
            "pnl": [1.0, 2.0, 3.0, -1.0],
            "entry_time": pd.date_range("2024-01-01", periods=4),
            "exit_time": pd.date_range("2024-01-08", periods=4),
        }
    )
    seen = {}
    real = trades_payload

    def spy(f, **kwargs):
        seen.update(kwargs)
        return real(f, **kwargs)

    monkeypatch.setattr("tradeflow.services.analysis.trades_payload", spy)
    from tradeflow.cli import _print_exit_concentration

    class Result:
        trades = frame

    _print_exit_concentration(Result())

    assert seen == {"max_rows": None, "jsonable": False}
    # The un-coerced rows carry pandas Timestamps and numpy scalars rather than the
    # strings and floats the stored form has. The analytics has to read both, or the
    # cheap path would quietly measure less than the storage path does.
    cheap = trade_analytics(real(frame, max_rows=None, jsonable=False))
    stored = trade_analytics(real(frame, max_rows=None))
    assert cheap["duration"]["median"] == stored["duration"]["median"] == pytest.approx(7.0)
    assert cheap["overall"]["net_pnl"] == stored["overall"]["net_pnl"] == pytest.approx(5.0)


def test_a_result_with_no_exit_reason_column_converts_nothing():
    """The short-circuit the old groupby had and the converged path lost."""
    import pandas as pd

    from tradeflow.cli import _print_exit_concentration

    class Result:
        trades = pd.DataFrame({"pnl": [1.0, 2.0]})

    called = []
    import tradeflow.services.analysis as analysis_mod

    original = analysis_mod.trades_payload
    analysis_mod.trades_payload = lambda *a, **k: called.append(1) or original(*a, **k)
    try:
        _print_exit_concentration(Result())
    finally:
        analysis_mod.trades_payload = original

    assert called == []
