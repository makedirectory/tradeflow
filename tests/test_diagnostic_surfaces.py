"""The diagnostics added this session, each covered where nothing covered it.

These are read-only reports, which is exactly why they are easy to leave untested and
dangerous to leave wrong: a number nobody checks is worse than no number, because it
gets quoted. Every one of these had a defect that a test of this shape would have
caught — a metric read from the wrong key, a flag applied on one output path only, a
concentration nobody could see without writing a query by hand.
"""

import json

import pytest

from tests.fakes import FakeMarketData
from tradeflow.marketdata.client import MarketDataClient


# --- fill stress ------------------------------------------------------------------
def _stress(margins=(0.0, 25.0)):
    from datetime import datetime

    from tradeflow.services import analysis

    symbols = [f"S{i}" for i in range(4)]
    client = MarketDataClient(FakeMarketData([*symbols, "SPY"], n=200, freq="1D"))
    return analysis.run_fill_stress(
        client,
        "demo_trend",
        symbols,
        datetime(2024, 1, 2),
        datetime(2024, 8, 1),
        capital=100_000.0,
        margins=margins,
    )


def test_the_stress_counts_the_trades_it_actually_ran():
    """The bug: it read `trades` where the metric is `total_trades`, and defaulted the
    miss to 0 — so it reported 0 trades on every row of a 1952-trade run. A wrong key
    plus a zero default is indistinguishable from a real answer."""
    report = _stress()

    assert all(point["trades"] is not None for point in report["points"])
    assert any(point["trades"] > 0 for point in report["points"])


def test_a_wider_margin_never_admits_more_fills():
    """Requiring the price to trade further through a target can only remove exits."""
    report = _stress(margins=(0.0, 50.0))
    counts = [point["trades"] for point in report["points"]]

    assert counts[-1] <= counts[0]


def test_the_stress_journals_nothing():
    """One candidate under stated assumptions, not new candidates — recording them
    would inflate the multiple-testing count the deflated Sharpe deflates against."""
    from tradeflow.engine.backtest import ACCOUNTING_VERSION
    from tradeflow.store.trials import TrialStore

    with TrialStore() as store:
        before = len(store.list_trials(accounting=ACCOUNTING_VERSION))
    _stress()
    with TrialStore() as store:
        assert len(store.list_trials(accounting=ACCOUNTING_VERSION)) == before


def test_a_missing_trade_count_renders_as_unknown_not_zero(capsys):
    """Absent is not zero, in the rendering as well as the data."""
    from unittest import mock

    from tradeflow.cli import _print_fill_stress

    with mock.patch(
        "tradeflow.services.analysis.run_fill_stress",
        return_value={
            "points": [{"margin_bps": 0.0, "sharpe_ratio": 1.0, "total_return": 5.0, "trades": None}],
            "survives_to_bps": None,
        },
    ):
        args = mock.Mock(
            start=None,
            end=None,
            capital=1.0,
            benchmark="SPY",
            commission_bps=1.0,
            impact_eta=0.3,
            borrow_bps=50.0,
        )
        _print_fill_stress(None, "demo_trend", [], args, None)

    printed = capsys.readouterr().out
    assert "—" in printed and "touch only" in printed
    assert "does not survive" in printed


# --- exit concentration -----------------------------------------------------------
def _result_with_exits(pairs):
    """A stand-in result carrying just the trade table the diagnostic reads."""
    import pandas as pd

    class Result:
        trades = pd.DataFrame({"exit_reason": [r for r, _ in pairs], "pnl": [p for _, p in pairs]})

    return Result()


def test_a_book_whose_gain_is_one_exit_path_is_told_so(capsys):
    """This finding needed a hand-written query against a 1952-row trade table. A
    result that is a bet on one exit's fill assumption should say so on the page."""
    from tradeflow.cli import _print_exit_concentration

    _print_exit_concentration(_result_with_exits([("TAKE_PROFIT", 500.0)] * 9 + [("SIGNAL", -20.0)]))

    printed = capsys.readouterr().out
    assert "TAKE_PROFIT" in printed
    assert "Nearly all of the gain comes from TAKE_PROFIT" in printed


def test_a_book_with_spread_gains_is_not_warned_about(capsys):
    """Both directions: a warning that fires on every result is one nobody reads."""
    from tradeflow.cli import _print_exit_concentration

    _print_exit_concentration(_result_with_exits([("TAKE_PROFIT", 100.0)] * 5 + [("SIGNAL", 100.0)] * 5))

    assert "Nearly all of the gain" not in capsys.readouterr().out


def test_a_result_with_no_trades_prints_nothing(capsys):
    from tradeflow.cli import _print_exit_concentration

    _print_exit_concentration(_result_with_exits([]))

    assert capsys.readouterr().out == ""


# --- trials show --json --trades-limit --------------------------------------------
def _trial(rows):
    return {"id": "t1", "trades": {"columns": ["a"], "rows": [[i] for i in range(rows)]}}


def test_the_json_form_honours_the_trade_limit():
    """The flag was read only on the text path, so JSON dumped every stored trade —
    for a real trial, thousands of rows nobody asked for."""
    from tradeflow.cli import _limit_trial_trades

    limited = _limit_trial_trades(_trial(1952), 3)

    assert len(limited["trades"]["rows"]) == 3


def test_a_truncated_payload_says_what_it_dropped():
    """A truncated payload that does not say so looks like the whole table."""
    from tradeflow.cli import _limit_trial_trades

    marker = _limit_trial_trades(_trial(1952), 3)["trades"]["truncated"]

    assert marker == {"shown": 3, "total": 1952, "flag": "--trades-limit"}


def test_a_payload_under_the_limit_is_untouched():
    """Both directions, and it must not carry a truncation marker it did not earn."""
    from tradeflow.cli import _limit_trial_trades

    whole = _limit_trial_trades(_trial(2), 25)

    assert len(whole["trades"]["rows"]) == 2
    assert "truncated" not in whole["trades"]


def test_a_trial_with_no_trades_survives_the_limit():
    from tradeflow.cli import _limit_trial_trades

    assert _limit_trial_trades({"id": "t1"}, 3) == {"id": "t1"}


def test_the_limited_payload_is_still_json():
    """It is the JSON path; a marker that breaks serialization defeats the point."""
    from tradeflow.cli import _limit_trial_trades

    json.dumps(_limit_trial_trades(_trial(50), 5), default=str)


# --- execution-report command -----------------------------------------------------
def test_the_execution_report_command_runs_over_an_empty_ledger(capsys, tmp_path):
    """A session that placed nothing is the most likely first run of this command, and
    it must report that rather than fail."""
    from unittest import mock

    from tradeflow.cli import cmd_execution_report

    args = mock.Mock(ledger=str(tmp_path / "ledger.jsonl"), json=False, orders=False)
    cmd_execution_report(args)

    assert "Execution quality" in capsys.readouterr().out


def test_the_execution_report_emits_valid_json(capsys, tmp_path):
    from unittest import mock

    from tradeflow.cli import cmd_execution_report
    from tradeflow.execution.ledger import CUMULATIVE, PositionLedger

    ledger = PositionLedger(tmp_path / "ledger.jsonl")
    ledger.record_intent(
        "AAA",
        "buy",
        1,
        order_id="o1",
        decision_id="d1",
        plan={"side": "buy", "qty": 1, "reference_price": 100.0},
    )
    ledger.record_fill("AAA", "buy", 1, order_id="o1", basis=CUMULATIVE, fill_price=100.5)

    args = mock.Mock(ledger=str(tmp_path / "ledger.jsonl"), json=True, orders=False)
    cmd_execution_report(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["slippage"]["n_measured"] == 1
    assert payload["orders"][0]["symbol"] == "AAA"


# --- net cap candidates -----------------------------------------------------------
def test_candidate_caps_always_offer_one_above_the_observed_maximum():
    """A percentile ladder alone can sit entirely under a fat tail, and then reports
    that no cap leaves the book intact when one obviously does."""
    from tradeflow.analytics.exposure import candidate_caps

    caps = candidate_caps({"p95": 0.05, "max": 0.90})

    assert max(caps) >= 0.90


def test_candidate_caps_are_empty_when_no_tilt_was_carried():
    """Every positive cap is unbinding, so offering candidates would invent a choice."""
    from tradeflow.analytics.exposure import candidate_caps

    assert candidate_caps({"p95": 0.0, "max": 0.0}) == []


def test_candidate_caps_are_sorted_and_positive():
    from tradeflow.analytics.exposure import candidate_caps

    caps = candidate_caps({"p95": 0.30, "max": 0.42})

    assert caps == sorted(caps) and all(cap > 0 for cap in caps)


# --- the feed reaches the provider through the factory ----------------------------
def test_the_factory_passes_the_feed_through():
    """Pinning the feed is only useful if it survives construction — the mismatch it
    exists to close is between the two halves of one provider."""
    from tradeflow.brokers.alpaca.factory import build_market_data

    assert build_market_data("k", "s", feed="iex")._feed == "iex"
    assert build_market_data("k", "s")._feed is None


# --- execution diagnostics that feed a promotion gate ------------------------------
def _execution(filled=0, rounded=0, below=0):
    from tradeflow.engine.backtest import _Execution

    return _Execution(filled=filled, rounded_to_zero=rounded, below_min_notional=below)


def test_unfillable_counts_entries_that_never_opened_at_all():
    """It feeds `max_unfillable_pct`, which is a promotion gate — a book that opens
    every name slightly smaller is a different proposition from one that silently never
    opens a quarter of them, and only this number tells them apart."""
    assert _execution(filled=3, rounded=1).unfillable_pct() == pytest.approx(25.0)


def test_both_ways_an_entry_can_fail_to_open_are_counted():
    """Rounding to zero and falling under a venue floor are different causes of the
    same outcome; counting one would understate the gate's input."""
    assert _execution(filled=2, rounded=1, below=1).unfillable_pct() == pytest.approx(50.0)


def test_a_book_that_opened_everything_reports_nothing_unfillable():
    assert _execution(filled=10).unfillable_pct() == 0.0


def test_no_attempted_entries_is_zero_rather_than_a_division():
    """A run that never tried has nothing unfillable, and must not raise on the way to
    saying so — this value is read on the promotion path."""
    assert _execution().unfillable_pct() == 0.0


# --- the net-cap block only applies to a book that has a tilt ----------------------
def test_the_net_cap_derivation_is_skipped_for_a_long_only_book(capsys):
    """A long-only book's net *is* its gross, so a cap on it would be a second name for
    a limit that already exists — printing one invites a reader to set both."""
    from tradeflow.cli import _print_net_cap_derivation

    class Result:
        legs = {"long": {"trades": 12}}  # no short leg
        exposure = {
            "samples": 100,
            "net_abs": {"median": 0.5, "p90": 0.8, "p95": 0.9, "p99": 0.95, "max": 1.0},
            "candidates": [],
        }

    _print_net_cap_derivation(Result(), {"max_gross_exposure": 0.9})

    assert capsys.readouterr().out == ""


def test_the_net_cap_derivation_prints_for_a_book_that_traded_both_sides(capsys):
    """Both directions: the diagnostic exists for exactly this book."""
    from tradeflow.analytics.exposure import candidate_caps
    from tradeflow.cli import _print_net_cap_derivation

    stats = {"median": 0.20, "p90": 0.30, "p95": 0.32, "p99": 0.37, "max": 0.42}

    class Result:
        legs = {"long": {"trades": 12}, "short": {"trades": 9}}
        exposure = {
            "samples": 400,
            "net_abs": stats,
            "net_signed_mean": 0.18,
            "gross_max": 0.8,
            "candidates": [
                {"cap": cap, "binding_rate": 0.0, "above_observed_max": True} for cap in candidate_caps(stats)
            ],
        }

    _print_net_cap_derivation(Result(), {"max_gross_exposure": 0.9})

    assert "Directional tilt actually carried" in capsys.readouterr().out


# --- an offline scan must say what it could see -----------------------------------
def test_an_offline_scan_says_its_universe_is_only_as_current_as_the_cache():
    """Nothing errors when coverage ends before the clock — the newest cached bar just
    becomes "the latest", and a universe chosen from stale bars looks exactly like one
    chosen from fresh ones. That is the case that has to announce itself."""
    from datetime import datetime

    from tradeflow.analytics.reporting import format_offline_scan_notice

    notice = format_offline_scan_notice(datetime(2026, 8, 21))

    assert "offline" in notice.lower()
    assert "cache" in notice.lower()


# --- the leg decomposition has to survive the path a user actually takes -----------
def _both_legs():
    """A legs payload with both sides trading — the only case `_leg_lines` renders."""
    side = {
        "return_pct": 4.0,
        "volatility_pct": 1.5,
        "max_drawdown_pct": -3.0,
        "beta": 0.5,
        "benchmark_correlation": 0.4,
        "trades": 10,
        "cost": 120.0,
    }
    return {"long": dict(side), "short": dict(side, beta=-0.6, trades=8)}


def test_the_leg_decomposition_reaches_the_log_and_not_only_the_formatter(caplog):
    """It did not. `log_backtest_report` accepted `legs` and dropped it on the way to
    `format_backtest_report`, whose `_leg_lines` block is the only thing that prints it
    — so the decomposition was fully covered by tests calling the formatter directly
    and rendered nothing at all from `backtest`, the one surface that passes it.

    Rendering *through* the logging wrapper is the whole point of this test; asserting
    against `format_backtest_report` is what let the defect ship.
    """
    import logging

    from tradeflow.analytics.reporting import log_backtest_report

    with caplog.at_level(logging.INFO, logger="tradeflow.analytics.reporting"):
        log_backtest_report({"sharpe_ratio": 1.0}, 100_000.0, 110_000.0, legs=_both_legs())

    assert "Legs (diagnostic" in caplog.text
    assert "short" in caplog.text


def test_a_long_only_book_still_logs_no_leg_block(caplog):
    """The other direction: forwarding `legs` must not start printing an empty table for
    a book with nothing to decompose."""
    import logging

    from tradeflow.analytics.reporting import log_backtest_report

    with caplog.at_level(logging.INFO, logger="tradeflow.analytics.reporting"):
        log_backtest_report({"sharpe_ratio": 1.0}, 100_000.0, 110_000.0, legs={"long": {"trades": 12}})

    assert "Legs (diagnostic" not in caplog.text


# --- an execution check renders in its own unit, not the formatter's assumption ----
def test_every_execution_check_a_real_run_produces_declares_its_unit():
    """Walks a *real* verdict rather than a hand-written list of names: that is what
    stops the checks and the formatter drifting apart. A check reaching a surface
    without an entry here renders under the formatter's default assumption, which is
    how a position count printed as "1.00%"."""
    from tradeflow.analytics.performance import EXECUTION_VALUE_KINDS, execution_verdict

    verdict = execution_verdict(
        {
            "positions_filled": 4,
            "positions_rounded_to_zero": 1,
            "positions_below_min_notional": 0,
            "rounding_drag_pct": 2.0,
            "unfillable_pct": 20.0,
            "max_positions": 1,
            "universe_size": 40,
            "gross_profit": 1000.0,
            "total_cost": 100.0,
        }
    )

    assert set(verdict["checks"]) <= set(EXECUTION_VALUE_KINDS)
    assert set(verdict["checks"]) >= {
        "rounding_drag",
        "unfillable_entries",
        "book_breadth",
        "cost_share_of_gross",
    }


def test_a_position_count_is_not_rendered_as_a_percentage():
    """`book_breadth`'s value is a count of positions. Every check was formatted as
    `{value}% vs {threshold}%`, so it read as "a maximum of 1.00% positions"."""
    from tradeflow.analytics.performance import format_execution_value

    assert format_execution_value("book_breadth", 1.0) == "1"
    assert format_execution_value("book_breadth", 5) == "5"
    assert format_execution_value("rounding_drag", 2.5) == "2.50%"
    assert format_execution_value("an_unknown_future_check", 2.5) == "2.50%"


def test_the_breadth_remedy_is_not_printed_at_a_book_that_already_has_breadth():
    """The note hardcoded "max_positions is 1, the shipped default" whatever the value
    was, so a five-position run passed the check and still told the reader to go and
    change a config that was already right."""
    from tradeflow.analytics.performance import execution_verdict

    def note(max_positions):
        return execution_verdict(
            {
                "positions_filled": 4,
                "max_positions": max_positions,
                "universe_size": 40,
            }
        )["checks"]["book_breadth"]["note"]

    assert "shipped default" in note(1)
    assert "at most 1 of 40" in note(1)
    assert "shipped default" not in note(5)
    assert "at most 5 of 40" in note(5)
