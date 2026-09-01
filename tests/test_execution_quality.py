"""Execution telemetry: what was decided, what was sent, what the venue did.

The ledger already proved quantity and sign correct. This is the layer above it —
price, latency, cost — and its whole value is that the numbers mean what they say.
Two rules do most of the work here: a measure that can be missing must never be
rendered as zero, and a modelled cost must never be added to an observed one.
"""

import pytest

from tradeflow.analytics.execution_quality import (
    cost_summary,
    decline_summary,
    execution_report,
    fill_summary,
    latency_summary,
    slippage_summary,
)
from tradeflow.execution import decision as decisions
from tradeflow.execution.ledger import CUMULATIVE, PositionLedger, slippage_bps


@pytest.fixture
def ledger(tmp_path):
    return PositionLedger(tmp_path / "ledger.jsonl")


def _submit(ledger, symbol, side, qty, reference, order_id, **plan_kwargs):
    plan = decisions.OrderPlan(
        side=side, qty=qty, reference_price=reference, client_order_id=f"c-{order_id}", **plan_kwargs
    )
    decision = decisions.allow(symbol, "BUY", f"entered {qty}", (decisions.BROKER,), None, plan)
    ledger.record_decision(decision)
    ledger.record_intent(
        symbol, side, qty, order_id=order_id, decision_id=decision.decision_id, plan=plan.as_dict()
    )
    return decision


def _fill(ledger, symbol, side, qty, order_id, price, at=None, **kwargs):
    ledger.record_fill(
        symbol, side, qty, order_id=order_id, basis=CUMULATIVE, fill_price=price, filled_at=at, **kwargs
    )


# --- slippage sign ----------------------------------------------------------------
def test_a_buy_that_paid_more_is_positive():
    """Positive is always worse, whichever way the trade went."""
    assert slippage_bps("buy", 100.0, 100.1) == pytest.approx(10.0)


def test_a_buy_that_paid_less_is_negative():
    assert slippage_bps("buy", 100.0, 99.9) == pytest.approx(-10.0)


def test_a_sell_that_received_less_is_positive():
    """The case an unsigned measure gets backwards: a short filling below reference is
    adverse, and must not cancel out a bad buy in the same average."""
    assert slippage_bps("sell", 100.0, 99.9) == pytest.approx(10.0)


def test_a_sell_that_received_more_is_negative():
    assert slippage_bps("sell", 100.0, 100.1) == pytest.approx(-10.0)


@pytest.mark.parametrize("reference,fill", [(None, 100.0), (100.0, None), (0.0, 100.0)])
def test_slippage_is_unmeasurable_rather_than_zero(reference, fill):
    """Absent is not zero: a fill with no price must not read as one that filled
    exactly on reference."""
    assert slippage_bps("buy", reference, fill) is None


# --- the join ---------------------------------------------------------------------
def test_a_lifecycle_joins_decision_to_intent_to_fill(ledger):
    """The question the ledger exists to answer: what did the strategy expect, what did
    we submit, what did the broker fill?"""
    decision = _submit(ledger, "MSFT", "buy", 1, 500.73, "o1", stop_loss=440.0, take_profit=560.0)
    _fill(ledger, "MSFT", "buy", 1, "o1", 500.91)

    (row,) = ledger.lifecycles()

    assert row["decision_id"] == decision.decision_id
    assert (row["symbol"], row["submitted_qty"], row["filled_qty"]) == ("MSFT", 1.0, 1.0)
    assert row["reference_price"] == 500.73 and row["fill_price"] == 500.91
    assert row["slippage_bps"] == pytest.approx(3.595, abs=0.01)
    assert (row["stop_loss"], row["take_profit"]) == (440.0, 560.0)
    assert row["client_order_id"] == "c-o1"


def test_partial_fills_report_the_final_quantity_not_their_sum(ledger):
    """Cumulative reporting again — the last fill is the whole truth about the order."""
    _submit(ledger, "NKE", "sell", 31, 78.40, "o3")
    _fill(ledger, "NKE", "sell", 12, "o3", 78.30, status="partially_filled")
    _fill(ledger, "NKE", "sell", 31, "o3", 78.22)

    (row,) = ledger.lifecycles()

    assert row["filled_qty"] == 31.0
    assert row["n_fill_events"] == 2  # both events are still visible
    assert row["fill_price"] == 78.22


def test_an_order_that_never_filled_still_appears(ledger):
    """An order with no fill is the most interesting row on the page, and a join keyed
    off fills would drop it entirely."""
    _submit(ledger, "CVX", "buy", 5, 150.0, "o4")

    (row,) = ledger.lifecycles()

    assert row["filled_qty"] == 0.0
    assert row["fill_price"] is None and row["slippage_bps"] is None


def test_the_join_survives_a_restart(tmp_path):
    """Derived on read, not written at fill time: a process that dies between
    submitting and filling still has both halves on disk."""
    path = tmp_path / "ledger.jsonl"
    _submit(PositionLedger(path), "MSFT", "buy", 1, 500.0, "o1")
    _fill(PositionLedger(path), "MSFT", "buy", 1, "o1", 501.0)

    (row,) = PositionLedger(path).lifecycles()

    assert row["slippage_bps"] == pytest.approx(20.0)


def test_orders_from_different_decisions_do_not_merge(ledger):
    _submit(ledger, "MSFT", "buy", 1, 500.0, "o1")
    _submit(ledger, "AMGN", "buy", 2, 430.0, "o2")

    assert {row["symbol"] for row in ledger.lifecycles()} == {"MSFT", "AMGN"}


# --- summaries --------------------------------------------------------------------
def _rows(**overrides):
    base = {
        "symbol": "AAA",
        "side": "buy",
        "submitted_qty": 1.0,
        "filled_qty": 1.0,
        "reference_price": 100.0,
        "fill_price": 100.1,
        "slippage_bps": 10.0,
        "decision_to_fill_ms": 500.0,
        "cost_estimate": None,
        "broker_fee": None,
    }
    return {**base, **overrides}


def test_a_fill_with_no_price_is_counted_as_unmeasured(capsys):
    """A clean-looking average over two of twenty fills is worse than no average."""
    summary = slippage_summary([_rows(), _rows(slippage_bps=None, fill_price=None)])

    assert summary["n_filled"] == 2
    assert summary["n_measured"] == 1
    assert summary["n_unmeasured"] == 1


def test_the_worst_fill_is_named(capsys):
    summary = slippage_summary(
        [_rows(symbol="AAA", slippage_bps=2.0), _rows(symbol="BBB", slippage_bps=40.0)]
    )

    assert summary["worst_bps"] == 40.0 and summary["worst_symbol"] == "BBB"


def test_no_measurable_fills_reports_nothing_rather_than_zero():
    summary = slippage_summary([_rows(slippage_bps=None)])

    assert summary["mean_bps"] is None and summary["median_bps"] is None


def test_a_fill_timed_before_its_decision_is_not_a_latency():
    """Negative elapsed means the clocks disagree, not that a fill preceded its
    request. Averaging it in would drag the median negative and look like speed."""
    summary = latency_summary([_rows(decision_to_fill_ms=500.0), _rows(decision_to_fill_ms=-1_287_158.0)])

    assert summary["n_measured"] == 1
    assert summary["n_clock_skew"] == 1
    assert summary["median_ms"] == 500.0


def test_unfilled_orders_are_counted_and_shrink_the_fill_ratio():
    summary = fill_summary([_rows(), _rows(filled_qty=0.0, fill_price=None)])

    assert summary["n_unfilled"] == 1
    assert summary["fill_ratio"] == pytest.approx(100.1 / 200.0)


def test_a_partial_fill_is_counted_as_partial():
    summary = fill_summary([_rows(submitted_qty=10.0, filled_qty=4.0)])

    assert summary["n_partial"] == 1


def test_nothing_submitted_has_no_fill_ratio():
    """0.0 would read as "nothing filled"; there is simply no ratio."""
    assert fill_summary([])["fill_ratio"] is None


def test_a_modelled_cost_is_never_added_to_a_broker_fee():
    """One is a prediction and one is an observation. Summing them turns the
    prediction into evidence."""
    summary = cost_summary([_rows(cost_estimate=1.50, broker_fee=0.35)])

    assert summary["model_cost_estimate"] == 1.50
    assert summary["broker_fees"] == 0.35


def test_a_venue_that_reports_no_fees_is_not_a_venue_that_charged_zero():
    """Every paper fill looks like this, and reading it as zero would make a live book
    look more expensive than the paper one for no real reason."""
    summary = cost_summary([_rows(cost_estimate=1.50)])

    assert summary["fees_reported"] is False
    assert summary["broker_fees"] is None


def test_declines_are_counted_by_reason_worst_first():
    """Why a strategy did nothing is the question that leaves no other trace."""
    counts = decline_summary(
        [{"reason": "book is full"}, {"reason": "gross exposure capped"}, {"reason": "gross exposure capped"}]
    )

    assert list(counts) == ["gross exposure capped", "book is full"]
    assert counts["gross exposure capped"] == 2


def test_a_decline_with_no_reason_is_not_dropped():
    """Absent is not zero here either: an unexplained refusal still happened."""
    assert decline_summary([{}]) == {"unknown": 1}


def test_the_report_ties_the_whole_session_together(ledger):
    _submit(ledger, "MSFT", "buy", 1, 500.73, "o1", cost_estimate=0.25)
    _fill(ledger, "MSFT", "buy", 1, "o1", 500.91)
    ledger.record_decision(
        decisions.decline("WMT", "BUY", "gross exposure capped", (decisions.POSITION_LIMITS,))
    )

    report = execution_report(ledger.lifecycles(), ledger.declines())

    assert report["slippage"]["n_measured"] == 1
    assert report["fills"]["n_orders"] == 1
    assert report["costs"]["model_cost_estimate"] == 0.25
    assert report["declines"] == {"gross exposure capped": 1}
