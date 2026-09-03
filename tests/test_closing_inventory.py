"""What a stopped session reports it was holding.

A run whose every reconciliation agreed with the broker at eight positions printed a
closing summary listing seven. The ledger was right; the summary was reading the
strategy's in-memory book, which a reconciliation sweep rebuilds wholesale — and a
sweep landing between an entry's submission and its fill leaves that book short a
position for the rest of the interval.
"""

import pytest

from tradeflow.cli import _print_closing_inventory
from tradeflow.execution.ledger import CUMULATIVE, PositionLedger
from tradeflow.services.registry import STRATEGIES
from tradeflow.strategies import signals


@pytest.fixture
def ledger(tmp_path):
    return PositionLedger(tmp_path / "ledger.jsonl")


def _strategy(positions=None):
    strategy = STRATEGIES["demo_trend"].create_with_defaults()
    strategy.positions = positions or {}
    return strategy


def _held(side, qty):
    return {"side": side, "qty": qty, "entry_price": 100.0, "stop_loss": 0.0, "take_profit": 0.0}


def test_the_summary_reports_a_fill_the_in_memory_book_lost(capsys, ledger):
    """The reported bug, reproduced end to end.

    MSFT was entered, a sweep replaced the book before the broker had the position,
    then the fill arrived. The ledger has it; the book does not.
    """
    ledger.record_adoption("COP", "buy", 8)
    ledger.record_fill("MSFT", "buy", 1, order_id="o1", basis=CUMULATIVE)
    book_missing_msft = {"COP": _held(signals.BUY, 8)}

    _print_closing_inventory(ledger, _strategy(book_missing_msft))

    printed = capsys.readouterr().out
    assert "2 position(s)" in printed
    assert "MSFT" in printed


def test_the_summary_names_which_source_it_read(capsys, ledger):
    """A count with no provenance is what made the original wrong number credible."""
    ledger.record_adoption("COP", "buy", 8)

    _print_closing_inventory(ledger, _strategy())

    assert "per the ledger" in capsys.readouterr().out


def test_shorts_are_reported_as_shorts(capsys, ledger):
    ledger.record_adoption("NKE", "sell", 31)

    _print_closing_inventory(ledger, _strategy())

    printed = capsys.readouterr().out
    assert "NKE" in printed and "short" in printed and "31" in printed


def test_it_never_claims_nothing_was_opened(capsys, ledger):
    """The line this replaced said "No new positions were opened" after a session that
    opened six. Silence about an open book is the worst thing to print at the moment
    somebody is deciding whether to intervene."""
    ledger.record_fill("MSFT", "buy", 1, order_id="o1", basis=CUMULATIVE)

    _print_closing_inventory(ledger, _strategy())

    printed = capsys.readouterr().out
    assert "still open at the broker" in printed
    assert "Nothing was flattened" in printed


def test_with_no_ledger_it_falls_back_and_says_so(capsys):
    """--no-ledger is allowed, so the weaker source must still be reported — labelled,
    not silently substituted."""
    _print_closing_inventory(None, _strategy({"COP": _held(signals.BUY, 8)}))

    printed = capsys.readouterr().out
    assert "COP" in printed
    assert "this process's own book" in printed


def test_an_empty_book_says_so_without_asserting_it_is_flat(capsys, ledger):
    """This process holding nothing is not the same as the account holding nothing."""
    _print_closing_inventory(ledger, _strategy())

    assert "Check the broker" in capsys.readouterr().out


def test_a_broken_ledger_does_not_break_the_exit(capsys, ledger):
    """A summary must not raise over the real outcome of the run."""

    def explode():
        raise OSError("disk gone")

    ledger.expected_positions = explode

    _print_closing_inventory(ledger, _strategy({"COP": _held(signals.BUY, 8)}))

    assert "COP" in capsys.readouterr().out
