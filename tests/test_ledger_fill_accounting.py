"""Fill accounting: the arithmetic that turns broker events into an expected book.

This is the failure the existing suite could not surface. Its fakes emitted one
`fill` per order, with a side the ledger never had to read — so both defects were
invisible: a cumulative quantity summed across events, and a side that defaulted to
"buy" and recorded every short as a long.

The fixtures here therefore emit what a broker actually emits: running totals across
partial fills, repeated events, bracket legs with their own order ids, and both sides.
"""

import pytest

from tradeflow.execution.ledger import CUMULATIVE, INCREMENTAL, PositionLedger


@pytest.fixture
def ledger(tmp_path):
    return PositionLedger(tmp_path / "ledger.jsonl")


def _fills(ledger, order_id, symbol, side, running):
    """Emit one order's life as a broker reports it: a cumulative total per event."""
    for i, total in enumerate(running):
        ledger.record_fill(
            symbol,
            side,
            total,
            order_id=order_id,
            status="partially_filled" if i < len(running) - 1 else "filled",
            basis=CUMULATIVE,
        )


# --- the reported session ---------------------------------------------------------
#: What the broker actually held, against the event stream that produced it. Every
#: sequence here sums (the old, wrong way) to the number the live run reported.
SESSION = [
    ("COP", "buy", [5, 8, 8], 8, 21),
    ("CRM", "buy", [1, 4, 4], 4, 9),
    ("SCHW", "buy", [9, 10], 10, 19),
    ("WMT", "sell", [2, 11], -11, 13),
    ("NKE", "sell", [31], -31, 31),
]


@pytest.mark.parametrize("symbol,side,running,held,summed", SESSION)
def test_a_cumulative_fill_stream_nets_to_what_the_broker_holds(ledger, symbol, side, running, held, summed):
    """The bug, one symbol at a time.

    Alpaca re-reports an order's running total on every partial fill and again on the
    final fill. Summing those events counts the same shares repeatedly: COP filled 8
    across three reports and the ledger claimed 21.
    """
    _fills(ledger, "o1", symbol, side, running)

    assert ledger.expected_positions() == {symbol: held}
    assert sum(running) == summed  # the number the broken accounting produced


def test_the_whole_reported_session_reconciles_clean(ledger):
    """All five together, against a broker holding exactly what it held that day."""

    class Position:
        def __init__(self, symbol, qty):
            self.symbol, self.qty, self.is_long = symbol, abs(qty), qty > 0

    class Broker:
        def list_positions(self):
            return [Position(s, held) for s, _, _, held, _ in SESSION]

    for i, (symbol, side, running, _, _) in enumerate(SESSION):
        _fills(ledger, f"o{i}", symbol, side, running)

    report = ledger.reconcile(Broker())

    assert report.clean, report.summary()


# --- sign ------------------------------------------------------------------------
def test_a_short_fill_is_recorded_negative(ledger):
    """The sign half of the bug: TradeUpdate carried no side at all, so the handler's
    `getattr(update, "side", "buy")` resolved to buy for every fill ever recorded."""
    _fills(ledger, "o1", "NKE", "sell", [31])

    assert ledger.expected_positions() == {"NKE": -31}


def test_a_long_fill_is_recorded_positive(ledger):
    """Both directions — a sign fix that inverted everything would pass the test above."""
    _fills(ledger, "o1", "COP", "buy", [8])

    assert ledger.expected_positions() == {"COP": 8}


@pytest.mark.parametrize("word", ["sell", "SELL", "Sell", "short"])
def test_every_spelling_of_a_short_side_is_read_as_short(ledger, word):
    """The side arrives as a broker's enum value, whose case is not ours to assume."""
    _fills(ledger, "o1", "NKE", word, [10])

    assert ledger.expected_positions() == {"NKE": -10}


# --- idempotence ------------------------------------------------------------------
def test_a_repeated_event_does_not_double_the_position(ledger):
    """A stream reconnect can replay history. Because a cumulative report is the whole
    truth about its order, replaying it must be a no-op rather than an addition."""
    _fills(ledger, "o1", "COP", "buy", [8])
    _fills(ledger, "o1", "COP", "buy", [8])

    assert ledger.expected_positions() == {"COP": 8}


def test_a_missed_intermediate_partial_still_lands_on_the_truth(ledger):
    """Only the last report matters, so a dropped middle event costs nothing."""
    _fills(ledger, "o1", "COP", "buy", [8])  # the 5 was never delivered

    assert ledger.expected_positions() == {"COP": 8}


def test_events_arriving_out_of_order_take_the_last_written(ledger):
    """The file is the state and it is append-only, so 'last' means last recorded."""
    _fills(ledger, "o1", "COP", "buy", [8, 5])

    assert ledger.expected_positions() == {"COP": 5}


# --- several orders in one symbol -------------------------------------------------
def test_two_separate_orders_in_one_symbol_add_up(ledger):
    """Collapsing by order must not collapse across orders — that would lose a genuine
    second entry, the mirror-image error of the one being fixed."""
    _fills(ledger, "o1", "COP", "buy", [3, 5])
    _fills(ledger, "o2", "COP", "buy", [2, 3])

    assert ledger.expected_positions() == {"COP": 8}


def test_a_bracket_leg_nets_against_the_entry_it_closes(ledger):
    """Entry and protective legs carry their own order ids, so a stop that fills nets
    the position flat with no special-casing anywhere."""
    _fills(ledger, "entry", "COP", "buy", [8])
    _fills(ledger, "stop", "COP", "sell", [8])

    assert ledger.expected_positions() == {}  # flat, not a zero entry


def test_a_partial_exit_leaves_the_remainder(ledger):
    _fills(ledger, "entry", "COP", "buy", [10])
    _fills(ledger, "stop", "COP", "sell", [4])

    assert ledger.expected_positions() == {"COP": 6}


def test_flat_and_never_traded_look_the_same(ledger):
    """A symbol netting to zero is dropped rather than recorded as a zero position."""
    _fills(ledger, "entry", "COP", "buy", [8])
    _fills(ledger, "stop", "COP", "sell", [8])

    assert "COP" not in ledger.expected_positions()


# --- closes -----------------------------------------------------------------------
def test_a_close_zeroes_the_symbol(ledger):
    _fills(ledger, "o1", "COP", "buy", [8])
    ledger.record_close("COP")

    assert ledger.expected_positions() == {}


def test_a_fill_after_a_close_starts_a_new_position(ledger):
    """A close zeroes what came before it, not what comes after — re-entering the same
    symbol later in a session must not be swallowed."""
    _fills(ledger, "o1", "COP", "buy", [8])
    ledger.record_close("COP")
    _fills(ledger, "o2", "COP", "buy", [3])

    assert ledger.expected_positions() == {"COP": 3}


def test_a_close_does_not_touch_other_symbols(ledger):
    _fills(ledger, "o1", "COP", "buy", [8])
    _fills(ledger, "o2", "NKE", "sell", [5])
    ledger.record_close("COP")

    assert ledger.expected_positions() == {"NKE": -5}


# --- other event types must not count ---------------------------------------------
def test_intent_alone_is_not_a_position(ledger):
    """Intent is what was submitted, not what happened. Counting it would make every
    rejected order a phantom holding."""
    ledger.record_intent("COP", "buy", 8, order_id="o1")

    assert ledger.expected_positions() == {}


def test_intent_and_its_fill_are_not_added_together(ledger):
    ledger.record_intent("COP", "buy", 8, order_id="o1")
    _fills(ledger, "o1", "COP", "buy", [8])

    assert ledger.expected_positions() == {"COP": 8}


# --- basis ------------------------------------------------------------------------
def test_incremental_fills_are_summed_not_collapsed(ledger):
    """A broker reporting per-event shares is the other convention, and collapsing
    those to the last report would under-count exactly as summing over-counts."""
    for qty in (5, 3):
        ledger.record_fill("COP", "buy", qty, order_id="o1", basis=INCREMENTAL)

    assert ledger.expected_positions() == {"COP": 8}


def test_a_fill_with_no_order_id_stands_alone(ledger):
    """Nothing to collapse on, so each event must count once rather than overwrite."""
    ledger.record_fill("COP", "buy", 5, basis=CUMULATIVE)
    ledger.record_fill("COP", "buy", 3, basis=CUMULATIVE)

    assert ledger.expected_positions() == {"COP": 8}


# --- pre-fix records --------------------------------------------------------------
def test_a_ledger_written_before_the_fix_is_reported_not_silently_reinterpreted(ledger, caplog):
    """Records written before fill accounting had a basis also defaulted every side to
    buy. Their numbers cannot be recovered, so a reconcile over them must say so — a
    quiet reinterpretation would produce a confident number with no meaning."""
    ledger._append({"event": "fill", "symbol": "COP", "side": "buy", "qty": 8, "order_id": "o1"})

    class Broker:
        def list_positions(self):
            return []

    with caplog.at_level("ERROR"):
        ledger.reconcile(Broker())

    assert "before fill accounting" in caplog.text
    assert "Archive" in caplog.text


def test_a_ledger_written_after_the_fix_says_nothing_about_legacy_records(ledger, caplog):
    """Both directions: a warning that always fires is one nobody reads."""
    _fills(ledger, "o1", "COP", "buy", [8])

    class Broker:
        def list_positions(self):
            return []

    with caplog.at_level("ERROR"):
        ledger.reconcile(Broker())

    assert "before fill accounting" not in caplog.text


# --- durability -------------------------------------------------------------------
def test_the_replay_survives_a_restart(tmp_path):
    """The file is the state: a fresh process must recover the same expectation."""
    path = tmp_path / "ledger.jsonl"
    _fills(PositionLedger(path), "o1", "COP", "buy", [5, 8])

    assert PositionLedger(path).expected_positions() == {"COP": 8}


def test_a_torn_final_line_does_not_erase_the_ledger(tmp_path):
    """Killed mid-write. A small gap must not become total amnesia."""
    path = tmp_path / "ledger.jsonl"
    _fills(PositionLedger(path), "o1", "COP", "buy", [8])
    with path.open("a") as fh:
        fh.write('{"event": "fill", "symbol": "NK')

    assert PositionLedger(path).expected_positions() == {"COP": 8}
