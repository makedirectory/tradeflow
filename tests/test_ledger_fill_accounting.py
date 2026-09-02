"""Fill accounting: the arithmetic that turns broker events into an expected book.

This is the failure the existing suite could not surface. Its fakes emitted one
`fill` per order, with a side the ledger never had to read — so both defects were
invisible: a cumulative quantity summed across events, and a side that defaulted to
"buy" and recorded every short as a long.

The fixtures here therefore emit what a broker actually emits: running totals across
partial fills, repeated events, bracket legs with their own order ids, and both sides.
"""

import json

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


# --- adoption ---------------------------------------------------------------------
#: The book a resumed session found already open at the broker.
ADOPTED = [
    ("COP", "buy", 8),
    ("CRM", "buy", 4),
    ("CVX", "buy", 5),
    ("NKE", "sell", 31),
    ("SCHW", "buy", 10),
    ("WMT", "sell", 11),
]


class _Broker:
    def __init__(self, held):
        self._held = held

    def list_positions(self):
        class P:
            def __init__(self, symbol, qty):
                self.symbol, self.qty, self.is_long = symbol, abs(qty), qty > 0

        return [P(s, q) for s, q in self._held.items()]


def test_an_adopted_book_reconciles_clean(ledger):
    """The bug: live adopted six positions into its in-memory book and the durable
    ledger never heard of them, so the next sweep called all six unexpected — noise
    exactly where a real divergence has to stand out."""
    for symbol, side, qty in ADOPTED:
        ledger.record_adoption(symbol, side, qty)
    held = {s: (q if side == "buy" else -q) for s, side, q in ADOPTED}

    report = ledger.reconcile(_Broker(held))

    assert report.clean, report.summary()


def test_adoption_records_a_short_negative(ledger):
    ledger.record_adoption("NKE", "sell", 31)

    assert ledger.expected_positions() == {"NKE": -31}


def test_adoption_takes_the_magnitude_whatever_sign_the_broker_reports(ledger):
    """A broker reports a short's quantity negative; the side is what decides."""
    ledger.record_adoption("NKE", "sell", -31)

    assert ledger.expected_positions() == {"NKE": -31}


def test_adoption_replaces_what_came_before_rather_than_adding_to_it(ledger):
    """A baseline, not a fill. The broker's holding at that moment is the whole truth
    about the symbol, so adding to a stale belief would double it."""
    _fills(ledger, "o1", "COP", "buy", [8])
    ledger.record_adoption("COP", "buy", 8)

    assert ledger.expected_positions() == {"COP": 8}


def test_adoption_can_correct_a_ledger_that_had_drifted(ledger):
    """The recovery path: whatever the ledger believed, the broker's number wins."""
    _fills(ledger, "o1", "COP", "buy", [21])
    ledger.record_adoption("COP", "buy", 8)

    assert ledger.expected_positions() == {"COP": 8}


def test_fills_after_an_adoption_still_count(ledger):
    """An adoption is a baseline, not a freeze — trading continues from it."""
    ledger.record_adoption("COP", "buy", 8)
    _fills(ledger, "o2", "COP", "buy", [3])

    assert ledger.expected_positions() == {"COP": 11}


def test_an_exit_after_an_adoption_nets_against_it(ledger):
    ledger.record_adoption("COP", "buy", 8)
    _fills(ledger, "exit", "COP", "sell", [8])

    assert ledger.expected_positions() == {}


def test_adopting_nothing_leaves_a_previously_closed_symbol_closed(ledger):
    _fills(ledger, "o1", "COP", "buy", [8])
    ledger.record_close("COP")

    assert ledger.expected_positions() == {}


def test_adoption_does_not_disturb_other_symbols(ledger):
    _fills(ledger, "o1", "NKE", "sell", [31])
    ledger.record_adoption("COP", "buy", 8)

    assert ledger.expected_positions() == {"NKE": -31, "COP": 8}


def test_a_later_adoption_supersedes_an_earlier_one(ledger):
    """Restarting twice must not accumulate baselines."""
    ledger.record_adoption("COP", "buy", 8)
    ledger.record_adoption("COP", "buy", 5)

    assert ledger.expected_positions() == {"COP": 5}


def test_adoption_of_a_flat_symbol_records_nothing_held(ledger):
    """Absent is not zero, and zero is not a holding."""
    ledger.record_adoption("COP", "buy", 0)

    assert ledger.expected_positions() == {}


def test_an_adoption_is_not_counted_as_a_legacy_record(ledger, caplog):
    """It carries no basis because it is not a fill; that must not trip the pre-fix
    warning, which would then fire on every resumed session."""
    ledger.record_adoption("COP", "buy", 8)

    with caplog.at_level("ERROR"):
        ledger.reconcile(_Broker({"COP": 8}))

    assert "before fill accounting" not in caplog.text


# --- record shape -----------------------------------------------------------------
def test_every_record_carries_the_shape_it_was_written_in(ledger):
    """This file is append-only with nothing behind it to rebuild from, so a record
    written badly is readable but not fixable, forever. Two compatibility decisions were
    already improvised per-field — a fill's `basis`, a decision's `reason_code` — and
    each cost a live session before it was noticed. A version is what stops the third
    being improvised: a reader can ask what shape it is looking at rather than inferring
    it from which keys happen to be present."""
    from tradeflow.execution.ledger import LEDGER_VERSION

    ledger.record_fill("AAA", "buy", 1, order_id="o1", basis=CUMULATIVE)
    ledger.record_intent("AAA", "buy", 1, order_id="o1")
    ledger.record_adoption("AAA", "buy", 1)
    ledger.record_close("AAA")

    written = [json.loads(line) for line in ledger.path.read_text().splitlines() if line.strip()]
    assert written and all(record["v"] == LEDGER_VERSION for record in written)


def test_a_pre_version_record_is_counted_apart(ledger):
    """A file that mixes shapes is the normal case for anything append-only, and the
    counts are what let an operator decide whether to archive rather than guess."""
    ledger.path.write_text('{"event": "fill", "symbol": "AAA", "side": "buy", "qty": 8}\n')
    ledger.record_fill("BBB", "sell", 2, order_id="o1", basis=CUMULATIVE)

    summary = ledger.version_summary()

    assert summary["pre-version"] == 1
    assert summary["1"] == 1


def test_an_empty_ledger_summarises_to_nothing(ledger):
    assert ledger.version_summary() == {}


def test_the_version_does_not_disturb_the_replay(ledger):
    """The stamp is metadata; adding it must not change a single expected position."""
    _fills(ledger, "o1", "COP", "buy", [5, 8])

    assert ledger.expected_positions() == {"COP": 8.0}
