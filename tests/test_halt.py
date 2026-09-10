"""The kill switch.

Two properties carry the weight. A halt must **block entries and never exits** — a
switch that trapped the book would be one nobody dares pull, and it would deadlock a
flatten against its own gate. And **absent is not halted**: a missing or corrupt state
file means no halt was recorded, because the reverse default turns an unrelated disk
problem into a silent trading freeze.
"""

import json

import pytest

from tests.fakes import FailingBroker, FakeBroker
from tradeflow.brokers.base import Position
from tradeflow.brokers.errors import BrokerUnavailableError
from tradeflow.demo.strategies import DemoTrendStrategy
from tradeflow.execution.flatten import flatten
from tradeflow.execution.halt import ALL, HaltState
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.strategies import signals


@pytest.fixture
def halts(tmp_path):
    return HaltState(tmp_path / "halts.json")


def _position(symbol="AAA"):
    return Position(
        symbol=symbol,
        qty=10.0,
        side="long",
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=1000.0,
        unrealized_pl=0.0,
    )


# --- state ------------------------------------------------------------------
def test_a_halt_is_not_in_force_until_it_is_set(halts):
    assert halts.is_halted() is False
    assert halts.active() is None


def test_a_halt_survives_a_new_reader(tmp_path):
    """Durability is the whole point: a restarted engine must still see it."""
    HaltState(tmp_path / "halts.json").set("bad data", actor="cli")
    assert HaltState(tmp_path / "halts.json").is_halted() is True


def test_a_halt_records_who_and_why(halts):
    halt = halts.set("feed looked wrong", actor="andy")
    assert halt.reason == "feed looked wrong"
    assert halt.actor == "andy"
    assert halt.set_at


def test_a_global_halt_covers_every_scope(halts):
    halts.set("everything off", actor="cli", scope=ALL)
    assert halts.is_halted("DemoTrendStrategy") is True


def test_a_scoped_halt_leaves_other_strategies_alone(halts):
    halts.set("just this one", actor="cli", scope="DemoTrendStrategy")
    assert halts.is_halted("DemoTrendStrategy") is True
    # Any other scope. A literal rather than a second shipped class: the engine now
    # ships exactly one strategy, and the scoping rule is about the name, not the class.
    assert halts.is_halted("SomeOtherStrategy") is False


def test_lifting_a_halt_reports_whether_one_was_in_force(halts):
    assert halts.clear() is False  # nothing to lift
    halts.set("stop", actor="cli")
    assert halts.clear() is True
    assert halts.is_halted() is False


def test_a_corrupt_state_file_means_no_halt_not_a_permanent_freeze(tmp_path):
    """The reverse default would turn an unrelated disk problem into a trading
    freeze nobody chose and nobody can explain."""
    path = tmp_path / "halts.json"
    path.write_text("{not json at all")
    assert HaltState(path).is_halted() is False


def test_a_malformed_record_is_skipped_without_losing_the_others(tmp_path):
    path = tmp_path / "halts.json"
    path.write_text(json.dumps({"all": {"no_reason_field": True}, "Strat": {"reason": "kept"}}))
    state = HaltState(path)
    assert state.is_halted("Strat") is True
    assert state.active("Strat").reason == "kept"


# --- what the trader does with it -------------------------------------------
def _trader(broker, halts):
    return LiveTrader(
        broker,
        DemoTrendStrategy.create_with_defaults(),
        respect_market_hours=False,
        halt_state=halts,
    )


def test_a_halt_refuses_a_new_entry(halts):
    broker = FakeBroker()
    halts.set("stop", actor="cli")

    _trader(broker, halts).handle_signal("AAA", signals.BUY, 100.0)

    assert broker.orders == []


def test_a_halt_never_refuses_an_exit(halts):
    """A switch that trapped the book is one nobody dares pull — and it would
    deadlock a flatten against its own gate."""
    broker = FakeBroker(positions=[_position()])
    halts.set("stop", actor="cli")
    trader = _trader(broker, halts)
    trader.sync_strategy_book()

    trader.handle_signal("AAA", signals.CLOSE_BUY, 100.0)

    assert broker.closed == ["AAA"]


def test_lifting_the_halt_allows_entries_again(halts):
    """The other direction: a halt must be reversible, or it is just an outage."""
    broker = FakeBroker()
    halts.set("stop", actor="cli")
    trader = _trader(broker, halts)
    trader.handle_signal("AAA", signals.BUY, 100.0)
    assert broker.orders == []

    halts.clear()
    trader.handle_signal("AAA", signals.BUY, 100.0)
    assert len(broker.orders) == 1


def test_a_strategy_scoped_halt_stops_only_that_strategy(halts):
    broker = FakeBroker()
    halts.set("this one misbehaves", actor="cli", scope="DemoTrendStrategy")

    _trader(broker, halts).handle_signal("AAA", signals.BUY, 100.0)

    assert broker.orders == []


# --- flatten ----------------------------------------------------------------
def test_flatten_halts_cancels_and_closes(halts):
    broker = FakeBroker(positions=[_position()])

    report = flatten(broker, reason="drill", actor="cli", halt_state=halts)

    assert report.complete
    assert halts.is_halted() is True
    assert broker.positions == {}


def test_flatten_halts_before_it_closes_anything(halts):
    """A running engine re-enters on the next bar; cancelling and closing while it
    still believes it may trade is a race the engine can win."""
    observed = {}

    class Watching(FakeBroker):
        def close_all_positions(self, cancel_orders=True):
            observed["halted_first"] = halts.is_halted()
            return super().close_all_positions(cancel_orders)

    flatten(Watching(positions=[_position()]), reason="drill", halt_state=halts)

    assert observed["halted_first"] is True


def test_flatten_still_closes_positions_when_cancelling_orders_fails(halts):
    """A partial flatten is bad; stopping halfway and leaving positions open is
    worse."""
    broker = FailingBroker(positions=[_position()])
    broker.failures["cancel_all_orders"] = BrokerUnavailableError("timeout")

    report = flatten(broker, reason="drill", halt_state=halts)

    assert report.close_submitted is True
    assert report.orders_cancelled is False
    assert not report.complete
    assert report.failures


def test_an_incomplete_flatten_says_so_rather_than_reporting_success(halts):
    broker = FailingBroker(positions=[_position()])
    broker.failures["close_all_positions"] = BrokerUnavailableError("timeout")

    report = flatten(broker, reason="drill", halt_state=halts)

    assert not report.complete
    assert "INCOMPLETE" in report.summary()
    assert halts.is_halted() is True  # the halt still stands


# --- submitted is not observed -----------------------------------------------------
class QueueingBroker(FakeBroker):
    """A broker that accepts the close and keeps the positions.

    Exactly what a real venue does outside market hours: `close_all_positions` returns
    successfully, the orders queue, and the account still holds everything until the
    open. Observed in practice — a flatten reported every position closed and the
    account still held twelve of them eight minutes later.
    """

    def close_all_positions(self, cancel_orders: bool = True) -> bool:
        return True  # accepted, and nothing is closed


def test_a_queued_close_is_not_reported_as_a_closed_position(halts):
    """The defect this exists to stop: `positions_closed` was set when the *request*
    returned, so a flatten whose orders merely queued printed a complete report and a
    reassuring next step, while the book was untouched."""
    from tradeflow.execution.flatten import PENDING

    broker = QueueingBroker(positions=[_position()])

    report = flatten(broker, reason="drill", halt_state=halts)

    assert report.close_submitted is True  # the request was accepted
    assert report.observed == PENDING  # and nothing was observed closed
    assert not report.complete
    assert report.remaining == [_position().symbol]
    assert report.checked_at

    text = report.summary()
    assert "NOT FLAT YET" in text
    assert "close orders submitted    : yes" in text
    assert "positions observed closed : pending" in text
    # It must not claim the terminal state anywhere.
    assert "Flat, and confirmed" not in text


def test_a_confirmed_flatten_says_flat_and_names_when_it_looked(halts):
    """Both directions: the guard must still recognise the case it exists to permit,
    and the confirmation has to carry the instant it was taken — a flat report with no
    timestamp is a claim with no evidence behind it."""
    from tradeflow.execution.flatten import CLOSED

    report = flatten(FakeBroker(positions=[_position()]), reason="drill", halt_state=halts)

    assert report.observed == CLOSED
    assert report.complete
    assert report.remaining == []
    assert report.checked_at
    text = report.summary()
    assert "positions observed closed : yes" in text
    assert "Flat, and confirmed by a broker read" in text


def test_an_unreadable_account_is_never_reported_as_flat(halts):
    """Absent is not zero, in the place it would hurt most. A confirming read that
    failed must not render as a clean book."""
    from tradeflow.execution.flatten import UNKNOWN

    broker = FailingBroker(positions=[_position()])
    broker.failures["list_positions"] = BrokerUnavailableError("timeout")

    report = flatten(broker, reason="drill", halt_state=halts)

    assert report.observed == UNKNOWN
    assert not report.complete
    assert "UNCONFIRMED" in report.summary()
    assert "positions observed closed : unknown" in report.summary()


def test_a_failed_close_is_distinguished_from_a_queued_one(halts):
    """`no` and `pending` are different operator situations: one means the venue never
    took the request, the other means it took it and has not filled it."""
    from tradeflow.execution.flatten import NOT_CLOSED

    broker = FailingBroker(positions=[_position()])
    broker.failures["close_all_positions"] = BrokerUnavailableError("timeout")

    report = flatten(broker, reason="drill", halt_state=halts)

    assert report.close_submitted is False
    assert report.observed == NOT_CLOSED
    assert not report.complete
