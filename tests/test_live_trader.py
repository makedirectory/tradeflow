"""Execution-layer tests using an in-memory FakeBroker."""

from src.brokers.base import Position
from src.execution.live_trader import LiveTrader
from src.strategies import signals
from src.strategies.volume_spike import VolumeSpikeStrategy
from tests.fakes import FakeBroker


def _trader(broker):
    return LiveTrader(broker, VolumeSpikeStrategy.create_with_defaults())


def test_entry_submits_bracket_order():
    broker = FakeBroker(buying_power=100_000)
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)

    assert len(broker.orders) == 1
    order = broker.orders[0]
    assert order["type"] == "bracket" and order["symbol"] == "AAA"
    assert order["stop_loss"] < 100.0 < order["take_profit"]
    assert order["qty"] > 0


def test_entry_skipped_when_position_exists():
    pos = Position("AAA", qty=5, side="long", avg_entry_price=90,
                   current_price=100, market_value=500, unrealized_pl=50)
    broker = FakeBroker(positions=[pos])
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)
    assert broker.orders == []


def test_exit_closes_matching_position():
    pos = Position("AAA", qty=5, side="long", avg_entry_price=90,
                   current_price=100, market_value=500, unrealized_pl=50)
    broker = FakeBroker(positions=[pos])
    _trader(broker).handle_signal("AAA", signals.CLOSE_BUY, price=100.0)
    assert broker.closed == ["AAA"]


def test_exit_ignored_when_side_mismatches():
    pos = Position("AAA", qty=5, side="long", avg_entry_price=90,
                   current_price=100, market_value=500, unrealized_pl=50)
    broker = FakeBroker(positions=[pos])
    _trader(broker).handle_signal("AAA", signals.CLOSE_SELL, price=100.0)  # closing a short
    assert broker.closed == []


def test_hold_is_noop():
    broker = FakeBroker()
    _trader(broker).handle_signal("AAA", signals.HOLD, price=100.0)
    assert broker.orders == [] and broker.closed == []


def test_insufficient_buying_power_blocks_entry():
    broker = FakeBroker(buying_power=10.0)  # can't afford one $100 share
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)
    assert broker.orders == []
