"""Execution-layer tests using an in-memory FakeBroker."""

import pytest

from src.brokers.base import AccountSnapshot, OrderSide, Position
from src.execution.live_trader import LiveTrader
from src.execution.sizing import BetaSizer, PortfolioWeightSizer, RiskBasedSizer
from src.strategies import signals
from src.strategies.volume_spike import VolumeSpikeStrategy
from tests.fakes import FakeBroker


def _trader(broker, sizer=None):
    return LiveTrader(broker, VolumeSpikeStrategy.create_with_defaults(), sizer=sizer)


# --- position sizers --------------------------------------------------------
def _account(equity=100_000.0):
    return AccountSnapshot(cash=equity, equity=equity, buying_power=equity, portfolio_value=equity)


def test_portfolio_weight_sizer():
    sizer = PortfolioWeightSizer({"AAA": 0.25})
    assert sizer.size("AAA", 100.0, _account()) == 250.0  # 0.25 * 100k / 100
    assert sizer.size("BBB", 100.0, _account()) == 0.0  # unfunded symbol


def test_risk_based_sizer_delegates_to_strategy():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    sizer = RiskBasedSizer(strategy)
    assert sizer.size("AAA", 100.0, _account()) == strategy.calculate_position_size(100_000.0, 100.0)


def test_beta_sizer_scales_inversely_with_beta():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    account = _account()
    # beta 1.0 is the neutral baseline; beta 2.0 should roughly halve the size.
    base = BetaSizer(strategy, {"AAA": 1.0}).size("AAA", 100.0, account)
    high_beta = BetaSizer(strategy, {"AAA": 2.0}).size("AAA", 100.0, account)
    assert high_beta == pytest.approx(base / 2.0)


def test_beta_sizer_uses_default_for_unknown_symbol():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    account = _account()
    unknown = BetaSizer(strategy, {}, default_beta=1.0).size("ZZZ", 100.0, account)
    neutral = BetaSizer(strategy, {"ZZZ": 1.0}).size("ZZZ", 100.0, account)
    assert unknown == neutral


def test_beta_sizer_clamps_extreme_beta():
    strategy = VolumeSpikeStrategy.create_with_defaults()
    account = _account()
    # Beyond max_abs_beta, sizes should match the clamp (not keep shrinking).
    at_cap = BetaSizer(strategy, {"AAA": 4.0}, max_abs_beta=4.0).size("AAA", 100.0, account)
    beyond = BetaSizer(strategy, {"AAA": 99.0}, max_abs_beta=4.0).size("AAA", 100.0, account)
    assert beyond == pytest.approx(at_cap)


def test_live_trader_sizes_from_portfolio_weights():
    broker = FakeBroker(buying_power=100_000)  # equity == buying_power in the fake
    _trader(broker, sizer=PortfolioWeightSizer({"AAA": 0.5})).handle_signal("AAA", signals.BUY, 100.0)
    assert broker.orders[0]["qty"] == 500  # floor(0.5 * 100k / 100)


def test_entry_submits_bracket_order():
    broker = FakeBroker(buying_power=100_000)
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)

    assert len(broker.orders) == 1
    order = broker.orders[0]
    assert order["type"] == "bracket" and order["symbol"] == "AAA"
    assert order["stop_loss"] < 100.0 < order["take_profit"]
    assert order["qty"] > 0


def test_entry_skipped_when_position_exists():
    pos = Position(
        "AAA", qty=5, side="long", avg_entry_price=90, current_price=100, market_value=500, unrealized_pl=50
    )
    broker = FakeBroker(positions=[pos])
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)
    assert broker.orders == []


def test_exit_closes_matching_position():
    pos = Position(
        "AAA", qty=5, side="long", avg_entry_price=90, current_price=100, market_value=500, unrealized_pl=50
    )
    broker = FakeBroker(positions=[pos])
    _trader(broker).handle_signal("AAA", signals.CLOSE_BUY, price=100.0)
    assert broker.closed == ["AAA"]


def test_exit_ignored_when_side_mismatches():
    pos = Position(
        "AAA", qty=5, side="long", avg_entry_price=90, current_price=100, market_value=500, unrealized_pl=50
    )
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


def test_closed_market_blocks_orders():
    broker = FakeBroker(buying_power=100_000, market_open=False)
    _trader(broker).handle_signal("AAA", signals.BUY, price=100.0)
    assert broker.orders == []


def test_market_hours_can_be_disabled():
    broker = FakeBroker(buying_power=100_000, market_open=False)
    LiveTrader(broker, VolumeSpikeStrategy.create_with_defaults(), respect_market_hours=False).handle_signal(
        "AAA", signals.BUY, price=100.0
    )
    assert len(broker.orders) == 1


# --- order management -------------------------------------------------------
def test_pending_order_blocks_duplicate_entry():
    broker = FakeBroker(buying_power=100_000)
    trader = _trader(broker)
    trader.handle_signal("AAA", signals.BUY, 100.0)  # places a bracket order
    trader.handle_signal("AAA", signals.BUY, 100.0)  # order pending -> must not double-submit
    assert len(broker.orders) == 1


def test_exit_cancels_resting_orders_before_closing():
    pos = Position(
        "AAA", qty=5, side="long", avg_entry_price=90, current_price=100, market_value=500, unrealized_pl=50
    )
    broker = FakeBroker(positions=[pos])
    # Simulate a resting bracket leg for the symbol.
    broker.submit_bracket_order("AAA", 5, OrderSide.SELL, 95, 110)
    assert broker.list_open_orders("AAA")  # precondition

    _trader(broker).handle_signal("AAA", signals.CLOSE_BUY, 100.0)
    assert broker.closed == ["AAA"]
    assert broker.list_open_orders("AAA") == []  # resting legs canceled
