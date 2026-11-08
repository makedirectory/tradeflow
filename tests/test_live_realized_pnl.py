"""The live trader's running realized P&L, which sets the pool it sizes from.

Related: ``tests/test_sizing_parity.py`` compares the resulting size with the backtest's.
These pin the bookkeeping itself, including the shapes a plausibly-wrong version gets
wrong: a short, a partial close counted twice, and a close the book had already forgotten.
"""

import pytest

from quantred.brokers.base import AccountSnapshot, Position
from quantred.execution.live_trader import LiveTrader
from quantred.strategies import signals
from tests.fakes import FakeBroker
from tests.test_backtest import ScriptedStrategy

_CONFIG = {"position_limits": {"max_positions": 5, "max_position_size": 1e9, "max_total_risk": 5.0}}


def _trader(*positions):
    broker = FakeBroker(positions=list(positions))
    broker.account = AccountSnapshot(cash=1e5, equity=1e5, buying_power=4e5, portfolio_value=1e5)
    trader = LiveTrader(broker, ScriptedStrategy([], _CONFIG), capital=10_000.0, respect_market_hours=False)
    trader.sync_strategy_book()
    return broker, trader


def test_a_signal_close_is_priced_after_the_book_has_forgotten_the_position():
    broker, trader = _trader(Position("AAA", 10.0, "long", 100.0, 90.0, 900.0, -100.0))
    assert trader.handle_signal("AAA", signals.CLOSE_BUY, price=90.0).allowed
    trader.refresh_position("AAA", 90.0)
    assert trader.realized_pnl == pytest.approx(-100.0)


def test_a_short_that_rises_is_a_loss():
    broker, trader = _trader(Position("AAA", 10.0, "short", 100.0, 110.0, 1100.0, -100.0))
    assert trader.handle_signal("AAA", signals.CLOSE_SELL, price=110.0).allowed
    trader.refresh_position("AAA", 110.0)
    assert trader.realized_pnl == pytest.approx(-100.0)


def test_partial_closing_fills_are_counted_once_each():
    """A broker-side exit that fills 4 then the remaining 6: the second event must not
    re-count the first four."""
    broker, trader = _trader(Position("AAA", 10.0, "long", 100.0, 110.0, 1100.0, 100.0))
    broker.positions["AAA"] = Position("AAA", 6.0, "long", 100.0, 110.0, 660.0, 60.0)
    trader.refresh_position("AAA", 110.0)  # 4 sold at 110
    assert trader.realized_pnl == pytest.approx(40.0)
    del broker.positions["AAA"]
    trader.refresh_position("AAA", 120.0)  # last 6 at 120
    assert trader.realized_pnl == pytest.approx(40.0 + 120.0)


def test_a_fill_without_a_price_or_that_grows_the_position_realizes_nothing():
    broker, trader = _trader(Position("AAA", 10.0, "long", 100.0, 90.0, 900.0, -100.0))
    broker.positions["AAA"] = Position("AAA", 15.0, "long", 98.0, 90.0, 1350.0, -120.0)
    trader.refresh_position("AAA", 98.0)  # an add
    del broker.positions["AAA"]
    trader.refresh_position("AAA", None)  # a close whose price is unknown: a gap, not a zero
    assert trader.realized_pnl == 0.0
