"""What an entry is sized against, on both clocks.

The backtest sized each entry against free cash - the capital less what is already
deployed, so every open position shrinks the next. Live sized against
`min(venue buying power, capital)`, which on a margin or paper account stays equal to the
capital however much is open. At a capital of 10,000, a 1% risk and a 4% stop, three
simultaneous $50 entries came out at 2,500 / 1,850 / 1,400 in the backtest and
2,500 / 2,500 / 2,500 live: the same declared book deployed more per name, and more in
total, than the one that had been validated. With no capital at all, live sized on the
account's margin buying power.

Each clock's own tests passed. The load-bearing test feeds one sequence of entries to both
and compares the notionals.
"""

from datetime import datetime

import pandas as pd
import pytest

from quantred.brokers.base import AccountSnapshot
from quantred.engine.backtest import BacktestEngine
from quantred.execution.live_trader import LiveTrader
from quantred.execution.sizing import deployable_pool, free_buying_power
from quantred.marketdata.client import MarketDataClient
from quantred.strategies import signals
from quantred.utils.timeutils import NEW_YORK
from tests.fakes import DictMarketData, FakeBroker
from tests.test_backtest import ScriptedStrategy

CAPITAL = 10_000.0
PRICE = 50.0
SYMBOLS = ["AAA", "BBB", "CCC"]
_CONFIG = {
    "risk_per_trade": 0.01,
    "stop_loss": 0.04,
    "take_profit": 0.50,
    "position_limits": {"max_positions": 10, "max_position_size": 1e9, "max_total_risk": 5.0},
}


def _backtest_notionals():
    index = pd.date_range("2024-01-02", periods=4, freq="D", tz=NEW_YORK)
    frame = pd.DataFrame([{"open": PRICE, "high": PRICE, "low": PRICE, "close": PRICE, "volume": 1}] * 4, index=index)
    strategy = ScriptedStrategy([signals.BUY, signals.HOLD, signals.HOLD, signals.HOLD], _CONFIG)
    client = MarketDataClient(DictMarketData({s: frame for s in SYMBOLS}))
    result = BacktestEngine(strategy, client).run(SYMBOLS, datetime(2024, 1, 2), datetime(2024, 1, 10), CAPITAL)
    trades = result.trades.set_index("symbol")
    return [float(trades.loc[s, "size"] * trades.loc[s, "entry_price"]) for s in SYMBOLS]


def _live_notionals(capital, equity=CAPITAL, buying_power=4 * CAPITAL):
    """The same three entries, in the same order, through the live admission path -
    against a margin account whose buying power is a multiple of its equity."""
    broker = FakeBroker()
    broker.account = AccountSnapshot(
        cash=equity, equity=equity, buying_power=buying_power, portfolio_value=equity
    )
    strategy = ScriptedStrategy([], _CONFIG)
    trader = LiveTrader(broker, strategy, capital=capital, respect_market_hours=False)
    for symbol in SYMBOLS:
        decision = trader.handle_signal(symbol, signals.BUY, price=PRICE)
        assert decision.allowed, decision.reason
    return [order["qty"] * PRICE for order in broker.orders]


def test_the_backtest_notionals_are_what_the_comparison_is_pinned_to():
    """Worked by hand so the parity test below cannot agree with a wrong backtest: each
    entry risks 1% of what is still free at a 4% stop, in whole $50 shares."""
    assert _backtest_notionals() == [2500.0, 1850.0, 1400.0]


def test_live_sizes_each_entry_against_what_the_book_has_left_like_the_backtest():
    assert _live_notionals(capital=CAPITAL) == _backtest_notionals()


def test_live_with_no_capital_sizes_on_equity_not_margin_buying_power():
    """No capital was configured, so the pool is the account's equity - the backtest's
    book can never deploy more than its capital - and the venue's 4x lending limit is not
    what the risk percentage is a percentage of."""
    assert _live_notionals(capital=None) == _backtest_notionals()


def test_live_on_a_larger_account_is_still_held_to_its_capital():
    assert _live_notionals(capital=CAPITAL, equity=10 * CAPITAL, buying_power=40 * CAPITAL) == _backtest_notionals()


# --- the shared rule itself, both directions ----------------------------------------
def test_free_buying_power_is_the_pool_less_what_is_held():
    assert free_buying_power(10_000.0, 2_500.0) == 7_500.0


def test_the_venue_only_ever_lowers_the_answer():
    assert free_buying_power(10_000.0, 0.0, venue_buying_power=3_000.0) == 3_000.0
    assert free_buying_power(10_000.0, 0.0, venue_buying_power=40_000.0) == 10_000.0


def test_an_over_deployed_book_has_nothing_left_never_a_negative_amount():
    assert free_buying_power(10_000.0, 12_000.0) == 0.0
    assert free_buying_power(10_000.0, 0.0, venue_buying_power=-5.0) == 0.0


def test_the_pool_is_the_capital_capped_at_equity_or_the_equity_alone():
    assert deployable_pool(3_000.0, 8_000.0) == 3_000.0
    assert deployable_pool(30_000.0, 8_000.0) == 8_000.0
    assert deployable_pool(30_000.0, None) == 30_000.0
    assert pytest.approx(deployable_pool(30_000.0)) == 30_000.0
