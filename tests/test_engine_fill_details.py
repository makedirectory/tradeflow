"""What a fill is priced and levelled from.

Four places the backtest computed something from a price other than the one that
applies, each found by comparing it with what the live path, or the fill itself, uses.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantred.brokers.base import AccountSnapshot
from quantred.costs.parametric import ParametricCostModel
from quantred.engine.backtest import BacktestEngine, _benchmark_returns
from quantred.execution.live_trader import LiveTrader
from quantred.marketdata.client import MarketDataClient
from quantred.strategies import signals
from quantred.utils.timeutils import NEW_YORK
from tests.fakes import DictMarketData, FakeBroker
from tests.test_backtest import ScriptedStrategy, _frame

B, H = signals.BUY, signals.HOLD
CAPITAL = 100_000.0
#: Commission only, so a cost is a known fraction of the notional that traded.
COMMISSION = ParametricCostModel(commission_bps=10.0, default_spread_bps=0.0, impact_eta=0.0)


def _bar(open_, high, low, close, volume=1):
    row = {"open": open_, "high": high, "low": low, "close": close}
    return row if volume is None else {**row, "volume": volume}


def _run(rows, per_bar_signals, cost_model=None):
    client = MarketDataClient(DictMarketData({"AAA": _frame(rows)}))
    return BacktestEngine(ScriptedStrategy(per_bar_signals), client, cost_model=cost_model).run(
        ["AAA"], datetime(2024, 1, 2), datetime(2024, 1, 10), CAPITAL
    )


def test_a_stop_is_costed_on_the_level_it_filled_at_not_the_bar_s_open():
    """The exit's cost was priced before the exit price was known, on the open. A stop
    that fills 10% below the open was charged as if the full notional had traded.
    10 shares in at 100, out at the 90 stop: 10 bp of 1,000 plus 10 bp of 900."""
    rows = [_bar(100, 100, 100, 100), _bar(100, 100.5, 100, 100), _bar(100, 100, 89, 91)]
    trade = _run(rows, [B, H, H], COMMISSION).trades.iloc[0]
    assert (trade["exit_reason"], trade["exit_price"]) == ("STOP_LOSS", 90.0)
    assert trade["cost"] == pytest.approx(0.001 * (10 * 100 + 10 * 90), rel=1e-12)


def test_bars_without_volume_are_still_charged_commission():
    """No volume column means no liquidity context, which removes the impact term. It
    used to remove every cost, so the run was gross while labelled net."""
    rows = [_bar(100, 100, 100, 100, None), _bar(100, 100.5, 100, 100, None), _bar(100, 101, 100, 101, None)]
    result = _run(rows, [B, H, H], COMMISSION)
    assert result.total_cost == pytest.approx(0.001 * (10 * 100 + 10 * 101), rel=1e-12)


def test_size_and_bracket_come_from_the_signal_s_close_on_both_clocks():
    """Live submits a bracket with its stop and target already set, from the close the
    signal came from; the backtest set them from the open the order filled at. On a gap
    the two stops sat at different prices: here the signal closes at 100 and the fill is
    at 104, so a 10% stop is 90 live and was 93.6 in the backtest, which stopped out on
    a low of 93 that live's order would have ridden through. Size differed the same way."""
    rows = [_bar(100, 100, 100, 100), _bar(104, 104.5, 93, 104), _bar(104, 105, 103, 104)]
    trade = _run(rows, [B, H, H]).trades.iloc[0]

    broker = FakeBroker()
    broker.account = AccountSnapshot(
        cash=CAPITAL, equity=CAPITAL, buying_power=CAPITAL, portfolio_value=CAPITAL
    )
    trader = LiveTrader(broker, ScriptedStrategy([]), capital=CAPITAL, respect_market_hours=False)
    assert trader.handle_signal("AAA", B, price=100.0).allowed
    order = broker.orders[0]

    assert order["stop_loss"] == 90.0  # by hand: 10% under the 100 close
    assert trade["exit_reason"] == "END_OF_PERIOD"  # a low of 93 is above that stop
    assert trade["entry_price"] == 104  # the fill is still the open
    assert trade["size"] == order["qty"] == 10  # 1,000 of notional at the 100 close


def _benchmark(n=4):
    index = pd.date_range("2024-01-02", periods=n, freq="D", tz=NEW_YORK)
    closes = pd.Series([100.0, 102.0, 101.0, 103.0][:n], index=index)
    opens = pd.Series([99.0, 100.0, 102.5, 101.0][:n], index=index)
    return index, closes, opens


def test_a_fold_s_first_step_pairs_with_the_benchmark_s_first_bar_open_to_close():
    """After a warmup the book can fill at the first bar's open, so its first-step return
    spans that bar's open to close. The benchmark's was left out, so a fold's return
    covered a bar its benchmark's did not."""
    index, closes, opens = _benchmark()
    returns = _benchmark_returns(closes, [None, *index[1:]], opens)  # index[0] is warmup
    assert returns.loc[1] == pytest.approx(102.0 / 100.0 - 1.0, rel=1e-12)
    assert returns.loc[2] == pytest.approx(101.0 / 102.0 - 1.0, rel=1e-12)


def test_with_no_bar_before_it_the_first_step_stays_unpaired():
    """Without a warmup a signal fills one bar later, so the book cannot hold through the
    first step, and regressing its zero on the benchmark's move would bias the beta."""
    index, closes, opens = _benchmark()
    returns = _benchmark_returns(closes, [None, *index], opens)
    assert np.isnan(returns.loc[1])
    assert returns.loc[2] == pytest.approx(102.0 / 100.0 - 1.0, rel=1e-12)
