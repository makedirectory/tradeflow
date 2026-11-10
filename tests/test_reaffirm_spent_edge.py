"""An entry edge that was acted on is not re-stated, on either clock.

Signals are edge-triggered and the backtest acts on each edge once: a position closed by
its own target leaves the book flat while the score still implies the direction, and the
backtest stays flat until the score leaves and crosses again. Live re-states an entry the
book lacks, which exists for an edge the process never saw. It also fired for an edge
the process had already traded, so live bought again one bar after every take-profit
- a trade no backtest of the same bars contains. Each clock's own tests passed; the
load-bearing test runs one set of bars through both and compares the entries.
"""

from datetime import datetime

import pandas as pd

from quantred.engine.backtest import BacktestEngine
from quantred.execution.live_trader import LiveTrader
from quantred.marketdata.client import MarketDataClient
from quantred.strategies import signals
from quantred.utils.timeutils import NEW_YORK
from tests.fakes import DictMarketData, FakeBroker, RecordingBroker, ScriptedStrategy

CONFIG = {"pivot": 100.0, "risk_per_trade": 0.02, "stop_loss": 0.03, "take_profit": 0.06}

#: Crosses the pivot on bar 1, runs through the target on bar 3, and stays above the
#: pivot to the end: one edge, one target exit, and a score that never stops saying long.
BARS = [
    (99, 99, 99, 99),
    (100, 101, 100, 101),
    (101, 102, 101, 102),
    (102, 108, 102, 108),
    (108, 109, 108, 109),
    (109, 110, 109, 110),
    (110, 111, 110, 111),
]


def _frame(rows):
    index = pd.date_range("2024-01-02", periods=len(rows), freq="B", tz=NEW_YORK)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).assign(volume=1e6)


def _backtest_entries(frame):
    client = MarketDataClient(DictMarketData({"AAA": frame}))
    result = BacktestEngine(ScriptedStrategy(dict(CONFIG)), client).run(
        ["AAA"], frame.index[0], frame.index[-1], 100_000.0
    )
    return len(result.trades), list(result.trades["exit_reason"])


def _live_entries(frame):
    """The same bars through the live path. The broker fills nothing by itself, so the
    target is played by hand: on the bar whose high reaches it the broker is flat, and
    the trader re-reads the symbol exactly as it does on a fill."""
    strategy = ScriptedStrategy(dict(CONFIG))
    trader = LiveTrader(FakeBroker(), strategy, respect_market_hours=False)
    entries = 0
    for timestamp, bar in frame.iterrows():
        held = strategy.positions.get("AAA")
        if held is not None and bar["high"] >= held["take_profit"]:
            trader.broker.open_orders_list.clear()  # the bracket is done once a leg fills
            trader.refresh_position("AAA")  # broker holds nothing: the book goes flat
        signal = strategy.process_bar("AAA", bar.to_dict(), timestamp)
        if signal in signals.ENTRY_SIGNALS:
            decision = trader.handle_signal("AAA", signal, bar["close"], timestamp)
            entries += int(decision.allowed)
    return entries


def test_live_enters_as_often_as_the_backtest_when_a_target_closes_a_position_mid_trend():
    frame = _frame(BARS)
    trades, reasons = _backtest_entries(frame)
    assert (trades, reasons) == (1, ["TAKE_PROFIT"])  # the scenario is what it claims to be
    assert _live_entries(frame) == trades


def test_a_fresh_crossing_after_a_spent_edge_is_taken_on_both_clocks():
    """Spent is not forever: once the score leaves the direction, the next crossing is a
    new edge, and both clocks trade it."""
    rows = [*BARS, (110, 110, 98, 98), (98, 99, 98, 99), (99, 102, 99, 102), (102, 103, 102, 103)]
    frame = _frame(rows)
    trades, _ = _backtest_entries(frame)
    assert trades == 2
    assert _live_entries(frame) == trades


def test_an_edge_the_process_never_saw_is_still_restated():
    """What re-affirmation is for: started mid-trend, the first live bar opens the position."""
    strategy = ScriptedStrategy(dict(CONFIG))
    frame = _frame(BARS)
    strategy.warm_up("AAA", frame.iloc[:3])  # the crossing is inside the warm-up history
    timestamp, bar = frame.index[3], frame.iloc[3]
    assert strategy.process_bar("AAA", bar.to_dict(), timestamp) == signals.BUY


def _trader(broker, limits):
    strategy = ScriptedStrategy({**CONFIG, "position_limits": limits})
    return strategy, LiveTrader(broker, strategy, respect_market_hours=False)


def test_an_entry_the_book_refused_is_spent_as_the_backtest_declines_it_once():
    limits = {"max_positions": 1, "max_position_size": 100_000.0, "max_total_risk": 0.05}
    strategy, trader = _trader(FakeBroker(), limits)
    strategy.positions["AAA"] = {"side": signals.BUY, "qty": 10, "entry_price": 100.0}
    decision = trader.handle_signal("BBB", signals.BUY, 101.0, datetime(2024, 1, 3))
    assert not decision.allowed and decision.reason_code == "book_full"
    del strategy.positions["AAA"]  # the slot frees later; the backtest never returns to BBB
    assert strategy._reaffirm("BBB", signals.HOLD, 1) == signals.HOLD
    assert strategy._reaffirm("BBB", signals.HOLD, 0) == signals.HOLD  # the score leaves long
    assert strategy._reaffirm("BBB", signals.HOLD, 1) == signals.BUY  # and a missed edge is new


def test_an_entry_the_broker_refused_stays_open_to_being_restated():
    """The backtest has no broker to refuse it, so this is a missed edge, not a spent one."""
    limits = {"max_positions": 5, "max_position_size": 100_000.0, "max_total_risk": 0.05}
    strategy, trader = _trader(RecordingBroker(reject_orders=True), limits)
    decision = trader.handle_signal("AAA", signals.BUY, 101.0, datetime(2024, 1, 3))
    assert not decision.allowed
    assert strategy._reaffirm("AAA", signals.HOLD, 1) == signals.BUY
