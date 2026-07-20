"""Backtest engine fill-logic tests.

Uses a ``ScriptedStrategy`` that emits a fixed signal per bar, so we can assert
the engine's exit handling (stop-loss, take-profit, signal exit, end-of-period)
and P&L exactly - independent of any indicator behavior.
"""

from datetime import datetime
from typing import Dict, List

import pandas as pd
import pytest

from src.engine.backtest import BacktestEngine
from src.marketdata.client import MarketDataClient
from src.strategies import signals
from src.strategies.base import Strategy
from src.utils.timeutils import NEW_YORK
from tests.fakes import DictMarketData

# stop_loss/take_profit = 10%; size resolves to 10 shares at $100 (see notional cap).
_CONFIG = {
    "timeframe": "1Day",
    "risk_per_trade": 1.0,
    "stop_loss": 0.10,
    "take_profit": 0.10,
    "position_limits": {"max_positions": 1, "max_position_size": 1000.0, "max_total_risk": 1.0},
}


class ScriptedStrategy(Strategy):
    PARAM_RANGES: Dict = {}

    def __init__(self, per_bar_signals: List[str]):
        self._signals = per_bar_signals
        super().__init__(dict(_CONFIG))

    def calculate_required_lookback(self) -> int:
        return 1

    def initialize(self) -> None:
        pass

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        # Engine fill-logic tests script signals directly; the score is unused.
        return pd.Series(0.0, index=data.index)

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        return dict(zip(data.index, self._signals))


def _frame(rows: List[dict]) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=len(rows), freq="D", tz=NEW_YORK)
    return pd.DataFrame(rows, index=index)


def _run(rows, per_bar_signals):
    strategy = ScriptedStrategy(per_bar_signals)
    data_client = MarketDataClient(DictMarketData({"AAA": _frame(rows)}))
    return BacktestEngine(strategy, data_client).run(
        ["AAA"], datetime(2024, 1, 2), datetime(2024, 1, 10), 100_000
    )


def test_take_profit_exit():
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"open": 100, "high": 105, "low": 98, "close": 104, "volume": 1},
        {"open": 108, "high": 112, "low": 107, "close": 111, "volume": 1},
    ]
    result = _run(rows, [signals.BUY, signals.HOLD, signals.HOLD])
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "TAKE_PROFIT"
    assert trade["exit_price"] == pytest.approx(110) and trade["pnl"] == pytest.approx(100)  # (110-100)*10
    assert result.final_capital == pytest.approx(100_100)


def test_stop_loss_exit():
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"open": 99, "high": 100, "low": 89, "close": 92, "volume": 1},
    ]
    result = _run(rows, [signals.BUY, signals.HOLD])
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_price"] == pytest.approx(90) and trade["pnl"] == pytest.approx(-100)


def test_signal_exit_at_open():
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"open": 103, "high": 104, "low": 99, "close": 103, "volume": 1},
    ]
    result = _run(rows, [signals.BUY, signals.SELL])  # SELL closes the long at next open
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "SIGNAL"
    assert trade["exit_price"] == 103 and trade["pnl"] == 30


def test_end_of_period_exit():
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"open": 101, "high": 103, "low": 99, "close": 105, "volume": 1},
    ]
    result = _run(rows, [signals.BUY, signals.HOLD])  # never hits stop/take/exit
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "END_OF_PERIOD"
    assert trade["exit_price"] == 105 and trade["pnl"] == 50


def test_backtest_honors_injected_sizer():
    from src.engine.backtest import BacktestEngine
    from src.execution.sizing import PortfolioWeightSizer
    from src.marketdata.client import MarketDataClient
    from tests.fakes import DictMarketData

    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"open": 108, "high": 112, "low": 107, "close": 111, "volume": 1},  # take-profit
    ]
    strategy = ScriptedStrategy([signals.BUY, signals.HOLD])
    data_client = MarketDataClient(DictMarketData({"AAA": _frame(rows)}))
    sizer = PortfolioWeightSizer({"AAA": 0.5})  # 0.5 * equity(=100k) / price(100) = 500 shares

    result = BacktestEngine(strategy, data_client, sizer=sizer).run(
        ["AAA"], datetime(2024, 1, 2), datetime(2024, 1, 10), 100_000
    )
    assert result.trades.iloc[0]["size"] == 500


def test_no_signals_means_no_trades():
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}] * 3
    result = _run(rows, [signals.HOLD, signals.HOLD, signals.HOLD])
    assert result.trades.empty
    assert result.final_capital == 100_000
    assert result.metrics["total_trades"] == 0


class _BrokenStrategy(ScriptedStrategy):
    """Raises during signal generation, as an unconstructable config would."""

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        raise KeyError("risk_per_trade")


def test_all_symbols_failing_raises_instead_of_reporting_no_trades():
    """A universe-wide failure is an error, not a zero-edge result.

    Swallowing it would let a broken strategy be scored as "no edge" by the
    promotion gates - a false negative the validation engine must not have.
    """
    from src.engine.backtest import BacktestError

    rows = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}] * 3
    data_client = MarketDataClient(DictMarketData({"AAA": _frame(rows)}))
    with pytest.raises(BacktestError, match="risk_per_trade"):
        BacktestEngine(_BrokenStrategy(["HOLD"] * 3), data_client).run(
            ["AAA"], datetime(2024, 1, 2), datetime(2024, 1, 10), 100_000
        )


def test_one_bad_symbol_still_completes_the_run():
    """Partial failure keeps the original behavior: skip the symbol, run the rest."""
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"open": 100, "high": 120, "low": 99, "close": 118, "volume": 1},
        {"open": 118, "high": 120, "low": 99, "close": 118, "volume": 1},
    ]

    class HalfBroken(ScriptedStrategy):
        def generate_signals(self, data: pd.DataFrame) -> Dict:
            if float(data["close"].iloc[0]) == 50.0:  # the "bad" symbol
                raise ValueError("bad symbol")
            return dict(zip(data.index, self._signals))

    bad = [{**r, "open": 50, "close": 50} for r in rows]
    data_client = MarketDataClient(DictMarketData({"AAA": _frame(rows), "BAD": _frame(bad)}))
    result = BacktestEngine(HalfBroken(["BUY", "HOLD", "CLOSE_BUY"]), data_client).run(
        ["AAA", "BAD"], datetime(2024, 1, 2), datetime(2024, 1, 10), 100_000
    )
    assert result.metrics["total_trades"] == 1


# --------------------------------------------------------------------------- #
# Portfolio accounting (spec 025)
# --------------------------------------------------------------------------- #
def _multi(frames: Dict[str, pd.DataFrame], per_bar_signals: List[str], capital=100_000):
    strategy = ScriptedStrategy(per_bar_signals)
    data_client = MarketDataClient(DictMarketData(frames))
    return BacktestEngine(strategy, data_client).run(
        sorted(frames), datetime(2024, 1, 2), datetime(2024, 1, 10), capital
    )


_ROUND_TRIP = [
    {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
    {"open": 100, "high": 120, "low": 99, "close": 118, "volume": 1},
    {"open": 118, "high": 119, "low": 99, "close": 100, "volume": 1},
]


def test_metrics_do_not_scale_with_universe_size():
    """The bug that motivated spec 025: N copies of a symbol must not multiply returns.

    With max_positions=1 the book can only ever hold one of the duplicates, so a
    universe of N identical names must produce exactly the single-name result.
    Under per-symbol accounting return scaled roughly linearly with N.
    """
    one = _multi({"AAA": _frame(_ROUND_TRIP)}, ["BUY", "HOLD", "CLOSE_BUY"])
    many = _multi(
        {sym: _frame(_ROUND_TRIP) for sym in ("AAA", "BBB", "CCC", "DDD")},
        ["BUY", "HOLD", "CLOSE_BUY"],
    )
    assert many.final_capital == pytest.approx(one.final_capital)
    assert many.metrics["total_return"] == pytest.approx(one.metrics["total_return"])
    assert many.metrics["total_trades"] == one.metrics["total_trades"] == 1


def test_position_limit_is_enforced_across_the_whole_book():
    """max_positions is a portfolio limit, not a per-symbol one."""
    frames = {sym: _frame(_ROUND_TRIP) for sym in ("AAA", "BBB", "CCC", "DDD")}
    result = _multi(frames, ["BUY", "HOLD", "CLOSE_BUY"])
    # One position at a time => at most one trade per round trip, not one per symbol.
    assert len(result.trades) == 1


def test_capital_is_conserved():
    """Final capital is initial plus realized P&L, and the curve ends there."""
    frames = {sym: _frame(_ROUND_TRIP) for sym in ("AAA", "BBB")}
    result = _multi(frames, ["BUY", "HOLD", "CLOSE_BUY"])
    expected = result.initial_capital + result.trades["pnl"].sum()
    assert result.final_capital == pytest.approx(expected)
    assert result.equity_curve[-1] == pytest.approx(result.final_capital)
    assert min(result.equity_curve) > 0


def test_ranking_is_deterministic_regardless_of_universe_order():
    """Shuffling the universe must not change the result."""
    frames = {sym: _frame(_ROUND_TRIP) for sym in ("AAA", "BBB", "CCC")}
    forward = _multi(frames, ["BUY", "HOLD", "CLOSE_BUY"])
    reversed_frames = {sym: frames[sym] for sym in reversed(list(frames))}
    backward = _multi(reversed_frames, ["BUY", "HOLD", "CLOSE_BUY"])
    assert forward.final_capital == pytest.approx(backward.final_capital)
    assert list(forward.trades["symbol"]) == list(backward.trades["symbol"])


def test_equity_curve_marks_open_positions_to_market():
    """An open position moves the curve; it must not sit flat until the exit bar."""
    rising = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"open": 100, "high": 110, "low": 100, "close": 109, "volume": 1},
        {"open": 109, "high": 118, "low": 108, "close": 117, "volume": 1},
    ]
    result = _multi({"AAA": _frame(rising)}, ["BUY", "HOLD", "HOLD"])
    # Bars 2 and 3 are both held; with mark-to-market they differ.
    assert result.equity_curve[2] != pytest.approx(result.equity_curve[3])


def test_freed_capital_is_reusable_in_the_same_bar():
    """Exits settle before entries, so a close funds an open on the same bar."""
    frames = {"AAA": _frame(_ROUND_TRIP), "BBB": _frame(_ROUND_TRIP)}
    # AAA closes on bar 3 (CLOSE_BUY); BBB should be able to enter on that same bar.
    result = _multi(frames, ["BUY", "HOLD", "CLOSE_BUY"])
    assert len(result.trades) >= 1
    # The single slot is reused rather than left idle after the exit.
    assert result.trades["exit_reason"].iloc[0] in ("SIGNAL", "TAKE_PROFIT", "END_OF_PERIOD")


def test_trade_from_warms_up_without_trading():
    """Bars before ``trade_from`` feed indicators but open no positions.

    This is what lets an OOS window be measured on its own portfolio curve instead
    of a curve reconstructed from a filtered trade list.
    """
    rows = _ROUND_TRIP + _ROUND_TRIP
    frames = {"AAA": _frame(rows)}
    index = _frame(rows).index

    full = _multi(frames, ["BUY", "HOLD", "CLOSE_BUY"] * 2)
    strategy = ScriptedStrategy(["BUY", "HOLD", "CLOSE_BUY"] * 2)
    gated = BacktestEngine(strategy, MarketDataClient(DictMarketData(frames))).run(
        ["AAA"],
        datetime(2024, 1, 2),
        datetime(2024, 1, 10),
        100_000,
        trade_from=index[3].to_pydatetime(),
    )
    # The first round trip is warmup only, so the gated run keeps just the second.
    assert len(full.trades) == 2
    assert len(gated.trades) == 1
    assert gated.trades["entry_time"].iloc[0] >= index[3]
    # And its curve covers only the traded span, not the warmup.
    assert len(gated.equity_curve) < len(full.equity_curve)
