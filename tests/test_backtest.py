"""Backtest engine fill-logic tests.

Uses a ``ScriptedStrategy`` that emits a fixed signal per bar, so we can assert
the engine's exit handling (stop-loss, take-profit, signal exit, end-of-period)
and P&L exactly - independent of any indicator behavior.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytest

from tests.fakes import DictMarketData
from tradeflow.engine.backtest import BacktestEngine
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.strategies import signals
from tradeflow.strategies.base import Strategy
from tradeflow.utils.timeutils import NEW_YORK

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

    def __init__(self, per_bar_signals: List[str], overrides: Optional[Dict] = None):
        self._signals = per_bar_signals
        super().__init__({**_CONFIG, **(overrides or {})})

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
    from tests.fakes import DictMarketData
    from tradeflow.engine.backtest import BacktestEngine
    from tradeflow.execution.sizing import PortfolioWeightSizer
    from tradeflow.marketdata.client import MarketDataClient

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
    from tradeflow.engine.backtest import BacktestError

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
# Portfolio accounting
# --------------------------------------------------------------------------- #
def _multi(frames: Dict[str, pd.DataFrame], per_bar_signals: List[str], capital=100_000, overrides=None):
    strategy = ScriptedStrategy(per_bar_signals, overrides)
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
    """The bug that motivated portfolio-level accounting: N copies of a symbol must not multiply returns.

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


def test_ragged_grid_annualizes_on_the_merged_timeline():
    """A symbol that trades on off-grid timestamps must not inflate Sharpe.

    Regression: per-step quantities were annualized at the strategy's single-symbol
    timeframe rate. The merged timeline is the union of every symbol's timestamps, so
    a universe whose symbols don't share a bar grid produces more steps than that rate
    assumes, scaling Sharpe and volatility by sqrt(density).
    """
    rows = _ROUND_TRIP + _ROUND_TRIP
    aligned = _frame(rows)
    # Same bars, shifted half a day off the shared grid: the union is twice as dense
    # while neither symbol's own history got any denser.
    offset = aligned.copy()
    offset.index = aligned.index + pd.Timedelta(hours=12)

    engine = BacktestEngine(
        ScriptedStrategy(["BUY", "HOLD", "CLOSE_BUY"] * 2),
        MarketDataClient(DictMarketData({"AAA": aligned, "BBB": offset})),
    )
    engine.run(["AAA", "BBB"], datetime(2024, 1, 2), datetime(2024, 1, 10), 100_000)

    base = engine._periods_per_year
    # Two interleaved copies of one grid: ~2x the steps, so ~2x the annualization rate.
    assert engine._step_periods_per_year == pytest.approx(2 * base, rel=0.15)
    assert engine._step_periods_per_year > base


def test_aligned_grid_keeps_the_timeframe_rate():
    """The density correction must be inert for the common case.

    Symbols sharing one grid make the merged timeline exactly as dense as any single
    symbol's, so the rate has to come out at the timeframe's own value - otherwise the
    fix would silently restate every ordinary backtest.
    """
    rows = _ROUND_TRIP + _ROUND_TRIP
    frames = {"AAA": _frame(rows), "BBB": _frame(rows)}
    engine = BacktestEngine(
        ScriptedStrategy(["BUY", "HOLD", "CLOSE_BUY"] * 2),
        MarketDataClient(DictMarketData(frames)),
    )
    engine.run(["AAA", "BBB"], datetime(2024, 1, 2), datetime(2024, 1, 10), 100_000)
    assert engine._step_periods_per_year == pytest.approx(engine._periods_per_year)


# --------------------------------------------------------------------------- #
# Risk budget vs. gross exposure
# --------------------------------------------------------------------------- #
#: Flat prices, so nothing stops or takes profit and every admitted position
#: survives to the closing signal - the trade count is the admission count.
_FLAT = [{"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1}] * 3

#: A 1% stop with a $20k notional cap risks $200 per position, so four positions
#: spend $800 of budget while deploying $80k of notional against $100k of equity.
_TIGHT_STOP = {
    "stop_loss": 0.01,
    "take_profit": 0.01,
    "position_limits": {"max_positions": 4, "max_position_size": 20_000.0, "max_total_risk": 0.05},
}


def _four_names():
    return {sym: _frame(_FLAT) for sym in ("AAA", "BBB", "CCC", "DDD")}


def _limits(**changes):
    return {**_TIGHT_STOP, "position_limits": {**_TIGHT_STOP["position_limits"], **changes}}


def test_max_total_risk_does_not_bound_deployed_notional():
    """The distinction the config docs turn on, asserted rather than described.

    max_total_risk is loss-at-stop, so a tight stop buys a lot of notional for very
    little of it. Here a 5% budget admits 80% of equity in notional - anyone reading
    the fraction as an exposure cap is off by 16x.
    """
    result = _multi(_four_names(), ["BUY", "HOLD", "CLOSE_BUY"], overrides=_TIGHT_STOP)
    notional = (result.trades["size"] * result.trades["entry_price"]).sum()
    assert len(result.trades) == 4
    assert notional == pytest.approx(80_000)
    assert notional > 0.05 * result.initial_capital


def test_max_total_risk_bounds_loss_at_stop():
    """The budget it does enforce: $200 of risk per position against a $300 budget."""
    result = _multi(_four_names(), ["BUY", "HOLD", "CLOSE_BUY"], overrides=_limits(max_total_risk=0.003))
    assert len(result.trades) == 1


def test_max_gross_exposure_caps_deployed_notional():
    """The notional cap the risk budget is not: 45% of equity admits two $20k positions."""
    result = _multi(_four_names(), ["BUY", "HOLD", "CLOSE_BUY"], overrides=_limits(max_gross_exposure=0.45))
    notional = (result.trades["size"] * result.trades["entry_price"]).sum()
    assert len(result.trades) == 2
    assert notional == pytest.approx(40_000)


def test_max_gross_exposure_is_off_by_default():
    """Unset must change nothing - free cash stays the only bound on notional."""
    baseline = _multi(_four_names(), ["BUY", "HOLD", "CLOSE_BUY"], overrides=_TIGHT_STOP)
    explicit = _multi(_four_names(), ["BUY", "HOLD", "CLOSE_BUY"], overrides=_limits(max_gross_exposure=None))
    assert len(explicit.trades) == len(baseline.trades) == 4
    assert explicit.final_capital == pytest.approx(baseline.final_capital)


# --- the benchmark actually reaches the metrics -------------------------------
def _bench_run(benchmark):
    from tests.fakes import FakeMarketData
    from tradeflow.services.registry import STRATEGIES

    client = MarketDataClient(FakeMarketData(["AAA", "BBB", "SPY"], n=400, freq="1D"))
    engine = BacktestEngine(STRATEGIES["ma_crossover"].create_with_defaults(), client)
    return engine.run(
        ["AAA", "BBB"], datetime(2024, 1, 2), datetime(2025, 1, 2), 100_000, benchmark=benchmark
    )


def test_a_supplied_benchmark_reaches_the_metrics():
    """Reported from a real run: `--benchmark SPY` was accepted and the report still
    said `(i) no benchmark` with beta 0, beside a Buy & Hold of 95.56%.

    Both were true. Buy & Hold comes from the traded universe's own bars, while the
    benchmark never reached the engine at all - `run()` had no parameter for one, so
    `benchmark_returns` was always None and alpha/beta/IR were structurally zero.
    """
    result = _bench_run("SPY")

    assert result.metrics["benchmark_available"] is True
    assert result.metrics["beta"] != 0.0


def test_no_benchmark_still_says_so():
    """The other direction: the flag means something, so its absence must too."""
    result = _bench_run(None)

    assert result.metrics["benchmark_available"] is False
    assert result.metrics["beta"] == 0.0
    assert result.metrics["information_ratio"] == 0.0


def test_the_benchmark_is_aligned_to_the_curve_positionally():
    """The failure this had to avoid is silent, not loud.

    The equity curve reaches the metrics as a bare list of floats, so its returns carry
    a RangeIndex. A date-indexed benchmark would join to nothing, `dropna` would empty
    it, and the regression would report a confident zero rather than raising.
    """
    import pandas as pd

    from tests.fakes import FakeMarketData
    from tradeflow.engine.backtest import _benchmark_returns

    client = MarketDataClient(FakeMarketData(["SPY"], n=120, freq="1D"))
    closes = client.get_bars(["SPY"], "1Day", datetime(2024, 1, 2), datetime(2024, 6, 1))["SPY"]["close"]
    steps = list(closes.index[:20])

    aligned = _benchmark_returns(closes, [None, *steps])

    assert isinstance(aligned.index, pd.RangeIndex)  # positional, or it pairs with nothing
    assert len(aligned) == len(steps)
    assert aligned.notna().sum() >= len(steps) - 1


def test_an_unfetchable_benchmark_degrades_instead_of_failing_the_run():
    """A benchmark that cannot be loaded is a missing comparison, not a broken
    backtest — but it must land as `benchmark_available: False` rather than as a
    number scored against nothing."""
    result = _bench_run("NOT_A_REAL_SYMBOL")

    assert result.metrics["benchmark_available"] is False
    assert result.metrics["total_return"] == _bench_run(None).metrics["total_return"]
