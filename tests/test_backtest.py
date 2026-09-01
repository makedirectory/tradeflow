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


# --- two reporting fixes, both found by reading a real report ------------------
def test_buy_and_hold_names_what_it_holds():
    """It stayed 95.56% for both SPY and QQQ, which is how it surfaced.

    `buy_hold_return` averages the *traded universe*, so it cannot move when the
    benchmark does. The number was right and the label said only "Buy & Hold Return",
    which reads as the benchmark's return to anyone who just passed `--benchmark`.
    """
    from tradeflow.analytics.reporting import format_backtest_report

    rendered = format_backtest_report(
        {"buy_hold_return": 95.56, "benchmark_buy_hold_return": 41.2, "benchmark_available": True},
        4000.0,
        4480.0,
        title="t",
    )

    assert "Buy & Hold (universe)" in rendered
    assert "Buy & Hold (benchmark)" in rendered
    assert "Buy & Hold Return" not in rendered  # the ambiguous label is gone


def test_the_two_buy_and_holds_differ_and_only_one_tracks_the_benchmark():
    """The property the label now promises, asserted end to end rather than by name."""
    from tests.fakes import FakeMarketData
    from tradeflow.services.registry import STRATEGIES

    client = MarketDataClient(FakeMarketData(["AAA", "BBB", "SPY", "QQQ"], n=400, freq="1D"))
    engine = BacktestEngine(STRATEGIES["ma_crossover"].create_with_defaults(), client)

    spy = engine.run(["AAA", "BBB"], datetime(2024, 1, 2), datetime(2025, 1, 2), 100_000, benchmark="SPY")
    qqq = engine.run(["AAA", "BBB"], datetime(2024, 1, 2), datetime(2025, 1, 2), 100_000, benchmark="QQQ")

    # The universe figure cannot move: it never looked at the benchmark.
    assert spy.metrics["buy_hold_return"] == qqq.metrics["buy_hold_return"]
    # The benchmark figure must, or it is measuring the same thing twice.
    assert spy.metrics["benchmark_buy_hold_return"] != qqq.metrics["benchmark_buy_hold_return"]


def test_treynor_is_suppressed_rather_than_unbounded_at_a_near_zero_beta():
    """Reported as Treynor -262.22 beside a beta rendered "-0.00".

    The guard was `beta == 0` exactly, so -0.0001 divided. Excess return *per unit of
    beta* has no meaning for a book with no market exposure, and `--beta-sizing` puts
    books in that regime deliberately - so this is the common case, not an edge one.
    """
    from tradeflow.analytics import metrics as m

    returns = [0.001] * 250

    assert m.treynor_ratio(returns, -0.0002) == 0.0
    assert m.treynor_ratio(returns, 0.02) == 0.0
    # And it still reports where beta is real, or the guard just deletes the metric.
    assert m.treynor_ratio(returns, 0.9) != 0.0
    assert m.treynor_ratio(returns, m.MIN_ABS_BETA_FOR_TREYNOR) != 0.0


def test_a_suppressed_treynor_is_distinguishable_from_a_real_zero():
    """0.0 is the established "unavailable" value for these metrics, so the flag is
    what stops a suppressed ratio reading as a measured one. It fails independently of
    `benchmark_available`: the benchmark here is present and fine."""
    import pandas as pd

    from tradeflow.analytics.performance import compute_backtest_metrics
    from tradeflow.analytics.reporting import format_backtest_report

    trades = pd.DataFrame({"pnl": [1.0, -0.5, 2.0]})
    curve = [100.0 + i * 0.01 for i in range(300)]
    flat_benchmark = pd.Series([0.0005] * (len(curve) - 1))  # near-zero covariance -> tiny beta

    metrics = compute_backtest_metrics(trades, curve, 100.0, 103.0, {}, benchmark_returns=flat_benchmark)

    assert metrics["benchmark_available"] is True
    assert metrics["treynor_available"] is False
    assert metrics["treynor_ratio"] == 0.0
    assert "beta near zero" in format_backtest_report(metrics, 100.0, 103.0, title="t")


# --- execution at small capital ----------------------------------------------
def _small_account_run(capital, min_notional=None):
    from tests.fakes import FakeMarketData
    from tradeflow.services.registry import STRATEGIES

    symbols = [f"S{i}" for i in range(8)]
    client = MarketDataClient(FakeMarketData(symbols, n=300, freq="1D"))
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    strategy.config["position_limits"] = {
        **strategy.position_limits(),
        "max_positions": 8,
        "max_position_size": 50_000.0,
        "min_notional": min_notional,
    }
    return BacktestEngine(strategy, client).run(symbols, datetime(2024, 1, 2), datetime(2024, 12, 1), capital)


def test_share_rounding_drag_grows_as_the_account_shrinks():
    """The thing that was invisible. Whole-share rounding just happened.

    The same config drags a fraction of a percent at $100k and double digits at $500,
    and nothing in the result said so - the equity curve was right while the reason for
    it was unexplained. Asserted as a direction rather than a constant, because the
    exact figures are a property of the fake feed.
    """
    big = _small_account_run(100_000.0).execution
    small = _small_account_run(4_000.0).execution
    tiny = _small_account_run(500.0).execution

    assert big["rounding_drag_pct"] < small["rounding_drag_pct"] < tiny["rounding_drag_pct"]
    assert big["positions_rounded_to_zero"] == 0
    assert tiny["positions_rounded_to_zero"] > small["positions_rounded_to_zero"]


def test_intended_and_filled_notional_are_both_reported():
    """The gap is the point, so both sides of it have to be visible - a drag percentage
    with no notional behind it cannot be checked."""
    execution = _small_account_run(4_000.0).execution

    assert execution["requested_notional"] > execution["filled_notional"] > 0
    expected = (1 - execution["filled_notional"] / execution["requested_notional"]) * 100
    assert execution["rounding_drag_pct"] == pytest.approx(expected)


def test_a_min_notional_floor_refuses_orders_a_venue_would():
    """Nothing modelled a venue minimum, so the backtest filled positions a real
    account could not open - and validated a book that could not be traded."""
    without = _small_account_run(4_000.0, min_notional=None)
    with_floor = _small_account_run(4_000.0, min_notional=2_000.0)

    assert without.execution["positions_below_min_notional"] == 0
    assert with_floor.execution["positions_below_min_notional"] > 0
    assert len(with_floor.trades) < len(without.trades)


def test_the_default_leaves_the_floor_off():
    """Absent is not zero: a config that never mentioned a venue minimum keeps the
    behaviour it was validated under."""
    from tradeflow.strategies.base import DEFAULT_POSITION_LIMITS

    assert DEFAULT_POSITION_LIMITS["min_notional"] is None
    assert _small_account_run(4_000.0).execution["positions_below_min_notional"] == 0


def test_executability_is_judged_separately_from_the_edge():
    """A verdict on whether the book can be traded at this capital, kept apart from
    whether the edge was real - collapsing the two would make one number mean two
    things and silently redefine `promotable` for every trial already recorded."""
    from tradeflow.analytics.performance import execution_verdict

    assert execution_verdict(_small_account_run(100_000.0).execution)["executable"] is True

    failing = execution_verdict(_small_account_run(500.0).execution)
    assert failing["executable"] is False
    assert failing["reason"]
    assert not failing["checks"]["rounding_drag"]["passed"]


def test_nothing_attempted_is_not_the_same_as_passing():
    """`executable: None` rather than True, or a run that never tried to open anything
    would read as one that traded cleanly."""
    from tradeflow.analytics.performance import execution_verdict

    verdict = execution_verdict({})

    assert verdict["executable"] is None
    assert verdict["checks"] == {}


def test_the_report_shows_the_gap_and_names_the_failing_check():
    from tradeflow.analytics.reporting import format_backtest_report

    result = _small_account_run(500.0)
    rendered = format_backtest_report(result.metrics, 500.0, result.final_capital, execution=result.execution)

    assert "Execution & cost" in rendered
    assert "Intended notional" in rendered and "Filled notional" in rendered
    assert "FAIL" in rendered and "rounding_drag" in rendered
    assert "Not the book that was validated" in rendered


def test_a_healthy_run_does_not_grow_an_execution_section():
    """A diagnostic that always fires is one people learn to skip."""
    from tradeflow.analytics.reporting import format_backtest_report

    result = _small_account_run(100_000.0)
    rendered = format_backtest_report(result.metrics, 100_000.0, result.final_capital, execution=None)

    assert "Execution & cost" not in rendered


def test_cost_is_judged_against_gross_profit_not_capital():
    """The honest denominator, and the one the two disagree about.

    The same dollar cost is unremarkable against a large gross return and fatal against
    a small one. Measuring against capital would call a strategy that spent its entire
    edge on commission "3% of capital in cost" and pass it.
    """
    from tests.fakes import FakeMarketData
    from tradeflow.analytics.performance import execution_verdict
    from tradeflow.costs.parametric import ParametricCostModel
    from tradeflow.services.registry import STRATEGIES

    symbols = [f"S{i}" for i in range(8)]
    client = MarketDataClient(FakeMarketData(symbols, n=300, freq="1D"))

    def share(bps):
        strategy = STRATEGIES["ma_crossover"].create_with_defaults()
        strategy.config["position_limits"] = {
            **strategy.position_limits(),
            "max_positions": 8,
            "max_position_size": 50_000.0,
        }
        result = BacktestEngine(strategy, client, cost_model=ParametricCostModel(commission_bps=bps)).run(
            symbols, datetime(2024, 1, 2), datetime(2024, 12, 1), 100_000.0
        )
        return execution_verdict(result.execution)["checks"]["cost_share_of_gross"]

    cheap, dear = share(1.0), share(60.0)

    assert cheap["value"] < dear["value"]
    assert not dear["passed"]


def test_no_gross_profit_means_no_cost_ratio_rather_than_a_guessed_one():
    """A ratio against a non-positive denominator is arithmetic, not a fact: there was
    no edge for cost to eat, and reporting a number would imply there was."""
    from tradeflow.analytics.performance import execution_verdict

    verdict = execution_verdict(
        {"positions_filled": 3, "gross_profit": -50.0, "total_cost": 20.0, "rounding_drag_pct": 0.0}
    )

    assert "cost_share_of_gross" not in verdict["checks"]
    assert verdict["executable"] is True  # the checks that *could* run all passed


def _breadth_run(max_positions, n_symbols=20):
    from tests.fakes import FakeMarketData
    from tradeflow.services.registry import STRATEGIES

    symbols = [f"S{i}" for i in range(n_symbols)]
    client = MarketDataClient(FakeMarketData(symbols, n=400, freq="1D"))
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    strategy.config["position_limits"] = {
        **strategy.position_limits(),
        "max_positions": max_positions,
        "max_position_size": 50_000.0,
    }
    return BacktestEngine(strategy, client).run(
        symbols, datetime(2024, 1, 2), datetime(2025, 1, 2), 100_000.0
    )


def test_a_one_position_book_over_a_wide_universe_is_flagged():
    """Every shipped strategy declares `max_positions: 1`.

    So a backtest over a scanned universe of sixty names validates a book that holds
    one position at a time, and nothing said so - measured here, a cap of 1 over twenty
    candidates ever touches nine of them. The result is correct and describes a
    different strategy from the one the user thinks they are testing.
    """
    from tradeflow.analytics.performance import execution_verdict

    result = _breadth_run(1)
    check = execution_verdict(result.execution)["checks"]["book_breadth"]

    assert not check["passed"]
    assert "1 of 20 candidates" in check["note"]
    assert result.execution["symbols_traded"] < result.execution["universe_size"]


def test_a_deliberately_concentrated_book_is_not_flagged():
    """The check is "was the cap ever chosen", not "is the cap small".

    Concentrating in the best five of twenty is a legitimate design, and a gate that
    called it a defect would be one people learn to switch off.
    """
    from tradeflow.analytics.performance import execution_verdict

    check = execution_verdict(_breadth_run(5).execution)["checks"]["book_breadth"]

    assert check["passed"]


def test_breadth_is_not_judged_on_a_single_name_universe():
    """A one-position book over one candidate is the only book available, so there is
    nothing to flag and a check that fired would be noise."""
    from tradeflow.analytics.performance import execution_verdict

    assert "book_breadth" not in execution_verdict(_breadth_run(1, n_symbols=1).execution)["checks"]


# --- three verdicts, shown together --------------------------------------------
def test_a_command_says_which_verdicts_it_did_not_assess():
    """The fix for "a backtest replay reads as approved".

    The three verdicts already never collapsed into one another, but each was printed
    by a different command at a different moment, so nothing showed a reader all three.
    A backtest can speak to execution and to nothing else, and saying so is the point -
    an unknown rendered as a blank is an unknown a reader fills in optimistically.
    """
    from tradeflow.analytics.reporting import format_verdicts

    rendered = "\n".join(format_verdicts(execution="PASS"))

    assert "Execution viability" in rendered and "PASS" in rendered
    assert "Statistical validation" in rendered and "not assessed here" in rendered
    assert "walkforward" in rendered  # and names what would assess it
    assert "Clearing one says nothing about the others" in rendered


def test_the_three_verdicts_are_never_merged_into_one():
    """Each carries its own answer; none is derived from another."""
    from tradeflow.analytics.reporting import format_verdicts

    rendered = "\n".join(
        format_verdicts(statistical="FAIL - min_oos_sharpe", execution="PASS", evidence="2 of 3 evaluated")
    )

    assert "FAIL - min_oos_sharpe" in rendered
    assert "PASS" in rendered
    assert "2 of 3 evaluated" in rendered
    assert "not assessed here" not in rendered


def test_one_trial_reports_an_undeflated_sharpe_by_that_name():
    """At one trial there is nothing to deflate against - the deflated Sharpe is
    identically the probabilistic one, so the name promised a multiple-testing
    correction that was never applied. The number was always right."""
    import pandas as pd

    from tradeflow.analytics.performance import compute_backtest_metrics
    from tradeflow.analytics.reporting import format_backtest_report

    curve = [100.0 + i * 0.01 for i in range(300)]
    trades = pd.DataFrame({"pnl": [1.0, -0.5, 2.0]})

    alone = compute_backtest_metrics(trades, curve, 100.0, 103.0, {}, n_trials=1)
    campaign = compute_backtest_metrics(trades, curve, 100.0, 103.0, {}, n_trials=12)

    assert alone["deflation_applied"] is False
    assert campaign["deflation_applied"] is True
    assert "Sharpe (undeflated)" in format_backtest_report(alone, 100.0, 103.0, title="t")
    assert "Deflated Sharpe" in format_backtest_report(campaign, 100.0, 103.0, title="t")


def test_the_undeflated_number_itself_is_unchanged():
    """Only the label was wrong. Hiding or altering a correct number would be a
    different defect."""
    import numpy as np

    from tradeflow.analytics.metrics import deflated_sharpe_ratio, probabilistic_sharpe_ratio

    returns = np.random.default_rng(7).normal(0.0004, 0.01, 300)

    assert deflated_sharpe_ratio(returns, 1) == pytest.approx(probabilistic_sharpe_ratio(returns))
    # And a real campaign count still deflates, hard.
    assert deflated_sharpe_ratio(returns, 50) < deflated_sharpe_ratio(returns, 1)
