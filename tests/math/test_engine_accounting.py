"""Math contract: what the backtest engine books.

Scripted signals on hand-built bars, so every fill price, P&L, cost and benchmark pairing
is computed here by arithmetic and compared with what the engine reports.

Related existing tests: ``tests/test_backtest.py`` (fill logic), ``tests/test_signal_causality.py``,
``tests/test_benchmark_portfolio.py``, ``tests/test_cost_spread.py``.

Defects this module guards:

* a stop (or target) gapped through filled at the level instead of at the open;
* the benchmark return series was labelled one step early, pairing every strategy
  return with the next bar's benchmark return (a book holding the benchmark itself
  reported a beta near zero);
* costs priced with the liquidity of the fill's own bar;
* costs charged on one leg only, so the trade ledger and the book disagreed.
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantred.costs.parametric import ParametricCostModel
from quantred.engine.backtest import BacktestEngine
from quantred.marketdata.client import MarketDataClient
from quantred.strategies import signals
from quantred.utils.timeutils import NEW_YORK
from tests.fakes import DictMarketData
from tests.test_backtest import ScriptedStrategy

START, END = datetime(2024, 1, 2), datetime(2025, 12, 31)
B, H, CB, CS = signals.BUY, signals.HOLD, signals.CLOSE_BUY, signals.CLOSE_SELL


def _bars(rows, start="2024-01-02"):
    index = pd.date_range(start, periods=len(rows), freq="D", tz=NEW_YORK)
    cols = ["open", "high", "low", "close", "volume"]
    return pd.DataFrame([dict(zip(cols, r if len(r) == 5 else (*r, 1))) for r in rows], index=index)


def _run(frames, sigs, overrides=None, capital=100_000.0, cost_model=None, benchmark=None):
    strategy = ScriptedStrategy(sigs, overrides or {})
    client = MarketDataClient(DictMarketData(frames))
    symbols = [s for s in frames if s != benchmark]
    return BacktestEngine(strategy, client, cost_model=cost_model).run(
        symbols, START, END, capital, benchmark=benchmark
    )


# 10 shares at $100: stop/target 10% either side (see ScriptedStrategy's config)
ENTRY_BAR = (100, 100, 100, 100)


# --------------------------------------------------------------------------- #
# Fill prices: stops and targets are resting orders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bar, price, reason",
    [
        ((95, 96, 85, 90), 90.0, "STOP_LOSS"),  # trades through the level inside the bar: fills at 90
        ((80, 82, 78, 79), 80.0, "STOP_LOSS"),  # opens beyond the stop: fills at the open, not 90
        ((90, 92, 89, 91), 90.0, "STOP_LOSS"),  # opens exactly on the level
        ((108, 112, 107, 111), 110.0, "TAKE_PROFIT"),  # target reached inside the bar
        ((115, 118, 114, 116), 115.0, "TAKE_PROFIT"),  # target gapped through: the better open
    ],
)
def test_long_exit_prices(bar, price, reason):
    """A long entered at 100 with stop 90 and target 110. Guards stops filling at the stop
    level through a gap: a long stopped at 90 on a bar opening at 80 lost 20, not 10."""
    result = _run({"AAA": _bars([ENTRY_BAR, ENTRY_BAR, bar, ENTRY_BAR])}, [B, H, H, H])
    trade = result.trades.iloc[0]
    assert (trade["entry_price"], trade["size"]) == (100.0, 10)
    assert trade["exit_reason"] == reason and trade["exit_price"] == pytest.approx(price)
    assert trade["pnl"] == pytest.approx((price - 100.0) * 10)


@pytest.mark.parametrize(
    "bar, price, reason",
    [
        ((105, 115, 104, 112), 110.0, "STOP_LOSS"),  # short stop is 110
        ((125, 126, 122, 124), 125.0, "STOP_LOSS"),  # gapped up through the stop: the worse open
        ((95, 96, 88, 89), 90.0, "TAKE_PROFIT"),  # target 90 reached inside the bar
        ((85, 86, 83, 84), 85.0, "TAKE_PROFIT"),  # gapped down through the target: the better open
    ],
)
def test_short_exit_prices(bar, price, reason):
    result = _run({"AAA": _bars([ENTRY_BAR, ENTRY_BAR, bar, ENTRY_BAR])}, [signals.SELL, H, H, H])
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == reason and trade["exit_price"] == pytest.approx(price)
    assert trade["pnl"] == pytest.approx((100.0 - price) * 10)


def test_the_entry_bar_is_checked_against_its_own_stop():
    """Signal decided on bar 0, filled at bar 1's open; bar 1's low is below the stop, so
    the position is stopped that same bar (it used to carry on as if it had not)."""
    result = _run({"AAA": _bars([ENTRY_BAR, (100, 101, 85, 99), ENTRY_BAR, ENTRY_BAR])}, [B, H, H, H])
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "STOP_LOSS" and trade["exit_price"] == pytest.approx(90.0)
    assert trade["entry_time"] == trade["exit_time"]


def test_a_signal_acts_on_the_next_bar_open_not_the_bar_that_produced_it():
    frame = _bars([(100, 100, 100, 100), (102, 102, 102, 102), (104, 104, 104, 104), (106, 106, 106, 106)])
    trade = _run({"AAA": frame}, [B, H, CB, H]).trades.iloc[0]
    assert trade["entry_price"] == 102.0  # signalled on bar 0, filled at bar 1's open
    assert trade["exit_price"] == 106.0  # exit signalled on bar 2, filled at bar 3's open
    assert trade["exit_reason"] == "SIGNAL"


# --------------------------------------------------------------------------- #
# Conservation: Sum of trade P&L is the change in the book, net of cost
# --------------------------------------------------------------------------- #
def _random_world(seed, n_bars=120, symbols=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(seed)
    frames = {}
    for sym in symbols:
        close = 100 * np.cumprod(1 + rng.normal(0, 0.012, n_bars))
        open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.004, n_bars))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n_bars))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n_bars))
        vol = rng.uniform(5e5, 2e6, n_bars)
        frames[sym] = _bars(list(zip(open_, high, low, close, vol)))
    return frames


class _PerSymbolStrategy(ScriptedStrategy):
    """Different scripted signals per symbol: the engine asks generate_signals once per name."""

    def __init__(self, scripts, overrides=None):
        super().__init__([], overrides)
        self._scripts = scripts
        self._calls = 0
        self._order = sorted(scripts)

    def generate_signals(self, data):
        symbol = self._order[self._calls % len(self._order)]
        self._calls += 1
        return dict(zip(data.index, self._scripts[symbol]))


def _random_signals(seed, n_bars, symbols):
    rng = np.random.default_rng(seed)
    pool = [B, H, H, H, CB, signals.SELL, CS]
    return {s: [pool[i] for i in rng.integers(0, len(pool), n_bars)] for s in symbols}


def _multi_run(seed, cost_model):
    frames = _random_world(seed)
    scripts = _random_signals(seed + 1, 120, frames)
    strategy = _PerSymbolStrategy(
        scripts,
        {"position_limits": {"max_positions": 3, "max_position_size": 20_000.0, "max_total_risk": 1.0}},
    )
    client = MarketDataClient(DictMarketData(frames))
    return BacktestEngine(strategy, client, cost_model=cost_model).run(list(frames), START, END, 100_000.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_trade_pnl_is_the_book_change_and_gross_less_cost(seed):
    result = _multi_run(seed, ParametricCostModel(annual_borrow_bps=200.0))
    t = result.trades
    assert len(t) > 5 and (t["side"] == signals.SELL).any()
    direction = np.where(t["side"] == B, 1.0, -1.0)
    gross = (t["exit_price"] - t["entry_price"]) * t["size"] * direction
    assert t["gross_pnl"].to_numpy() == pytest.approx(gross.to_numpy(), rel=1e-12)
    assert t["pnl"].to_numpy() == pytest.approx((t["gross_pnl"] - t["cost"]).to_numpy(), abs=1e-9)
    # The book ends exactly where the ledger says, and the marked curve agrees with it.
    assert result.final_capital == pytest.approx(100_000.0 + t["pnl"].sum(), abs=1e-6)
    assert result.equity_curve[-1] == pytest.approx(result.final_capital, abs=1e-6)
    assert result.total_cost == pytest.approx(t["cost"].sum(), rel=1e-12)
    assert result.gross_final_capital - result.final_capital == pytest.approx(result.total_cost, abs=1e-9)
    assert result.total_cost > 0


def test_cost_free_run_has_zero_cost_and_pnl_equals_gross():
    result = _multi_run(4, None)
    assert result.total_cost == 0.0
    assert (result.trades["pnl"] == result.trades["gross_pnl"]).all()


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_round_trip_is_charged_on_both_legs_with_the_models_own_spread():
    """Flat bars with no ADV history: each leg costs notional * (commission + spread/2)."""
    model = ParametricCostModel(commission_bps=3.0, default_spread_bps=12.0)
    flat = _bars([(100, 100, 100, 100, 1)] * 5)
    trade = _run({"AAA": flat}, [B, H, CB, H, H], cost_model=model).trades.iloc[0]
    per_leg = 1_000.0 * (3e-4 + 12e-4 / 2)
    assert trade["cost"] == pytest.approx(2 * per_leg, rel=1e-12)
    assert trade["pnl"] == pytest.approx(-2 * per_leg, rel=1e-12)


def test_costs_scale_exactly_with_the_linear_terms_when_there_is_no_impact():
    """Cost stress k x: commission and spread both scaled by k, impact off. The total must
    be exactly k times the base (the spread used to stay put, so 3x cost ~1.7x)."""
    flat = _bars([(100, 100, 100, 100, 1)] * 5)
    sig = [B, H, CB, H, H]
    base = _run({"AAA": flat}, sig, cost_model=ParametricCostModel(1.0, 5.0, 0.0)).total_cost
    for k in (2.0, 3.0, 5.0):
        stressed = _run({"AAA": flat}, sig, cost_model=ParametricCostModel(1.0 * k, 5.0 * k, 0.0)).total_cost
        assert stressed == pytest.approx(k * base, rel=1e-12)
    assert base == pytest.approx(2 * 1_000.0 * (1e-4 + 5e-4 / 2), rel=1e-12)


def test_impact_is_priced_with_the_liquidity_known_before_the_fills_bar():
    """Entry at bar 25: ADV and volatility are the trailing 20 bars ending bar 24. A volume
    spike on the fill bar itself must not change what the fill cost."""
    rng = np.random.default_rng(11)
    n = 40
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    open_ = np.concatenate([[100.0], close[:-1]])
    vol = rng.uniform(8e3, 1.2e4, n)
    rows = list(zip(open_, np.maximum(open_, close) + 0.1, np.minimum(open_, close) - 0.1, close, vol))
    sigs = [H] * n
    sigs[24], sigs[34] = B, CB
    model = ParametricCostModel(
        commission_bps=0.0, default_spread_bps=0.0, impact_eta=0.3, annual_borrow_bps=0.0
    )
    overrides = {
        "stop_loss": 0.9,
        "take_profit": 9.0,
        "position_limits": {"max_positions": 1, "max_position_size": 5_000.0, "max_total_risk": 1.0},
    }
    trade = _run({"AAA": _bars(rows)}, sigs, overrides, cost_model=model).trades.iloc[0]
    shares = trade["size"]

    def expected_cost(volume, shares):
        def leg(bar, price):
            adv = np.mean(volume[bar - 20 : bar])  # bars bar-20 .. bar-1
            rets = close[bar - 20 : bar] / close[bar - 21 : bar - 1] - 1  # pct_change over the same window
            return 0.3 * rets.std(ddof=1) * math.sqrt(shares / adv) * shares * price

        return leg(25, open_[25]) + leg(35, open_[35])

    index = _bars(rows).index
    assert trade["entry_time"] == index[25] and trade["exit_time"] == index[35]
    assert trade["cost"] == pytest.approx(expected_cost(vol, shares), rel=1e-9)

    # A spike on the exit's fill bar cannot change the exit's price of liquidity ...
    spiked = [list(r) for r in rows]
    spiked[35][4] *= 1000.0
    again = _run({"AAA": _bars([tuple(r) for r in spiked])}, sigs, overrides, cost_model=model).trades.iloc[0]
    assert again["cost"] == pytest.approx(trade["cost"], rel=1e-12)
    # ... while a spike on the entry bar is history by the time the exit is priced, and
    # legitimately moves the exit leg (and only that one).
    vol2 = vol.copy()
    vol2[25] *= 1000.0
    spiked = [list(r) for r in rows]
    spiked[25][4] = vol2[25]
    moved = _run({"AAA": _bars([tuple(r) for r in spiked])}, sigs, overrides, cost_model=model).trades.iloc[0]
    assert moved["cost"] == pytest.approx(expected_cost(vol2, shares), rel=1e-9)


def test_short_borrow_accrues_pro_rata_over_the_steps_held():
    model = ParametricCostModel(commission_bps=0.0, default_spread_bps=0.0, annual_borrow_bps=252.0)
    flat = _bars([(100, 100, 100, 100, 1)] * 12)
    trades = {}
    for held, exit_signal_bar in ((2, 4), (6, 8)):
        sigs = [H] * 12
        sigs[1], sigs[exit_signal_bar] = signals.SELL, CS
        trades[held] = _run({"AAA": flat}, sigs, cost_model=model).trades.iloc[0]
    # entry fills at step 2; exit fills at step 5 or 9. Borrow = 2.52%/yr * $1,000 * steps/252
    assert trades[2]["cost"] == pytest.approx(0.0252 * 1_000 * 3 / 252, rel=1e-12)
    assert trades[6]["cost"] == pytest.approx(0.0252 * 1_000 * 7 / 252, rel=1e-12)
    assert trades[6]["cost"] / trades[2]["cost"] == pytest.approx(7 / 3, rel=1e-12)
    assert (
        _run({"AAA": flat}, [H, B, H, H, CB] + [H] * 7, cost_model=model).total_cost == 0.0
    )  # longs pay none


# --------------------------------------------------------------------------- #
# Benchmark pairing
# --------------------------------------------------------------------------- #
def _hold_world(n=40, seed=5):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.01, n)
    steps[0] = 0.0  # the first close is exactly 100, so 10 shares cost the whole $1,000
    close = 100 * np.cumprod(1 + steps)
    open_ = np.concatenate([[100.0], close[:-1]])  # continuous: each open is the prior close
    rows = [(o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1) for o, c in zip(open_, close)]
    return _bars(rows)


def test_a_book_holding_the_benchmark_has_beta_one_alpha_zero_and_r_squared_one():
    """Guards the off-by-one labelling: paired with the next bar's benchmark return, the same
    book read a beta near zero."""
    frame = _hold_world()
    overrides = {
        "stop_loss": 0.5,
        "take_profit": 50.0,
        "position_limits": {"max_positions": 1, "max_position_size": 1000.0, "max_total_risk": 1.0},
    }
    sigs = [B] + [H] * (len(frame) - 1)
    result = _run({"AAA": frame, "BMK": frame.copy()}, sigs, overrides, capital=1000.0, benchmark="BMK")
    assert result.trades.iloc[0]["size"] == 10  # the whole capital: equity tracks the price exactly
    m = result.metrics
    assert m["benchmark_available"] is True
    assert m["beta"] == pytest.approx(1.0, abs=1e-9)
    assert m["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert m["r_squared"] == pytest.approx(1.0, abs=1e-9)
    # the pairing itself: strategy return at step j is the benchmark's return over the same step
    eq = np.array(result.equity_curve)
    strat = eq[2:] / eq[1:-1] - 1
    bench = result.benchmark_returns.dropna()
    assert bench.index[0] == 2 and len(bench) == len(strat)
    assert bench.to_numpy() == pytest.approx(strat, abs=1e-12)


def test_zero_tracking_error_information_ratio_is_unavailable():
    frame = _hold_world()
    overrides = {
        "stop_loss": 0.5,
        "take_profit": 50.0,
        "position_limits": {"max_positions": 1, "max_position_size": 1000.0, "max_total_risk": 1.0},
    }
    result = _run(
        {"AAA": frame, "BMK": frame.copy()}, [B] + [H] * 39, overrides, capital=1000.0, benchmark="BMK"
    )
    assert result.metrics["information_ratio"] == 0.0


def test_benchmark_buy_hold_is_compounded_over_the_measured_steps():
    frame = _hold_world(30, 6)
    result = _run(
        {"AAA": frame, "BMK": frame.copy()},
        [B] + [H] * 29,
        {"stop_loss": 0.5, "take_profit": 50.0},
        benchmark="BMK",
    )
    closes = frame["close"].to_numpy()
    # steps covered: first mark after bar 0 through bar 29 -> close[29]/close[0] - 1
    assert result.metrics["benchmark_buy_hold_return"] == pytest.approx(
        (closes[-1] / closes[0] - 1) * 100, rel=1e-9
    )


def test_metrics_are_absent_without_a_benchmark_not_zero_alpha():
    result = _run({"AAA": _hold_world()}, [B] + [H] * 39, {"stop_loss": 0.5, "take_profit": 50.0})
    assert result.metrics["benchmark_available"] is False and result.metrics["treynor_available"] is False
