"""Unit tests for the small, pure building blocks."""

import numpy as np
import pandas as pd
import pytest

from src.analytics import metrics
from src.marketdata.timeframe import DAY, HOUR, MINUTE, Timeframe
from src.optimization.param_space import ParameterSpace
from src.utils import numeric


# --- timeframe --------------------------------------------------------------
@pytest.mark.parametrize(
    "text,amount,unit",
    [("5Min", 5, MINUTE), ("1Day", 1, DAY), ("2h", 2, HOUR), ("15min", 15, MINUTE)],
)
def test_timeframe_parse(text, amount, unit):
    tf = Timeframe.parse(text)
    assert tf.amount == amount and tf.unit == unit


def test_timeframe_parse_rejects_garbage():
    for bad in ("", "Min", "0Day", "5fortnights"):
        with pytest.raises(ValueError):
            Timeframe.parse(bad)


def test_timeframe_pandas_offset():
    assert Timeframe.parse("5Min").to_pandas_offset() == "5min"
    assert Timeframe.parse("1Day").to_pandas_offset() == "1D"


# --- numeric ----------------------------------------------------------------
def test_round_price_sub_dollar_gets_more_precision():
    assert numeric.round_price(123.456) == 123.46
    assert numeric.round_price(0.123456) == 0.1235


def test_round_quantity():
    assert numeric.round_quantity(3.9) == 3
    assert numeric.round_quantity(3.9, allow_fractional=True) == 3.9


def test_step_decimals():
    assert numeric.step_decimals(1) == 0
    assert numeric.step_decimals(0.05) == 2


# --- metrics ----------------------------------------------------------------
def test_max_drawdown_known():
    assert metrics.max_drawdown([100, 120, 90, 130]) == pytest.approx(0.25)


def test_profit_factor_and_win_rate():
    pnl = pd.Series([10, -5, 20, -5])
    assert metrics.profit_factor(pnl) == pytest.approx(3.0)
    assert metrics.win_rate(pnl) == pytest.approx(0.5)


def test_sharpe_zero_for_constant_returns():
    assert metrics.sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


# --- beta -------------------------------------------------------------------
def test_beta_of_amplified_benchmark():
    from src.indicators.indicators import calculate_beta

    rng = np.random.default_rng(0)
    benchmark_ret = rng.normal(0, 0.01, 300)
    benchmark = pd.Series(100 * np.cumprod(1 + benchmark_ret))
    symbol = pd.Series(100 * np.cumprod(1 + 2 * benchmark_ret))  # moves ~2x the benchmark
    assert calculate_beta(symbol, benchmark) == pytest.approx(2.0, abs=0.05)


def test_beta_neutral_on_degenerate_input():
    from src.indicators.indicators import calculate_beta

    flat = pd.Series([100.0] * 10)
    assert calculate_beta(flat, flat) == 1.0          # flat benchmark -> neutral
    assert calculate_beta(pd.Series([100.0]), pd.Series([100.0])) == 1.0  # too few points


# --- parameter space --------------------------------------------------------
def _space():
    return ParameterSpace({
        "a": {"type": "int", "min": 1, "max": 3, "step": 1, "default": 1},
        "b": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.5, "default": 0.0},
        "fixed": {"type": "float", "min": 0, "max": 1, "default": 0.5},  # no step -> not searched
    })


def test_grid_size_and_searchable():
    space = _space()
    assert space.searchable == ["a", "b"]
    assert space.dimensions == 2
    grid = space.grid()
    assert len(grid) == 3 * 3  # a in {1,2,3} x b in {0,0.5,1}
    assert all(point["fixed"] == 0.5 for point in grid)  # fixed param carried through


def test_unit_vector_round_trip():
    space = _space()
    params = {"a": 2, "b": 0.5, "fixed": 0.5}
    recovered = space.from_unit_vector(space.to_unit_vector(params))
    assert recovered["a"] == 2 and recovered["b"] == pytest.approx(0.5)


def test_random_samples_are_on_grid():
    space = _space()
    samples = space.random_samples(20, np.random.default_rng(0))
    for sample in samples:
        assert sample["a"] in (1, 2, 3)
        assert sample["b"] in (0.0, 0.5, 1.0)
