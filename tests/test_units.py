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


# --- new metric primitives ---------------------------------------
def test_cagr_known():
    # Doubling over exactly one year -> 100% CAGR; over two years -> ~41.4%.
    assert metrics.cagr([100, 200], years=1.0) == pytest.approx(1.0)
    assert metrics.cagr([100, 200], years=2.0) == pytest.approx(2**0.5 - 1)


def test_cagr_degenerate():
    assert metrics.cagr([100], years=1.0) == 0.0
    assert metrics.cagr([100, 200], years=0.0) == 0.0
    assert metrics.cagr([-1, 200], years=1.0) == 0.0  # non-positive start


def test_max_drawdown_duration_known():
    # peak at idx0(100), underwater for 3 points, recovers at the last -> 3.
    assert metrics.max_drawdown_duration([100, 95, 90, 99, 101]) == 3


def test_ulcer_index_zero_for_monotonic_curve():
    assert metrics.ulcer_index([100, 101, 102, 103]) == 0.0
    assert metrics.ulcer_index([100, 90, 100]) > 0.0


def test_value_at_risk_and_cvar():
    returns = [-0.10, -0.05, 0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    var = metrics.value_at_risk(returns, level=0.90)
    cvar = metrics.conditional_var(returns, level=0.90)
    assert var > 0  # reported as a positive loss
    assert cvar >= var  # tail loss is at least as deep as the VaR threshold


def test_consecutive_streaks():
    pnl = [1, 2, -1, -1, -1, 3, -1, 4, 5]
    assert metrics.consecutive(pnl, winning=True) == 2
    assert metrics.consecutive(pnl, winning=False) == 3


def test_probabilistic_sharpe_in_unit_interval():
    rng = np.random.default_rng(1)
    good = pd.Series(rng.normal(0.01, 0.01, 250))  # clearly positive Sharpe
    flat = pd.Series(rng.normal(0.0, 0.01, 250))  # no edge
    psr_good = metrics.probabilistic_sharpe_ratio(good)
    psr_flat = metrics.probabilistic_sharpe_ratio(flat)
    assert 0.0 <= psr_flat <= psr_good <= 1.0
    assert psr_good > 0.9  # strong, long sample -> high confidence


def test_deflated_sharpe_drops_with_more_trials():
    rng = np.random.default_rng(2)
    # Modest edge (per-period Sharpe ~0.1) so deflation is visible rather than saturated.
    returns = pd.Series(rng.normal(0.001, 0.01, 250))
    dsr_one = metrics.deflated_sharpe_ratio(returns, n_trials=1)
    dsr_many = metrics.deflated_sharpe_ratio(returns, n_trials=1000)
    assert 0.0 <= dsr_many < dsr_one <= 1.0  # more trials -> harder to clear


def test_norm_cdf_ppf_round_trip():
    for p in (0.05, 0.5, 0.975):
        assert metrics.norm_cdf(metrics.norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_alpha_beta_of_amplified_benchmark():
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0, 0.01, 300))
    strat = 2 * bench + 0.001  # beta 2, small constant alpha, perfect fit
    alpha, beta, r2 = metrics.alpha_beta(strat, bench)
    assert beta == pytest.approx(2.0, abs=0.01)
    assert alpha == pytest.approx(0.001, abs=1e-4)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_information_ratio_zero_when_tracking_benchmark():
    bench = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01])
    assert metrics.information_ratio(bench, bench) == 0.0  # no active return


def test_new_primitives_handle_degenerate_input():
    assert metrics.cagr([], years=1.0) == 0.0
    assert metrics.ulcer_index([]) == 0.0
    assert metrics.max_drawdown_duration([]) == 0
    assert metrics.value_at_risk([]) == 0.0
    assert metrics.probabilistic_sharpe_ratio([0.01]) == 0.0
    assert metrics.deflated_sharpe_ratio([0.01], n_trials=5) == 0.0
    assert metrics.alpha_beta([0.01], [0.01]) == (0.0, 0.0, 0.0)
    assert metrics.kelly_criterion([]) == 0.0


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
    assert calculate_beta(flat, flat) == 1.0  # flat benchmark -> neutral
    assert calculate_beta(pd.Series([100.0]), pd.Series([100.0])) == 1.0  # too few points


# --- parameter space --------------------------------------------------------
def _space():
    return ParameterSpace(
        {
            "a": {"type": "int", "min": 1, "max": 3, "step": 1, "default": 1},
            "b": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.5, "default": 0.0},
            "fixed": {"type": "float", "min": 0, "max": 1, "default": 0.5},  # no step -> not searched
        }
    )


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
