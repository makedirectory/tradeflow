"""Math contract: returns, growth, risk-adjusted ratios and drawdown shape.

Every expectation here is computed in the test from first principles (plain loops,
hand-derived literals) and never by calling another function of the package, so a fix
that bends a formula cannot also bend its own check.

Related tests that already pin one mechanic each and are left where they are:
``tests/test_downside_deviation.py`` (semideviation, Sortino),
``tests/test_aggregate_annualisation.py`` (the walk-forward clock),
``tests/test_units.py``.

Defects this module guards:

* the downside deviation was the std of only the losing returns about their own mean;
* Calmar and Sterling took the magnitude of the whole ratio, so a losing run read positive;
* the dated equity curve had a row per calendar day, shrinking the per-period std by
  about sqrt(5/7) and understating an annualised Sharpe by the same factor;
* a Treynor ratio was reported for a beta indistinguishable from zero.
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantred.analytics import metrics as m
from quantred.analytics import performance
from quantred.marketdata.timeframe import Timeframe

SQRT_252 = math.sqrt(252)


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


# --------------------------------------------------------------------------- #
# Known answers
# --------------------------------------------------------------------------- #
# r = [+2%, -1%, +2%, -1%]: mean 0.005; every deviation is +/-0.015 so the sample
# variance is 4 * 2.25e-4 / 3 = 3e-4 and std = 0.01732...
R4 = [0.02, -0.01, 0.02, -0.01]


def test_sharpe_known_answer():
    # per-period 0.005 / sqrt(3e-4) = 1/(2*sqrt(3));  annual: sqrt(252)/(2*sqrt(3)) = sqrt(84)/2
    assert m.sharpe_ratio(R4) == pytest.approx(math.sqrt(84) / 2, rel=1e-12)
    assert m.sharpe_ratio(R4) == pytest.approx(4.58257569495584, rel=1e-12)


def test_sharpe_scales_with_sqrt_of_periods_per_year():
    assert m.sharpe_ratio(R4, 52) / m.sharpe_ratio(R4, 252) == pytest.approx(math.sqrt(52 / 252), rel=1e-12)


def test_annualised_volatility_known_answer():
    # sqrt(3e-4) * sqrt(252) = sqrt(0.0756)
    assert m.annualized_volatility(R4) == pytest.approx(math.sqrt(0.0756), rel=1e-12)


def test_downside_deviation_and_sortino_known_answer():
    """Only the two -1% periods fall short of zero: mean(min(r,0)^2) = 2e-4/4 = 5e-5, so
    the semideviation is sqrt(5e-5). Guards the old std-of-losers form, which would give
    0.0 here (two identical losses have no dispersion)."""
    semi = math.sqrt(5e-5)
    assert m.downside_deviation(R4) == pytest.approx(semi * SQRT_252, rel=1e-12)
    # Sortino = sqrt(252) * 0.005 / sqrt(5e-5) = sqrt(126)
    assert m.sortino_ratio(R4) == pytest.approx(math.sqrt(126), rel=1e-12)
    assert m.sortino_ratio(R4) == pytest.approx(11.224972160321824, rel=1e-12)


def test_sortino_equals_loop_reference_on_random_data():
    r = np.random.default_rng(1).normal(0.0003, 0.012, 500).tolist()
    shortfall_sq = [min(x, 0.0) ** 2 for x in r]
    semi = math.sqrt(sum(shortfall_sq) / len(r))
    assert m.sortino_ratio(r) == pytest.approx(SQRT_252 * _mean(r) / semi, rel=1e-12)


def test_cagr_known_answer_on_the_curve_alone():
    # 100 -> 121 in two years is exactly 10% a year
    assert m.cagr([100.0, 110.0, 121.0], 2.0) == pytest.approx(0.10, rel=1e-12)


def test_cagr_inside_the_metrics_uses_365_25_day_years():
    """Elapsed time is calendar days over 365.25, not 365 and not trading days."""
    trades = pd.DataFrame({"pnl": [21.0]})
    start, end = datetime(2020, 1, 1), datetime(2022, 1, 1)  # 731 days
    out = performance.compute_backtest_metrics(trades, [100.0, 121.0], 100.0, 121.0, {}, start=start, end=end)
    years = 731 / 365.25
    assert out["cagr"] == pytest.approx((1.21 ** (1 / years) - 1) * 100, rel=1e-12)
    assert out["total_return"] == pytest.approx(21.0, rel=1e-12)


def test_cagr_without_a_window_falls_back_to_the_curve_length_in_trading_days():
    trades = pd.DataFrame({"pnl": [21.0]})
    curve = [100.0] * 253 + [121.0]  # 253 steps
    out = performance.compute_backtest_metrics(trades, curve, 100.0, 121.0, {})
    assert out["cagr"] == pytest.approx((1.21 ** (252 / 253) - 1) * 100, rel=1e-12)


def test_calmar_and_sterling_keep_the_sign_of_the_return():
    """A losing run's ratio is negative; taking abs() of the whole ratio reported it positive."""
    assert m.calmar_ratio(0.10, 0.20) == pytest.approx(0.5)
    assert m.calmar_ratio(0.10, -0.20) == pytest.approx(0.5)  # drawdown sign is the caller's choice
    assert m.calmar_ratio(-0.10, 0.20) == pytest.approx(-0.5)
    assert m.sterling_ratio(-0.10, 0.05) == pytest.approx(-2.0)
    assert m.sterling_ratio(0.10, 0.05) == pytest.approx(2.0)
    assert math.isinf(m.calmar_ratio(0.1, 0.0)) and math.isinf(m.sterling_ratio(0.1, 0.0))


def test_treynor_known_answer_and_suppression_near_zero_beta():
    r = [0.001, 0.002, 0.003]
    assert m.treynor_ratio(r, 2.0) == pytest.approx(0.002 * 252 / 2.0, rel=1e-12)
    # Below the floor there is no exposure worth dividing by: the documented unavailable value
    assert m.treynor_ratio(r, 0.04) == 0.0
    assert m.treynor_ratio(r, -0.0001) == 0.0
    assert m.treynor_ratio(r, 0.05) == pytest.approx(0.002 * 252 / 0.05, rel=1e-12)  # boundary admitted


# Equity 100,110,99,105,90,95,120: running peak 100,110,110,110,110,110,120
EQUITY = [100.0, 110.0, 99.0, 105.0, 90.0, 95.0, 120.0]


def test_drawdown_series_known_answer():
    peak = 100.0
    expected = []
    for x in EQUITY:
        peak = max(peak, x)
        expected.append((x - peak) / peak)
    got = m.drawdown_series(EQUITY).tolist()
    assert got == pytest.approx(expected, abs=1e-15)
    # trough 90 against peak 110
    assert min(got) == pytest.approx(-20 / 110, rel=1e-12)


def test_max_drawdown_and_duration_known_answer():
    assert m.max_drawdown(EQUITY) == pytest.approx(2 / 11, rel=1e-12)
    # below the 110 peak at indices 2..5 inclusive: four consecutive periods underwater
    assert m.max_drawdown_duration(EQUITY) == 4


def test_ulcer_index_known_answer():
    dd_pct = [0, 0, -10.0, -100 * 5 / 110, -100 * 20 / 110, -100 * 15 / 110, 0]
    assert m.ulcer_index(EQUITY) == pytest.approx(math.sqrt(sum(d * d for d in dd_pct) / 7), rel=1e-12)


def test_recovery_and_martin_ratios():
    assert m.recovery_factor(500.0, 250.0) == pytest.approx(2.0)
    assert math.isinf(m.recovery_factor(500.0, 0.0))
    assert m.martin_ratio(0.12, 6.0) == pytest.approx(2.0)  # cagr in percent over the ulcer index
    assert m.martin_ratio(0.12, 0.0) == 0.0


def test_underwater_duration_is_the_longest_run_not_the_total():
    curve = [10, 9, 10, 8, 7, 6, 10, 9]  # underwater: idx1 (1), then idx3..5 (3), then idx7 (1)
    assert m.max_drawdown_duration(curve) == 3


# --------------------------------------------------------------------------- #
# Benchmark-relative
# --------------------------------------------------------------------------- #
def test_exact_linear_relation_recovers_alpha_beta_and_unit_r_squared():
    b = np.random.default_rng(3).normal(0, 0.01, 200)
    r = 2.0 * b + 0.001
    alpha, beta, r2 = m.alpha_beta(r, b)
    assert beta == pytest.approx(2.0, abs=1e-12)
    assert alpha == pytest.approx(0.001, abs=1e-12)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_holding_the_benchmark_gives_beta_one_alpha_zero_r_squared_one():
    b = np.random.default_rng(4).normal(0.0004, 0.01, 300)
    assert m.alpha_beta(b, b) == pytest.approx((0.0, 1.0, 1.0), abs=1e-12)


def test_alpha_beta_matches_least_squares_on_noisy_data():
    rng = np.random.default_rng(5)
    b = rng.normal(0.0003, 0.01, 400)
    r = 0.0002 + 0.7 * b + rng.normal(0, 0.005, 400)
    slope, intercept = np.polyfit(b, r, 1)
    ss_res = float(np.sum((r - (intercept + slope * b)) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    alpha, beta, r2 = m.alpha_beta(r, b)
    assert (alpha, beta, r2) == pytest.approx((intercept, slope, 1 - ss_res / ss_tot), rel=1e-9)


def test_alpha_beta_pairs_only_dates_present_in_both_series():
    idx = pd.date_range("2024-01-01", periods=6)
    b = pd.Series([0.01, -0.02, 0.015, 0.0, 0.01, -0.01], index=idx)
    r = pd.Series(2.0 * b.to_numpy(), index=idx)
    r.iloc[2] = np.nan
    b2 = b.copy()
    b2.iloc[4] = np.nan
    _, beta, r2 = m.alpha_beta(r, b2)
    assert beta == pytest.approx(2.0, abs=1e-12) and r2 == pytest.approx(1.0, abs=1e-12)


def test_information_ratio_is_annualised_mean_over_std_of_the_difference():
    rng = np.random.default_rng(6)
    r, b = rng.normal(0.001, 0.01, 250), rng.normal(0.0005, 0.008, 250)
    active = (r - b).tolist()
    assert m.information_ratio(r, b) == pytest.approx(SQRT_252 * _mean(active) / _std(active), rel=1e-12)
    assert m.information_ratio(b, b) == 0.0  # zero tracking error is unavailable, not infinite


# --------------------------------------------------------------------------- #
# Clocks and units
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tf, ppy",
    [
        ("1Day", 252.0),
        ("1Week", 52.0),
        ("1Hour", 1638.0),  # 252 * 6.5
        ("5Min", 19656.0),  # 252 * 6.5 * 60 / 5
        ("15Min", 6552.0),
        ("30Min", 3276.0),
    ],
)
def test_periods_per_year_table(tf, ppy):
    assert Timeframe.parse(tf).periods_per_year() == pytest.approx(ppy, rel=1e-12)


def test_dated_equity_curve_is_on_business_days_and_conserves_pnl():
    """A weekend exit is booked on the Friday bin; no calendar-day rows. A daily index
    inserted zero-return weekends, understating the annualised Sharpe by sqrt(5/7)."""
    exits = pd.to_datetime(["2024-01-05 16:00", "2024-01-06 12:00", "2024-01-09 16:00"])  # Fri, Sat, Tue
    trades = pd.DataFrame({"exit_time": exits, "pnl": [10.0, 5.0, -3.0]})
    curve = performance.build_dated_equity_curve(trades, 1000.0)
    assert (curve.index.dayofweek < 5).all()
    # start row one business day before the first bin, then Fri(10+5), Mon(0), Tue(-3)
    assert curve.tolist() == [1000.0, 1015.0, 1015.0, 1012.0]
    assert curve.iloc[-1] == 1000.0 + sum(trades["pnl"])


def test_sharpe_on_the_dated_curve_is_a_trading_day_sharpe():
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2022-01-03", periods=400)
    pnl = rng.normal(30, 300, len(days))
    trades = pd.DataFrame({"exit_time": days + pd.Timedelta(hours=16), "pnl": pnl})
    curve = performance.build_dated_equity_curve(trades, 100_000.0)
    r = (curve.iloc[1:].to_numpy() / curve.iloc[:-1].to_numpy() - 1).tolist()
    assert len(r) == 400
    assert m.sharpe_ratio(m.returns_from_equity(curve)) == pytest.approx(
        SQRT_252 * _mean(r) / _std(r), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# Trade statistics
# --------------------------------------------------------------------------- #
PNL = [100.0, -50.0, 200.0, -25.0, 0.0, 75.0]


def test_trade_statistics_known_answers():
    # wins: 100, 200, 75 (sum 375); losses: -50, -25 (sum -75); one flat trade
    assert m.win_rate(PNL) == pytest.approx(3 / 6)
    assert m.profit_factor(PNL) == pytest.approx(375 / 75)
    assert m.expectancy(PNL) == pytest.approx(300 / 6)
    assert m.payoff_ratio(PNL) == pytest.approx((375 / 3) / (75 / 2))
    assert m.gain_to_pain_ratio(PNL) == pytest.approx(300 / 75)
    w, r = 3 / 6, (375 / 3) / (75 / 2)
    assert m.kelly_criterion(PNL) == pytest.approx(w - (1 - w) / r)
    assert m.system_quality_number(PNL) == pytest.approx(math.sqrt(6) * 50.0 / _std(PNL))
    assert m.consecutive(PNL, winning=True) == 1 and m.consecutive(PNL, winning=False) == 1
    assert m.consecutive([1, 1, -1, -1, -1, 1], winning=False) == 3


def test_trade_statistics_without_losses_use_the_documented_infinite_or_zero_values():
    wins = [1.0, 2.0]
    assert math.isinf(m.profit_factor(wins)) and math.isinf(m.payoff_ratio(wins))
    assert math.isinf(m.gain_to_pain_ratio(wins))
    assert m.kelly_criterion(wins) == 0.0  # degenerate input is not a bet size


# --------------------------------------------------------------------------- #
# Edge cases: the documented unavailable value, never a silent NaN or inf
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("series", [[], [0.01], [0.0] * 10])
def test_ratios_on_degenerate_series_are_finite(series):
    """Empty, one observation, all-zero: zero spread means the ratio is
    unavailable (0.0), and nothing leaks NaN or inf into a printed metric."""
    for fn in (m.sharpe_ratio, m.sortino_ratio, m.annualized_volatility, m.downside_deviation):
        assert math.isfinite(fn(series)), fn.__name__
    for fn in (m.skewness, m.kurtosis, m.value_at_risk, m.conditional_var, m.probabilistic_sharpe_ratio):
        assert math.isfinite(fn(series)), fn.__name__


def test_all_negative_returns_give_a_negative_sharpe_and_a_positive_downside():
    r = [-0.01, -0.02, -0.015, -0.005]
    assert m.sharpe_ratio(r) < 0
    assert m.sortino_ratio(r) < 0
    assert m.downside_deviation(r) == pytest.approx(
        math.sqrt(_mean([x * x for x in r])) * SQRT_252, rel=1e-12
    )


def test_sortino_with_no_losses_is_the_unavailable_value_not_infinity():
    assert m.sortino_ratio([0.01, 0.02, 0.0, 0.03]) == 0.0


def test_nans_are_ignored_by_the_return_ratios_not_propagated():
    clean = [0.01, -0.02, 0.03, 0.0, 0.01]
    dirty = [0.01, float("nan"), -0.02, 0.03, 0.0, 0.01]
    assert m.sharpe_ratio(dirty) == pytest.approx(m.sharpe_ratio(clean), rel=1e-12)
    assert m.annualized_volatility(dirty) == pytest.approx(m.annualized_volatility(clean), rel=1e-12)


def test_returns_from_equity_and_empty_curves():
    assert m.returns_from_equity([100.0, 110.0, 99.0]).tolist() == pytest.approx([0.10, -0.10])
    assert m.returns_from_equity([100.0]).empty and m.returns_from_equity([]).empty
    assert m.max_drawdown([]) == 0.0 and m.max_drawdown_duration([5.0]) == 0
    assert m.ulcer_index([]) == 0.0
    assert m.cagr([100.0], 1.0) == 0.0 and m.cagr([100.0, 110.0], 0.0) == 0.0
    assert m.cagr([100.0, -5.0], 1.0) == 0.0  # a wiped-out curve has no meaningful growth rate


def test_zero_trade_run_reports_the_empty_metrics_without_nan():
    out = performance.compute_backtest_metrics(pd.DataFrame(), [100.0], 100.0, 100.0, {})
    assert out["benchmark_available"] is False
    for key, value in out.items():
        if isinstance(value, float):
            assert math.isfinite(value), key
