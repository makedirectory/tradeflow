"""The walk-forward aggregate is one clock end to end: trading days.

The out-of-sample aggregate has no single simulation behind it, so it is rebuilt from
closed-trade P&L. That curve used to carry a row for every calendar day, weekends
included, with a zero return on each, and was then annualized at 252 per year. Two
weekend zeros per week shrink the per-period standard deviation by about sqrt(5/7), so
the annualized Sharpe came out about 16% low, while the Deflated Sharpe's trial
variance was converted to per-trading-day units - a numerator and a benchmark in
different units.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from quantred.analytics import performance
from quantred.marketdata.client import MarketDataClient
from quantred.optimization.walk_forward import WalkForwardValidator
from tests.fakes import FakeMarketData
from tests.test_walk_forward import SYMBOLS, PeriodicStrategy

CAPITAL = 100_000.0


def _daily_exit_trades(n_days: int = 756) -> pd.DataFrame:
    """One trade closing on every business day, with a modest positive drift."""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    pnl = rng.normal(40.0, 400.0, n_days)
    return pd.DataFrame({"exit_time": days + pd.Timedelta(hours=16), "pnl": pnl})


def _reference_returns(trades: pd.DataFrame) -> np.ndarray:
    """Independent construction: equity after each exit, returns on the prior equity."""
    equity = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(trades["pnl"].to_numpy())])
    return equity[1:] / equity[:-1] - 1.0


def _validator() -> WalkForwardValidator:
    return WalkForwardValidator(
        PeriodicStrategy,
        MarketDataClient(FakeMarketData(SYMBOLS, n=60, freq="1D")),
        initial_capital=CAPITAL,
        seed=1,
    )


def test_dated_equity_curve_has_no_weekend_rows():
    curve = performance.build_dated_equity_curve(_daily_exit_trades(), CAPITAL)
    assert (curve.index.dayofweek < 5).all()


def test_aggregate_sharpe_matches_closed_form_on_trading_days():
    trades = _daily_exit_trades()
    r = _reference_returns(trades)
    expected = math.sqrt(252) * r.mean() / r.std(ddof=1)

    got = _validator()._metrics_for_trades(trades, None, None, 1, None)["sharpe_ratio"]

    # The calendar-day construction this replaced read low by about sqrt(252/365).
    assert abs(got / expected - 1.0) < 1e-9


def test_trial_variance_and_aggregate_share_per_trading_day_units():
    """Trial Sharpes and the aggregate come from one series; the DSR is checked against
    an independent computation of the deflation in per-trading-day units."""
    trades = _daily_exit_trades()
    r = _reference_returns(trades)
    slices = np.array_split(r, 12)
    annualized = [math.sqrt(252) * s.mean() / s.std(ddof=1) for s in slices]
    per_day = [s.mean() / s.std(ddof=1) for s in slices]

    var_trial = WalkForwardValidator._per_period_sr_variance(annualized)
    assert math.isclose(var_trial, float(np.var(per_day, ddof=1)), rel_tol=1e-12)

    n_trials = 50
    got = _validator()._metrics_for_trades(trades, None, None, n_trials, var_trial)

    nd = NormalDist()
    euler = 0.5772156649015329
    sr_star = math.sqrt(var_trial) * (
        (1 - euler) * nd.inv_cdf(1 - 1 / n_trials) + euler * nd.inv_cdf(1 - 1 / (n_trials * math.e))
    )
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=0)
    skew, kurt = float((z**3).mean()), float((z**4).mean())
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    expected_dsr = nd.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom))

    assert abs(got["deflated_sharpe_ratio"] - expected_dsr) < 1e-6
    # The aggregate's annualized Sharpe is the same per-day quantity times sqrt(252).
    assert abs(got["sharpe_ratio"] / math.sqrt(252) - sr) < 1e-9
