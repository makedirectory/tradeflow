"""What the walk-forward diagnostics are called must be what they compute.

* The Monte Carlo "Sharpe" was ``sqrt(N_trades) * mean / std`` of per-trade P&L, a
  t-statistic that grows with trade count however little the strategy earns per year.
  It is now an annualized Sharpe of resampled daily P&L.
* The "PBO" ranks folds, not configurations across symmetric partitions, and the
  bootstrap "IR" is a Sharpe of raw OOS returns. Both are named for what they are.
"""

import math

import numpy as np
import pandas as pd

from quantred.cli import build_parser
from quantred.marketdata.client import MarketDataClient
from quantred.optimization.walk_forward import WalkForwardValidator
from tests.fakes import FakeMarketData
from tests.test_walk_forward import SYMBOLS, PeriodicStrategy

CAPITAL = 100_000.0


def _validator() -> WalkForwardValidator:
    return WalkForwardValidator(
        PeriodicStrategy,
        MarketDataClient(FakeMarketData(SYMBOLS, n=60, freq="1D")),
        initial_capital=CAPITAL,
        seed=1,
    )


def _daily_pnl() -> np.ndarray:
    return np.random.default_rng(9).normal(30.0, 300.0, 120)


def _trades(pnl: np.ndarray, per_day: int) -> pd.DataFrame:
    """The same daily P&L, split over ``per_day`` trades closing that day."""
    days = pd.bdate_range("2022-01-03", periods=len(pnl))
    rows = [
        {"exit_time": d + pd.Timedelta(hours=16), "pnl": p / per_day}
        for d, p in zip(days, pnl)
        for _ in range(per_day)
    ]
    return pd.DataFrame(rows)


def test_monte_carlo_sharpe_is_the_annualized_sharpe_of_daily_pnl():
    pnl = _daily_pnl()
    # One block spanning the whole series leaves exactly one possible resample: the
    # series itself, so every percentile equals the closed-form annualized Sharpe.
    mc = _validator().monte_carlo([_trades(pnl, 1)], n_resamples=50, block=len(pnl))
    expected = math.sqrt(252) * pnl.mean() / pnl.std(ddof=1)
    assert abs(mc["sharpe_p05"] - expected) < 1e-9
    assert abs(mc["sharpe_p50"] - expected) < 1e-9
    assert mc["periods"] == len(pnl)


def test_monte_carlo_sharpe_does_not_grow_with_trade_count():
    pnl = _daily_pnl()
    one = _validator().monte_carlo([_trades(pnl, 1)], n_resamples=200, block=10)
    five = _validator().monte_carlo([_trades(pnl, 5)], n_resamples=200, block=10)
    assert math.isclose(one["sharpe_p50"], five["sharpe_p50"], rel_tol=1e-9)
    assert math.isclose(one["sharpe_p05"], five["sharpe_p05"], rel_tol=1e-9)


def test_monte_carlo_with_too_little_data_says_how_little():
    mc = _validator().monte_carlo([_trades(_daily_pnl()[:5], 1)], block=20)
    assert mc == {"samples": 0, "periods": 5}


def test_the_rank_failure_estimate_is_not_called_a_probability_of_overfitting():
    parser = build_parser()
    wf = next(
        a for a in parser._subparsers._group_actions[0].choices["walkforward"]._actions if a.dest == "pbo"
    )
    assert "heuristic" in wf.help and "not a true" in wf.help
    assert "heuristic" in WalkForwardValidator.estimate_pbo.__doc__.lower()
