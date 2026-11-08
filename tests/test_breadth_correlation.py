"""Breadth is deflated by the correlation of the returns the IC was measured on.

The IC is computed on benchmark-residual returns, but breadth used the risk model's
total-return correlation, which includes the market move the residual already removed.
In a one-factor world that over-deflated effective breadth many times over, and made the
reported breadth change with ``--risk-model``.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from quantred.analytics.information import effective_breadth, mean_residual_correlation
from quantred.marketdata.client import MarketDataClient
from quantred.services import analysis
from tests.fakes import DictMarketData, make_ohlcv

N, T = 50, 3000


def _one_factor_returns():
    rng = np.random.default_rng(8)
    market = pd.Series(rng.normal(0.0, 0.01, T))
    betas = rng.uniform(0.6, 1.4, N)
    idio = rng.normal(0.0, 0.01, (T, N))
    returns = pd.DataFrame(np.outer(market, betas) + idio, columns=[f"S{i}" for i in range(N)])
    return returns, market


def test_residual_correlation_is_near_zero_where_total_correlation_is_high():
    returns, market = _one_factor_returns()
    total = returns.corr().to_numpy()
    total_mean = float((total.sum() - N) / (N * (N - 1)))
    assert total_mean > 0.3  # what the old code deflated breadth by

    rho = mean_residual_correlation(returns, market)
    assert abs(rho) < 0.02  # the idiosyncratic parts are independent by construction

    naive = effective_breadth(N, 12.0, total_mean)["br_eff"]
    residual = effective_breadth(N, 12.0, rho)["br_eff"]
    assert residual / naive > 10  # the over-deflation the fix removes


def test_residual_correlation_matches_an_independent_regression():
    returns, market = _one_factor_returns()
    returns["S1"] = returns["S0"] * 0.5 + returns["S1"] * 0.5  # genuine residual dependence
    r = returns.to_numpy()
    m = market.to_numpy()
    beta = ((r - r.mean(0)) * (m - m.mean())[:, None]).sum(0) / ((m - m.mean()) ** 2).sum()
    resid = r - np.outer(m, beta)
    corr = np.corrcoef(resid, rowvar=False)
    expected = (corr.sum() - N) / (N * (N - 1))
    assert abs(mean_residual_correlation(returns, market) - expected) < 1e-9


def test_reported_rho_bar_does_not_depend_on_the_risk_model():
    syms = [f"S{i}" for i in range(12)]
    data = {s: make_ohlcv(n=600, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    client = MarketDataClient(DictMarketData(data))
    kw = dict(n_points=20, n_trials=10)
    args = (client, "demo_trend", syms, datetime(2023, 1, 1), datetime(2024, 12, 31))
    shrunk = analysis.compute_information(*args, risk_model="shrinkage", **kw)
    factor = analysis.compute_information(*args, risk_model="factor", **kw)
    assert shrunk["rho_bar"] == factor["rho_bar"]
