"""Optimizer tests (grid, random, and the optional Bayesian path)."""

from datetime import datetime

import pytest

from src.marketdata.client import MarketDataClient
from src.optimization.optimizer import ParameterOptimizer
from src.strategies.volume_spike import VolumeSpikeStrategy
from tests.fakes import FakeMarketData

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2024, 2, 1)


def _optimizer():
    return ParameterOptimizer(
        VolumeSpikeStrategy, MarketDataClient(FakeMarketData(SYMBOLS)), initial_capital=100_000
    )


def test_random_search_returns_ranked_results():
    result = _optimizer().random_search(SYMBOLS, START, END, objective="sharpe_ratio", n_samples=6)
    assert not result.results.empty
    assert "sharpe_ratio" in result.results.columns
    # Results are sorted best-first.
    assert result.results["sharpe_ratio"].iloc[0] == result.results["sharpe_ratio"].max()
    assert set(result.best_params) <= set(VolumeSpikeStrategy.PARAM_RANGES)


def test_grid_search_capped_by_max_evals():
    result = _optimizer().grid_search(SYMBOLS, START, END, objective="total_return", max_evals=4)
    assert len(result.results) <= 4


def test_bayesian_search_runs():
    pytest.importorskip("sklearn")
    result = _optimizer().optimize_bayesian(
        SYMBOLS, START, END, objective="sharpe_ratio", n_initial=3, n_iterations=2, n_candidates=32
    )
    assert result.objective == "sharpe_ratio"
    assert not result.results.empty
