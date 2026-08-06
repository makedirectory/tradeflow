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


def test_best_params_retains_pinned_parameters():
    """Pinned params (default, no min/max/step) survive into ``best_params``.

    Callers construct a strategy directly from ``best_params``; dropping the pinned
    entries would hand back a config the strategy cannot be built from - which shows
    up much later as a walk-forward run that mysteriously places no trades.
    """
    from src.optimization.param_space import ParameterSpace

    ranges = {
        "searched": {"type": "int", "min": 1, "max": 3, "step": 1, "default": 2},
        "pinned": {"type": "float", "default": 0.02},
    }
    space = ParameterSpace(ranges)
    assert space.searchable == ["searched"]

    grid = space.grid()
    assert all(combo["pinned"] == 0.02 for combo in grid)
