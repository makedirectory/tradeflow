"""Optimizer tests (grid, random, and the optional Bayesian path)."""

from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.optimization.optimizer import ParameterOptimizer
from tradeflow.services.registry import STRATEGIES
from tradeflow.strategies.volume_spike import VolumeSpikeStrategy

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
    from tradeflow.optimization.param_space import ParameterSpace

    ranges = {
        "searched": {"type": "int", "min": 1, "max": 3, "step": 1, "default": 2},
        "pinned": {"type": "float", "default": 0.02},
    }
    space = ParameterSpace(ranges)
    assert space.searchable == ["searched"]

    grid = space.grid()
    assert all(combo["pinned"] == 0.02 for combo in grid)


def test_best_params_are_plain_python_scalars():
    """They are read off a DataFrame row, so they arrived as numpy scalars — which
    NumPy 2 reprs as `np.int64(50)`. The CLI printed that back as the chosen config,
    and every other consumer had to re-normalize it."""
    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=400, freq="1D"))
    optimizer = ParameterOptimizer(STRATEGIES["ma_crossover"], client, initial_capital=100_000)

    result = optimizer.random_search(
        ["AAA", "BBB"], datetime(2024, 1, 2), datetime(2025, 1, 2), "sharpe_ratio", n_samples=3
    )

    assert result.best_params
    for name, value in result.best_params.items():
        assert type(value) in (int, float, str, bool), f"{name} is {type(value).__name__}"
    assert "np." not in repr(result.best_params)
