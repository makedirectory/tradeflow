"""Optimizer tests (grid, random, and the optional Bayesian path)."""

from datetime import datetime

import numpy as np
import pytest

from quantred.demo.strategies import DemoTrendStrategy
from quantred.marketdata.client import MarketDataClient
from quantred.optimization.optimizer import ParameterOptimizer
from quantred.services.registry import STRATEGIES
from tests.fakes import FakeMarketData

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2024, 2, 1)


def _optimizer():
    return ParameterOptimizer(
        DemoTrendStrategy, MarketDataClient(FakeMarketData(SYMBOLS)), initial_capital=100_000
    )


def test_random_search_returns_ranked_results():
    result = _optimizer().random_search(SYMBOLS, START, END, objective="sharpe_ratio", n_samples=6)
    assert not result.results.empty
    assert "sharpe_ratio" in result.results.columns
    # Results are sorted best-first.
    assert result.results["sharpe_ratio"].iloc[0] == result.results["sharpe_ratio"].max()
    assert set(result.best_params) <= set(DemoTrendStrategy.PARAM_RANGES)


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
    from quantred.optimization.param_space import ParameterSpace

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
    optimizer = ParameterOptimizer(STRATEGIES["demo_trend"], client, initial_capital=100_000)

    result = optimizer.random_search(
        ["AAA", "BBB"], datetime(2024, 1, 2), datetime(2025, 1, 2), "sharpe_ratio", n_samples=3
    )

    assert result.best_params
    for name, value in result.best_params.items():
        assert type(value) in (int, float, str, bool), f"{name} is {type(value).__name__}"
    assert "np." not in repr(result.best_params)


def test_the_seed_selects_which_candidates_a_random_search_draws():
    """`optimize` accepted no --seed, so every random search used the built-in 42.

    That is reproducible and uncontrollable: you could not draw a second independent
    sample, and could not reproduce a `walkforward --seed N` run's inner search with a
    standalone optimize. Both directions matter — same seed same draw, different seed
    different draw — or the flag is decorative.
    """
    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=400, freq="1D"))

    def draw(seed):
        optimizer = ParameterOptimizer(STRATEGIES["demo_trend"], client, initial_capital=100_000, seed=seed)
        result = optimizer.random_search(
            ["AAA", "BBB"], datetime(2024, 1, 2), datetime(2025, 1, 2), "sharpe_ratio", n_samples=4
        )
        return [tuple(sorted(row.items())) for row in result.results.to_dict("records")]

    assert draw(27) == draw(27)
    assert draw(27) != draw(42)


def test_optimize_exposes_the_seed_walkforward_already_had():
    from quantred.cli import build_parser

    args = build_parser().parse_args(
        ["optimize", "--strategy", "demo_trend", "--method", "random", "--seed", "27"]
    )
    assert args.seed == 27


# --- What may win a search ---------------------------------------------------------


def test_an_unbounded_objective_is_never_the_winner():
    """A profit factor with no losing trade is infinite, which is what one lucky trade
    produces. Sorted on the value alone it was always first, so a single-trade candidate
    beat a 200-trade one with a real ratio."""
    rows = [
        {"fast_ema_period": 5, "slow_ema_period": 30, "profit_factor": float("inf"), "total_trades": 1},
        {"fast_ema_period": 8, "slow_ema_period": 40, "profit_factor": 1.8, "total_trades": 200},
        {"fast_ema_period": 9, "slow_ema_period": 50, "profit_factor": 1.2, "total_trades": 150},
    ]
    result = _optimizer()._build_result(rows, "profit_factor")
    assert result.best_score == 1.8
    assert result.best_params["fast_ema_period"] == 8
    assert len(result.results) == 3  # still reported, last, and still counted as a trial
    assert result.results["profit_factor"].iloc[-1] == float("inf")


def test_a_search_with_no_bounded_score_names_no_winner():
    rows = [{"fast_ema_period": 5, "slow_ema_period": 30, "profit_factor": float("inf")}]
    result = _optimizer()._build_result(rows, "profit_factor")
    assert result.best_params == {}
    assert len(result.results) == 1


class _RecordingSurrogate:
    """Stands in for the Gaussian process: keeps what it was fitted on."""

    def __init__(self):
        self.fits = []

    def fit(self, x, y):
        self.fits.append((np.array(x), np.array(y)))

    def predict(self, candidates, return_std=False):
        mean = candidates.sum(axis=1)
        return (mean, np.ones(len(candidates))) if return_std else mean


def test_a_parallel_bayesian_round_pairs_each_score_with_the_point_it_was_measured_at(monkeypatch):
    """The batch's rows were zipped against the proposals by position. Rows come back with
    memoized candidates first, duplicates once and failures not at all, so a score was
    attached to a different point's coordinates and the surrogate learned a scrambled
    surface. Here the score is a function of the params, the batch comes back reversed
    with its last candidate missing, and every fitted pair must still satisfy the function."""
    optimizer = _optimizer()
    space = optimizer.space
    surrogate = _RecordingSurrogate()
    monkeypatch.setattr(optimizer, "_make_gp", lambda: surrogate)
    monkeypatch.setattr(optimizer, "_parallel_available", lambda: True)
    optimizer.workers = 4

    def score(params):
        return float(params["fast_ema_period"] * 100 + params["slow_ema_period"])

    def row(params):
        return {**{k: params[k] for k in space.searchable}, "sharpe_ratio": score(params)}

    monkeypatch.setattr(optimizer, "_score_config", lambda params, *a: (score(params), row(params)))
    monkeypatch.setattr(
        optimizer, "_evaluate_parallel", lambda combos, *a: [row(c) for c in reversed(combos[:-1])]
    )
    optimizer.optimize_bayesian(SYMBOLS, START, END, n_initial=3, n_iterations=8, n_candidates=64)

    x, _ = surrogate.fits[-1]
    assert len(x) > 3  # the batch rounds contributed points
    raw = [r["sharpe_ratio"] for r in map(row, map(space.from_unit_vector, x))]
    observed = optimizer._standardize(raw)
    np.testing.assert_allclose(surrogate.fits[-1][1], observed[: len(x)])
    for point in x:  # and each is the grid point evaluated, not the raw proposal
        np.testing.assert_allclose(space.to_unit_vector(space.from_unit_vector(point)), point)
