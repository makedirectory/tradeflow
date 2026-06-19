"""Parameter optimization for strategies.

Tunes a strategy's parameters by backtesting candidate configs and ranking them
by an objective metric. Three search methods, increasing in sophistication:

* :meth:`grid_search`    - exhaustive (or capped) sweep of the step grid
* :meth:`random_search`  - random step-aligned sampling
* :meth:`optimize_bayesian` - trains a Gaussian-Process *surrogate model* of the
  objective and proposes promising configs (the "train a model to align params"
  approach). Requires scikit-learn (optional ``optimize`` extra).

Runs serially for determinism and simplicity; each evaluation is an independent
:class:`BacktestEngine` run, so this is trivially parallelisable later.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd

from src.engine.backtest import BacktestEngine
from src.marketdata.client import MarketDataClient
from src.optimization.param_space import ParameterSpace
from src.strategies.base import Strategy

logger = logging.getLogger(__name__)

# Stand-in for +inf objective values (e.g. profit_factor with no losses) so
# ranking and the surrogate model stay numerically well-behaved.
_INF_SENTINEL = 1e6


@dataclass
class OptimizationResult:
    """Outcome of a parameter search."""

    best_params: Dict[str, Any]
    best_score: float
    objective: str
    results: pd.DataFrame  # one row per evaluated config (searched params + metrics)


class ParameterOptimizer:
    """Searches a strategy's parameter space for the best backtest objective."""

    def __init__(
        self,
        strategy_class: Type[Strategy],
        data_client: MarketDataClient,
        initial_capital: float = 100_000.0,
        seed: int = 42,
    ):
        self.strategy_class = strategy_class
        self.data_client = data_client
        self.initial_capital = initial_capital
        self.space = ParameterSpace(strategy_class.PARAM_RANGES)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Search methods
    # ------------------------------------------------------------------ #
    def grid_search(
        self, symbols, start, end, objective="sharpe_ratio", max_evals=None
    ) -> OptimizationResult:
        total = self.space.grid_size()
        # The full grid is combinatorial: for many-parameter spaces it can be
        # billions of points, so never materialise it when we're going to cap.
        # When the grid exceeds the budget, randomly sample the grid instead.
        if max_evals and total > max_evals:
            logger.info("Grid has %d configs; randomly sampling %d", total, max_evals)
            combos = self.space.random_samples(max_evals, self._rng)
        else:
            combos = self.space.grid()
        return self._evaluate_all(combos, symbols, start, end, objective)

    def random_search(
        self, symbols, start, end, objective="sharpe_ratio", n_samples=50
    ) -> OptimizationResult:
        combos = self.space.random_samples(n_samples, self._rng)
        return self._evaluate_all(combos, symbols, start, end, objective)

    def optimize_bayesian(
        self,
        symbols,
        start,
        end,
        objective="sharpe_ratio",
        n_initial=8,
        n_iterations=20,
        exploration=0.1,
        n_candidates=512,
    ) -> OptimizationResult:
        """Bayesian optimization with a Gaussian-Process surrogate (UCB acquisition)."""
        gp = self._make_gp()  # raises a friendly error if scikit-learn is missing

        rows: List[Dict[str, Any]] = []
        x_obs: List[np.ndarray] = []
        y_obs: List[float] = []

        for params in self.space.random_samples(n_initial, self._rng):
            score, row = self._score_config(params, symbols, start, end, objective)
            if row is not None:
                rows.append(row)
                x_obs.append(self.space.to_unit_vector(params))
                y_obs.append(score)

        if not x_obs:
            logger.warning("No successful initial evaluations; aborting Bayesian search")
            return self._build_result(rows, objective)

        for i in range(n_iterations):
            gp.fit(np.vstack(x_obs), self._standardize(y_obs))
            candidates = self._rng.random((n_candidates, self.space.dimensions))
            mean, std = gp.predict(candidates, return_std=True)
            best = candidates[int(np.argmax(mean + exploration * std))]

            params = self.space.from_unit_vector(best)
            score, row = self._score_config(params, symbols, start, end, objective)
            logger.info("Bayesian iter %d/%d -> %s=%.4f", i + 1, n_iterations, objective, score)
            if row is not None:
                rows.append(row)
                x_obs.append(best)
                y_obs.append(score)

        return self._build_result(rows, objective)

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #
    def _evaluate_all(self, combos, symbols, start, end, objective) -> OptimizationResult:
        logger.info("Evaluating %d parameter configs on %d symbols", len(combos), len(symbols))
        rows: List[Dict[str, Any]] = []
        for i, params in enumerate(combos, 1):
            _, row = self._score_config(params, symbols, start, end, objective)
            if row is not None:
                rows.append(row)
            if i % 10 == 0 or i == len(combos):
                logger.info("  ... %d/%d evaluated", i, len(combos))
        return self._build_result(rows, objective)

    def _score_config(self, params, symbols, start, end, objective):
        """Backtest one config; return (objective_score, results_row) or (-inf, None)."""
        try:
            metrics = self._backtest(params, symbols, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.error("Evaluation failed for %s: %s", params, exc)
            return float("-inf"), None
        row = {**{k: params[k] for k in self.space.searchable}, **metrics}
        return self._finite(metrics.get(objective, float("-inf"))), row

    def _backtest(self, params: Dict[str, Any], symbols, start: datetime, end: datetime) -> Dict[str, float]:
        strategy = self.strategy_class(dict(params))
        result = BacktestEngine(strategy, self.data_client).run(symbols, start, end, self.initial_capital)
        return result.metrics

    def _build_result(self, rows: List[Dict[str, Any]], objective: str) -> OptimizationResult:
        df = pd.DataFrame(rows)
        if df.empty or objective not in df.columns:
            return OptimizationResult({}, float("-inf"), objective, df)
        df = df.sort_values(objective, ascending=False).reset_index(drop=True)
        best = df.iloc[0]
        best_params = {k: best[k] for k in self.space.searchable if k in df.columns}
        logger.info("Best %s = %.4f with %s", objective, best[objective], best_params)
        return OptimizationResult(best_params, float(best[objective]), objective, df)

    @staticmethod
    def _finite(value: float) -> float:
        if value == float("inf"):
            return _INF_SENTINEL
        if value == float("-inf"):
            return -_INF_SENTINEL
        return float(value)

    def _standardize(self, y: List[float]) -> np.ndarray:
        arr = np.array([self._finite(v) for v in y], dtype="float64")
        std = arr.std()
        return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()

    @staticmethod
    def _make_gp():
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Bayesian optimization requires scikit-learn. Install the optional extra:\n"
                "    uv sync --extra optimize"
            ) from exc
        return GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=3, random_state=42)
