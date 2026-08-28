"""Parameter optimization for strategies - excellent at finding the config that
would have made you rich last year. Whether it works *next* year is what
walk-forward validation is for.

Tunes a strategy's parameters by backtesting candidate configs and ranking them
by an objective metric. Three search methods, increasing in sophistication:

* :meth:`grid_search`    - exhaustive (or capped) sweep of the step grid
* :meth:`random_search`  - random step-aligned sampling
* :meth:`optimize_bayesian` - trains a Gaussian-Process *surrogate model* of the
  objective and proposes promising configs (the "train a model to align params"
  approach). Requires scikit-learn (optional ``optimize`` extra).

Runs serially for determinism and simplicity; each evaluation is an independent
:class:`BacktestEngine` run, so this is trivially parallelizable later.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from tradeflow.costs.base import CostModel
from tradeflow.engine.backtest import BacktestEngine
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.optimization.param_space import ParameterSpace
from tradeflow.strategies.base import Strategy

logger = logging.getLogger(__name__)

# Stand-in for +inf objective values (e.g. profit_factor with no losses) so
# ranking and the surrogate model stay numerically well-behaved.
_INF_SENTINEL = 1e6


def _plain(value):
    """A DataFrame cell as a plain Python scalar.

    A row read off a DataFrame carries numpy scalars, and under NumPy 2 those repr as
    ``np.int64(50)`` - which is what the CLI printed back as the chosen config, and
    what every other consumer then had to re-normalize. Converting once here keeps a
    search result an ordinary dict.
    """
    return value.item() if hasattr(value, "item") else value


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
        cost_model: Optional[CostModel] = None,
        *,
        trial_store: Optional[Any] = None,
        strategy_name: Optional[str] = None,
        cost_key: Optional[Dict[str, Any]] = None,
        accounting: Optional[int] = None,
        force: bool = False,
        workers: Optional[int] = None,
        data_spec: Optional[Any] = None,
    ):
        self.strategy_class = strategy_class
        self.data_client = data_client
        self.initial_capital = initial_capital
        #: Charged on every simulated fill. ``None`` searches gross returns, which
        #: reliably favors the highest-turnover config in the space.
        self.cost_model = cost_model
        self.space = ParameterSpace(strategy_class.PARAM_RANGES)
        self._rng = np.random.default_rng(seed)
        #: Optional per-candidate memoization: a
        #: search that re-evaluates a config already scored this campaign - a
        #: real possibility with random sampling, or a resumed/extended search -
        #: is served from the trial store instead of re-simulated. All-or-nothing:
        #: `trial_store` is `None` unless the caller opts in (CLI/MCP wiring), so
        #: every other caller (tests, ad hoc scripts) is unaffected.
        self.trial_store = trial_store
        self.strategy_name = strategy_name
        self.cost_key = cost_key or {}
        self.accounting = accounting
        self.force = force
        #: Candidate evaluation is embarrassingly parallel, but only *execution*
        #: parallelizes: memoization is still checked here, in the parent, before
        #: anything is dispatched, and nothing a worker returns is written by the
        #: worker. ``workers <= 1`` runs the original sequential path untouched.
        from tradeflow.optimization.parallel import resolve_workers

        self.workers = resolve_workers(workers)
        #: How a worker builds its own data client (a live one cannot be pickled to
        #: a spawned process). Required for parallel execution; without it, a
        #: parallel request degrades to sequential rather than guessing.
        self.data_spec = data_spec
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Search methods
    # ------------------------------------------------------------------ #
    def grid_search(
        self, symbols, start, end, objective="sharpe_ratio", max_evals=None
    ) -> OptimizationResult:
        total = self.space.grid_size()
        # The full grid is combinatorial: for many-parameter spaces it can be
        # billions of points, so never materialize it when we're going to cap.
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

        # The surrogate proposes one point at a time by design, so parallelism comes
        # from asking it for a *batch* per round: propose `workers` points, evaluate
        # them together, refit, repeat. Same evaluation budget, fewer rounds.
        batch = self.workers if self._parallel_available() else 1
        remaining = n_iterations
        round_index = 0
        while remaining > 0:
            gp.fit(np.vstack(x_obs), self._standardize(y_obs))
            picks = self._propose_batch(gp, min(batch, remaining), exploration, n_candidates)
            round_index += 1

            if len(picks) == 1:
                params = self.space.from_unit_vector(picks[0])
                score, row = self._score_config(params, symbols, start, end, objective)
                logger.info("Bayesian round %d -> %s=%.4f", round_index, objective, score)
                if row is not None:
                    rows.append(row)
                    x_obs.append(picks[0])
                    y_obs.append(score)
            else:
                combos = [self.space.from_unit_vector(pick) for pick in picks]
                batch_rows = self._evaluate_parallel(combos, symbols, start, end, objective)
                logger.info("Bayesian round %d evaluated %d candidates", round_index, len(batch_rows))
                for pick, row in zip(picks, batch_rows):
                    rows.append(row)
                    x_obs.append(pick)
                    y_obs.append(self._finite(row.get(objective, float("-inf"))))
            remaining -= len(picks)

        return self._build_result(rows, objective)

    def _propose_batch(self, gp, size: int, exploration: float, n_candidates: int) -> List[np.ndarray]:
        """Propose ``size`` distinct points from one surrogate fit.

        The acquisition function has no memory of what it just proposed, so asking it
        `size` times returns the same point `size` times. This applies the standard
        constant-liar trick: after picking a point, penalize its neighborhood so the
        next pick explores elsewhere, exactly as a sequential run would once that
        point's (unknown) result came back.
        """
        candidates = self._rng.random((n_candidates, self.space.dimensions))
        mean, std = gp.predict(candidates, return_std=True)
        acquisition = mean + exploration * std
        picks: List[np.ndarray] = []
        for _ in range(size):
            index = int(np.argmax(acquisition))
            picks.append(candidates[index])
            # The lie: assume this point comes back unremarkable, and suppress the
            # region around it so the batch spreads rather than clustering.
            distance = np.linalg.norm(candidates - candidates[index], axis=1)
            acquisition = acquisition - np.exp(-((distance / 0.1) ** 2)) * (acquisition.max() + 1.0)
        return picks

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #
    def _evaluate_all(self, combos, symbols, start, end, objective) -> OptimizationResult:
        logger.info("Evaluating %d parameter configs on %d symbols", len(combos), len(symbols))
        if self._parallel_available():
            return self._build_result(
                self._evaluate_parallel(combos, symbols, start, end, objective), objective
            )
        rows: List[Dict[str, Any]] = []
        for i, params in enumerate(combos, 1):
            _, row = self._score_config(params, symbols, start, end, objective)
            if row is not None:
                rows.append(row)
            if i % 10 == 0 or i == len(combos):
                logger.info("  ... %d/%d evaluated", i, len(combos))
        return self._build_result(rows, objective)

    def _parallel_available(self) -> bool:
        """Whether this search can actually run in parallel.

        A worker builds its own data client from a picklable recipe, so without one
        there is nothing to build. Falling back to sequential is the right failure:
        the answer is identical, only slower, and silently guessing at how to
        reconstruct someone's data client is how a parallel run ends up reading
        different bars than the sequential one did.
        """
        return self.workers > 1 and self.data_spec is not None

    def _evaluate_parallel(self, combos, symbols, start, end, objective) -> List[Dict[str, Any]]:
        """Dispatch candidates to workers; keep every write here in the parent.

        Memoization is resolved *before* dispatch, so a candidate already scored this
        campaign costs no compute and no worker slot. Workers only ever run the
        candidates that genuinely need running.
        """
        from tradeflow.optimization.parallel import EvalRequest, candidate_key, cost_spec, run_pool, summarize

        rows: List[Dict[str, Any]] = []
        requests: List[EvalRequest] = []
        spec = cost_spec(self.cost_model)
        for params in combos:
            cached = self._find_cached(params, symbols, start, end, objective)
            if cached is not None:
                rows.append({**{k: params[k] for k in self.space.searchable}, **cached})
                continue
            requests.append(
                EvalRequest(
                    key=candidate_key(self.strategy_name or "", params, symbols, start, end),
                    strategy=self.strategy_name or "",
                    params=dict(params),
                    symbols=tuple(symbols),
                    start=start,
                    end=end,
                    capital=self.initial_capital,
                    cost=spec,
                    data=self.data_spec,
                    base_seed=self.seed,
                )
            )

        done = [0]

        def _progress(_result) -> None:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(requests):
                logger.info("  ... %d/%d evaluated (%d workers)", done[0], len(requests), self.workers)

        report = run_pool(requests, self.workers, on_result=_progress)
        logger.info("Parallel evaluation: %s", summarize(report))
        for result in report.results:
            if result.get("error"):
                # A crashed candidate is still a configuration that was tried; it is
                # reported, not hidden, and it does not stop the campaign.
                logger.error("Evaluation failed for %s: %s", result["params"], result["error"])
                continue
            searched = {k: result["params"][k] for k in self.space.searchable if k in result["params"]}
            rows.append({**searched, **result["metrics"]})
        return rows

    def _score_config(self, params, symbols, start, end, objective):
        """Backtest one config; return (objective_score, results_row) or (-inf, None)."""
        try:
            metrics = self._backtest(params, symbols, start, end, objective)
        except Exception as exc:  # noqa: BLE001
            logger.error("Evaluation failed for %s: %s", params, exc)
            return float("-inf"), None
        row = {**{k: params[k] for k in self.space.searchable}, **metrics}
        return self._finite(metrics.get(objective, float("-inf"))), row

    def _backtest(
        self, params: Dict[str, Any], symbols, start: datetime, end: datetime, objective: Optional[str] = None
    ) -> Dict[str, float]:
        cached = self._find_cached(params, symbols, start, end, objective)
        if cached is not None:
            return cached
        strategy = self.strategy_class(dict(params))
        result = BacktestEngine(strategy, self.data_client, cost_model=self.cost_model).run(
            symbols, start, end, self.initial_capital
        )
        return result.metrics

    def _find_cached(
        self, params: Dict[str, Any], symbols, start, end, objective: Optional[str]
    ) -> Optional[Dict[str, float]]:
        """A stored metrics dict for this exact candidate, or ``None``.

        Only serves a memo when the requested ``objective`` is one of the trial
        store's denormalized headline fields - the store only ever persists a
        fixed subset (see ``tradeflow.store.trials``), so silently serving a memo
        missing the requested objective would make a real candidate look like
        ``-inf`` and drop out of the ranking. An objective the memo can't answer
        is treated as a miss (a real run), never a wrong answer.
        """
        if self.trial_store is None or self.force or not self.strategy_name:
            return None
        from tradeflow.optimization.config_store import current_git_sha

        found = self.trial_store.find(
            strategy=self.strategy_name,
            params={**params, "_cost": self.cost_key},
            symbols=symbols,
            window_start=start,
            window_end=end,
            accounting=self.accounting,
            git_sha=current_git_sha(),
        )
        if found is None:
            return None
        import json

        metrics = dict(json.loads(found["metrics_json"] or "{}"))
        if not metrics or (objective is not None and objective not in metrics):
            return None
        metrics["_memoized_from"] = found["id"]
        metrics["_memoized_ts"] = found["ts"]
        return metrics

    def _build_result(self, rows: List[Dict[str, Any]], objective: str) -> OptimizationResult:
        df = pd.DataFrame(rows)
        if df.empty or objective not in df.columns:
            return OptimizationResult({}, float("-inf"), objective, df)
        # A total order, not just a sort by score. Ranking on the objective alone
        # leaves tied candidates in whatever order they were evaluated, so a parallel
        # run completing in a different order could pick a different "best" config
        # from an identical set of results. Breaking ties on the searched parameter
        # values makes the winner a property of the results, not of the schedule.
        tie_break = [k for k in self.space.searchable if k in df.columns]
        df = df.sort_values(
            [objective, *tie_break],
            ascending=[False, *[True] * len(tie_break)],
            kind="mergesort",
        ).reset_index(drop=True)
        best = df.iloc[0]
        # Layer the searched values over the full defaults: callers construct a
        # strategy straight from best_params, so dropping *pinned* params (declared
        # with a default but no min/max/step) would yield an unconstructable config.
        searched = {k: _plain(best[k]) for k in self.space.searchable if k in df.columns}
        best_params = {**self.space.defaults, **searched}
        logger.info("Best %s = %.4f with %s", objective, best[objective], searched)
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
