"""Walk-forward / out-of-sample validation.

The honest fitness function for the whole automation effort. The optimizer
(:class:`~src.optimization.optimizer.ParameterOptimizer`) tunes parameters on an
in-sample (IS) window; this module makes the chosen config prove itself on an
out-of-sample (OOS) window the optimizer never saw, across rolling/anchored
folds, with a holdout slice scored exactly once at the end.

Why it matters: the moment an agent can run hundreds of optimizations and read
the resulting Sharpe, it becomes a machine for *discovering noise*. Optimizing on
one slice and measuring on another is the structural defense (see the
walk-forward page in the engineering docs).

Key correctness properties:
* **No fold-boundary leakage.** Each OOS backtest fetches ``warmup`` bars before
  ``oos_start`` so indicators are valid, but only trades entered at/after
  ``oos_start`` are counted. The embargo (>= lookback) separates IS from OOS.
* **Variable-length folds ⇒ CAGR/annualized metrics**, never raw
  total return.
* **The holdout is sacred** - computed first, subtracted from the fold region,
  never passed to any optimizer call.
* **Determinism** - the optimizer seed is threaded so a run is reproducible.
* **Prefetch once, slice per fold** - the full window is fetched a single
  time and sliced in memory for every fold/OOS run.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from src.analytics import performance
from src.analytics.metrics import TRADING_DAYS_PER_YEAR
from src.costs.base import CostModel
from src.engine.backtest import BacktestEngine
from src.marketdata.base import BarHandler, MarketDataProvider
from src.marketdata.client import MarketDataClient
from src.marketdata.timeframe import Timeframe
from src.optimization.optimizer import ParameterOptimizer
from src.strategies.base import Strategy

logger = logging.getLogger(__name__)

#: Promotion gates. Config-driven so thresholds are tunable
#: without code changes; uses *median* (not mean) for WFE/Sharpe so one lucky
#: fold can't inflate the verdict.
#:
#: Calibration note (spec 025). Portfolio accounting changed how two of these are
#: *measured*, so some thresholds had to move just to keep meaning what they meant:
#:
#: * **Sharpe scale shifted ~1.19x.** Measured over 12 runs (2 strategies x 3
#:   universes x 2 windows), holding trades fixed and varying only the curve
#:   construction: mark-to-market Sharpe ran 1.04-1.27x the old realized-P&L Sharpe
#:   (median 1.19). The old curve booked P&L as a spike at exit, so its volatility
#:   was overstated. ``min_oos_sharpe`` is rescaled 1.0 -> 1.2 to preserve the
#:   original bar; leaving it at 1.0 would have quietly made the gate ~16% easier.
#: * **Drawdown scale held** (median 1.04x over the same runs), and the gate is a
#:   ratio of two same-construction numbers, so ``max_dd_ratio`` is unchanged.
#:
#: What deliberately did NOT move: ``min_oos_trades``. Portfolio accounting reduced
#: trade counts sharply (one book, ``max_positions`` slots, instead of every symbol
#: trading its own full capital), so strategies now clear it far less often. That is
#: a genuine loss of evidence, not a measurement artifact - the sample really is
#: smaller. Lowering a statistical-power floor because results got worse is exactly
#: the gate-fitting this engine exists to prevent. Note the shipped strategies set
#: ``max_positions: 1``, which caps concurrency and makes 100 OOS trades hard to
#: reach; that is a strategy-config question, not a threshold question.
DEFAULT_GATES: Dict[str, float] = {
    "min_oos_sharpe": 1.2,  # median OOS Sharpe across folds (rescaled, see above)
    "min_oos_profit_factor": 1.3,  # median OOS profit factor (trade-level, unaffected)
    "min_wfe": 0.4,  # median walk-forward efficiency ...
    "wfe_relaxed": 0.3,  # ... or this if OOS Sharpe also clears target
    "max_dd_ratio": 1.5,  # OOS max drawdown <= this x IS max drawdown
    "min_oos_trades": 100,  # statistical-power floor (total OOS trades)
    "max_param_sensitivity_loss": 0.25,  # <=25% Sharpe loss under +-10% perturbation
    "min_deflated_sharpe": 0.5,  # DSR with n_trials_total
}


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class Fold:
    index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime


@dataclass
class FoldResult:
    fold: Fold
    is_best_params: Dict[str, Any]
    is_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    oos_trades: int
    n_trials: int


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    oos_aggregate: Dict[str, float]
    holdout: Optional[Dict[str, float]]
    holdout_params: Optional[Dict[str, Any]]
    efficiency: float
    degradation: Dict[str, float]
    n_trials_total: int
    objective: str
    pbo: Optional[float] = None
    monte_carlo: Optional[Dict[str, float]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # --- derived summaries used by the gates ---
    def median_oos(self, key: str) -> float:
        values = [fr.oos_metrics.get(key, 0.0) for fr in self.folds]
        return float(np.median(values)) if values else 0.0

    def median_efficiency(self) -> float:
        ratios = []
        for fr in self.folds:
            is_obj = fr.is_metrics.get(self.objective, 0.0)
            oos_obj = fr.oos_metrics.get(self.objective, 0.0)
            if is_obj not in (0.0, float("inf"), float("-inf")):
                ratios.append(oos_obj / is_obj)
        return float(np.median(ratios)) if ratios else 0.0

    def total_oos_trades(self) -> int:
        return int(sum(fr.oos_trades for fr in self.folds))

    def gate_report(self, gates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Evaluate the promotion gates; a config is promotable only if all pass."""
        g = {**DEFAULT_GATES, **(gates or {})}
        median_sharpe = self.median_oos("sharpe_ratio")
        median_pf = self.median_oos("profit_factor")
        median_wfe = self.median_efficiency()
        total_trades = self.total_oos_trades()
        oos_dd = self.oos_aggregate.get("max_drawdown", 0.0)
        is_dd = float(np.median([fr.is_metrics.get("max_drawdown", 0.0) for fr in self.folds] or [0.0]))
        dsr = self.oos_aggregate.get("deflated_sharpe_ratio", 0.0)

        checks: Dict[str, Dict[str, Any]] = {
            "oos_sharpe": _check(median_sharpe, ">=", g["min_oos_sharpe"]),
            "oos_profit_factor": _check(median_pf, ">=", g["min_oos_profit_factor"]),
            "walk_forward_efficiency": {
                "value": median_wfe,
                "threshold": g["min_wfe"],
                "passed": median_wfe >= g["min_wfe"]
                or (median_sharpe >= g["min_oos_sharpe"] and median_wfe >= g["wfe_relaxed"]),
            },
            "oos_drawdown_vs_is": {
                "value": oos_dd,
                "threshold": g["max_dd_ratio"] * is_dd if is_dd else float("inf"),
                "passed": is_dd == 0.0 or oos_dd <= g["max_dd_ratio"] * is_dd,
            },
            "min_oos_trades": _check(total_trades, ">=", g["min_oos_trades"]),
            "deflated_sharpe": _check(dsr, ">", g["min_deflated_sharpe"]),
        }

        sensitivity = self.diagnostics.get("parameter_sensitivity")
        if sensitivity is not None:
            loss = sensitivity.get("max_sharpe_loss", 0.0)
            checks["parameter_sensitivity"] = _check(loss, "<=", g["max_param_sensitivity_loss"])
        leakage = self.diagnostics.get("leakage_probe")
        if leakage is not None:
            checks["leakage_probe"] = {
                "value": leakage.get("passed"),
                "threshold": True,
                "passed": bool(leakage.get("passed")),
            }

        promotable = all(c["passed"] for c in checks.values())
        return {"promotable": promotable, "checks": checks}


def _check(value: float, op: str, threshold: float) -> Dict[str, Any]:
    passed = value >= threshold if op == ">=" else value > threshold if op == ">" else value <= threshold
    return {"value": float(value), "threshold": float(threshold), "passed": bool(passed)}


# --------------------------------------------------------------------------- #
# Prefetch+slice data provider
# --------------------------------------------------------------------------- #
class _PrefetchedProvider(MarketDataProvider):
    """Serves slices of an in-memory ``{symbol: DataFrame}`` prefetched once."""

    def __init__(self, frames: Dict[str, pd.DataFrame]):
        self._frames = frames

    def get_bars(self, symbols, timeframe: Timeframe, start, end) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self._frames.get(symbol)
            if df is None or df.empty:
                continue
            lo, hi = _coerce_tz(start, df.index), _coerce_tz(end, df.index)
            out[symbol] = df.loc[(df.index >= lo) & (df.index <= hi)].copy()
        return out

    async def stream_bars(self, symbols, handler: BarHandler) -> None:  # pragma: no cover
        raise NotImplementedError("prefetched provider does not stream")

    def supports_streaming(self) -> bool:
        return False


def _coerce_tz(when: datetime, index: pd.DatetimeIndex) -> pd.Timestamp:
    ts = pd.Timestamp(when)
    if index.tz is not None:
        ts = ts.tz_localize(index.tz) if ts.tzinfo is None else ts.tz_convert(index.tz)
    elif ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
class WalkForwardValidator:
    """Orchestrates per-fold IS optimization + leakage-safe OOS evaluation."""

    def __init__(
        self,
        strategy_class: Type[Strategy],
        data_client: MarketDataClient,
        initial_capital: float = 100_000.0,
        seed: int = 42,
        gates: Optional[Dict[str, float]] = None,
        cost_model: Optional[CostModel] = None,
    ):
        self.strategy_class = strategy_class
        self.data_client = data_client
        self.initial_capital = initial_capital
        self.seed = seed
        self.gates = gates
        #: Charged on every simulated fill, in-sample and out. Gross-return validation
        #: systematically promotes turnover the strategy could not afford live.
        self.cost_model = cost_model

        # Read timeframe + lookback from a default instance (drives warmup/embargo).
        defaults = {
            name: spec["default"] for name, spec in strategy_class.PARAM_RANGES.items() if "default" in spec
        }
        probe = strategy_class(dict(defaults))
        self.timeframe = Timeframe.parse(probe.config["timeframe"])
        self.lookback_bars = int(probe.config.get("required_lookback_periods", 20))

    # ------------------------------------------------------------------ #
    # Fold construction
    # ------------------------------------------------------------------ #
    def build_folds(
        self,
        start: datetime,
        end: datetime,
        *,
        mode: str = "anchored",
        n_folds: Optional[int] = None,
        train_days: Optional[int] = None,
        test_days: Optional[int] = None,
        embargo_days: Optional[int] = None,
        holdout_days: int = 0,
    ) -> "tuple[List[Fold], Optional[tuple[datetime, datetime]]]":
        """Construct folds and (optionally) a disjoint holdout window.

        The holdout is carved off the *end* first and never re-enters the fold
        region, so it is provably disjoint from every IS/OOS window.
        """
        embargo = embargo_days if embargo_days is not None else self.default_embargo_days()
        holdout = None
        region_end = end
        if holdout_days > 0:
            holdout_start = end - timedelta(days=holdout_days)
            holdout = (holdout_start, end)
            region_end = holdout_start

        region_days = (region_end - start).days
        if region_days <= embargo:
            raise ValueError("Walk-forward region is too short for the requested embargo/holdout")

        if train_days is None or test_days is None:
            folds_n = n_folds or 4
            # train + folds_n * test + embargo == region_days  with train == test.
            test_days = max(int((region_days - embargo) / (folds_n + 1)), 1)
            train_days = test_days

        folds: List[Fold] = []
        index = 0
        is_end = start + timedelta(days=train_days)
        while True:
            is_start = start if mode == "anchored" else is_end - timedelta(days=train_days)
            oos_start = is_end + timedelta(days=embargo)
            oos_end = oos_start + timedelta(days=test_days)
            if oos_end > region_end:
                break
            folds.append(Fold(index, is_start, is_end, oos_start, oos_end))
            index += 1
            is_end = is_end + timedelta(days=test_days)

        if not folds:
            raise ValueError("No folds generated; widen the window or reduce train/test/embargo")
        return folds, holdout

    def default_embargo_days(self) -> int:
        """Lookback expressed in calendar days."""
        bars_per_day = self._bars_per_day()
        return max(math.ceil(self.lookback_bars / bars_per_day), 1)

    def _bars_per_day(self) -> float:
        tf = self.timeframe
        if tf.unit == "day":
            return 1.0 / tf.amount
        if tf.unit == "week":
            return 1.0 / (5.0 * tf.amount)
        if tf.unit == "hour":
            return 6.5 / tf.amount
        return 6.5 * 60.0 / tf.amount  # minutes

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        *,
        mode: str = "anchored",
        n_folds: Optional[int] = None,
        train_days: Optional[int] = None,
        test_days: Optional[int] = None,
        embargo_days: Optional[int] = None,
        holdout_days: int = 0,
        method: str = "grid",
        objective: str = "sharpe_ratio",
        max_evals: int = 50,
        pbo: bool = False,
        monte_carlo: bool = False,
        parameter_sensitivity: bool = False,
        leakage_probe: bool = False,
        n_trials_offset: int = 0,
    ) -> WalkForwardResult:
        embargo = embargo_days if embargo_days is not None else self.default_embargo_days()
        folds, holdout = self.build_folds(
            start,
            end,
            mode=mode,
            n_folds=n_folds,
            train_days=train_days,
            test_days=test_days,
            embargo_days=embargo,
            holdout_days=holdout_days,
        )
        warmup_days = embargo  # warmup buffer == embargo, by construction >= lookback

        # Prefetch the whole window once (plus warmup) and slice per fold.
        fetch_start = start - timedelta(days=warmup_days)
        frames = self.data_client.get_bars(symbols, self.timeframe, fetch_start, end)
        sliced = MarketDataClient(_PrefetchedProvider(frames))

        all_trial_sharpes: List[float] = []
        fold_results: List[FoldResult] = []
        oos_trade_frames: List[pd.DataFrame] = []

        for fold in folds:
            opt = ParameterOptimizer(
                self.strategy_class, sliced, self.initial_capital, self.seed, self.cost_model
            )
            is_result = self._optimize(opt, symbols, fold.is_start, fold.is_end, method, objective, max_evals)
            if not is_result.best_params:
                logger.warning("Fold %d produced no valid IS config; skipping", fold.index)
                continue

            is_metrics = self._row_metrics(is_result)
            all_trial_sharpes.extend(self._trial_sharpes(is_result))

            strategy = self.strategy_class(dict(is_result.best_params))
            oos_metrics, oos_trades = self._oos_backtest(
                strategy,
                sliced,
                symbols,
                fold.oos_start,
                fold.oos_end,
                warmup_days,
                n_trials=len(is_result.results),
            )
            oos_trade_frames.append(oos_trades)
            fold_results.append(
                FoldResult(
                    fold=fold,
                    is_best_params=dict(is_result.best_params),
                    is_metrics=is_metrics,
                    oos_metrics=oos_metrics,
                    oos_trades=int(len(oos_trades)),
                    n_trials=len(is_result.results),
                )
            )

        # ``n_trials_offset`` lets a research session accumulate the
        # multiple-testing count across many walk-forward runs, so the Deflated
        # Sharpe reflects every config tried this session, not just this run.
        n_trials_total = sum(fr.n_trials for fr in fold_results) + n_trials_offset
        var_trial_sr = self._per_period_sr_variance(all_trial_sharpes)

        oos_aggregate = self._aggregate_oos(oos_trade_frames, folds, n_trials_total, var_trial_sr)
        efficiency = self._efficiency(fold_results, objective)
        degradation = self._degradation(fold_results)

        # Choose the production candidate by optimizing over the entire non-holdout
        # region, then score the holdout exactly once.
        holdout_metrics, holdout_params = None, None
        if holdout is not None and fold_results:
            holdout_params, holdout_metrics = self._holdout(
                sliced, symbols, start, holdout, warmup_days, method, objective, max_evals
            )

        result = WalkForwardResult(
            folds=fold_results,
            oos_aggregate=oos_aggregate,
            holdout=holdout_metrics,
            holdout_params=holdout_params,
            efficiency=efficiency,
            degradation=degradation,
            n_trials_total=n_trials_total,
            objective=objective,
        )

        # Optional, costlier diagnostics.
        if fold_results:
            best = fold_results[-1].is_best_params
            if parameter_sensitivity:
                result.diagnostics["parameter_sensitivity"] = self.parameter_sensitivity(
                    sliced, symbols, best, folds[-1], warmup_days
                )
            if leakage_probe:
                result.diagnostics["leakage_probe"] = self.leakage_probe(
                    sliced, frames, symbols, best, folds[-1], warmup_days
                )
            if monte_carlo:
                result.monte_carlo = self.monte_carlo(oos_trade_frames)
            if pbo:
                result.pbo = self.estimate_pbo(oos_trade_frames, fold_results, objective)
        return result

    def evaluate_config(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        params: Dict[str, Any],
        *,
        mode: str = "anchored",
        n_folds: Optional[int] = 4,
        train_days: Optional[int] = None,
        test_days: Optional[int] = None,
        embargo_days: Optional[int] = None,
        objective: str = "sharpe_ratio",
        n_trials_offset: int = 0,
        strategy_class: Optional[Type[Strategy]] = None,
    ) -> WalkForwardResult:
        """Validate a *fixed* config out-of-sample across folds (no per-fold search).

        Used by the research agent: the agent proposes a specific
        config, and this scores that exact config OOS, fold by fold, leakage-safe.
        It counts as **one** trial for the multiple-testing correction; pass the
        running session count via ``n_trials_offset`` so the Deflated Sharpe
        reflects every config tried so far.
        """
        cls = strategy_class or self.strategy_class
        embargo = embargo_days if embargo_days is not None else self.default_embargo_days()
        folds, _ = self.build_folds(
            start,
            end,
            mode=mode,
            n_folds=n_folds,
            train_days=train_days,
            test_days=test_days,
            embargo_days=embargo,
            holdout_days=0,
        )
        warmup_days = embargo
        frames = self.data_client.get_bars(symbols, self.timeframe, start - timedelta(days=warmup_days), end)
        sliced = MarketDataClient(_PrefetchedProvider(frames))

        fold_results: List[FoldResult] = []
        oos_trade_frames: List[pd.DataFrame] = []
        for fold in folds:
            is_result = BacktestEngine(cls(dict(params)), sliced, cost_model=self.cost_model).run(
                symbols, fold.is_start, fold.is_end, self.initial_capital
            )
            oos_metrics, oos_trades = self._oos_backtest(
                cls(dict(params)), sliced, symbols, fold.oos_start, fold.oos_end, warmup_days, n_trials=1
            )
            oos_trade_frames.append(oos_trades)
            fold_results.append(
                FoldResult(
                    fold=fold,
                    is_best_params=dict(params),
                    is_metrics=is_result.metrics,
                    oos_metrics=oos_metrics,
                    oos_trades=int(len(oos_trades)),
                    n_trials=1,
                )
            )

        n_trials_total = 1 + n_trials_offset  # one distinct config evaluated this call
        oos_aggregate = self._aggregate_oos(oos_trade_frames, folds, n_trials_total, None)
        return WalkForwardResult(
            folds=fold_results,
            oos_aggregate=oos_aggregate,
            holdout=None,
            holdout_params=None,
            efficiency=self._efficiency(fold_results, objective),
            degradation=self._degradation(fold_results),
            n_trials_total=n_trials_total,
            objective=objective,
        )

    def score_window(
        self,
        symbols: List[str],
        window_start: datetime,
        window_end: datetime,
        params: Dict[str, Any],
        *,
        embargo_days: Optional[int] = None,
        n_trials: int = 1,
        var_of_trial_sr: Optional[float] = None,
        strategy_class: Optional[Type[Strategy]] = None,
    ) -> Dict[str, float]:
        """Leakage-safe metrics for a fixed config over a single window (e.g. holdout)."""
        cls = strategy_class or self.strategy_class
        warmup_days = embargo_days if embargo_days is not None else self.default_embargo_days()
        frames = self.data_client.get_bars(
            symbols, self.timeframe, window_start - timedelta(days=warmup_days), window_end
        )
        sliced = MarketDataClient(_PrefetchedProvider(frames))
        metrics, _ = self._oos_backtest(
            cls(dict(params)),
            sliced,
            symbols,
            window_start,
            window_end,
            warmup_days,
            n_trials=n_trials,
            var_of_trial_sr=var_of_trial_sr,
        )
        return metrics

    # ------------------------------------------------------------------ #
    # Per-fold helpers
    # ------------------------------------------------------------------ #
    def _optimize(self, opt, symbols, is_start, is_end, method, objective, max_evals):
        if method == "grid":
            return opt.grid_search(symbols, is_start, is_end, objective, max_evals=max_evals)
        if method == "random":
            return opt.random_search(symbols, is_start, is_end, objective, n_samples=max_evals)
        return opt.optimize_bayesian(symbols, is_start, is_end, objective)

    def _oos_backtest(
        self, strategy, client, symbols, oos_start, oos_end, warmup_days, n_trials, var_of_trial_sr=None
    ):
        """Backtest with warmup, trading only from ``oos_start``.

        ``trade_from`` keeps the warmup bars out of the book, so the engine's own
        portfolio equity curve *is* the OOS curve — mark-to-market, open positions
        included. Reconstructing one from a filtered trade list (the fallback in
        :meth:`_metrics_for_trades`) can only see realized P&L at exit.
        """
        fetch_start = oos_start - timedelta(days=warmup_days)
        result = BacktestEngine(strategy, client, cost_model=self.cost_model).run(
            symbols, fetch_start, oos_end, self.initial_capital, trade_from=oos_start
        )
        oos_trades = _filter_trades_from(result.trades, oos_start)  # belt and braces
        metrics = self._metrics_for_trades(
            oos_trades, oos_start, oos_end, n_trials, var_of_trial_sr, equity=result.equity_curve
        )
        return metrics, oos_trades

    def _metrics_for_trades(self, trades, start, end, n_trials, var_of_trial_sr, equity=None):
        """Metrics for a trade set, preferring a real portfolio curve when available.

        ``equity=None`` falls back to accumulating realized P&L, which is all that is
        possible when the trades come from *several* folds and no single simulation
        produced them (see :meth:`_aggregate_oos`).
        """
        # The two curve sources run on different clocks: the engine samples per bar,
        # the realized-P&L fallback resamples to calendar days. Annualize accordingly.
        if equity is None:
            equity = performance.build_equity_curve(trades, self.initial_capital)
            periods_per_year = TRADING_DAYS_PER_YEAR
        else:
            periods_per_year = self.timeframe.periods_per_year()
        final = self.initial_capital + (trades["pnl"].sum() if not trades.empty else 0.0)
        return performance.compute_backtest_metrics(
            trades,
            equity,
            self.initial_capital,
            final,
            {},
            start=start,
            end=end,
            periods_per_year=periods_per_year,
            n_trials=n_trials,
            var_of_trial_sr=var_of_trial_sr,
        )

    def _aggregate_oos(self, oos_trade_frames, folds, n_trials_total, var_trial_sr):
        trades = _concat_trades(oos_trade_frames)
        span_start = min((f.oos_start for f in folds), default=None)
        span_end = max((f.oos_end for f in folds), default=None)
        return self._metrics_for_trades(trades, span_start, span_end, n_trials_total, var_trial_sr)

    def _holdout(self, client, symbols, region_start, holdout, warmup_days, method, objective, max_evals):
        holdout_start, holdout_end = holdout
        # Optimize over everything before the holdout (the production training set).
        opt = ParameterOptimizer(
            self.strategy_class, client, self.initial_capital, self.seed, self.cost_model
        )
        final = self._optimize(opt, symbols, region_start, holdout_start, method, objective, max_evals)
        if not final.best_params:
            return None, None
        strategy = self.strategy_class(dict(final.best_params))
        metrics, _ = self._oos_backtest(
            strategy, client, symbols, holdout_start, holdout_end, warmup_days, n_trials=len(final.results)
        )
        return dict(final.best_params), metrics

    # ------------------------------------------------------------------ #
    # Aggregate diagnostics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_metrics(is_result) -> Dict[str, float]:
        if is_result.results.empty:
            return {}
        row = is_result.results.iloc[0]
        return {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in row.items()
            if k in performance.METRIC_KEYS
        }

    @staticmethod
    def _trial_sharpes(is_result) -> List[float]:
        if "sharpe_ratio" in is_result.results.columns:
            return [float(v) for v in is_result.results["sharpe_ratio"].to_numpy() if np.isfinite(v)]
        return []

    @staticmethod
    def _per_period_sr_variance(annualized_sharpes: List[float]) -> Optional[float]:
        """Variance of *per-period* trial Sharpes (the DSR needs per-period units)."""
        if len(annualized_sharpes) < 2:
            return None
        per_period = np.array(annualized_sharpes) / math.sqrt(TRADING_DAYS_PER_YEAR)
        return float(np.var(per_period, ddof=1))

    def _efficiency(self, fold_results, objective) -> float:
        is_objs = [fr.is_metrics.get(objective, 0.0) for fr in fold_results]
        oos_objs = [fr.oos_metrics.get(objective, 0.0) for fr in fold_results]
        is_objs = [v for v in is_objs if v not in (float("inf"), float("-inf"))]
        oos_objs = [v for v in oos_objs if v not in (float("inf"), float("-inf"))]
        mean_is = np.mean(is_objs) if is_objs else 0.0
        if mean_is == 0:
            return 0.0
        return float(np.mean(oos_objs) / mean_is)

    @staticmethod
    def _degradation(fold_results) -> Dict[str, float]:
        keys = ("sharpe_ratio", "sortino_ratio", "profit_factor", "max_drawdown", "cagr")
        out: Dict[str, float] = {}
        for key in keys:
            is_vals = [fr.is_metrics.get(key, 0.0) for fr in fold_results]
            oos_vals = [fr.oos_metrics.get(key, 0.0) for fr in fold_results]
            is_vals = [v for v in is_vals if v not in (float("inf"), float("-inf"))]
            oos_vals = [v for v in oos_vals if v not in (float("inf"), float("-inf"))]
            if is_vals and oos_vals:
                out[key] = float(np.mean(is_vals) - np.mean(oos_vals))
        return out

    # ------------------------------------------------------------------ #
    # Optional robustness diagnostics
    # ------------------------------------------------------------------ #
    def parameter_sensitivity(
        self, client, symbols, best_params, fold, warmup_days, perturbation: float = 0.10
    ) -> Dict[str, float]:
        """Perturb each searched param +-``perturbation`` and re-backtest the OOS.

        A robust optimum loses little Sharpe; a config perched on a spike is
        overfit. Returns the worst observed Sharpe loss fraction.
        """
        space = ParameterOptimizer(
            self.strategy_class, client, self.initial_capital, self.seed, self.cost_model
        ).space
        base = self._oos_sharpe(client, symbols, best_params, fold, warmup_days)
        worst_loss = 0.0
        details: Dict[str, float] = {}
        for name in space.searchable:
            spec = space.param_ranges[name]
            for direction in (1 - perturbation, 1 + perturbation):
                candidate = dict(best_params)
                value = best_params[name] * direction
                value = min(spec["max"], max(spec["min"], value))
                candidate[name] = int(round(value)) if spec["type"] == "int" else value
                sharpe = self._oos_sharpe(client, symbols, candidate, fold, warmup_days)
                loss = (base - sharpe) / abs(base) if base else 0.0
                details[f"{name}:{'+' if direction > 1 else '-'}"] = sharpe
                worst_loss = max(worst_loss, loss)
        return {"base_sharpe": base, "max_sharpe_loss": float(worst_loss), "perturbed": details}

    def _oos_sharpe(self, client, symbols, params, fold, warmup_days) -> float:
        strategy = self.strategy_class(dict(params))
        metrics, _ = self._oos_backtest(
            strategy, client, symbols, fold.oos_start, fold.oos_end, warmup_days, n_trials=1
        )
        return metrics.get("sharpe_ratio", 0.0)

    def leakage_probe(self, client, frames, symbols, best_params, fold, warmup_days) -> Dict[str, Any]:
        """Re-run the OOS with bars shifted forward; identical results => leakage.

        Operationalizes the fold-boundary leakage check: if the strategy reads future data, shifting
        the feed forward leaves results unchanged. A *clean* strategy's results
        change materially, so we **fail** when they don't.
        """
        strategy = self.strategy_class(dict(best_params))
        _, base_trades = self._oos_backtest(
            strategy, client, symbols, fold.oos_start, fold.oos_end, warmup_days, n_trials=1
        )

        shifted = {s: df.copy() for s, df in frames.items()}
        for df in shifted.values():
            # Shift OHLCV forward by 5 bars: bar t now carries data from t+5.
            for col in ("open", "high", "low", "close", "volume"):
                if col in df:
                    df[col] = df[col].shift(-5)
            df.dropna(inplace=True)
        shifted_client = MarketDataClient(_PrefetchedProvider(shifted))
        strategy2 = self.strategy_class(dict(best_params))
        _, shifted_trades = self._oos_backtest(
            strategy2, shifted_client, symbols, fold.oos_start, fold.oos_end, warmup_days, n_trials=1
        )

        same_count = len(base_trades) == len(shifted_trades)
        # If both empty we can't conclude; treat as pass (nothing traded).
        if base_trades.empty and shifted_trades.empty:
            return {"passed": True, "reason": "no OOS trades to probe"}
        identical = same_count and _pnl_close(base_trades, shifted_trades)
        return {
            "passed": not identical,
            "base_trades": int(len(base_trades)),
            "shifted_trades": int(len(shifted_trades)),
        }

    def monte_carlo(
        self, oos_trade_frames, n_resamples: int = 1000, block: int = 20, seed: int = 0
    ) -> Dict[str, float]:
        """Block-bootstrap the OOS trade sequence -> distribution of outcomes.

        Reports the 5th-percentile Sharpe ("how bad is a plausibly unlucky run").
        """
        trades = _concat_trades(oos_trade_frames)
        pnl = trades["pnl"].to_numpy() if not trades.empty else np.array([])
        if len(pnl) < block:
            return {"samples": 0}
        rng = np.random.default_rng(seed)
        sharpes, returns = [], []
        n_blocks = math.ceil(len(pnl) / block)
        for _ in range(n_resamples):
            starts = rng.integers(0, len(pnl) - block + 1, size=n_blocks)
            sample = np.concatenate([pnl[s : s + block] for s in starts])[: len(pnl)]
            std = sample.std()
            sharpes.append(math.sqrt(len(sample)) * sample.mean() / std if std > 0 else 0.0)
            returns.append(sample.sum())
        return {
            "samples": n_resamples,
            "sharpe_p05": float(np.percentile(sharpes, 5)),
            "sharpe_p50": float(np.percentile(sharpes, 50)),
            "return_p05": float(np.percentile(returns, 5)),
        }

    def estimate_pbo(self, oos_trade_frames, fold_results, objective) -> Optional[float]:
        """Probability of backtest overfitting via a simple CSCV-style estimate.

        Splits folds into IS/OOS halves over all combinations; PBO is the fraction
        of splits where the IS-best fold ranks below the OOS median. Needs >= 4
        folds to be meaningful.
        """
        from itertools import combinations

        n = len(fold_results)
        if n < 4:
            return None
        is_scores = np.array([fr.is_metrics.get(objective, 0.0) for fr in fold_results])
        oos_scores = np.array([fr.oos_metrics.get(objective, 0.0) for fr in fold_results])
        is_scores = np.nan_to_num(is_scores, posinf=1e6, neginf=-1e6)
        oos_scores = np.nan_to_num(oos_scores, posinf=1e6, neginf=-1e6)

        indices = range(n)
        below = total = 0
        for combo in combinations(indices, n // 2):
            train = list(combo)
            test = [i for i in indices if i not in combo]
            if not train or not test:
                continue
            best_train = train[int(np.argmax(is_scores[train]))]
            median_test = np.median(oos_scores[test])
            below += oos_scores[best_train] < median_test
            total += 1
        return float(below / total) if total else None


# --------------------------------------------------------------------------- #
# Trade-frame utilities
# --------------------------------------------------------------------------- #
def _filter_trades_from(trades: pd.DataFrame, oos_start: datetime) -> pd.DataFrame:
    """Keep trades whose entry is at/after ``oos_start`` (tz-aware safe)."""
    if trades.empty or "entry_time" not in trades:
        return trades
    entry = pd.to_datetime(trades["entry_time"])
    if getattr(entry.dt, "tz", None) is not None:
        entry = entry.dt.tz_localize(None)
    mask = (
        entry >= pd.Timestamp(oos_start).tz_localize(None)
        if pd.Timestamp(oos_start).tzinfo
        else entry >= pd.Timestamp(oos_start)
    )
    return trades[mask.to_numpy()].reset_index(drop=True)


def _concat_trades(frames: List[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    if "exit_time" in combined:
        combined = combined.sort_values("exit_time").reset_index(drop=True)
    return combined


def _pnl_close(a: pd.DataFrame, b: pd.DataFrame, tol: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    pa, pb = a["pnl"].to_numpy(), b["pnl"].to_numpy()
    return bool(np.allclose(np.sort(pa), np.sort(pb), atol=tol))
