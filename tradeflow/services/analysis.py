"""Analysis services: scan, backtest, optimize, walk-forward, summarize bars.

Each function takes a data-only :class:`MarketDataClient`, runs an existing
engine/optimizer/walk-forward path, and returns a compact, JSON-serializable
dict. Large outputs (trade tables, full optimization grids) are written to an
artifact file and referenced by path - never inlined.
"""

import contextlib
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from tradeflow.alphas import (
    DEFAULT_IC,
    Alpha,
    AlphaContext,
    panel_to_alphas,
    refine_alpha,
    scanner_scorer,
    signal_scorer,
    strategy_scorer,
)
from tradeflow.analytics import metrics as m
from tradeflow.analytics import performance
from tradeflow.data import (
    ClientBarSource,
    FeaturePanel,
    add_factor_exposure_features,
    add_risk_features,
    add_score_feature,
)
from tradeflow.engine.backtest import ACCOUNTING_VERSION, BacktestEngine
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.marketdata.timeframe import Timeframe
from tradeflow.optimization.optimizer import ParameterOptimizer
from tradeflow.optimization.walk_forward import WalkForwardValidator
from tradeflow.services.audit import new_run_id
from tradeflow.services.registry import resolve_strategy_class
from tradeflow.settings import state_root

logger = logging.getLogger(__name__)

#: Where trade tables / optimization grids are written.
ARTIFACT_DIR = state_root() / "logs" / "artifacts"

#: Cap on rows returned inline from an optimization (the rest go to CSV).
TOP_N = 10

#: A lagged blend whose added turnover costs more than this per year (a conservative
#: heuristic) isn't recommended — the IR uplift rarely justifies it.
_BLEND_COST_CEILING = 0.02


def _strategy(strategy_name: str, config: Optional[Dict[str, Any]] = None):
    """Instantiate a strategy from defaults, overlaid with ``config`` overrides."""
    cls = resolve_strategy_class(strategy_name)
    params = {name: spec["default"] for name, spec in cls.PARAM_RANGES.items() if "default" in spec}
    if config:
        params.update(config)
    return cls(params)


def _build_cost_model(
    gross: bool,
    commission_bps: float,
    impact_eta: float,
    participation_cap: float,
    borrow_bps: float,
):
    """Shared cost-model construction for run_backtest/run_optimization/
    run_walk_forward - so a search or validation prices trades the same way a
    live backtest does. ``None`` (i.e. ``gross=True``) reliably favors the
    highest-turnover config, so this must reach every entrypoint that can run a
    search or a validation, not just run_backtest."""
    from tradeflow.costs import ParametricCostModel

    if gross:
        return None
    return ParametricCostModel(
        commission_bps=commission_bps,
        impact_eta=impact_eta,
        participation_cap=participation_cap,
        annual_borrow_bps=borrow_bps,
    )


def limits_key(limits: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The book limits folded into a trial's dedup key, or ``{}`` when none are set.

    One definition, used by every surface. Limits are not tunable params, so before
    this they went through no identity at all and two runs differing only in
    ``max_gross_exposure`` hashed alike. Adding it to one surface and not the other
    would be worse than leaving it out of both: a trial recorded over the CLI would
    stop being found by MCP, which is the promise :func:`_find_cached_trial` makes.

    Unset limits are omitted rather than recorded as null, so a config that never
    mentioned them keys exactly as it did before this existed.
    """
    declared = {name: value for name, value in (limits or {}).items() if value is not None}
    return {"_limits": declared} if declared else {}


def walk_forward_recipe(
    *,
    mode: str,
    n_folds: Optional[int],
    train_days: Optional[int],
    test_days: Optional[int],
    embargo_days: Optional[int],
    holdout_days: int,
    method: str,
    objective: str,
    max_evals: int,
    seed: int,
    cost_key: Dict[str, Any],
    limits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A walk-forward's memoization key: the *validation recipe*, not the params.

    The chosen params are not known until the search runs, so what identifies a repeat
    here is the recipe - same window, same folds, same objective, same cost, same book.

    One definition, reached from both surfaces, because the alternative was tried: the
    book limits were folded into ``run_backtest``'s key and not into this one, so two
    validations differing only in ``max_positions`` hashed alike and the second was
    answered from the first - reporting a one-position validation as an eight-position
    book. That is the failure a walk-forward exists to rule out, so the fold belongs
    here rather than at each call site where it can be forgotten again.
    """
    return {
        "mode": mode,
        "n_folds": n_folds,
        "train_days": train_days,
        "test_days": test_days,
        "embargo_days": embargo_days,
        "holdout_days": holdout_days,
        "method": method,
        "objective": objective,
        "max_evals": max_evals,
        "seed": seed,
        "_cost": cost_key,
        **limits_key(limits),
    }


def _cost_key(gross: bool, commission_bps: float, impact_eta: float, borrow_bps: float) -> Dict[str, Any]:
    """The cost-model assumptions folded into a trial's dedup key - same shape as
    the CLI's ``main._cost_key`` so a trial recorded via one surface is found by
    the other. Two runs differing only in a cost flag must never collide as "the
    same trial"."""
    if gross:
        return {"gross": True}
    return {
        "gross": False,
        "commission_bps": commission_bps,
        "impact_eta": impact_eta,
        "borrow_bps": borrow_bps,
    }


def _tunable_params(strategy) -> Dict[str, Any]:
    """The strategy's own tunable knobs (its ``config`` narrowed to
    ``PARAM_RANGES`` keys) - what identifies a trial, not the incidental config
    keys (timeframe, lookback, position limits) a strategy also carries."""
    return {k: strategy.config[k] for k in strategy.PARAM_RANGES if k in strategy.config}


@contextlib.contextmanager
def _open_trial_store():
    """A trial store against the current journal, or ``None`` on any failure to
    open one - same fail-safe contract as the CLI's ``main._open_trial_store``.
    v1 of the trial store is passive and derived: a broken store must never
    break the command it's attached to, memoization included."""
    from tradeflow.services import audit
    from tradeflow.store.trials import TrialStore, db_path_for_journal

    journal_path = audit.DEFAULT_TRIAL_JOURNAL
    try:
        store = TrialStore(db_path_for_journal(journal_path), journal_path=journal_path)
    except Exception:  # noqa: BLE001
        logger.warning("Trial store unavailable; memoization disabled for this run", exc_info=True)
        yield None
        return
    try:
        yield store
    finally:
        store.close()


def _find_cached_trial(
    strategy: str,
    params: Dict[str, Any],
    symbols,
    start,
    end,
    accounting: int,
    require_trades: bool = False,
) -> Optional[Dict[str, Any]]:
    """Same lookup CLI's ``main._find_cached_trial`` does, so a trial run over
    MCP and one run over the CLI dedup against each other identically."""
    from tradeflow.optimization.config_store import current_git_sha

    with _open_trial_store() as store:
        if store is None:
            return None
        return store.find(
            strategy=strategy,
            params=params,
            symbols=symbols,
            window_start=start,
            window_end=end,
            accounting=accounting,
            require_trades=require_trades,
            git_sha=current_git_sha(),
        )


def run_scan(
    data_client: MarketDataClient,
    scanner: str,
    symbols: List[str],
    config: Optional[Dict[str, Any]] = None,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Run a universe scanner; return the flagged ``(symbol, signal)`` pairs."""
    from tradeflow.scanners.symbol_scanner import SymbolScanner, resolve_scan_clock

    flagged = SymbolScanner(data_client, scanner, config).scan(symbols, as_of=as_of)
    return {
        "scanner": scanner,
        "candidates": list(symbols),
        # The clock the scan actually resolved at, not the argument that asked for it.
        "as_of": resolve_scan_clock(as_of).isoformat(),
        "flagged": [{"symbol": s, "signal": sig} for s, sig in flagged],
        "flagged_count": len(flagged),
    }


def run_backtest(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    capital: float = 100_000.0,
    config: Optional[Dict[str, Any]] = None,
    beta_sizing: bool = False,
    benchmark: str = "SPY",
    gross: bool = False,
    take_profit_margin_bps: float = 0.0,
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
    force: bool = False,
    journal: bool = True,
) -> Dict[str, Any]:
    """Backtest a strategy; return the full metrics dict + a path to the trades CSV.

    Metrics are **net of transaction cost** by default (commission + half-spread +
    square-root impact); pass ``gross=True`` to disable cost for attribution. Trades
    are NOT inlined (could be thousands of rows); read the CSV if needed.

    Journals one trial into the research journal / trial store (the same one
    ``python main.py backtest`` writes) so this run counts toward the campaign's
    multiple-testing total — an agent driving this over MCP is still bound by the
    same Deflated Sharpe honesty as the CLI. Before running, an exact prior trial
    is served instead (labeled ``memoized``) unless ``force=True``, which re-runs
    and appends a new trial rather than overwriting.
    """
    from tradeflow.services.sizing import build_beta_sizer

    run_id = new_run_id()
    strat = _strategy(strategy, config)
    dedup_params = {
        **_tunable_params(strat),
        "_cost": _cost_key(gross, commission_bps, impact_eta, borrow_bps),
        **limits_key(strat.position_limits()),
    }

    if not force:
        cached = _find_cached_trial(
            strategy, dedup_params, symbols, start, end, ACCOUNTING_VERSION, require_trades=True
        )
        if cached is not None:
            metrics = json.loads(cached["metrics_json"] or "{}")
            return {
                "run_id": run_id,
                "strategy": strategy,
                "symbols": list(symbols),
                "window": {"start": start.isoformat(), "end": end.isoformat()},
                "memoized": True,
                "trial_id": cached["id"],
                "trial_ts": cached["ts"],
                "note": "Served from an identical prior trial, not re-run. Pass force=True to re-verify.",
                "metrics": _jsonable(metrics),
            }

    sizer = build_beta_sizer(data_client, strat, symbols, benchmark, as_of=start) if beta_sizing else None
    cost_model = _build_cost_model(gross, commission_bps, impact_eta, participation_cap, borrow_bps)
    result = BacktestEngine(
        strat,
        data_client,
        sizer=sizer,
        cost_model=cost_model,
        take_profit_margin_bps=take_profit_margin_bps,
    ).run(symbols, start, end, capital, benchmark=benchmark)

    # `journal=False` exists for re-runs of a config that is already a candidate -
    # a cost-stress curve is one candidate under several stated assumptions, and
    # recording each point would inflate the multiple-testing total the deflated
    # Sharpe deflates against. It must never be the default: a run that quietly does
    # not count is how a campaign loses track of what it tried.
    if journal:
        from tradeflow.services.audit import journal_trial

        journal_trial(
            "backtest",
            strategy=strategy,
            symbols=symbols,
            start=start,
            end=end,
            params=dedup_params,
            metrics=result.metrics,
        )

    trades_csv = None
    if not result.trades.empty:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        trades_csv = str(ARTIFACT_DIR / f"backtest_{run_id}.csv")
        result.trades.to_csv(trades_csv, index=False)

    return backtest_payload(
        result,
        run_id=run_id,
        strategy=strategy,
        symbols=symbols,
        start=start,
        end=end,
        capital=capital,
        gross=gross,
        benchmark=benchmark,
        trades_csv=trades_csv,
    )


def trades_payload(frame, *, max_rows: int = 5000) -> Optional[Dict[str, Any]]:
    """A trade DataFrame as the ``{columns, rows}`` payload the trial store keeps.

    ``None`` when there is no frame at all - distinct from a run that genuinely
    made no trades, which is an empty ``rows`` list.

    ``max_rows`` is a deliberate ceiling on what one trial may store, and hitting
    it is *recorded* (``truncated``/``total_rows``), never silent: a truncated
    table that looks complete is worse than no table.
    """
    if frame is None:
        return None
    total = int(len(frame))
    kept = frame.head(max_rows) if total > max_rows else frame
    payload = {
        "columns": [str(c) for c in kept.columns],
        "rows": _jsonable(kept.values.tolist()),
        "total_rows": total,
    }
    if total > max_rows:
        payload["truncated"] = True
    return payload


def backtest_payload(
    result,
    *,
    run_id: str,
    strategy: str,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    capital: float,
    gross: bool,
    benchmark: Optional[str] = None,
    trades_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """A ``BacktestResult`` as the JSON-serializable dict every surface consumes.

    One shape, one place: the CLI holds the result *object* and the service returns
    a *dict*, and before this they built that dict twice. Anything that renders a
    backtest - the terminal, an HTML report, an agent over MCP - now sees exactly
    the same fields. Trades stay out of it (thousands of rows) and go to the CSV
    path instead.
    """
    return {
        "run_id": run_id,
        "strategy": strategy,
        "symbols": list(symbols),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_capital": capital,
        "final_capital": result.final_capital,
        # Named here so a report cannot show "Benchmark —" beside metrics that were
        # in fact scored against one, or the reverse.
        "benchmark": benchmark if result.metrics.get("benchmark_available") else None,
        "gross": gross,
        "total_cost": result.total_cost,
        "gross_final_capital": result.gross_final_capital,
        "cost_drag_pct": (result.total_cost / capital * 100.0) if capital else 0.0,
        "metrics": _jsonable(result.metrics),
        # The gap between the intended book and the tradeable one, plus the verdict on
        # it - separate from `metrics` because it judges executability at this capital,
        # not whether the edge was real.
        "execution": _jsonable(getattr(result, "execution", {}) or {}),
        "executability": _jsonable(performance.execution_verdict(getattr(result, "execution", None))),
        "total_trades": int(len(result.trades)),
        "trades_csv": trades_csv,
        "resolved_config": _jsonable(result.strategy_config),
    }


def _worker_data_spec(workers: Optional[int], cache_dir: Optional[Any], offline: bool):
    """How worker processes should build their own data clients, or ``None`` when
    the run is sequential.

    Parallel execution is **cache-backed by construction**: a live client cannot
    cross a process boundary, and N workers independently fetching the same bars
    from the vendor is strictly worse than one warmed local cache. So asking for
    workers implies the bar cache, whether or not the caller passed ``--cache``.
    """
    from tradeflow.optimization.parallel import DataSpec, resolve_workers

    if resolve_workers(workers) <= 1:
        return None
    return DataSpec(kind="cache", cache_dir=str(cache_dir) if cache_dir else None, offline=offline)


def run_optimization(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    method: str = "grid",
    objective: str = "sharpe_ratio",
    max_evals: int = 50,
    seed: int = 42,
    capital: float = 100_000.0,
    gross: bool = False,
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
    force: bool = False,
    workers: Optional[int] = None,
    cache_dir: Optional[Any] = None,
    offline: bool = False,
) -> Dict[str, Any]:
    """Search a strategy's parameters IN-SAMPLE; return best params + top-N rows.

    WARNING for the caller: these are in-sample results from selecting the best of
    many configs - NOT evidence of edge. Validate with ``run_walk_forward`` before
    trusting any of this; ``best_score`` will almost always look good here.

    Net of transaction cost by default (commission + half-spread + square-root
    impact); pass ``gross=True`` to search gross returns instead - gross search
    reliably favors the highest-turnover config.

    Each evaluated config is journaled as its own trial (a 50-point search is 50
    trials, matching ``python main.py optimize``), unless it's served from the
    trial store first (an identical prior candidate - real with random sampling
    or a resumed search) - ``force=True`` disables that per-candidate memoization.

    ``workers`` (default sequential) evaluates candidates across that many worker
    processes. Only *execution* parallelizes: memoization is still resolved here
    before dispatch, and every journal write still happens in this process after the
    results come back, so the campaign's trial count is unaffected by how the work
    was scheduled. A parallel run and a sequential run of the same search produce
    the same trials and the same winner.
    """
    from tradeflow.services.audit import journal_trial

    run_id = new_run_id()
    cls = resolve_strategy_class(strategy)
    cost_model = _build_cost_model(gross, commission_bps, impact_eta, participation_cap, borrow_bps)
    cost_key = _cost_key(gross, commission_bps, impact_eta, borrow_bps)
    data_spec = _worker_data_spec(workers, cache_dir, offline)
    if data_spec is not None:
        # Warm once, in this process, before any dispatch: N cold workers would
        # otherwise request the same ranges simultaneously, multiplying the API
        # calls (and the rate-limit exposure) by N for one set of bars.
        from tradeflow.optimization.parallel import warm_for

        timeframe = cls.create_with_defaults().config.get("timeframe", "1Day")
        warm_for(data_spec, symbols, timeframe, start, end)
    with _open_trial_store() as trial_store:
        opt = ParameterOptimizer(
            cls,
            data_client,
            initial_capital=capital,
            seed=seed,
            cost_model=cost_model,
            trial_store=trial_store,
            strategy_name=strategy,
            cost_key=cost_key,
            accounting=ACCOUNTING_VERSION,
            force=force,
            workers=workers,
            data_spec=data_spec,
        )
        if method == "grid":
            result = opt.grid_search(symbols, start, end, objective, max_evals=max_evals)
        elif method == "random":
            result = opt.random_search(symbols, start, end, objective, n_samples=max_evals)
        else:
            result = opt.optimize_bayesian(symbols, start, end, objective)

    results_csv = None
    total = len(result.results)
    top: List[Dict[str, Any]] = []
    n_memoized = 0
    if not result.results.empty:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        results_csv = str(ARTIFACT_DIR / f"optimize_{run_id}.csv")
        result.results.to_csv(results_csv, index=False)
        top = [_jsonable(row) for row in result.results.head(TOP_N).to_dict("records")]

        searchable = opt.space.searchable
        defaults = opt.space.defaults
        for row in result.results.to_dict("records"):
            if "_memoized_from" in row:
                # Already exists as its own trial; re-journaling it would double-count
                # the exact repeat this spec exists to stop.
                n_memoized += 1
                continue
            searched = {k: row[k] for k in searchable if k in row}
            metrics = {k: v for k, v in row.items() if k not in searchable}
            journal_trial(
                "optimize",
                strategy=strategy,
                symbols=symbols,
                start=start,
                end=end,
                params={**defaults, **searched, "_cost": cost_key},
                metrics=metrics,
                objective=objective,
            )

    return {
        "run_id": run_id,
        "strategy": strategy,
        "method": method,
        "objective": objective,
        "symbols": list(symbols),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "best_params": _jsonable(result.best_params),
        "best_score": result.best_score,
        "n_trials": total,
        "n_memoized": n_memoized,
        "top": top,
        "truncated": max(total - len(top), 0),
        "results_csv": results_csv,
        "seed": seed,
        "gross": gross,
        "note": "IN-SAMPLE. Selecting the best of many configs inflates these. "
        "Validate out-of-sample with run_walk_forward (it applies the Deflated Sharpe).",
    }


#: A screen never writes a trial. Stated on the payload rather than only in prose,
#: because the whole point of the command is that asking a cheap question stays cheap:
#: every journaled trial raises the deflated-Sharpe bar for its family permanently, and
#: a researcher who cannot afford to ask will either stop asking or stop looking.
SCREEN_JOURNALING_NOTE = (
    "Reconnaissance, not evidence: nothing here was journaled, so none of it counts "
    "toward the campaign's multiple-testing total and none of it is promotable. "
    "Confirm a single point to turn it into a recorded trial."
)


def screen_ranges(strategy_class, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A strategy's declared ranges with per-parameter narrowing applied.

    An override supplies any of ``min``/``max``/``step`` and inherits the rest, so
    narrowing one axis cannot silently change a parameter's type or drop its default.
    Unknown names are refused rather than ignored: a typo that quietly screens the
    full range looks exactly like a screen that found nothing where you looked.
    """
    ranges = {name: dict(spec) for name, spec in (strategy_class.PARAM_RANGES or {}).items()}
    for name, override in (overrides or {}).items():
        if name not in ranges:
            raise ValueError(f"{strategy_class.__name__} declares no parameter {name!r} to narrow")
        unknown = set(override) - {"min", "max", "step"}
        if unknown:
            raise ValueError(f"Cannot override {sorted(unknown)} for {name!r}; only min/max/step")
        ranges[name] = {**ranges[name], **override}
        if ranges[name]["min"] > ranges[name]["max"]:
            raise ValueError(f"Narrowed range for {name!r} is empty: min {ranges[name]['min']} > max")
    return ranges


def run_screen(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    *,
    method: str = "grid",
    objective: str = "sharpe_ratio",
    max_evals: int = 50,
    seed: int = 42,
    capital: float = 100_000.0,
    gross: bool = False,
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
    position_limits: Optional[Dict[str, Any]] = None,
    param_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Sweep a parameter space cheaply, and report the distribution rather than a winner.

    **Nothing here is journaled.** A screen is reconnaissance: it answers "is there
    anything in this family at all" without spending statistical budget, because every
    journaled trial raises the deflated-Sharpe bar for its ``(strategy, universe,
    accounting)`` family permanently. Use ``confirm_screen_point`` to turn exactly one
    selected point into a recorded trial.

    One process, one data fetch, N evaluations: the window is fetched once and every
    candidate is scored against the same in-memory frames.

    The result leads with the distribution - n, median, spread, positive rate - and any
    best point is reported beside what the best of that many draws is worth under the
    null. A leaderboard without that is the selection bias the deflated Sharpe exists
    to prevent, one layer up. Per-parameter gradients are included for the same reason:
    a positive rate that moves coherently with a parameter is different evidence from
    the same count of positive points scattered at random, and it can point off the
    edge of the searched space, which a best-point report never does.
    """
    from tradeflow.analytics import screening
    from tradeflow.optimization.param_space import ParameterSpace
    from tradeflow.optimization.walk_forward import PrefetchedProvider

    run_id = new_run_id()
    cls = resolve_strategy_class(strategy)
    space = ParameterSpace(screen_ranges(cls, param_ranges), getattr(cls, "PARAM_CONSTRAINTS", ()) or ())
    cost_model = _build_cost_model(gross, commission_bps, impact_eta, participation_cap, borrow_bps)

    # Fetched once, then served from memory to every evaluation. A screen that refetched
    # per candidate would be the thing it exists to replace: a shell loop over N runs.
    timeframe = cls.create_with_defaults().config.get("timeframe", "1Day")
    frames = data_client.get_bars(list(symbols), Timeframe.parse(timeframe), start, end)
    sliced = MarketDataClient(PrefetchedProvider(frames))

    optimizer = ParameterOptimizer(
        cls,
        sliced,
        initial_capital=capital,
        seed=seed,
        cost_model=cost_model,
        # No trial store, deliberately: a screen neither records trials nor is served
        # one. Memoizing against journaled evidence would make some points cheap and
        # others not, and put a "reused" caveat on a sweep that is not evidence anyway.
        trial_store=None,
        workers=workers,
        space=space,
        position_limits=position_limits,
    )

    grid_total = space.grid_size()
    if method == "random":
        result = optimizer.random_search(list(symbols), start, end, objective, n_samples=max_evals)
        requested = max_evals
    else:
        result = optimizer.grid_search(list(symbols), start, end, objective, max_evals=max_evals)
        requested = min(max_evals, grid_total) if max_evals else grid_total

    # Analytics run on the raw records, rendering on the JSON-safe copy. `_jsonable`
    # turns a non-finite float into the *string* "-inf" so a payload can round-trip,
    # and a failed evaluation reaching the distribution as text rather than as a
    # dropped value is exactly the silent miscount the summary exists to prevent.
    raw_rows = result.results.to_dict("records") if not result.results.empty else []
    rows = [_jsonable(row) for row in raw_rows]
    scores = [row[objective] for row in raw_rows if row.get(objective) is not None]

    results_csv = None
    if rows:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        results_csv = str(ARTIFACT_DIR / f"screen_{run_id}.csv")
        result.results.to_csv(results_csv, index=False)

    distribution = screening.score_distribution(scores)
    baseline = screening.noise_baseline(scores, objective)
    best_raw = (
        max(raw_rows, key=lambda r: r.get(objective, float("-inf")), default=None) if raw_rows else None
    )
    best_row = _jsonable(best_raw) if best_raw is not None else None

    return {
        "run_id": run_id,
        "strategy": strategy,
        "method": method,
        "objective": objective,
        "symbols": list(symbols),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "journaled": False,
        "note": SCREEN_JOURNALING_NOTE,
        "searched": {
            "parameters": list(space.searchable),
            "constraints": list(space.constraints.describe()),
            "grid_size": grid_total,
            "unconstrained_grid_size": space.unconstrained_grid_size(),
            "requested": requested,
            "evaluated": len(rows),
            # Never silent: a sweep that covered a fraction of its grid and said
            # nothing reads as a sweep that covered all of it.
            "sampled_from_grid": bool(method == "grid" and grid_total > requested),
        },
        "distribution": distribution,
        "noise_baseline": baseline,
        "best_point": best_row,
        "gradients": screening.gradients(raw_rows, list(space.searchable), objective),
        "position_limits": dict(position_limits) if position_limits else None,
        "results_csv": results_csv,
        "seed": seed,
        "gross": gross,
    }


def confirm_screen_point(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    params: Dict[str, Any],
    *,
    capital: float = 100_000.0,
    gross: bool = False,
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
    position_limits: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Re-run **exactly one** screened point as a proper, journaled trial.

    Exactly one is the constraint that matters. A confirm that could take a set would
    be a screen that journals, which reintroduces through the back door the budget
    problem the screen exists to solve - and it would journal the *best* of N, which is
    the one selection a sweep cannot support.

    Delegates to :func:`run_backtest`, so a confirmed point is indistinguishable from
    the same backtest run directly: same dedup identity, same memoization, same journal
    record. A separate journaling path here would be a second definition of what a
    trial is.
    """
    config = dict(params)
    if position_limits:
        config["position_limits"] = dict(position_limits)
    result = run_backtest(
        data_client,
        strategy,
        symbols,
        start,
        end,
        capital=capital,
        config=config,
        gross=gross,
        commission_bps=commission_bps,
        impact_eta=impact_eta,
        participation_cap=participation_cap,
        borrow_bps=borrow_bps,
        force=force,
        journal=True,
    )
    return {**result, "confirmed_params": _jsonable(params), "journaled": True}


def run_walk_forward(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    mode: str = "anchored",
    # ``None``, not 4, and the same default the CLI has always had. Both resolve to
    # four folds - ``build_folds`` falls back to ``n_folds or 4`` - so this is a
    # *keying* question, not a validation one: the recipe recorded ``None`` from one
    # surface and ``4`` from the other for the identical validation, so a walk-forward
    # run over the CLI was not found again over MCP. ``None`` is also the honest value
    # when ``train_days``/``test_days`` are given, where the fold count is derived and
    # this parameter has no effect at all. Aligning on ``None`` rather than ``4`` keeps
    # every walk-forward already in the store findable: a memoization miss is not free,
    # it journals a fresh trial and permanently raises the deflation bar for the family.
    n_folds: Optional[int] = None,
    train_days: Optional[int] = None,
    test_days: Optional[int] = None,
    embargo_days: Optional[int] = None,
    holdout_days: int = 0,
    method: str = "grid",
    objective: str = "sharpe_ratio",
    max_evals: int = 50,
    seed: int = 42,
    capital: float = 100_000.0,
    include_pbo: bool = False,
    include_monte_carlo: bool = False,
    parameter_sensitivity: bool = False,
    leakage_probe: bool = False,
    gates: Optional[Dict[str, float]] = None,
    n_trials_offset: int = 0,
    gross: bool = False,
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
    force: bool = False,
    workers: Optional[int] = None,
    cache_dir: Optional[Any] = None,
    offline: bool = False,
    benchmark: Optional[str] = None,
    position_limits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Honest evaluation: optimize IS, score OOS across folds, gate the verdict.

    This is the advancement criterion - returns the OOS aggregate, efficiency,
    degradation, per-fold summary, holdout (if requested), the Deflated Sharpe
    (with n_trials across all folds), and the promotion-gate pass/fail + overall
    ``promotable``. ``include_pbo`` is expensive and defaults off.

    Net of transaction cost by default, in-sample and out - pass ``gross=True``
    to validate gross returns instead, which systematically promotes turnover
    the strategy could not afford live.

    Journals one *validated* trial (the OOS aggregate — matching
    ``python main.py walkforward``), unless an identical prior validation is
    served instead (same recipe: mode/folds/method/objective/max_evals/seed/cost
    over the same window — the chosen params aren't known until the search runs,
    so those, not params, are what identifies a repeat here). ``force=True``
    bypasses that and re-runs, appending a new trial.

    ``benchmark`` and ``position_limits`` are the same two the CLI passes. Without
    the first, every fold reports ``benchmark_available: False`` and the
    benchmark-relative promotion prerequisites are never evaluated - silently, which
    is the worst way for a gate not to run. Without the second, a config asking for
    eight positions is validated at whatever the strategy class declares, so the
    validated book and the deployed one are different books.

    ``workers`` parallelizes each fold's in-sample candidate search — folds
    themselves stay sequential, since the candidates are where the work is and the
    per-fold progress stays readable. Journaling is unaffected: this process still
    records exactly one validated trial when the whole run finishes.
    """
    from tradeflow.optimization.config_store import current_git_sha
    from tradeflow.services.audit import journal_trial

    run_id = new_run_id()
    cls = resolve_strategy_class(strategy)
    cost_model = _build_cost_model(gross, commission_bps, impact_eta, participation_cap, borrow_bps)
    cost_key = _cost_key(gross, commission_bps, impact_eta, borrow_bps)
    wf_data_spec = _worker_data_spec(workers, cache_dir, offline)
    if wf_data_spec is not None:
        from tradeflow.optimization.parallel import warm_for

        timeframe = cls.create_with_defaults().config.get("timeframe", "1Day")
        warm_for(wf_data_spec, symbols, timeframe, start, end)
    recipe = walk_forward_recipe(
        mode=mode,
        n_folds=n_folds,
        train_days=train_days,
        test_days=test_days,
        embargo_days=embargo_days,
        holdout_days=holdout_days,
        method=method,
        objective=objective,
        max_evals=max_evals,
        seed=seed,
        cost_key=cost_key,
        limits=position_limits,
    )

    with _open_trial_store() as trial_store:
        if not force and trial_store is not None:
            cached = trial_store.find(
                strategy=strategy,
                params=recipe,
                symbols=symbols,
                window_start=start,
                window_end=end,
                accounting=ACCOUNTING_VERSION,
                git_sha=current_git_sha(),
            )
            if cached is not None:
                metrics = json.loads(cached["metrics_json"] or "{}")
                return {
                    "run_id": run_id,
                    "strategy": strategy,
                    "symbols": list(symbols),
                    "window": {"start": start.isoformat(), "end": end.isoformat()},
                    "memoized": True,
                    "trial_id": cached["id"],
                    "trial_ts": cached["ts"],
                    "note": "Served from an identical prior validation (same recipe), not re-run. "
                    "Pass force=True to re-verify — per-fold detail isn't retained, only the "
                    "OOS aggregate below.",
                    "oos_aggregate": _jsonable(metrics),
                    "promotable": bool(cached["promotable"]) if cached["promotable"] is not None else None,
                    "efficiency": cached["efficiency"],
                    "n_trials_total": cached["n_trials_in_session"],
                }

        validator = WalkForwardValidator(
            cls,
            data_client,
            initial_capital=capital,
            seed=seed,
            gates=gates,
            cost_model=cost_model,
            trial_store=trial_store,
            strategy_name=strategy,
            cost_key=cost_key,
            accounting=ACCOUNTING_VERSION,
            force=force,
            workers=workers,
            data_spec=wf_data_spec,
            benchmark=benchmark,
            position_limits=position_limits,
        )
        result = validator.run(
            symbols,
            start,
            end,
            mode=mode,
            n_folds=n_folds,
            train_days=train_days,
            test_days=test_days,
            embargo_days=embargo_days,
            holdout_days=holdout_days,
            method=method,
            objective=objective,
            max_evals=max_evals,
            pbo=include_pbo,
            monte_carlo=include_monte_carlo,
            parameter_sensitivity=parameter_sensitivity,
            leakage_probe=leakage_probe,
            n_trials_offset=n_trials_offset,
        )

    if result.folds:
        chosen = result.holdout_params or result.folds[-1].is_best_params
        gate_report = result.gate_report(gates)
        journal_trial(
            "walkforward",
            strategy=strategy,
            symbols=symbols,
            start=start,
            end=end,
            params=dict(chosen),
            metrics=result.oos_aggregate,
            objective=objective,
            extra={
                "n_trials": result.n_trials_total,
                "promotable": gate_report["promotable"],
                "efficiency": result.median_efficiency(),
            },
            returns=result.oos_returns,
            dedup_params=recipe,
        )

    return walk_forward_payload(
        result,
        run_id=run_id,
        strategy=strategy,
        symbols=symbols,
        start=start,
        end=end,
        mode=mode,
        objective=objective,
        method=method,
        gross=gross,
        seed=seed,
        gates=gates,
    )


def _draft_rejection(run_id: str, kind: str, code: str, exc: Exception, *, hygiene: bool) -> Dict[str, Any]:
    """The one shape every draft entry point returns when it cannot proceed.

    ``error_kind`` separates the two cases rather than flattening them into "invalid",
    because they call for opposite responses and only one of them is about the draft:

    * ``invalid_draft`` - the source was rejected. Rewrite it.
    * ``validator_error`` - the validator itself failed on this input. The draft may
      be fine; this is a defect here, and reporting it as invalid code would send an
      agent rewriting something that was never the problem.

    Neither consumes a trial, and the note says so: a caller tracking its own
    multiple-testing budget cannot tell that from an absent metrics block.
    """
    return {
        "run_id": run_id,
        "valid": False,
        "kind": kind,
        "error": str(exc),
        "error_kind": "invalid_draft" if hygiene else "validator_error",
        "code_hash": draft_code_hash(code),
        "note": "Nothing was run and no trial was journaled.",
    }


def validate_draft_strategy_code(code: str, class_name: Optional[str] = None) -> Dict[str, Any]:
    """Validate generated/private strategy source without registering or running it.

    Answers with a verdict, never an exception: a tool whose only job is to say
    whether code is valid has failed if invalid code makes it raise.
    """
    from tradeflow.research.sandbox import HygieneError, load_strategy_from_code

    run_id = new_run_id()
    try:
        cls = load_strategy_from_code(code, class_name=class_name)
    except HygieneError as exc:
        return _draft_rejection(run_id, "strategy", code, exc, hygiene=True)
    except Exception as exc:  # noqa: BLE001 - a validator that crashes answers nothing
        logger.warning("Draft strategy validation failed unexpectedly", exc_info=True)
        return _draft_rejection(run_id, "strategy", code, exc, hygiene=False)
    return {
        "run_id": run_id,
        "valid": True,
        "kind": "strategy",
        "class_name": cls.__name__,
        "description": (cls.__doc__ or "").strip().split("\n", 1)[0],
        "timeframe": getattr(cls, "TIMEFRAME", ""),
        "param_ranges": _jsonable(cls.PARAM_RANGES),
        "code_hash": draft_code_hash(code),
    }


def validate_draft_scanner_code(code: str, class_name: Optional[str] = None) -> Dict[str, Any]:
    """Validate generated/private scanner source without registering or running it.

    Answers with a verdict, never an exception - see
    :func:`validate_draft_strategy_code`.
    """
    from tradeflow.research.sandbox import HygieneError, load_scanner_from_code

    run_id = new_run_id()
    try:
        cls = load_scanner_from_code(code, class_name=class_name)
    except HygieneError as exc:
        return _draft_rejection(run_id, "scanner", code, exc, hygiene=True)
    except Exception as exc:  # noqa: BLE001 - a validator that crashes answers nothing
        logger.warning("Draft scanner validation failed unexpectedly", exc_info=True)
        return _draft_rejection(run_id, "scanner", code, exc, hygiene=False)
    return {
        "run_id": run_id,
        "valid": True,
        "kind": "scanner",
        "class_name": cls.__name__,
        "description": (cls.__doc__ or "").strip().split("\n", 1)[0],
        "timeframe": getattr(cls, "TIMEFRAME", ""),
        "param_ranges": _jsonable(cls.PARAM_RANGES),
        "code_hash": draft_code_hash(code),
    }


#: Cost multiples the stress curve walks. 1x is the run's own assumptions, so the
#: curve always contains its own baseline and the reader can see the slope rather than
#: a single pass/fail. Points rather than one 3x check because *where* an edge dies is
#: the useful part: a strategy that survives 5x and one that dies just past 1x are both
#: "passes at 1x" and are not the same proposition.
DEFAULT_COST_STRESS_MULTIPLES = (1.0, 2.0, 3.0, 5.0)


def run_scanner_drift(
    data_client: MarketDataClient,
    scanner: str,
    symbols: List[str],
    as_of: datetime,
    *,
    config: Optional[Dict[str, Any]] = None,
    offsets_days: Sequence[int] = (-1, -2, -5),
    saved_universe: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """How much the selected universe moves when the scan clock moves.

    A universe that turns over 40% across one session is a different object from one
    that is stable, and nothing said which you had. That matters more than it looks: a
    validated config carries the universe its scanner *resolved*, so if the scan is
    unstable the book you get on the day you deploy is not the book that was validated,
    and no promotion gate would notice.

    Needs no new machinery - `--as-of` already resolves a scanner at an arbitrary
    historical clock, so this is that seam asked several times and differenced.

    ``saved_universe`` compares against what a config recorded rather than against
    another scan, which is the question a deployment actually has: *is the file still
    describing today's universe?*
    """
    from tradeflow.scanners.symbol_scanner import resolve_scan_clock

    run_id = new_run_id()
    baseline = {symbol for symbol, _ in _scan_at(data_client, scanner, symbols, config, as_of)}

    comparisons: List[Dict[str, Any]] = []
    for offset in offsets_days:
        when = as_of + timedelta(days=offset)
        other = {symbol for symbol, _ in _scan_at(data_client, scanner, symbols, config, when)}
        comparisons.append(
            {
                "offset_days": int(offset),
                "as_of": resolve_scan_clock(when).isoformat(),
                "size": len(other),
                **_universe_overlap(baseline, other),
            }
        )

    saved = None
    if saved_universe is not None:
        saved = {
            "size": len(set(saved_universe)),
            **_universe_overlap(set(saved_universe), baseline),
        }

    drifts = [c["turnover_pct"] for c in comparisons]
    return {
        "run_id": run_id,
        "scanner": scanner,
        "as_of": resolve_scan_clock(as_of).isoformat(),
        "candidates": len(symbols),
        "baseline_size": len(baseline),
        "comparisons": comparisons,
        "saved_vs_current": saved,
        "max_turnover_pct": max(drifts) if drifts else 0.0,
        "note": "Turnover is the share of the baseline universe that changes at each "
        "clock. A config records the universe its scanner resolved, so drift is the "
        "gap between the book that was validated and the one a deployment would get.",
    }


def _scan_at(data_client, scanner: str, symbols, config, when):
    from tradeflow.scanners.symbol_scanner import SymbolScanner

    return SymbolScanner(data_client, scanner, config).scan(list(symbols), as_of=when)


def _universe_overlap(left: set, right: set) -> Dict[str, Any]:
    """Symmetric difference between two universes, as counts and a turnover share.

    Turnover is measured against ``left`` (the reference), so "40% turnover" reads as
    "40% of the universe I am comparing from is not in the other one" rather than as an
    unanchored symmetric statistic.
    """
    added, dropped = sorted(right - left), sorted(left - right)
    reference = len(left)
    turnover = float((len(added) + len(dropped)) / reference * 100.0) if reference else 0.0
    return {
        "overlap": len(left & right),
        "added": added,
        "dropped": dropped,
        "turnover_pct": turnover,
    }


#: Basis points a bar must trade *through* a take-profit before it counts as filled.
#: Zero is the historical assumption; the rest ask what the edge is worth without it.
DEFAULT_FILL_STRESS_MARGINS: Sequence[float] = (0.0, 5.0, 10.0, 25.0, 50.0)


def run_fill_stress(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    *,
    config: Optional[Dict[str, Any]] = None,
    capital: float = 100_000.0,
    benchmark: str = "SPY",
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    borrow_bps: float = 50.0,
    margins: Sequence[float] = DEFAULT_FILL_STRESS_MARGINS,
) -> Dict[str, Any]:
    """Re-run one config requiring the price to trade progressively further *through*
    each take-profit before it counts as filled.

    At zero - the historical assumption - a bar that merely touched the target filled
    at it, which models a resting limit order that is always first in the queue. For a
    strategy whose edge is concentrated in target exits that assumption is not a detail;
    it is the result. This makes it a number that can be moved rather than one that can
    only be believed.

    The trigger tightens; the fill price does not. A limit order that fills, fills at
    its limit - the question is whether it filled at all, not whether it filled worse.

    Journals nothing: the same config under stated assumptions, not new candidates.
    """
    run_id = new_run_id()
    points: List[Dict[str, Any]] = []
    for margin in margins:
        result = run_backtest(
            data_client,
            strategy,
            symbols,
            start,
            end,
            config=config,
            capital=capital,
            benchmark=benchmark,
            commission_bps=commission_bps,
            impact_eta=impact_eta,
            borrow_bps=borrow_bps,
            take_profit_margin_bps=margin,
            journal=False,
            force=True,
        )
        metrics = result.get("metrics") or {}
        points.append(
            {
                "margin_bps": float(margin),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "total_return": metrics.get("total_return", 0.0),
                # `total_trades`, and None rather than 0 when it is missing. Read from
                # the wrong key it reported 0 on every row of a 1952-trade run, and a
                # zero default is what made a wrong key look like a real answer.
                "trades": metrics.get("total_trades"),
            }
        )

    survives = [p["margin_bps"] for p in points if p["sharpe_ratio"] > 0 and p["total_return"] > 0]
    return {
        "run_id": run_id,
        "strategy": strategy,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "points": points,
        "base_sharpe": points[0].get("sharpe_ratio") if points else None,
        # The widest margin the edge still clears at, or None if it dies immediately.
        "survives_to_bps": max(survives) if survives else None,
    }


def run_cost_stress(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    *,
    config: Optional[Dict[str, Any]] = None,
    capital: float = 100_000.0,
    benchmark: str = "SPY",
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
    multiples: Sequence[float] = DEFAULT_COST_STRESS_MULTIPLES,
    axis: str = "all",
) -> Dict[str, Any]:
    """Re-run one config under progressively worse cost assumptions.

    An edge that clears its gates at 1bp and evaporates at 2bp is a different
    proposition from one that survives five times its assumed cost, and nothing
    distinguished them: a single cost assumption produces a single number, and the
    reader cannot tell how much of the result was the assumption.

    ``axis`` restricts what is scaled. ``"all"`` scales commission, impact and borrow
    together. ``"borrow"`` scales only the borrow rate, which is worth asking
    separately because a long-short book's exposure to it is qualitatively different
    from its exposure to commission - it is a carry on inventory rather than a toll on
    turnover, so it grows with holding period rather than with trading.

    Read-only research clock: re-runs a backtest per point and reports the curve.
    Journals nothing - these are the same config under stated assumptions, not new
    candidates, and recording them as trials would inflate the multiple-testing count
    that the deflated Sharpe deflates against.
    """
    run_id = new_run_id()
    points: List[Dict[str, Any]] = []
    for multiple in multiples:
        scale_turnover = multiple if axis in ("all", "turnover") else 1.0
        scale_borrow = multiple if axis in ("all", "borrow") else 1.0
        result = run_backtest(
            data_client,
            strategy,
            symbols,
            start,
            end,
            config=config,
            capital=capital,
            benchmark=benchmark,
            commission_bps=commission_bps * scale_turnover,
            impact_eta=impact_eta * scale_turnover,
            participation_cap=participation_cap,
            borrow_bps=borrow_bps * scale_borrow,
            journal=False,
            force=True,
        )
        metrics = result.get("metrics") or {}
        points.append(
            {
                "multiple": float(multiple),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "total_return": metrics.get("total_return", 0.0),
                "total_cost": result.get("total_cost", 0.0),
                "executability": result.get("executability", {}),
            }
        )

    survives = [p["multiple"] for p in points if p["sharpe_ratio"] > 0 and p["total_return"] > 0]
    return {
        "run_id": run_id,
        "strategy": strategy,
        "axis": axis,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "base_cost": {
            "commission_bps": commission_bps,
            "impact_eta": impact_eta,
            "borrow_bps": borrow_bps,
        },
        "points": points,
        # The headline: the largest multiple of its own assumed cost the edge survives.
        # 0.0 means it does not survive its own assumptions, which is worth saying
        # plainly rather than leaving to be read off a table.
        "survives_to_multiple": max(survives) if survives else 0.0,
        "note": "Each point re-runs the same config under scaled cost assumptions. "
        "Nothing is journaled: these are one candidate under stated assumptions, not "
        "new candidates, and counting them would inflate the multiple-testing total.",
    }


def run_draft_walk_forward(
    data_client: MarketDataClient,
    code: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    *,
    class_name: Optional[str] = None,
    mode: str = "anchored",
    n_folds: Optional[int] = None,  # see run_walk_forward: one default per surface
    train_days: Optional[int] = None,
    test_days: Optional[int] = None,
    embargo_days: Optional[int] = None,
    holdout_days: int = 0,
    method: str = "grid",
    objective: str = "sharpe_ratio",
    max_evals: int = 50,
    seed: int = 42,
    capital: float = 100_000.0,
    include_pbo: bool = False,
    include_monte_carlo: bool = False,
    parameter_sensitivity: bool = False,
    leakage_probe: bool = False,
    gates: Optional[Dict[str, float]] = None,
    gross: bool = False,
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
    journal: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Validate strategy source, then run it through the normal walk-forward gates.

    Draft code is never registered globally and never written into the public repo.
    When ``journal`` is true, the validated run is recorded under a stable
    ``draft:<ClassName>:<hash>`` strategy id so the campaign can still see that a
    generated/private candidate consumed a test.
    """
    from tradeflow.optimization.config_store import current_git_sha
    from tradeflow.research.sandbox import HygieneError, load_strategy_from_code
    from tradeflow.services.audit import journal_trial

    run_id = new_run_id()
    code_hash = draft_code_hash(code)
    # The same verdict the validators return. This entry point guarded nothing at
    # all, so even the anticipated rejection - a draft that simply does not pass
    # hygiene - came back as a raised exception rather than an answer, from the one
    # of the three tools that costs a trial to call.
    try:
        cls = load_strategy_from_code(code, class_name=class_name)
    except HygieneError as exc:
        return _draft_rejection(run_id, "strategy", code, exc, hygiene=True)
    except Exception as exc:  # noqa: BLE001 - see validate_draft_strategy_code
        logger.warning("Draft strategy validation failed unexpectedly", exc_info=True)
        return _draft_rejection(run_id, "strategy", code, exc, hygiene=False)
    draft_strategy = f"draft:{cls.__name__}:{code_hash}"
    cost_model = _build_cost_model(gross, commission_bps, impact_eta, participation_cap, borrow_bps)
    cost_key = _cost_key(gross, commission_bps, impact_eta, borrow_bps)
    # The draft's own identity (its source) on top of the same recipe every other
    # walk-forward is keyed by - a third copy of that dict is a third place for the
    # next thing folded into the key to be forgotten.
    recipe = {
        "code_hash": code_hash,
        "class_name": cls.__name__,
        **walk_forward_recipe(
            mode=mode,
            n_folds=n_folds,
            train_days=train_days,
            test_days=test_days,
            embargo_days=embargo_days,
            holdout_days=holdout_days,
            method=method,
            objective=objective,
            max_evals=max_evals,
            seed=seed,
            cost_key=cost_key,
        ),
    }

    with _open_trial_store() as trial_store:
        if journal and not force and trial_store is not None:
            cached = trial_store.find(
                strategy=draft_strategy,
                params=recipe,
                symbols=symbols,
                window_start=start,
                window_end=end,
                accounting=ACCOUNTING_VERSION,
                git_sha=current_git_sha(),
            )
            if cached is not None:
                metrics = json.loads(cached["metrics_json"] or "{}")
                return {
                    "run_id": run_id,
                    "strategy": draft_strategy,
                    "symbols": list(symbols),
                    "window": {"start": start.isoformat(), "end": end.isoformat()},
                    "memoized": True,
                    "trial_id": cached["id"],
                    "trial_ts": cached["ts"],
                    "note": "Served from an identical prior draft validation, not re-run. "
                    "Pass force=True to re-verify.",
                    "oos_aggregate": _jsonable(metrics),
                    "promotable": bool(cached["promotable"]) if cached["promotable"] is not None else None,
                    "efficiency": cached["efficiency"],
                    "n_trials_total": cached["n_trials_in_session"],
                    "draft": {
                        "class_name": cls.__name__,
                        "code_hash": code_hash,
                        "journaled": True,
                    },
                }

        validator = WalkForwardValidator(
            cls,
            data_client,
            initial_capital=capital,
            seed=seed,
            gates=gates,
            cost_model=cost_model,
            trial_store=None,
            strategy_name=None,
            cost_key=cost_key,
            accounting=ACCOUNTING_VERSION,
            force=True,
        )
        result = validator.run(
            symbols,
            start,
            end,
            mode=mode,
            n_folds=n_folds,
            train_days=train_days,
            test_days=test_days,
            embargo_days=embargo_days,
            holdout_days=holdout_days,
            method=method,
            objective=objective,
            max_evals=max_evals,
            pbo=include_pbo,
            monte_carlo=include_monte_carlo,
            parameter_sensitivity=parameter_sensitivity,
            leakage_probe=leakage_probe,
        )
    # Outside the trial-store block, and not conditioned on it: the store is a
    # derived memo cache, the journal is the campaign's multiple-testing record. A
    # store that will not open must not quietly turn a spent out-of-sample test into
    # an unrecorded one. Same placement as run_walk_forward, for the same reason.
    journaled = bool(journal and result.folds)
    if journaled:
        chosen = result.holdout_params or result.folds[-1].is_best_params
        gate_report = result.gate_report(gates)
        journal_trial(
            "walkforward",
            strategy=draft_strategy,
            symbols=symbols,
            start=start,
            end=end,
            params={**dict(chosen), "_draft": {"class_name": cls.__name__, "code_hash": code_hash}},
            metrics=result.oos_aggregate,
            objective=objective,
            extra={
                "n_trials": result.n_trials_total,
                "promotable": gate_report["promotable"],
                "efficiency": result.median_efficiency(),
            },
            returns=result.oos_returns,
            dedup_params=recipe,
        )

    payload = walk_forward_payload(
        result,
        run_id=run_id,
        strategy=draft_strategy,
        symbols=symbols,
        start=start,
        end=end,
        mode=mode,
        objective=objective,
        method=method,
        gross=gross,
        seed=seed,
        gates=gates,
    )
    payload["draft"] = {
        "class_name": cls.__name__,
        "code_hash": code_hash,
        "journaled": journaled,
        "note": "Draft source was validated and run in-memory; install it through entry points to use it by name.",
    }
    if journal and not journaled:
        # Asked for and not done. Saying so is the point: a caller counting its own
        # multiple-testing budget cannot infer this from a `false` alone.
        payload["draft"]["not_journaled_reason"] = (
            "the run produced no folds, so there was no out-of-sample result to record"
        )
    return payload


def draft_code_hash(code: str) -> str:
    """Short digest identifying draft source.

    Public because the MCP layer logs it against calls the journal records under
    ``draft:<ClassName>:<hash>``. That layer had its own byte-identical copy, so the
    two would have silently named different things the first time either changed - and
    what they name is how a draft's calls are tied to the trials it spent.
    """
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]


def walk_forward_payload(
    result,
    *,
    run_id: str,
    strategy: str,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    mode: str,
    objective: str,
    method: str,
    gross: bool,
    seed: Optional[int] = None,
    gates: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """A ``WalkForwardResult`` as the JSON-serializable dict every surface consumes.

    The counterpart to :func:`backtest_payload`, and for the same reason: the CLI
    runs the validator directly and the service returns a dict, so a renderer that
    wants "the walk-forward result" has to get the identical object either way.
    """
    folds = [
        {
            "index": fr.fold.index,
            "is_window": {"start": fr.fold.is_start.isoformat(), "end": fr.fold.is_end.isoformat()},
            "oos_window": {"start": fr.fold.oos_start.isoformat(), "end": fr.fold.oos_end.isoformat()},
            "is_best_params": _jsonable(fr.is_best_params),
            "is_sharpe": fr.is_metrics.get("sharpe_ratio", 0.0),
            "oos_sharpe": fr.oos_metrics.get("sharpe_ratio", 0.0),
            "oos_profit_factor": fr.oos_metrics.get("profit_factor", 0.0),
            "oos_trades": fr.oos_trades,
            "n_trials": fr.n_trials,
        }
        for fr in result.folds
    ]

    return {
        "run_id": run_id,
        "strategy": strategy,
        "symbols": list(symbols),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "mode": mode,
        "objective": objective,
        "method": method,
        "gross": gross,
        "folds": folds,
        "oos_aggregate": _jsonable(result.oos_aggregate),
        "efficiency": result.efficiency,
        "median_efficiency": result.median_efficiency(),
        "median_oos_sharpe": result.median_oos("sharpe_ratio"),
        "degradation": _jsonable(result.degradation),
        "holdout": _jsonable(result.holdout) if result.holdout else None,
        "holdout_params": _jsonable(result.holdout_params) if result.holdout_params else None,
        "n_trials_total": result.n_trials_total,
        "total_oos_trades": result.total_oos_trades(),
        "gate_report": _jsonable(result.gate_report(gates)),
        "diagnostics": _jsonable(result.diagnostics),
        "pbo": result.pbo,
        "monte_carlo": _jsonable(result.monte_carlo) if result.monte_carlo else None,
        "seed": seed,
    }


def compute_bootstrap_skill(
    oos_returns: Optional[pd.Series],
    strategy: str,
    symbols: List[str],
    n_trials_total: int,
    oos_aggregate: Dict[str, float],
    *,
    accounting: Optional[int] = None,
    B: int = 2000,
    block_length: Optional[float] = None,
    seed: int = 0,
    min_overlap: int = 60,
    journal_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """The bootstrap-skill report: this config's OWN zero-alpha bootstrap p,
    always shown next to the FAMILY p from White's Reality Check over every OOS
    return series the trial store has recorded for
    ``(strategy, universe, accounting)`` — replacing the Deflated Sharpe's
    assumed ``E[max]``/effective-trial-count with the actual trials.

    Call this AFTER the current trial has been journaled (``journal_trial``), so
    the family query below includes it — the whole point of the family test is
    to ask "is this trial's result still notable once every trial this campaign
    has tried is priced in," which requires this trial to already be one of them.

    Best-effort on the trial store, like every other trial-store touchpoint: a
    store-open failure (or too few trials with a usable stored return series)
    degrades to "own p only" — it never blocks the caller, and the report says
    so rather than silently omitting the family half. Own and family are always
    returned together (never one alone) — a great own p and a terrible family p
    is exactly the selection-luck signature.
    """
    from tradeflow.analytics import bootstrap as boot
    from tradeflow.analytics.metrics import TRADING_DAYS_PER_YEAR

    if oos_returns is None or len(oos_returns) < 8:
        return {
            "available": False,
            "note": "Not enough OOS periods for a bootstrap (need >= 8 daily observations).",
        }

    own = boot.bootstrap_null(
        oos_returns.to_numpy(),
        B=B,
        block_length=block_length,
        seed=seed,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )

    family: Dict[str, Any] = {"available": False}
    try:
        from tradeflow.engine.backtest import ACCOUNTING_VERSION
        from tradeflow.store.trials import DEFAULT_JOURNAL_PATH, TrialStore, db_path_for_journal

        jpath = Path(journal_path) if journal_path else DEFAULT_JOURNAL_PATH
        acct = accounting if accounting is not None else ACCOUNTING_VERSION
        with TrialStore(db_path_for_journal(jpath), journal_path=jpath) as store:
            panel = store.returns_panel(strategy, symbols, acct, min_overlap=min_overlap)
        if panel["n_used"] >= 2:
            matrix = np.array(panel["matrix"], dtype=float)
            fam = boot.reality_check(
                matrix,
                B=B,
                block_length=block_length,
                seed=seed,
                periods_per_year=TRADING_DAYS_PER_YEAR,
                trial_ids=panel["trial_ids"],
            )
            fam.update(
                available=True,
                n_attempted=panel["n_attempted"],
                n_with_returns=panel["n_with_returns"],
                n_used=panel["n_used"],
                n_excluded_short=panel["n_excluded_short"],
            )
            family = fam
        else:
            family = {
                "available": False,
                "n_attempted": panel["n_attempted"],
                "n_with_returns": panel["n_with_returns"],
                "n_used": panel["n_used"],
                "note": "Fewer than 2 trials in this family have a usable stored return series "
                "(need >= 2 sharing >= min_overlap common dates) — Reality Check needs a real panel.",
            }
    except Exception:  # noqa: BLE001 - best-effort, like every other trial-store touchpoint
        logger.warning("Bootstrap-skill family check unavailable (trial store)", exc_info=True)

    return {
        "available": True,
        "own": own,
        "family": family,
        "n_trials_total": n_trials_total,
        "parametric_cross_check": {
            "probabilistic_sharpe_ratio": oos_aggregate.get("probabilistic_sharpe_ratio", 0.0),
            "deflated_sharpe_ratio": oos_aggregate.get("deflated_sharpe_ratio", 0.0),
        },
        "verdict": _bootstrap_skill_verdict(own, family),
    }


def _bootstrap_skill_verdict(own: Dict[str, Any], family: Dict[str, Any]) -> str:
    if own.get("insufficient_data"):
        return "insufficient data for a bootstrap verdict"
    own_significant = own["p_value"] < 0.05
    if not family.get("available"):
        base = "individually significant" if own_significant else "NOT individually significant"
        return f"{base} (own test only — family-of-trials test unavailable, see n_used/n_attempted)"
    family_significant = family["family_p"] < 0.05
    if own_significant and not family_significant:
        return (
            "individually significant, NOT significant as a selected maximum — "
            "consistent with selection luck; needs fresh OOS data to distinguish."
        )
    if own_significant and family_significant:
        return (
            "significant both individually and as the family's best — the strongest verdict this test gives."
        )
    return "NOT individually significant (own test already fails; family test moot)."


def summarize_bars(
    data_client: MarketDataClient,
    symbols: List[str],
    timeframe: str = "1Day",
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Compact OHLCV stats per symbol for qualitative analysis (no raw bars).

    Descriptive only. NOTE for the caller: choosing symbols by their realized
    stats here and then backtesting them is look-ahead - universe selection is a
    research decision, not a metric to optimize.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    bars = data_client.get_bars(symbols, timeframe, start, end)

    out: Dict[str, Any] = {}
    for symbol in symbols:
        frame = bars.get(symbol)
        if frame is None or len(frame) < 2:
            out[symbol] = {"available": False}
            continue
        close = frame["close"]
        returns = close.pct_change().dropna()
        sma_fast = close.rolling(min(10, len(close))).mean().iloc[-1]
        sma_slow = close.rolling(min(30, len(close))).mean().iloc[-1]
        volume = frame["volume"]
        out[symbol] = {
            "available": True,
            "bars": int(len(frame)),
            "last_close": float(close.iloc[-1]),
            "period_return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
            "annualized_vol_pct": float(m.annualized_volatility(returns) * 100),
            "trend": "up" if sma_fast >= sma_slow else "down",
            "max_drawdown_pct": float(m.max_drawdown(close) * 100),
            "avg_volume": float(volume.mean()),
            "recent_volume_ratio": float(volume.iloc[-1] / volume.mean()) if volume.mean() else 0.0,
            "high": float(frame["high"].max()),
            "low": float(frame["low"].min()),
        }
    return {"timeframe": timeframe, "lookback_days": lookback_days, "symbols": out}


def compute_alphas(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    as_of: datetime,
    source: str = "strategy",
    scanner: str = "demo_volume",
    ic: float = DEFAULT_IC,
    benchmark: str = "SPY",
    neutralize: bool = False,
    neutralize_factors: Sequence[str] = (),
    lookback_days: int = 180,
    timeframe: Optional[str] = None,
    scaling: str = "case1",
    price_derived: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn a per-name view into ranked residual-return alphas, via a feature panel.

    Read-only research-clock flow: scans the universe as of ``as_of`` (leakage-safe),
    assembles a :class:`FeaturePanel` (risk + score columns), refines it into a
    comparable annualized forecast, and returns the ranked table. Produces no orders
    and saves no config.

    ``source`` selects the score column's origin: ``"strategy"`` uses the strategy's
    continuous conviction; ``"signal"`` uses its discrete BUY/SELL/HOLD as +1/-1/0;
    ``"scanner"`` uses the ``scanner``'s continuous signed strength.

    ``scaling`` picks the per-name scaling: ``"case1"`` = ``ω·IC·z`` (the default),
    ``"case2"`` = ``IC·c_g·z`` (no per-name vol multiply), or ``"auto"`` to let
    :func:`~tradeflow.alphas.refine.case_test` decide from trailing history
    (``price_derived`` is the base-rate default when the test can't decide). The case
    diagnostics — chosen case, R², both candidates' cross-sectional correlation — are
    echoed under ``case`` whenever the test runs so a wrong call is visible.
    """
    run_id = new_run_id()
    strat = _strategy(strategy, config)
    tf = timeframe or strat.config.get("timeframe", "1Day")
    periods_per_year = Timeframe.parse(tf).periods_per_year()

    # Scan bars point-in-time (the leakage guard lives in the source), then build
    # the cross-sectional panel: risk features + the chosen score column.
    bars = ClientBarSource(data_client).scan([*symbols, benchmark], tf, as_of, lookback_days)
    bench_frame = bars.get(benchmark)
    universe_bars = {sym: bars[sym] for sym in symbols if sym in bars}

    scorer = {
        "strategy": lambda: strategy_scorer(strat),
        "signal": lambda: signal_scorer(strat),
        "scanner": lambda: scanner_scorer(_scanner(scanner)),
    }.get(source)
    if scorer is None:
        raise ValueError(f"source must be 'strategy', 'signal', or 'scanner', got {source!r}")

    panel = FeaturePanel.for_universe(as_of, list(universe_bars))
    add_risk_features(panel, universe_bars, bench_frame, periods_per_year)
    if neutralize_factors:
        add_factor_exposure_features(panel, universe_bars, bench_frame, neutralize_factors)
    add_score_feature(panel, scorer(), universe_bars)

    # Case selection: only when the caller asks — the "case1" default keeps the base
    # refinement pipeline byte-for-byte (the equivalence guard) and cheap.
    case_diag = None
    chosen_scaling = scaling
    if scaling in ("auto", "case2") and panel.has("residual_vol"):
        case_diag = _run_case_test(universe_bars, scorer(), panel.get("residual_vol"), price_derived)
        chosen_scaling = f"case{case_diag['case']}" if scaling == "auto" else "case2"

    context = AlphaContext(
        ic=ic,
        neutralize=neutralize,
        neutralize_factors=tuple(neutralize_factors),
        scaling=chosen_scaling,
    )
    refine_alpha(panel, context)
    alphas = panel_to_alphas(panel, context)

    table = [
        {
            "symbol": a.symbol,
            "score": float(panel.get("score").get(a.symbol)),
            "z": a.raw_z,
            "beta": float(panel.get("beta").get(a.symbol)) if panel.has("beta") else 1.0,
            "residual_vol": a.residual_vol,
            "alpha": a.alpha,
        }
        for a in alphas
    ]

    return {
        "run_id": run_id,
        "strategy": strategy,
        "source": source,
        "scanner": scanner if source == "scanner" else None,
        "as_of": as_of.isoformat(),
        "timeframe": tf,
        "ic": ic,
        "benchmark": benchmark,
        "benchmark_available": bool(panel.meta.get("benchmark_available")),
        "neutralize": neutralize,
        "neutralize_factors": list(neutralize_factors),
        "neutralized_against": list(panel.meta.get("neutralized_against", [])),
        "universe_size": int(panel.get("score").notna().sum()) if panel.has("score") else 0,
        "low_confidence": bool(panel.meta.get("low_confidence")),
        "scaling": chosen_scaling,
        "case": _jsonable(case_diag) if case_diag else None,
        "shrink_chain": _jsonable(panel.meta.get("shrink_chain", [])),
        "alphas": _jsonable(table),
        "note": "Alphas are residual-return FORECASTS, annualized, scaled by an "
        "ASSUMED IC (a prior until it is measured from realized outcomes). Relative sizing across "
        "names is correct regardless of IC; the absolute scale is only as good as it. "
        "'case' picks Case-1 (ω·IC·z) vs Case-2 (IC·c_g·z) scaling; the IC-uncertainty "
        "level shrink engages only where a MEASURED IC is available (compute_information).",
    }


def compute_combined_alphas(
    data_client: MarketDataClient,
    signals: List[str],
    symbols: List[str],
    as_of: datetime,
    benchmark: str = "SPY",
    neutralize: bool = False,
    neutralize_factors: Sequence[str] = (),
    lookback_days: int = 365,
    timeframe: str = "1Day",
    horizon: int = 5,
    n_points: int = 12,
) -> Dict[str, Any]:
    """Combine several strategies' signals into one alpha by their IC and correlation.

    Read-only research-clock flow: measures each signal's IC and the signal
    correlation matrix over a trailing window (realized residual returns), shrinks the
    ICs by their estimation confidence, and combines them with GLS weights
    (``Ω⁻¹·IC``) so redundant signals split a weight rather than double-count. The
    combined score is scaled by the **measured** combined IC - replacing the
    single-signal assumed scalar, never applied twice. Returns the ranked alpha table
    plus the measured ICs, weights, and correlation matrix.
    """
    from tradeflow.alphas import combined_score, measure_signals, strategy_scorer

    strategies = [resolve_strategy_class(s) and s for s in signals]  # validate names
    run_id = new_run_id()
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()

    bars = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, as_of, lookback_days)
    bench_frame = bars.get(benchmark)
    universe_bars = {sym: bars[sym] for sym in symbols if sym in bars}
    if bench_frame is None or bench_frame.empty or not universe_bars:
        return {
            "run_id": run_id,
            "signals": strategies,
            "as_of": as_of.isoformat(),
            "universe_size": 0,
            "note": "Insufficient data: need a benchmark series and at least one scored name.",
        }

    scorers = {name: strategy_scorer(_strategy(name, None)) for name in strategies}
    measurement = measure_signals(
        universe_bars, scorers, bench_frame, as_of, horizon=horizon, n_points=n_points
    )

    panel = FeaturePanel.for_universe(as_of, list(universe_bars))
    panel.set("score", combined_score(universe_bars, scorers, measurement, as_of))
    add_risk_features(panel, universe_bars, bench_frame, periods_per_year)
    if neutralize_factors:
        add_factor_exposure_features(panel, universe_bars, bench_frame, neutralize_factors)
    # The combined, measured, shrunk IC replaces the assumed scalar (no double-scaling).
    context = AlphaContext(
        ic=measurement.combined_ic,
        neutralize=neutralize,
        neutralize_factors=tuple(neutralize_factors),
    )
    refine_alpha(panel, context)
    alphas = panel_to_alphas(panel, context)

    table = [
        {
            "symbol": a.symbol,
            "score": float(panel.get("score").get(a.symbol)),
            "z": a.raw_z,
            "beta": float(panel.get("beta").get(a.symbol)) if panel.has("beta") else 1.0,
            "residual_vol": a.residual_vol,
            "alpha": a.alpha,
        }
        for a in alphas
    ]

    return {
        "run_id": run_id,
        "signals": strategies,
        "as_of": as_of.isoformat(),
        "timeframe": timeframe,
        "benchmark": benchmark,
        "neutralize": neutralize,
        "neutralize_factors": list(neutralize_factors),
        "neutralized_against": list(panel.meta.get("neutralized_against", [])),
        "universe_size": int(panel.get("score").notna().sum()) if panel.has("score") else 0,
        "low_confidence": bool(panel.meta.get("low_confidence")),
        "n_periods": measurement.n_periods,
        "combined_ic": measurement.combined_ic,
        "signal_ics": _jsonable(measurement.ics),
        "signal_shrunk_ics": _jsonable(measurement.shrunk_ics),
        "signal_weights": _jsonable(measurement.weights),
        "signal_correlation": _jsonable(measurement.correlation.to_dict()),
        "alphas": _jsonable(table),
        # Shrink-chain audit: the "is the IC real" level shrink is OWNED HERE by the
        # combination's per-signal Bayesian shrink (the same g/(g+1) math as the
        # single-signal level shrink, T = n_periods), so it is NOT re-applied
        # post-combination — that would double-shrink and undertrade forever. The
        # combination owns credit-sharing (Ω⁻¹) AND, on this path, the level; the
        # single-signal level shrink owns it on the non-combined path. See the
        # Multi-signal and Continuous-alphas pages in the engineering docs.
        "shrink_chain": _jsonable(
            [
                {"step": "measure", "ics": measurement.ics},
                {
                    "step": "ic_uncertainty",
                    "owner": "combination_shrink",
                    "shrunk_ics": measurement.shrunk_ics,
                    "n_periods": measurement.n_periods,
                },
                {"step": "combine", "combined_ic": measurement.combined_ic},
            ]
        ),
        "note": "ICs and the signal correlation are MEASURED over the trailing window "
        "(not assumed) and shrunk by estimation confidence; redundant signals split a "
        "weight via Ω⁻¹. The IC-uncertainty level shrink is owned here by the per-signal "
        "Bayesian shrink (not re-applied — that would double-shrink). Measure on "
        "out-of-sample data for an honest combination.",
    }


def compute_information(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "demo_volume",
    benchmark: str = "SPY",
    neutralize: bool = True,
    neutralize_factors: Sequence[str] = (),
    ic_prior: float = DEFAULT_IC,
    horizon: int = 5,
    n_points: int = 24,
    n_trials: int = 1,
    timeframe: str = "1Day",
    risk_model: str = "shrinkage",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure a strategy's information coefficient, breadth, and information ratio.

    Read-only research-clock diagnostic (see the Information-analysis page in the
    engineering docs). At sampled rebalances it pairs the
    alpha forecast known *at* ``t`` with the realized **residual** return over
    ``(t, t+horizon]`` (strict forward alignment - rewarding skill, not beta), giving
    the IC time series (Pearson + rank), its t-stat, the effective breadth ``BR_eff``
    (deflated by the average correlation ρ̄ from Σ), and the **predicted vs realized
    IR** reconciliation - with the research-integrity guardrails (IR standard-error
    band, multiple-testing inflation, sanity ceiling) that keep a lucky backtest
    honest. Factor-vs-specific **risk** attribution is available via the factor model
    (``compute_risk(..., model='factor')``); realized-return attribution and capacity
    are smaller follow-ons.
    """
    from tradeflow.alphas import horizon as hz
    from tradeflow.alphas import refine
    from tradeflow.analytics import information as info
    from tradeflow.indicators import indicators

    run_id = new_run_id()
    strat = _strategy(strategy, config)
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    rebalances_per_year = periods_per_year / horizon

    # One scan over the window; per-rebalance slices reuse it (leakage-safe by <= t).
    bars = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, end, _window_days(start, end))
    bench = bars.get(benchmark)
    universe_bars = {sym: bars[sym] for sym in symbols if sym in bars}
    if bench is None or bench.empty or not universe_bars:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "periods": 0,
            "note": "Insufficient data: need a benchmark series and scored names.",
        }

    scorer = {
        "strategy": lambda: strategy_scorer(strat),
        "signal": lambda: signal_scorer(strat),
        "scanner": lambda: scanner_scorer(_scanner(scanner)),
    }[source]()
    ctx = AlphaContext(ic=ic_prior, neutralize=neutralize, neutralize_factors=tuple(neutralize_factors))

    index = bench.index
    lo, hi = _to_ts(start, index), _to_ts(end, index)
    window = index[(index >= lo) & (index <= hi)]
    points = _rebalance_points(len(window), horizon, n_points)

    pearson_ics, rank_ics, portfolio_returns = [], [], []
    factor_contribs, specific_contribs = [], []
    n_names_seen = []
    last_weights = None  # the most recent paper active book, for the bucket diagnostic
    for j in points:
        t, t_fwd = window[j], window[j + horizon]
        alpha = _alpha_cross_section(universe_bars, bench, scorer, periods_per_year, t, ctx)
        resid = _forward_residual_return(universe_bars, bench, t, t_fwd, indicators)
        aligned = pd.concat([alpha, resid], axis=1, keys=["alpha", "resid"]).dropna()
        if len(aligned) < 5:
            continue
        pearson_ics.append(info.pearson_ic(aligned["alpha"], aligned["resid"]))
        rank_ics.append(info.rank_ic(aligned["alpha"], aligned["resid"]))
        # Realized return of the paper alpha portfolio: standardized-alpha-weighted
        # residual return (scale cancels in the IR).
        z = aligned["alpha"] - aligned["alpha"].mean()
        if z.std() > 0:
            w = z / z.std()
            last_weights = w  # mean-zero ⇒ already an active book
            portfolio_returns.append(float(w @ aligned["resid"]))
            # Attribution: split that return into factor vs specific by projecting the
            # realized cross-section onto the factor exposures (the split closes exactly).
            split = _factor_attribution(w, universe_bars, bench, t, t_fwd, periods_per_year)
            if split is not None:
                factor_contribs.append(split[0])
                specific_contribs.append(split[1])
        n_names_seen.append(len(aligned))

    stats = info.ic_stats(pearson_ics)
    rank_stats = info.ic_stats(rank_ics)
    n_names = int(np.median(n_names_seen)) if n_names_seen else 0

    # ρ̄ from the risk model over the window (correlated bets deflate breadth).
    matrix = _build_covariance(risk_model, universe_bars, bench, periods_per_year)
    if matrix is not None and len(matrix.symbols) > 1:
        corr = matrix.correlation().to_numpy()
        rho_bar = float((corr.sum() - len(corr)) / (len(corr) * (len(corr) - 1)))
    else:
        rho_bar = 0.0

    breadth = info.effective_breadth(n_names, rebalances_per_year, rho_bar)
    pred_ir = info.predicted_ir(stats["mean_ic"], breadth["br_eff"])

    # IC-uncertainty level shrink: how much of the measured-IC level survives its own
    # estimation error. T_eff deflates the rebalance count by the horizon/spacing overlap
    # (raw count under-shrinks); this is the honest haircut a human applies to the
    # recommended_ic before feeding it back into the alpha scaling.
    spacing = float(np.mean(np.diff(points))) if len(points) > 1 else float(horizon)
    t_eff = hz.effective_sample_size(stats["periods"], horizon, spacing)
    shrink_factor = refine.level_shrink_factor(stats["mean_ic"], t_eff)
    shrink_chain = [
        {"step": "scale", "note": "ω·IC·z at the recommended (measured) IC"},
        {
            "step": "ic_uncertainty",
            "owner": "level_shrink",
            "ic": stats["mean_ic"],
            "t_eff": t_eff,
            "multiplier": shrink_factor,
        },
    ]

    # Equal-risk-contribution diagnostic: does the current paper book spread active
    # variance evenly across residual-vol buckets, or tilt into one?
    bucket_diag = None
    if matrix is not None and last_weights is not None and len(matrix.symbols) > 1:
        bucket_diag = info.risk_bucket_diagnostic(
            last_weights, matrix.sigma, matrix.symbols, matrix.volatilities()
        )

    realized_ir = 0.0
    if len(portfolio_returns) > 1 and np.std(portfolio_returns) > 0:
        realized_ir = float(
            np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(rebalances_per_year)
        )
    years = max((end - start).days / 365.25, 1e-9)

    return {
        "run_id": run_id,
        "strategy": strategy,
        "source": source,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "horizon_bars": horizon,
        "periods": stats["periods"],
        "low_sample": stats["periods"] < info.MIN_PERIODS,
        "mean_ic": stats["mean_ic"],
        "ic_vol": stats["ic_vol"],
        "ic_tstat": stats["ic_tstat"],
        "rank_ic": rank_stats["mean_ic"],
        "rank_ic_tstat": rank_stats["ic_tstat"],
        "n_names": n_names,
        "rho_bar": rho_bar,
        "breadth_effective": breadth["br_eff"],
        "breadth_naive": breadth["br_naive"],
        "predicted_ir": pred_ir,
        "realized_ir": realized_ir,
        "ir_standard_error": info.ir_standard_error(realized_ir, years),
        "multiple_testing_inflation": info.multiple_testing_inflation(n_trials),
        "n_trials": n_trials,
        "sanity_ceiling_breached": abs(realized_ir) > 2.0,
        "recommended_ic": stats["mean_ic"],  # feeds back into the alpha scaling — a human applies it
        "effective_t": t_eff,
        "level_shrink_factor": shrink_factor,  # keep this fraction of the naive alpha level
        "shrink_chain": _jsonable(shrink_chain),
        "risk_bucket_diagnostic": _jsonable(bucket_diag) if bucket_diag else None,
        # Attribution: the realized active return split into factor tilts vs genuine
        # name selection (they sum to the realized portfolio return per rebalance).
        "factor_return": float(np.mean(factor_contribs)) if factor_contribs else 0.0,
        "specific_return": float(np.mean(specific_contribs)) if specific_contribs else 0.0,
        "note": "IC measured as alpha-vs-forward-RESIDUAL-return (strict t→t+h, no "
        "look-ahead). predicted_IR = mean_IC·√BR_eff; BR_eff deflates the name count "
        "by ρ̄. An IC t-stat < 2, a realized IR within its standard-error band of 0, or "
        "a realized IR > 2 on public data all mean: not skill yet. factor_return vs "
        "specific_return attributes the realized active return; capacity is in the "
        "portfolio report. level_shrink_factor is the IC-uncertainty haircut on the "
        "recommended_ic LEVEL for its own estimation error (T_eff deflates for horizon "
        "overlap); risk_bucket_diagnostic flags a residual-vol tilt from mis-scaling.",
    }


#: The composite result's shape identifier. A consumer that renders or serializes
#: this object checks it rather than guessing from the keys present, so a shape
#: change fails loudly instead of half-rendering.
VERDICT_SCHEMA = "verdict/1"

#: Steps of the composite pipeline, in the order they run. ``combination`` is
#: conditional on more than one signal being given.
_VERDICT_STEPS = ("scan", "alphas", "combination", "portfolio", "information")


def run_verdict(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    *,
    config: Optional[Dict[str, Any]] = None,
    scanner: str = "demo_volume",
    signals: Optional[Sequence[str]] = None,
    source: str = "strategy",
    benchmark: str = "SPY",
    timeframe: str = "1Day",
    capital: Optional[float] = None,
    horizon: int = 5,
    n_points: int = 24,
    target_te: float = 0.04,
    max_weight: float = 0.25,
    max_names: Optional[int] = None,
    neutralize_factors: Sequence[str] = (),
    risk_model: str = "shrinkage",
    lookback_days: int = 365,
    gross: bool = False,
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
    force: bool = False,
    journal: bool = True,
) -> Dict[str, Any]:
    """Run the whole cross-sectional pipeline once and return one composite answer.

    Scan the universe, refine the signal into alphas, combine several signals when
    given, construct the cost-aware portfolio, and measure the information content
    - all against **one** resolved universe, one window, and one cost model, so the
    sections of the report are guaranteed to be describing the same thing. Running
    the five steps by hand gives no such guarantee: each command re-resolves its own
    universe and applies its own defaults, and the joined-up story can silently be
    five different stories.

    Answers "what does the cross-sectional pipeline say about this universe as of
    the window end" - a forecast and a proposed book, not a historical simulation.
    For "did this ever work", that is ``run_backtest``/``run_walk_forward``.

    A step that fails does not kill the run: the composite records what ran, what
    did not, and why, and the overall verdict for any incomplete run is
    ``incomplete`` - never a verdict assembled from the steps that happened to
    succeed. The verdict line itself is derived from the gates the steps already
    compute (the IC t-stat and its band, the sanity ceiling, expected active return
    net of cost), never from a fresh heuristic invented for the summary.

    The whole composite journals as **one** trial (kind ``verdict``), not one per
    step - five journal rows per run would inflate a campaign's multiple-testing
    total five-fold. An identical prior run is served from the trial store instead
    of re-run unless ``force``, on the same terms as the single-step commands.

    Returns a JSON-serializable dict with stable top-level keys: ``schema``,
    ``run_id``, ``inputs``, ``steps``, ``scan``, ``alphas``, ``combination``,
    ``portfolio``, ``information``, ``verdict``.
    """
    from tradeflow.engine.backtest import ACCOUNTING_VERSION
    from tradeflow.marketdata.session import session_client

    run_id = new_run_id()
    signal_list = [s for s in (signals or []) if s]
    inputs = {
        "strategy": strategy,
        "signals": signal_list,
        "source": source,
        "scanner": scanner,
        "candidates": list(symbols),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "timeframe": timeframe,
        "benchmark": benchmark,
        "capital": capital,
        "horizon": horizon,
        "risk_model": risk_model,
        "target_te": target_te,
        "neutralize_factors": list(neutralize_factors),
        "cost": _cost_key(gross, commission_bps, impact_eta, borrow_bps),
    }

    # Every input any step reads goes into the identity, not just the ones a caller
    # is likely to vary - two materially different composites must never collide as
    # "the same trial" just because the flags they differ on live inside a step.
    dedup_params = {
        **_tunable_params(_strategy(strategy, config)),
        "_verdict": {
            "signals": sorted(signal_list),
            "source": source,
            "scanner": scanner,
            "benchmark": benchmark,
            "timeframe": timeframe,
            "horizon": horizon,
            "n_points": n_points,
            "risk_model": risk_model,
            "target_te": target_te,
            "max_weight": max_weight,
            "max_names": max_names,
            "lookback_days": lookback_days,
            "neutralize_factors": sorted(neutralize_factors),
            "capital": capital,
        },
        "_cost": _cost_key(gross, commission_bps, impact_eta, borrow_bps),
    }

    if not force:
        cached = _find_cached_trial(strategy, dedup_params, symbols, start, end, ACCOUNTING_VERSION)
        if cached is not None:
            memoized = _load_verdict_artifact(cached["id"])
            if memoized is not None:
                memoized["memoized"] = True
                memoized["trial_id"] = cached["id"]
                memoized["trial_ts"] = cached["ts"]
                memoized["note"] = (
                    "Served from an identical prior verdict run, not re-run. Pass force=True to re-verify."
                )
                return memoized

    client, cache = session_client(data_client)
    result: Dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "kind": "verdict",
        "run_id": run_id,
        "memoized": False,
        "inputs": inputs,
        "steps": {},
        "scan": None,
        "alphas": None,
        "combination": None,
        "portfolio": None,
        "information": None,
    }
    steps: Dict[str, Any] = result["steps"]

    def _step(name: str, fn):
        """Run one step, recording its outcome. A failure is reported, not raised -
        a 30-second pipeline must not lose four completed sections to the fifth."""
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - any step failure is data, not a crash
            logger.warning("Verdict step %r failed", name, exc_info=True)
            steps[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            return None
        steps[name] = {"status": "ok"}
        return value

    # 1. One universe, resolved once, used by every step below. Every later step is
    #    handed this list rather than the candidates, so no step can quietly re-scan
    #    into a different universe than the one the report's header claims.
    if not scanner or scanner == "none":
        steps["scan"] = {"status": "skipped", "reason": "no scanner — candidates used as-is"}
        scan, universe = None, list(symbols)
    else:
        scan = _step("scan", lambda: run_scan(client, scanner, list(symbols), as_of=end))
        universe = [row["symbol"] for row in (scan or {}).get("flagged", [])]
        if not universe:
            # Same fallback the single-step commands use: an empty scan means "nothing
            # stood out today", not "there is nothing to analyze".
            universe = list(symbols)
            if scan is not None:
                scan["fell_back_to_candidates"] = True
    result["scan"] = scan
    inputs["universe"] = universe

    # 2. The forecast: one signal refined, or several combined.
    result["alphas"] = _step(
        "alphas",
        lambda: compute_alphas(
            client,
            strategy,
            universe,
            end,
            config=config,
            source=source,
            scanner=scanner,
            benchmark=benchmark,
            neutralize_factors=neutralize_factors,
            lookback_days=lookback_days,
            timeframe=timeframe,
        ),
    )
    if len(signal_list) > 1:
        result["combination"] = _step(
            "combination",
            lambda: compute_combined_alphas(
                client,
                list(signal_list),
                universe,
                end,
                benchmark=benchmark,
                neutralize_factors=neutralize_factors,
                lookback_days=lookback_days,
                timeframe=timeframe,
                horizon=horizon,
            ),
        )
    else:
        steps["combination"] = {"status": "skipped", "reason": "one signal — nothing to combine"}

    # 3. The proposed book, priced with the same cost model as everything else.
    result["portfolio"] = _step(
        "portfolio",
        lambda: construct_portfolio(
            client,
            strategy,
            universe,
            end,
            source=source,
            scanner=scanner,
            target_te=target_te,
            max_weight=max_weight,
            max_names=max_names,
            benchmark=benchmark,
            neutralize_factors=neutralize_factors,
            risk_model=risk_model,
            lookback_days=lookback_days,
            timeframe=timeframe,
            capital=capital,
            cost_aware=not gross,
            commission_bps=commission_bps,
            impact_eta=impact_eta,
            participation_cap=participation_cap,
            borrow_bps=borrow_bps,
        ),
    )

    # 4. Is the forecast information or luck? The campaign's own trial count is what
    #    makes the multiple-testing guardrail mean anything, so it comes from the
    #    store rather than from a caller's guess.
    n_trials = _campaign_trials(strategy, universe, ACCOUNTING_VERSION)
    result["information"] = _step(
        "information",
        lambda: compute_information(
            client,
            strategy,
            universe,
            start,
            end,
            config=config,
            source=source,
            scanner=scanner,
            benchmark=benchmark,
            neutralize_factors=neutralize_factors,
            horizon=horizon,
            n_points=n_points,
            n_trials=n_trials,
            timeframe=timeframe,
            risk_model=risk_model,
        ),
    )

    result["verdict"] = _verdict_gates(result)
    result["provenance"] = _verdict_provenance(inputs, cache, n_trials)

    if journal:
        _journal_verdict(result, strategy, universe, start, end, dedup_params)
    return result


def _campaign_trials(strategy: str, symbols: Sequence[str], accounting: int) -> int:
    """How many trials this ``(strategy, universe)`` family has already seen, plus
    the one about to run.

    Counting the current run makes the multiple-testing guardrail strictly
    conservative - the honest direction when the alternative is understating how
    many attempts produced the number on screen. Falls back to 1 when the store is
    unavailable: the guardrail degrades to "this is the only trial I can see",
    which is what it would have said before a store existed.
    """
    with _open_trial_store() as store:
        if store is None:
            return 1
        try:
            return int(store.family_count(strategy, symbols, accounting)) + 1
        except Exception:  # noqa: BLE001 - a passive store never breaks its caller
            logger.warning("Campaign trial count unavailable; using 1", exc_info=True)
            return 1


def code_version() -> str:
    """Which code produced a result, in whatever form is actually available.

    A checkout has a git SHA (working-tree aware, so a dirty tree is marked). An
    installed copy has no repository at all, and reporting "unknown" there defeats
    the point of a provenance block: a report that outlives its context must be able
    to say what made it, and for an installed copy the package version is exactly
    that answer.
    """
    from tradeflow.optimization.config_store import current_git_sha

    sha = current_git_sha()
    if sha:
        return sha
    from tradeflow import __version__

    return f"tradeflow {__version__}"


def _verdict_provenance(inputs: Dict[str, Any], cache, n_trials: int) -> Dict[str, Any]:
    """What a reader needs to not misread this run a month later: the git SHA, the
    campaign's trial count, and the measured proof that the steps really did share
    one fetch (requests issued vs. requests that reached the provider)."""
    return {
        "git_sha": code_version(),
        "generated_at": datetime.now().isoformat(),
        "n_trials": n_trials,
        "universe_size": len(inputs.get("universe") or []),
        "bar_requests": cache.stats() if cache is not None else None,
    }


def _verdict_gates(result: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the verdict from the gates the steps already computed.

    Every check here reads a number some step produced and compares it to that
    step's own threshold - nothing is re-derived, nothing is averaged. When the
    checks disagree the verdict is ``mixed`` and every check is shown; a summary
    that collapses disagreement into a single reassuring number is exactly the
    failure mode a one-line verdict invites.
    """
    from tradeflow.analytics import information as info

    steps = result.get("steps") or {}
    failed = [name for name, s in steps.items() if s.get("status") == "failed"]
    checks: Dict[str, Dict[str, Any]] = {}
    inf = result.get("information") or {}
    pf = result.get("portfolio") or {}
    diagnostics = pf.get("diagnostics") or {}

    if inf.get("periods"):
        tstat = float(inf.get("ic_tstat") or 0.0)
        checks["ic_tstat"] = {
            "value": tstat,
            "threshold": 2.0,
            "passed": abs(tstat) >= 2.0,
            "note": "IC t-stat below 2 is not distinguishable from luck",
        }
        realized, se = float(inf.get("realized_ir") or 0.0), float(inf.get("ir_standard_error") or 0.0)
        checks["ir_above_noise"] = {
            "value": realized,
            "threshold": se,
            "passed": abs(realized) > se,
            "note": "realized IR inside its own standard-error band is indistinguishable from zero",
        }
        checks["sanity_ceiling"] = {
            "value": realized,
            "threshold": 2.0,
            "passed": not bool(inf.get("sanity_ceiling_breached")),
            "note": "a realized IR above 2 on public data means suspect a bug or a leak, not skill",
        }
        checks["sample_size"] = {
            "value": int(inf.get("periods") or 0),
            "threshold": int(info.MIN_PERIODS),
            "passed": not bool(inf.get("low_sample")),
            "note": "too few rebalances to measure an IC with any confidence",
        }

    if pf.get("feasible"):
        net = diagnostics.get("expected_active_return_net")
        if net is None:
            net = diagnostics.get("expected_active_return")
        checks["net_of_cost_alpha"] = {
            "value": float(net or 0.0),
            "threshold": 0.0,
            "passed": float(net or 0.0) > 0.0,
            "note": "expected active return after the cost of trading into the book",
        }
    elif "portfolio" in steps and steps["portfolio"].get("status") == "ok":
        checks["portfolio_feasible"] = {
            "value": False,
            "threshold": True,
            "passed": False,
            "note": pf.get("binding_constraint") or "no feasible portfolio at these constraints",
        }

    if failed:
        # A report where the portfolio printed but the information step did not is an
        # invitation to act on unvalidated weights. There is no partial verdict.
        return {
            "verdict": "incomplete",
            "promotable": None,
            "summary": "incomplete — no verdict (steps failed: " + ", ".join(sorted(failed)) + ")",
            "failed_steps": sorted(failed),
            "checks": checks,
        }
    if not checks:
        return {
            "verdict": "incomplete",
            "promotable": None,
            "summary": "incomplete — no verdict (no gate produced a number)",
            "failed_steps": [],
            "checks": checks,
        }
    if "ic_tstat" not in checks:
        # The information step ran but measured nothing (too short a window, too few
        # sampleable rebalances). Whatever the portfolio says, there is no evidence
        # of skill to weigh it against - and a book with no skill evidence behind it
        # must never read as a pass.
        return {
            "verdict": "needs more data",
            "promotable": False,
            "summary": "needs more data — no measurable rebalances, so skill was never tested",
            "failed_steps": [],
            "checks": checks,
        }

    passed = [name for name, c in checks.items() if c["passed"]]
    failed_checks = [name for name, c in checks.items() if not c["passed"]]
    if not checks.get("sample_size", {"passed": True})["passed"]:
        verdict, summary = "needs more data", "needs more data — too few rebalances to judge"
    elif not failed_checks:
        verdict, summary = "promotable", "promotable — every gate passed"
    elif not passed:
        verdict, summary = "not promotable", "not promotable — no gate passed"
    else:
        verdict = "mixed"
        summary = (
            "mixed — passed: " + ", ".join(sorted(passed)) + "; failed: " + ", ".join(sorted(failed_checks))
        )
    return {
        "verdict": verdict,
        "promotable": verdict == "promotable",
        "summary": summary,
        "failed_steps": [],
        "checks": checks,
    }


def _verdict_weights_payload(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The book this run proposed, in the shape the trial store persists.

    Journaling the weights and the factor-exposure vector alongside the trial is
    what makes a result's holdings recoverable later without re-running the
    optimizer.
    """
    pf = result.get("portfolio") or {}
    weights = pf.get("weights")
    if not pf.get("feasible") or not weights:
        return None
    payload = {
        "as_of": pf.get("as_of"),
        "weights": {str(k): float(v) for k, v in weights.items()},
    }
    active = pf.get("active_weights")
    if active:
        payload["active_weights"] = {str(k): float(v) for k, v in active.items()}
    exposures = pf.get("exposures")
    if exposures:
        payload["exposures"] = _jsonable(exposures)
    return payload


def _verdict_artifact_path(trial_id: str) -> Path:
    """Where a journaled verdict's full composite object lives.

    Derived from the trial id rather than recorded anywhere, so nothing has to stay
    in sync: given a trial, its artifact is at a known place or it is absent.
    """
    return ARTIFACT_DIR / f"verdict_{trial_id}.json"


def _load_verdict_artifact(trial_id: str) -> Optional[Dict[str, Any]]:
    """A prior run's composite object, or ``None`` if it is missing or unreadable.

    ``None`` sends the caller down the re-run path, which is always safe; serving a
    half-loaded composite never is.
    """
    path = _verdict_artifact_path(trial_id)
    try:
        with path.open() as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("schema") == VERDICT_SCHEMA else None


def _journal_verdict(
    result: Dict[str, Any],
    strategy: str,
    universe: Sequence[str],
    start: datetime,
    end: datetime,
    dedup_params: Dict[str, Any],
) -> None:
    """Record the composite as exactly one trial, and keep its full object beside it.

    One row, not five: the steps ran as library calls, and journaling each one would
    quintuple the campaign's multiple-testing total for a single command.
    """
    from tradeflow.services.audit import journal_trial

    verdict = result.get("verdict") or {}
    inf = result.get("information") or {}
    diagnostics = (result.get("portfolio") or {}).get("diagnostics") or {}
    metrics = {
        "ic_tstat": inf.get("ic_tstat"),
        "mean_ic": inf.get("mean_ic"),
        "predicted_ir": inf.get("predicted_ir"),
        "realized_ir": inf.get("realized_ir"),
        "expected_active_return_net": diagnostics.get("expected_active_return_net"),
    }
    try:
        trial_id = journal_trial(
            "verdict",
            strategy=strategy,
            symbols=universe,
            start=start,
            end=end,
            params=_jsonable(dedup_params),
            metrics={k: v for k, v in metrics.items() if v is not None},
            extra={"verdict": verdict.get("verdict"), "promotable": verdict.get("promotable")},
            weights=_verdict_weights_payload(result),
        )
    except Exception:  # noqa: BLE001 - journaling is bookkeeping, not the answer
        logger.warning("Verdict trial journaling failed", exc_info=True)
        return
    result["trial_id"] = trial_id
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with _verdict_artifact_path(trial_id).open("w") as fh:
            json.dump(_jsonable(result), fh)
    except (OSError, TypeError, ValueError):
        logger.warning("Verdict artifact could not be written; a rerun will recompute", exc_info=True)


def compute_attribution(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "demo_volume",
    benchmark: str = "SPY",
    neutralize_factors: Sequence[str] = (),
    horizon: int = 5,
    n_points: int = 24,
    n_trials: int = 1,
    timeframe: str = "1Day",
    risk_model: str = "shrinkage",
    benchmark_holdings: str = "equal",
    benchmark_premium: float = 0.05,
    signals: Optional[Sequence[str]] = None,
    min_obs: int = 60,
    detail: bool = False,
    conditional: Optional[str] = None,
    conditional_lambda: Optional[float] = None,
    bootstrap_skill: bool = False,
    bootstrap_b: int = 2000,
    bootstrap_block_length: Optional[float] = None,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    """Attribute realized active return to systematic timing, risk factors, signals,
    and stock-picking - and confront the attributed t-stats with the same
    research-integrity guardrails ``compute_information`` applies to ICs.

    Read-only research-clock diagnostic. Mirrors ``compute_information``'s pattern
    exactly: at sampled rebalances it rebuilds a leakage-safe cross-section (bars
    strictly ``<= t``) - alpha (for the paper active book), risk-factor exposures,
    and a per-period covariance Σ(t) (for the canonical Σ-implied beta, "one β,
    everywhere") - then pairs it with the forward realized return over
    ``(t, t_fwd]``. There is no persisted weights/exposure history to consume (see
    the module-level deviation note next to ``compute_information``); this
    recomputes on the fly, the same as that function already does for alpha/IC.

    Each rebalance's active return is split, by an exact regression identity
    (:func:`tradeflow.analytics.attribution.attribute_period`), into: the systematic
    benchmark-timing bucket (``β_a(t)·r_B(t)``, further decomposed in aggregate into
    expected/surprise/timing), each risk factor (market/momentum/volatility/size),
    the strategy's own alpha as a signal column (plus any additional ``signals`` -
    other strategies' combined scores, so a ``--combine`` weight can be checked
    against its realized counterpart), and a specific (stock-picking) remainder.
    Every attributed t-stat uses a Bayesian-blended risk (short samples lean on the
    risk model instead of a wild few-point sample SD) and the whole ranked table is
    deflated by the same multiple-testing inflation ``compute_information`` applies
    to ICs - ranking ~8 attributed rows and quoting the best is exactly that trap,
    replayed here.

    ``conditional`` (default ``None`` / off) threads an EWMA/HAR-conditioned Σ(t)
    into the per-period covariance this function already rebuilds at every sampled
    rebalance; when set, the report adds ``te_by_regime`` — predicted TE (from
    Σ(t)) vs a realized-return-dispersion proxy, bucketed by the benchmark's own
    trailing realized-vol tercile as of each rebalance — the number that answers
    "does the tracking-error budget actually hold across vol regimes." This runs
    ONE Σ choice per call (conditional or not, not both side by side); the
    net-of-cost conditional-vs-unconditional comparison lives in
    ``run_conditional_risk_ab``.

    ``bootstrap_skill`` (default off) adds a nonparametric OWN p-value next to the
    parametric ``SE{IR}≈1/√Y`` verdict: a stationary block bootstrap of
    ``r_active_series`` under the imposed null (demeaned by its own estimated
    alpha), reported as ``bootstrap`` in the result and folded into ``verdict``.
    This is the *own* test only (a single track record, not a trial family) - the
    family Reality Check needs the trial store's stored trials and lives on
    ``run_walk_forward``'s ``--bootstrap-skill`` instead.
    """
    from tradeflow.analytics import attribution as attr
    from tradeflow.analytics import information as info
    from tradeflow.portfolio.benchmark import load_benchmark_weights, restrict_and_renormalize
    from tradeflow.risk.exposures import FACTOR_NAMES, build_factor_exposures

    run_id = new_run_id()
    strat = _strategy(strategy, None)
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    rebalances_per_year = periods_per_year / horizon

    bars = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, end, _window_days(start, end))
    bench = bars.get(benchmark)
    universe_bars = {sym: bars[sym] for sym in symbols if sym in bars}
    if bench is None or bench.empty or not universe_bars:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "periods": 0,
            "note": "Insufficient data: need a benchmark series and scored names.",
        }

    scorer = {
        "strategy": lambda: strategy_scorer(strat),
        "signal": lambda: signal_scorer(strat),
        "scanner": lambda: scanner_scorer(_scanner(scanner)),
    }[source]()
    ctx = AlphaContext(ic=DEFAULT_IC, neutralize=True, neutralize_factors=tuple(neutralize_factors))
    extra_scorers = {name: strategy_scorer(_strategy(name, None)) for name in (signals or ())}
    own_signal_col = f"alpha:{strategy}"

    index = bench.index
    lo, hi = _to_ts(start, index), _to_ts(end, index)
    window = index[(index >= lo) & (index <= hi)]
    points = _rebalance_points(len(window), horizon, n_points)

    risk_names = list(FACTOR_NAMES)
    signal_names = [own_signal_col, *extra_scorers]
    component_names = ["systematic", *risk_names, *signal_names, "specific"]
    series: Dict[str, List[float]] = {name: [] for name in component_names}
    r_active_series, r_bench_series, beta_a_series, psi2_series, bench_vol_series = [], [], [], [], []
    gross_series: List[float] = []  # the paper book's gross per rebalance, for the level scale

    for j in points:
        t, t_fwd = window[j], window[j + horizon]
        bars_t = {s: f.loc[f.index <= t] for s, f in universe_bars.items()}
        bars_t = {s: f for s, f in bars_t.items() if len(f) >= 2}
        bench_t = bench.loc[bench.index <= t]
        if len(bars_t) < 3 or bench_t.empty:
            continue

        matrix = _build_covariance(
            risk_model, bars_t, bench_t, periods_per_year, min_obs, conditional, conditional_lambda
        )
        if matrix is None or len(matrix.symbols) < 3:
            continue
        # Trailing realized benchmark vol as of t (causal — no forward data) — the
        # regime label the predicted-vs-realized TE split buckets by.
        bench_ret_t = bench_t["close"].pct_change().dropna()
        trailing_vol = (
            float(bench_ret_t.tail(max(horizon * 4, 20)).std() * np.sqrt(periods_per_year))
            if len(bench_ret_t) >= 5
            else float("nan")
        )
        raw_bench_w = load_benchmark_weights(benchmark_holdings, matrix.symbols)
        w_bench, _coverage = restrict_and_renormalize(raw_bench_w, matrix.symbols)
        if not w_bench:
            continue
        beta_per_name = matrix.implied_beta(w_bench)

        alpha = _alpha_cross_section(universe_bars, bench, scorer, periods_per_year, t, ctx)
        z = alpha - alpha.mean()
        if z.std() == 0 or z.dropna().empty:
            continue
        w_active = z / z.std()

        risk_x = build_factor_exposures(bars_t, bench_t, factors=risk_names)
        if risk_x.empty:
            continue

        signal_cols = {own_signal_col: z}
        for name, sc in extra_scorers.items():
            signal_cols[name] = _signal_cross_section(bars_t, sc)
        signal_x = pd.DataFrame(signal_cols).dropna(how="any")
        if signal_x.empty:
            continue

        r_raw = _forward_raw_return(universe_bars, t, t_fwd)
        bench_close = bench["close"]
        if t not in bench_close.index or t_fwd not in bench_close.index:
            continue
        r_bench = float(bench_close.loc[t_fwd] / bench_close.loc[t] - 1.0)

        result = attr.attribute_period(w_active, risk_x, r_raw, beta_per_name, r_bench, signal_x=signal_x)
        if result is None:
            continue

        series["systematic"].append(result.systematic)
        for name in risk_names:
            series[name].append(result.factor_contributions.get(name, 0.0))
        for name in signal_names:
            series[name].append(result.signal_contributions.get(name, 0.0))
        series["specific"].append(result.specific)
        r_active_series.append(result.r_active)
        r_bench_series.append(r_bench)
        beta_a_series.append(result.beta_a)
        w_vec = w_active.reindex(matrix.symbols).fillna(0.0).to_numpy()
        psi2_series.append(float(w_vec @ matrix.sigma @ w_vec))
        gross_series.append(float(np.abs(w_vec).sum()))
        bench_vol_series.append(trailing_vol)

    # The paper book is a z-scored alpha vector - mean-zero, unit cross-sectional SD,
    # deliberately not normalized to unit gross, because scale cancels in an IR. It
    # does not cancel in a *level*: with 61 names that book carries roughly 50x
    # notional, which is how a reported mean of 644%/yr and a tracking error of 374%
    # arise from a perfectly sound IR of 1.7. Printed as percentages they read as fund
    # returns, which is a trap rather than a diagnostic.
    #
    # So every level is divided by one constant - the book's mean gross - which leaves
    # every ratio built from these series exactly unchanged: IR is mean/vol and the
    # t-stat is mean/sd x sqrt(t), and a constant cancels in both. Levels become
    # readable as a unit-gross long-short book; IR and t-stat are bit-identical.
    # Per-period normalization would not be: a time-varying divisor reshapes the
    # series and moves the IR with it.
    gross_scale = float(np.mean(gross_series)) if gross_series else 0.0
    if gross_scale > 0:
        for key in list(series):
            series[key] = [value / gross_scale for value in series[key]]
        r_active_series = [value / gross_scale for value in r_active_series]
        beta_a_series = [value / gross_scale for value in beta_a_series]
        psi2_series = [value / (gross_scale**2) for value in psi2_series]  # a variance

    periods = len(r_active_series)
    if periods < 5:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "periods": periods,
            "note": "Insufficient overlapping history for attribution (need >= 5 rebalances "
            "with a buildable Σ, benchmark weights, and factor exposures).",
        }

    # T_eff-honest T0: the risk model's own min_obs, converted from
    # bars to this attribution's rebalance-period units.
    t0 = attr.prior_weight_t0(min_obs, horizon)
    psi2_bar = float(np.mean(psi2_series)) if psi2_series else 0.0
    n_rows = len(risk_names) + len(signal_names) + 2  # + timing + specific
    sigma2_prior_per_row = psi2_bar / n_rows if n_rows else 0.0

    rows: Dict[str, Any] = {}
    mu_b_period = benchmark_premium * horizon / periods_per_year
    split = attr.systematic_split(beta_a_series, r_bench_series, mu_b_period)
    rows["beta_expected"] = {
        "total": split["expected"],
        "note": "not skill (assumed premium x mean active beta)",
    }
    rows["beta_surprise"] = {
        "total": split["surprise"],
        "note": "not skill (benchmark outturn vs the assumed premium x mean active beta)",
    }
    rows["timing"] = {
        "total": split["timing"],
        **attr.series_stats(split["timing_series"], rebalances_per_year, sigma2_prior_per_row, t0),
    }
    for name in [*risk_names, *signal_names]:
        rows[name] = {
            "total": float(np.sum(series[name])),
            **attr.series_stats(series[name], rebalances_per_year, sigma2_prior_per_row, t0),
        }
    rows["specific"] = {
        "total": float(np.sum(series["specific"])),
        **attr.series_stats(series["specific"], rebalances_per_year, sigma2_prior_per_row, t0),
    }

    # Share of variance across the "real" (skill-claiming) rows only - an
    # approximation (rows correlate, so shares don't sum exactly to total ψ²).
    skill_rows = ["timing", *risk_names, *signal_names, "specific"]
    skill_series = {"timing": split["timing_series"], "specific": series["specific"]}
    skill_series.update({name: series[name] for name in [*risk_names, *signal_names]})
    variances = {
        name: (float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0) for name, vals in skill_series.items()
    }
    total_var = sum(variances.values())
    for name in skill_rows:
        rows[name]["share_of_variance"] = float(variances[name] / total_var) if total_var > 0 else 0.0

    r_portfolio_series = [rb + ra for rb, ra in zip(r_bench_series, r_active_series)]
    cumulation = attr.cumulate_top_down(
        {name: series[name] for name in ["systematic", *risk_names, *signal_names, "specific"]},
        r_active_series,
        r_portfolio_series,
        r_bench_series,
    )
    cumulation_unreliable = bool(
        abs(cumulation["honest_car"]) > 1e-9
        and abs(cumulation["delta_cp"]) > 0.2 * abs(cumulation["honest_car"])
    )

    total_active_ir = 0.0
    if periods > 1 and np.std(r_active_series) > 0:
        total_active_ir = float(
            np.mean(r_active_series) / np.std(r_active_series) * np.sqrt(rebalances_per_year)
        )
    years = max((end - start).days / 365.25, 1e-9)
    total_ir_se = info.ir_standard_error(total_active_ir, years)
    inflation = info.multiple_testing_inflation(n_trials)

    best_row = max(skill_rows, key=lambda name: abs(rows[name].get("t_stat", 0.0)))
    te_by_regime = _te_by_regime(psi2_series, r_active_series, bench_vol_series, rebalances_per_year)

    bootstrap_report = None
    verdict = (
        "distinguishable from luck"
        if abs(total_active_ir) / max(total_ir_se, 1e-9) >= 2
        else "NOT distinguishable from luck"
    )
    if bootstrap_skill:
        from tradeflow.analytics import bootstrap as boot

        bootstrap_report = boot.bootstrap_null(
            np.asarray(r_active_series, dtype=float),
            B=bootstrap_b,
            block_length=bootstrap_block_length,
            seed=bootstrap_seed,
            periods_per_year=rebalances_per_year,
        )
        if not bootstrap_report["insufficient_data"]:
            verdict += (
                f"; bootstrap own-p={bootstrap_report['p_value']:.3f} "
                f"(B={bootstrap_report['B']}, L={bootstrap_report['block_length']:.1f})"
            )

    result_dict: Dict[str, Any] = {
        "run_id": run_id,
        "strategy": strategy,
        "source": source,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "horizon_bars": horizon,
        "periods": periods,
        "low_sample": periods < info.MIN_PERIODS,
        "risk_factor_names": risk_names,
        "signal_names": signal_names,
        "conditional": conditional,
        "te_by_regime": _jsonable(te_by_regime),
        "rows": _jsonable(rows),
        "systematic_split": _jsonable({k: v for k, v in split.items() if k != "timing_series"}),
        "cumulation": _jsonable({k: v for k, v in cumulation.items()}),
        "cumulation_unreliable": cumulation_unreliable,
        "total_active_ir": total_active_ir,
        "total_active_ir_se": total_ir_se,
        "years": years,
        "years_to_significance": attr.years_to_significance(total_active_ir),
        "prob_positive_over_window": attr.prob_positive_over_years(total_active_ir, years),
        "n_rows": n_rows,
        "n_trials": n_trials,
        "multiple_testing_inflation": inflation,
        "best_row": best_row,
        "best_row_t_stat": rows[best_row].get("t_stat", 0.0),
        "sanity_ceiling_breached": abs(total_active_ir) > 2.0,
        "prior_weight_t0": t0,
        "sigma2_prior_per_row": sigma2_prior_per_row,
        "verdict": verdict,
        "bootstrap": _jsonable(bootstrap_report) if bootstrap_report else None,
        "note": "Every attributed row sums exactly to the realized active return per period "
        "(regression identity); the systematic bucket further splits (in aggregate) into "
        "expected/surprise (not skill) and timing (real, but noisy - always check its own "
        "t-stat). Per-row risk is a Bayesian blend of the risk model's structural prior and "
        "the row's own realized variance; a ranked table of "
        f"{n_rows} rows is a multiple-testing family - P(any |t|>2 in {n_trials} trials) = "
        f"{inflation:.2f} - so quoting the single best row (here: {best_row}) without that "
        "context is exactly the same trap the IC guardrails guard against. Cumulative active "
        "return is ΠR_P - ΠR_B, never Π(1+r_active); cumulation.delta_cp is the honest leftover "
        "from top-down chain-linking the per-period split, reported not hidden. te_by_regime "
        "buckets rebalances by the benchmark's own trailing realized-vol tercile and shows "
        "predicted TE (from Σ(t)) next to a realized-dispersion proxy per bucket — the number "
        "that says whether the tracking-error budget holds through a stress regime.",
    }
    if detail:
        result_dict["detail"] = _jsonable(
            {
                "r_active": r_active_series,
                "r_bench": r_bench_series,
                "beta_a": beta_a_series,
                **series,
            }
        )
    return result_dict


def _te_by_regime(
    psi2_series: List[float],
    r_active_series: List[float],
    bench_vol_series: List[float],
    rebalances_per_year: float,
) -> Dict[str, Any]:
    """Bucket rebalances by the benchmark's trailing realized-vol tercile (ex-post
    labels, report-time only — no look-ahead in the model itself) and compare, per
    bucket, the **predicted** TE (``sqrt(mean psi2)``, from the per-period Σ(t)
    already built for attribution) against a **realized** dispersion proxy
    (``std(r_active)·sqrt(rebalances_per_year)``) — the number that answers
    whether the tracking-error budget holds through a stress regime, or breaches
    it the way an unconditional Σ mechanically must.
    """
    vols = np.asarray(bench_vol_series, dtype=float)
    finite = np.isfinite(vols)
    if int(finite.sum()) < 6:
        return {}
    q1, q2 = np.quantile(vols[finite], [1 / 3, 2 / 3])
    labels = np.where(vols <= q1, "low", np.where(vols <= q2, "mid", "high"))

    psi2 = np.asarray(psi2_series, dtype=float)
    r_active = np.asarray(r_active_series, dtype=float)
    out: Dict[str, Any] = {}
    for label in ("low", "mid", "high"):
        mask = (labels == label) & finite
        n = int(mask.sum())
        if n == 0:
            out[label] = {"n": 0}
            continue
        predicted_te = float(np.sqrt(max(np.mean(psi2[mask]), 0.0)))
        realized_te = float(np.std(r_active[mask], ddof=1) * np.sqrt(rebalances_per_year)) if n > 1 else 0.0
        out[label] = {
            "n": n,
            "predicted_te": predicted_te,
            "realized_te": realized_te,
            "gap": realized_te - predicted_te,
        }
    return out


def _signal_cross_section(bars_t: Dict[str, pd.DataFrame], scorer) -> pd.Series:
    """Cross-sectional z-score of a raw scorer's output at ``t`` (bars already
    sliced to ``<= t`` by the caller) - the same winsorize -> zscore steps the
    alpha pipeline applies, used here as a combined-signal exposure column."""
    from tradeflow.alphas import refine

    raw: Dict[str, float] = {}
    for sym, frame in bars_t.items():
        if len(frame) < 2:
            continue
        val = scorer(frame)
        if val is not None and val == val:
            raw[sym] = float(val)
    s = pd.Series(raw)
    if len(s) < 2 or s.std() == 0:
        return pd.Series(dtype=float)
    return refine.zscore(refine.winsorize(s))


def run_scaling_ab(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "demo_volume",
    benchmark: str = "SPY",
    neutralize: bool = True,
    neutralize_factors: Sequence[str] = (),
    ic_prior: float = DEFAULT_IC,
    horizon: int = 5,
    n_points: int = 24,
    timeframe: str = "1Day",
    price_derived: bool = True,
) -> Dict[str, Any]:
    """A/B the two scalings walk-forward: realized IR under Case 1 vs Case 2.

    The regression-based :func:`~tradeflow.alphas.refine.case_test` is one cheap number; this
    is the ground-truth tiebreak. At each rebalance it builds
    the paper alpha book under **both** scalings (same z, same measured residual return;
    only the per-name vol multiply differs) and reports each book's realized information
    ratio, alongside the regression's recommendation. When the two disagree, trust the
    A/B — but note the IR standard-error band before acting on a small gap.
    """
    run_id = new_run_id()
    strat = _strategy(strategy, None)
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    rebalances_per_year = periods_per_year / horizon

    bars = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, end, _window_days(start, end))
    bench = bars.get(benchmark)
    universe_bars = {sym: bars[sym] for sym in symbols if sym in bars}
    if bench is None or bench.empty or not universe_bars:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "periods": 0,
            "note": "Insufficient data: need a benchmark series and scored names.",
        }

    scorer = {
        "strategy": lambda: strategy_scorer(strat),
        "signal": lambda: signal_scorer(strat),
        "scanner": lambda: scanner_scorer(_scanner(scanner)),
    }[source]()
    ctxs = {
        kind: AlphaContext(
            ic=ic_prior, neutralize=neutralize, neutralize_factors=tuple(neutralize_factors), scaling=kind
        )
        for kind in ("case1", "case2")
    }

    index = bench.index
    window = index[(index >= _to_ts(start, index)) & (index <= _to_ts(end, index))]
    points = _rebalance_points(len(window), horizon, n_points)

    returns: Dict[str, List[float]] = {"case1": [], "case2": []}
    for j in points:
        t, t_fwd = window[j], window[j + horizon]
        resid = _forward_residual_return(universe_bars, bench, t, t_fwd, indicators=_indicators())
        if resid.dropna().empty:
            continue
        for kind, ctx in ctxs.items():
            alpha = _alpha_cross_section(universe_bars, bench, scorer, periods_per_year, t, ctx)
            aligned = pd.concat([alpha, resid], axis=1, keys=["a", "r"]).dropna()
            if len(aligned) < 5:
                continue
            z = aligned["a"] - aligned["a"].mean()
            if z.std() > 0:
                returns[kind].append(float((z / z.std()) @ aligned["r"]))

    def _ir(r: List[float]) -> float:
        if len(r) > 1 and np.std(r) > 0:
            return float(np.mean(r) / np.std(r) * np.sqrt(rebalances_per_year))
        return 0.0

    case1_ir, case2_ir = _ir(returns["case1"]), _ir(returns["case2"])

    # Regression recommendation needs the per-name residual vol as of the window end.
    panel = FeaturePanel.for_universe(end, list(universe_bars))
    add_risk_features(panel, universe_bars, bench, periods_per_year)
    resid_vol = panel.get("residual_vol") if panel.has("residual_vol") else pd.Series(dtype=float)
    case_diag = _run_case_test(universe_bars, scorer, resid_vol, price_derived)

    ab_pick = "case2" if case2_ir > case1_ir else "case1"
    return {
        "run_id": run_id,
        "strategy": strategy,
        "source": source,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "horizon_bars": horizon,
        "periods": min(len(returns["case1"]), len(returns["case2"])),
        "case1_realized_ir": case1_ir,
        "case2_realized_ir": case2_ir,
        "regression_pick": f"case{case_diag['case']}",
        "ab_pick": ab_pick,
        "agree": bool(f"case{case_diag['case']}" == ab_pick),
        "case_test": _jsonable(case_diag),
        "note": "case1 = ω·IC·z, case2 = IC·c_g·z. A/B realized IR is the ground truth; "
        "the regression case_test is the cheap proxy. Compare the IR gap to its "
        "standard-error band before acting — a small gap is noise.",
    }


def _indicators():
    """The indicators module (deferred import to keep the module load light)."""
    from tradeflow.indicators import indicators

    return indicators


def compute_horizon(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "demo_volume",
    benchmark: str = "SPY",
    neutralize: bool = True,
    neutralize_factors: Sequence[str] = (),
    ic_prior: float = DEFAULT_IC,
    max_lag: int = 10,
    n_points: int = 20,
    timeframe: str = "1Day",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure an alpha's decay and recommend a rebalance cadence + lagged blend.

    Read-only research diagnostic (see the Information-horizon page in the engineering
    docs): measures the IC-vs-lag profile (the
    alpha at ``t`` vs the residual return realized ``n`` periods later, for
    ``n = 1..max_lag``), fits the per-period decay ``δ`` and half-life, derives the
    cadence that maximizes ``IC(Δt)·√(1/Δt)``, and computes the IR-maximizing
    current/lagged blend from ``δ`` and the signal's autocorrelation. The half-life is
    the holding period transaction cost should be amortized over.
    """
    from tradeflow.alphas import horizon as hz
    from tradeflow.indicators import indicators

    run_id = new_run_id()
    strat = _strategy(strategy, config)
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()

    bars = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, end, _window_days(start, end))
    bench = bars.get(benchmark)
    universe_bars = {sym: bars[sym] for sym in symbols if sym in bars}
    if bench is None or bench.empty or not universe_bars:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "periods": 0,
            "note": "Insufficient data: need a benchmark series and scored names.",
        }

    scorer = {
        "strategy": lambda: strategy_scorer(strat),
        "signal": lambda: signal_scorer(strat),
        "scanner": lambda: scanner_scorer(_scanner(scanner)),
    }[source]()
    ctx = AlphaContext(ic=ic_prior, neutralize=neutralize, neutralize_factors=tuple(neutralize_factors))

    index = bench.index
    window = index[(index >= _to_ts(start, index)) & (index <= _to_ts(end, index))]
    last = len(window) - max_lag - 1
    if last <= 30:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "periods": 0,
            "note": "Window too short for the requested max_lag.",
        }
    points = np.linspace(30, last, num=min(n_points, last - 30), dtype=int)

    ic_by_lag: Dict[int, List[float]] = {n: [] for n in range(1, max_lag + 1)}
    prev_alpha = None
    autocorrs: List[float] = []
    for j in points:
        t = window[j]
        alpha = _alpha_cross_section(universe_bars, bench, scorer, periods_per_year, t, ctx)
        if alpha.dropna().std() == 0 or alpha.dropna().empty:
            continue
        for n in range(1, max_lag + 1):
            resid = _forward_residual_return(universe_bars, bench, t, window[j + n], indicators)
            pair = pd.concat([alpha, resid], axis=1).dropna()
            if len(pair) >= 5 and pair.iloc[:, 1].std() > 0:
                ic_by_lag[n].append(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])))
        if prev_alpha is not None:
            joint = pd.concat([alpha, prev_alpha], axis=1).dropna()
            if len(joint) >= 5 and joint.iloc[:, 0].std() > 0 and joint.iloc[:, 1].std() > 0:
                autocorrs.append(float(joint.iloc[:, 0].corr(joint.iloc[:, 1])))
        prev_alpha = alpha

    ic_profile = {n: float(np.mean(v)) for n, v in ic_by_lag.items() if v}
    fit = hz.fit_decay(ic_profile)
    rho = float(np.mean(autocorrs)) if autocorrs else 0.0
    w_now, w_lag = hz.blend_weights(fit["delta"], rho) if fit["delta"] == fit["delta"] else (1.0, 0.0)
    cadence = hz.recommended_cadence(ic_profile)

    # Net-of-cost guard: the lagged leg adds turnover; price it and only recommend the
    # blend when it diversifies (adds independent info) and its annual cost is modest.
    from tradeflow.costs import ParametricCostModel

    rebalances_per_year = periods_per_year / max(cadence, 1)
    blend_cost = abs(w_lag) * ParametricCostModel().turnover_cost_rate() * rebalances_per_year
    blend_recommended = (w_lag > 1e-3) and (blend_cost < _BLEND_COST_CEILING)

    return {
        "run_id": run_id,
        "strategy": strategy,
        "source": source,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "ic_by_lag": _jsonable(ic_profile),
        "decay_delta": fit["delta"],
        "half_life": fit["half_life"],
        "decay_r_squared": fit["r_squared"],
        "peak_return_horizon": hz.peak_return_horizon(fit["half_life"]),
        "frequency_ir_curve": _jsonable(hz.frequency_ir_curve(ic_profile)),
        "recommended_cadence": cadence,
        "signal_autocorrelation": rho,
        "blend_weight_now": w_now,
        "blend_weight_lagged": w_lag,
        "blend_regime": "diversify" if w_lag > 1e-6 else "hedge" if w_lag < -1e-6 else "latest-only",
        "blend_annual_cost": blend_cost,
        "blend_recommended": blend_recommended,
        "half_life_lower": fit.get("half_life_lower"),
        "half_life_upper": fit.get("half_life_upper"),
        "decay_slope_se": fit.get("decay_slope_se"),
        "blend_superseded_by": "the aim-in-front partial-adjustment policy "
        "('allocate --policy aim'): this lagged-blend recommendation is a special case "
        "of that policy's per-signal decay discount (a two-point blend vs a continuous "
        "κ/(κ+φ) discount) - prefer --policy aim for new work; this report stays "
        "accurate on its own terms either way.",
        "note": "δ is the per-period IC decay (HL = half-life, with a "
        "half_life_lower/half_life_upper confidence band from the fit's own slope SE). "
        "Rebalance near the cadence that maximizes IC·√(1/Δt); amortize cost over the "
        "half-life. The lagged blend is recommended only when it diversifies and its "
        "turnover cost is modest.",
    }


def compute_risk(
    data_client: MarketDataClient,
    symbols: List[str],
    as_of: datetime,
    model: str = "shrinkage",
    benchmark: str = "SPY",
    lookback_days: int = 365,
    timeframe: str = "1Day",
    min_obs: int = 60,
    conditional: Optional[str] = None,
    conditional_lambda: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate the universe's covariance Σ and summarize its risk structure.

    Read-only research-clock flow: scans returns up to ``as_of`` (leakage-safe),
    estimates an annualized, well-conditioned Σ (``shrinkage`` = Ledoit–Wolf,
    ``factor`` = structural ``X F Xᵀ + Δ``, ``sample`` = raw), and returns a compact
    summary - shrinkage δ, condition number, mean correlation, equal-weight portfolio
    volatility, top risk contributors, and (factor model) the factor-vs-specific risk
    split. Σ itself is not inlined; this is the diagnostic the optimizer consumes.

    ``conditional`` (default ``None`` / **off** — the MZ/QLIKE evidence gate hasn't
    cleared this repo's own data yet, see ``evaluate_conditional_risk``) conditions
    Σ_t's volatilities via an EWMA (``"ewma"``) or HAR-lite (``"har"``) per-name
    forecast, holding the correlation structure fixed. When set, the report adds
    ``sigma_regime`` — the current conditional/unconditional vol ratio per name,
    the "how stressed is the book right now" diagnostic (the construction and
    cost-aware optimizer both consume Σ_t transparently; this is the number a
    human reads).
    """
    from tradeflow.risk import COVARIANCE_MODELS
    from tradeflow.risk.factor import FactorRiskMatrix

    if model not in COVARIANCE_MODELS:
        raise ValueError(f"model must be one of {sorted(COVARIANCE_MODELS)}, got {model!r}")

    run_id = new_run_id()
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    fetched = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, as_of, lookback_days)
    bars = {s: fetched[s] for s in symbols if s in fetched}
    matrix = _build_covariance(
        model, bars, fetched.get(benchmark), periods_per_year, min_obs, conditional, conditional_lambda
    )

    if matrix is None:
        return {
            "run_id": run_id,
            "model": model,
            "as_of": as_of.isoformat(),
            "universe_size": 0,
            "note": f"Insufficient history: no name has >= {min_obs} aligned returns.",
        }

    names = matrix.symbols
    weights = {sym: 1.0 / len(names) for sym in names}
    vols = matrix.volatilities()
    mcr = matrix.marginal_contribution_to_risk(weights)
    # Each name's contribution to the equal-weight portfolio vol (sums to the total).
    contribution = {sym: weights[sym] * mcr[sym] for sym in names}
    corr = matrix.correlation().to_numpy()
    n = len(names)
    mean_corr = float((corr.sum() - n) / (n * (n - 1))) if n > 1 else 0.0

    top = sorted(
        (
            {"symbol": s, "volatility": float(vols[s]), "risk_contribution": float(contribution[s])}
            for s in names
        ),
        key=lambda r: r["risk_contribution"],
        reverse=True,
    )[:TOP_N]

    result = {
        "run_id": run_id,
        "model": model,
        "as_of": as_of.isoformat(),
        "timeframe": timeframe,
        "universe_size": n,
        "shrinkage": matrix.shrinkage,
        "condition_number": matrix.condition_number(),
        "positive_definite": matrix.is_positive_definite(),
        "mean_correlation": mean_corr,
        "equal_weight_volatility": matrix.volatility(weights),
        "top_risk_contributors": _jsonable(top),
        "conditional": conditional,
        "note": "Σ is annualized and kept invertible (shrinkage δ, or a structural "
        "factor model). Risk is not additive — correlated names are one bet. This is "
        "the denominator the portfolio optimizer divides alpha by.",
    }
    if matrix.conditional_diagnostics:
        result["sigma_regime"] = _jsonable(matrix.conditional_diagnostics)

    # The factor model makes risk attributable: split the equal-weight portfolio's
    # variance into common-factor risk and idiosyncratic (specific) risk.
    if isinstance(matrix, FactorRiskMatrix):
        total_var = matrix.variance(weights)
        factor_var = matrix.factor_variance(weights)
        result["factor_names"] = matrix.factor_names
        result["factor_risk_share"] = float(factor_var / total_var) if total_var > 0 else 0.0
        result["specific_risk_share"] = float(1.0 - factor_var / total_var) if total_var > 0 else 0.0

    return result


def evaluate_conditional_risk(
    data_client: MarketDataClient,
    symbols: List[str],
    start: datetime,
    end: datetime,
    timeframe: str = "1Day",
    min_obs: int = 60,
    n_points: int = 60,
    conditional_lambda: Optional[float] = None,
) -> Dict[str, Any]:
    """The MZ/QLIKE evidence gate: per name AND pooled across the universe,
    compare EWMA / HAR / unconditional (expanding trailing) one-bar-ahead variance
    forecasts against realized ``r²`` — Mincer–Zarnowitz (``b`` near 1 is
    well-calibrated) and QLIKE (lower is better), split by realized-vol tercile.
    **This is the gate that decides whether ``--conditional`` is worth turning on
    for this repo's own data** — not a preference. Read-only, no orders, no
    feedback into any model.
    """
    from tradeflow.risk.conditional import evaluate_vol_forecasts, mincer_zarnowitz, qlike_loss

    run_id = new_run_id()
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    bars = ClientBarSource(data_client).scan(symbols, timeframe, end, _window_days(start, end))

    per_name: Dict[str, Any] = {}
    pooled_realized: List[float] = []
    pooled_forecasts: Dict[str, List[float]] = {"ewma": [], "har": [], "unconditional": []}
    for sym in symbols:
        frame = bars.get(sym)
        if frame is None or len(frame) < min_obs + 10:
            continue
        returns = frame["close"].pct_change().dropna()
        evaluation = evaluate_vol_forecasts(
            returns,
            min_obs=min_obs,
            n_points=n_points,
            lambda_=conditional_lambda,
            periods_per_year=periods_per_year,
        )
        if evaluation.n_points < 10:
            continue
        per_name[sym] = {
            "n_points": evaluation.n_points,
            "by_method": {
                m: {"qlike": e["qlike"], "mincer_zarnowitz": e["mincer_zarnowitz"]}
                for m, e in evaluation.by_method.items()
            },
        }
        pooled_realized.extend(evaluation.realized.tolist())
        for method in pooled_forecasts:
            pooled_forecasts[method].extend(evaluation.forecasts[method].tolist())

    if not per_name:
        return {
            "run_id": run_id,
            "n_names": 0,
            "note": "Insufficient history: no name has enough returns to evaluate "
            f"(need >= min_obs({min_obs}) + 10).",
        }

    realized_arr = np.array(pooled_realized)
    pooled: Dict[str, Any] = {}
    for method, values in pooled_forecasts.items():
        arr = np.array(values)
        pooled[method] = {
            "mincer_zarnowitz": mincer_zarnowitz(realized_arr, arr),
            "qlike": qlike_loss(realized_arr, arr),
        }

    ranked = sorted(pooled.items(), key=lambda kv: kv[1]["qlike"])
    best_method = ranked[0][0]
    uncond = pooled["unconditional"]
    best = pooled[best_method]
    # Both prongs required — QLIKE improvement
    # ALONE isn't the gate: a method that "wins" on QLIKE while its calibration (MZ b)
    # is farther from 1 than the unconditional baseline's is not honestly better, it's
    # noise. Both must point the same way for gate_passed=True.
    qlike_improves = best_method != "unconditional" and best["qlike"] < uncond["qlike"]
    mz_improves = abs(best["mincer_zarnowitz"]["b"] - 1.0) < abs(uncond["mincer_zarnowitz"]["b"] - 1.0)
    gate_passed = bool(qlike_improves and mz_improves)

    return {
        "run_id": run_id,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "n_names": len(per_name),
        "n_points_per_name": {s: v["n_points"] for s, v in per_name.items()},
        "pooled": _jsonable(pooled),
        "per_name": _jsonable(per_name),
        "best_method_pooled_qlike": best_method,
        "gate_passed": gate_passed,
        "note": "The evidence gate: QLIKE lower is better, "
        "Mincer-Zarnowitz b near 1.0 is well-calibrated. 'gate_passed' is TRUE only when "
        "the best conditional method (ewma/har) BOTH pools a lower QLIKE AND a better-"
        "calibrated MZ slope than the unconditional trailing baseline on THIS "
        "universe/window — a QLIKE nudge with worse calibration is noise, not a win. If "
        "it's FALSE, the honest reading is that conditioning doesn't earn its keep here, "
        "not a bug to force past.",
    }


def construct_portfolio(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    as_of: datetime,
    config: Optional[Dict[str, Any]] = None,
    source: str = "strategy",
    scanner: str = "demo_volume",
    target_te: float = 0.04,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
    max_names: Optional[int] = None,
    benchmark: str = "SPY",
    neutralize: bool = True,
    neutralize_factors: Sequence[str] = (),
    risk_model: str = "shrinkage",
    lookback_days: int = 365,
    timeframe: str = "1Day",
    capital: Optional[float] = None,
    current_weights: Optional[Dict[str, float]] = None,
    holding_period_years: float = 1.0 / 12.0,
    cost_aware: bool = True,
    benchmark_holdings: Optional[str] = None,
    benchmark_premium: float = 0.05,
    book: str = "long_only",
    gross_leverage: Optional[float] = None,
    short_max_weight: float = 0.0,
    conditional: Optional[str] = None,
    conditional_lambda: Optional[float] = None,
    posterior: Optional[str] = None,
    posterior_ic: Optional[float] = None,
    posterior_t_eff: Optional[float] = None,
    posterior_tau: Optional[float] = None,
    policy: Optional[str] = None,
    trade_rate: Optional[float] = None,
    decay_lookback_days: int = 365,
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
) -> Dict[str, Any]:
    """Construct the utility-maximizing portfolio from alphas and Σ.

    Read-only research-clock flow: scans the universe as of ``as_of``, builds
    benchmark-neutral alphas and an annualized covariance Σ, then maximizes
    ``αᵀw − λ·wᵀΣw − cost(w − w₀)`` over long-only, box-bounded, budgeted (and
    optionally cardinality-capped) weights, calibrating ``λ`` to ``target_te``.

    With ``cost_aware`` (default) the objective carries the transaction-cost term:
    name-specific linear turnover (commission + a high-low-range spread proxy) and,
    when ``capital`` is set, the √-impact term - so the optimizer trades a name's
    alpha against *that name's* cost and a no-trade band emerges from the cost itself.
    ``cost_aware=False`` recovers the cost-blind (gross) solve with an ex-post drag.

    ``benchmark_holdings`` makes the benchmark a **portfolio** (``w_B``) rather
    than the ``benchmark`` return series above (which stays a beta/vol regression
    input, orthogonal to this): ``"equal"`` for uniform weight over the
    Σ-covered universe, or a ``symbol,weight`` CSV/JSON holdings file. Tracking
    error, alpha neutralization, and the transfer coefficient all move into active
    space (``w_a = w − w_B``); ``benchmark_premium`` (``μ_B``, an assumed annual
    benchmark excess return) drives the reverse-optimization report (the consensus
    returns for which ``w_B`` is itself optimal). Without ``benchmark_holdings``
    this is a no-op - every quantity reduces byte-for-byte to the cash-relative
    behavior.

    ``book="market_neutral"`` relaxes the long-only box to
    ``[−short_max_weight, max_weight]`` and the budget to ``Σw=0``; ``gross_leverage``
    (``‖w‖₁ ≤ L``) is then mandatory - see
    :meth:`~tradeflow.portfolio.optimizer.MeanVarianceOptimizer.optimize`. Borrow carry on
    the short book is priced automatically from the cost model's flat default when
    ``cost_aware``; a per-name override belongs in a future ``CostInputs.borrow`` feed
    (v1 has no per-name borrow-rate source at the service layer). ``book="long_only"``
    (the default) is unaffected.

    Returns the proposed weights plus the Fundamental-Law report (IR*, predicted TE/IR,
    transfer coefficient, turnover, cost split). This is a **proposal** - no orders.

    ``conditional`` (default ``None`` / **off** — see ``evaluate_conditional_risk``
    for the evidence-gate finding that decided the default) conditions Σ's
    volatilities (EWMA/HAR) before the solve, so ``target_te`` is measured against
    *current* risk, not the trailing-window average — the whole point being that
    the optimizer sells into a vol spike to hold the TE budget (mechanically
    correct, but it pays the real transaction cost to do it; see ``sigma_regime``
    in the diagnostics for how stressed Σ_t is relative to the unconditional
    estimate).

    ``posterior="bl"`` (default ``None`` / off until validated OOS) blends the
    refined alphas with the consensus prior via Black–Litterman before the solve:
    names with no signal get a real, Σ-propagated posterior instead of being
    excluded outright, and view confidence is tied to
    ``posterior_ic``/``posterior_t_eff`` rather than baked only into magnitude.
    ``posterior_t_eff`` is required when set (τ is pinned to ``1/T_eff``, never
    tuned — pass the ``effective_t`` a prior ``compute_information`` call
    measured); ``posterior_ic`` defaults to the same assumed IC the refinement
    step used; ``posterior_tau`` overrides the pinned τ. The report gains a
    ``posterior`` section (per-name consensus/view/posterior/source table, plus
    τ-sensitivity) and ``shrink_chain`` gains a ``bl`` step - the IC-uncertainty
    haircut moves from the refine step (which stays unshrunk for this path,
    avoiding a double-shrink) into Ω.

    ``policy="aim"`` (default ``None`` / off until validated OOS net of cost
    against the plain myopic solve above - see ``info --policy-ab``) replaces the
    myopic "jump to this period's optimum" with an aim-in-front-of-the-target
    partial adjustment: the alpha is discounted by ``κ/(κ+φ)`` (``φ`` the
    strategy's own measured decay rate, conservatively from the *upper* half-life
    confidence bound over the trailing ``decay_lookback_days``), the aim portfolio
    is the cost-aware solve on that discounted alpha with cost zeroed, and the
    trade is ``κ`` of the gap to the aim, banded by the same no-trade-band
    machinery. ``κ`` is derived from the book's risk-aversion and cost curvature
    (see ``tradeflow/portfolio/policy.py``); ``trade_rate`` overrides it. Falls back to
    exactly the plain cost-aware solve above (not an error) when the cost
    curvature can't be pinned (no ``capital``, or too little turnover to fit one)
    - see ``diagnostics["aim_degraded"]``. Long-only only in v1; incompatible with
    ``book="market_neutral"`` or ``benchmark_holdings``.

    ``commission_bps``/``impact_eta``/``participation_cap``/``borrow_bps`` set the
    cost assumptions the objective prices turnover with. They default to the cost
    model's own defaults, so omitting them is exactly the previous behavior; a
    composite run passes the same values it gives every other step, so one report
    never mixes two cost worlds.
    """
    from tradeflow.costs import ParametricCostModel
    from tradeflow.portfolio.optimizer import MeanVarianceOptimizer

    run_id = new_run_id()
    strat = _strategy(strategy, config)
    tf = timeframe
    periods_per_year = Timeframe.parse(tf).periods_per_year()

    bars = ClientBarSource(data_client).scan([*symbols, benchmark], tf, as_of, lookback_days)
    bench_frame = bars.get(benchmark)
    universe_bars = {sym: bars[sym] for sym in symbols if sym in bars}

    scorer = {
        "strategy": lambda: strategy_scorer(strat),
        "signal": lambda: signal_scorer(strat),
        "scanner": lambda: scanner_scorer(_scanner(scanner)),
    }.get(source)
    if scorer is None:
        raise ValueError(f"source must be 'strategy', 'signal', or 'scanner', got {source!r}")

    # Alphas (the value) and Σ (the risk denominator), both as of the same moment.
    panel = FeaturePanel.for_universe(as_of, list(universe_bars))
    add_risk_features(panel, universe_bars, bench_frame, periods_per_year)
    if neutralize_factors:
        add_factor_exposure_features(panel, universe_bars, bench_frame, neutralize_factors)
    add_score_feature(panel, scorer(), universe_bars)
    alpha_ctx = AlphaContext(
        ic=DEFAULT_IC, neutralize=neutralize, neutralize_factors=tuple(neutralize_factors)
    )
    refine_alpha(panel, alpha_ctx)
    alphas = panel_to_alphas(panel, alpha_ctx)
    matrix = _build_covariance(
        risk_model, universe_bars, bench_frame, periods_per_year, 60, conditional, conditional_lambda
    )

    if not alphas or matrix is None:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "as_of": as_of.isoformat(),
            "feasible": False,
            "note": "Insufficient data for alphas and/or a covariance matrix.",
        }

    # The cost assumptions are parameters, not constants, so a caller running this
    # step alongside a backtest can price both the same way - the defaults are the
    # model's own, so an unparameterized call behaves exactly as before.
    cost_model = ParametricCostModel(
        commission_bps=commission_bps,
        impact_eta=impact_eta,
        participation_cap=participation_cap,
        annual_borrow_bps=borrow_bps,
    )
    # Per-name, as-of liquidity context (spread proxy, ADV$, daily vol) priced by the
    # same cost model the backtest uses - the optimizer and backtest share one model.
    cost_inputs = _cost_inputs(universe_bars, cost_model) if cost_aware else None

    benchmark_weights, benchmark_report = _resolve_benchmark_portfolio(
        benchmark_holdings, benchmark_premium, matrix
    )

    posterior_report = None
    if posterior is not None:
        alphas, posterior_report = _apply_bl_posterior(
            panel,
            alphas,
            matrix,
            as_of,
            benchmark_report,
            posterior,
            posterior_ic,
            posterior_t_eff,
            posterior_tau,
            alpha_ctx.ic,
        )

    if policy is not None and policy != "aim":
        raise ValueError(f"policy must be 'aim' or None, got {policy!r}")
    if policy == "aim" and (book != "long_only" or benchmark_holdings is not None):
        raise ValueError(
            "policy='aim' is long-only, cash-relative only in v1 - "
            "incompatible with book='market_neutral' or benchmark_holdings"
        )

    optimizer = MeanVarianceOptimizer(max_weight=max_weight, min_weight=min_weight, max_names=max_names)
    policy_report = None
    if policy == "aim":
        from tradeflow.portfolio import policy as policy_mod

        # Calendar-day arithmetic on a naive datetime - subtracting a timedelta from
        # a tz-AWARE as_of would shift the absolute instant across DST boundaries and
        # can land on a locally-ambiguous wall-clock hour (a real crash seen with the
        # synthetic demo provider); date_client-facing datetimes are naive elsewhere
        # in this module, and _to_ts() re-localizes downstream regardless.
        as_of_naive = as_of.replace(tzinfo=None) if as_of.tzinfo is not None else as_of
        decay = compute_horizon(
            data_client,
            strategy,
            symbols,
            as_of_naive - timedelta(days=decay_lookback_days),
            as_of_naive,
            source=source,
            scanner=scanner,
            benchmark=benchmark,
            neutralize_factors=neutralize_factors,
            timeframe=tf,
        )
        periods_per_rebalance = max(holding_period_years * periods_per_year, 1.0)
        hl_upper_bars = decay.get("half_life_upper", float("nan"))
        if hl_upper_bars != hl_upper_bars:  # NaN: too little decay history - no discount
            hl_upper_bars = float("inf")
        hl_upper_rebalances = policy_mod.half_life_in_rebalance_units(hl_upper_bars, periods_per_rebalance)
        phi = policy_mod.phi_from_half_life(hl_upper_rebalances)
        result = policy_mod.build_aim_portfolio(
            optimizer,
            alphas,
            matrix,
            phi=phi,
            trade_rate=trade_rate,
            target_te=target_te,
            current_weights=current_weights,
            cost_model=cost_model if cost_aware else None,
            cost_inputs=cost_inputs,
            capital=capital,
            holding_period_years=holding_period_years,
        )
        policy_report = {
            "phi_per_rebalance": phi,
            "periods_per_rebalance": periods_per_rebalance,
            "decay_half_life_bars": decay.get("half_life"),
            "decay_half_life_upper_bars": decay.get("half_life_upper"),
            "decay_r_squared": decay.get("decay_r_squared"),
        }
    else:
        result = optimizer.optimize(
            alphas,
            matrix,
            target_te=target_te,
            current_weights=current_weights,
            cost_model=cost_model if cost_aware else None,
            cost_inputs=cost_inputs,
            capital=capital,
            holding_period_years=holding_period_years,
            benchmark_weights=benchmark_weights,
            book=book,
            gross_leverage=gross_leverage,
            short_max_weight=short_max_weight,
        )

    if result.feasible:
        if not result.diagnostics.get("cost_aware"):
            # Cost-blind (gross) objective: still report the ex-post drag so the net figure
            # is visible. Same convention as the cost-aware path - a round-trip haircut on
            # the book for the headline (matching capacity), one-way rebalance drag in detail.
            # The round-trip book size is gross exposure (Σ|w|) - equals Σw=1 for a
            # long-only book, but a market-neutral book's Σw≈0 isn't the exposure to price.
            h = max(holding_period_years, 1e-9)
            rate = cost_model.turnover_cost_rate()
            expected = result.diagnostics["expected_active_return"]
            gross_book = sum(abs(v) for v in result.weights.values())
            one_way = result.diagnostics["turnover"] * rate / h
            round_trip = 2.0 * rate * gross_book / h
            result.diagnostics["cost_drag"] = one_way
            result.diagnostics["round_trip_cost"] = round_trip
            result.diagnostics["expected_active_return_net"] = expected - round_trip
            result.diagnostics["expected_active_return_net_oneway"] = expected - one_way
        # Capacity: the capital at which √-impact cost erases the gross alpha.
        result.diagnostics["capacity_capital"] = _capacity(
            result.weights, universe_bars, result.diagnostics["expected_active_return"], holding_period_years
        )
        if benchmark_report is not None and result.diagnostics.get("has_benchmark"):
            # Value-added identity: SR_P² ≈ SR_B² + IR² -
            # active management adds to the benchmark's own Sharpe in quadrature.
            # Predicted, not realized: SR_B from the assumed premium, IR from the
            # optimizer's own predicted_ir.
            sigma_b = float(np.sqrt(result.diagnostics["benchmark_variance"]))
            sr_b = (benchmark_report["premium"] / sigma_b) if sigma_b > 0 else 0.0
            ir = result.diagnostics["predicted_ir"]
            benchmark_report["value_added_identity"] = {
                "sr_benchmark": sr_b,
                "ir": ir,
                "sr_portfolio_predicted": float(np.sqrt(sr_b**2 + ir**2)),
            }

    holdings = []
    if result.feasible and capital:
        from tradeflow.utils.numeric import round_quantity

        last_close = {sym: float(frame["close"].iloc[-1]) for sym, frame in universe_bars.items()}
        for sym, weight in sorted(result.weights.items(), key=lambda kv: kv[1], reverse=True):
            price = last_close.get(sym, 0.0)
            dollars = weight * capital
            holdings.append(
                {
                    "symbol": sym,
                    "weight": weight,
                    "dollars": dollars,
                    "shares": round_quantity(dollars / price) if price > 0 else 0.0,
                }
            )

    out = {
        "run_id": run_id,
        "strategy": strategy,
        "source": source,
        "as_of": as_of.isoformat(),
        "timeframe": tf,
        "benchmark": benchmark,
        "target_te": target_te,
        "feasible": result.feasible,
        "binding_constraint": result.binding_constraint,
        "universe_size": len(alphas),
        "risk_model": risk_model,
        "conditional": conditional,
        "policy": policy,
        "shrinkage": matrix.shrinkage,
        "weights": _jsonable(dict(sorted(result.weights.items(), key=lambda kv: kv[1], reverse=True))),
        "active_weights": _jsonable(_active_weights(result.weights, benchmark_weights)),
        "exposures": _jsonable(_portfolio_exposures(panel, result.weights)),
        "holdings": _jsonable(holdings),
        "diagnostics": _jsonable(result.diagnostics),
        "benchmark_portfolio": _jsonable(benchmark_report),
        "posterior": _jsonable(posterior_report),
        "shrink_chain": _jsonable(panel.meta.get("shrink_chain", [])),
        "note": "PROPOSAL, not an order. Maximizes αᵀw − λ·wᵀΣw at the target tracking "
        "error; the transfer coefficient shows how much of IR* survives the constraints.",
    }
    if matrix.conditional_diagnostics:
        out["sigma_regime"] = _jsonable(matrix.conditional_diagnostics)
    if policy_report is not None:
        out["policy_report"] = _jsonable(policy_report)
    return out


def _active_weights(
    weights: Dict[str, float], benchmark_weights: Optional[Dict[str, float]]
) -> Optional[Dict[str, float]]:
    """``w − w_B`` per name, or ``None`` when the run had no benchmark portfolio.

    ``None`` and "all zeros" are different facts - the first means nobody asked for
    active space, the second means the book *is* the benchmark - so the absent case
    stays absent rather than becoming a vector of zeros.
    """
    if not benchmark_weights:
        return None
    names = set(weights) | set(benchmark_weights)
    return {sym: float(weights.get(sym, 0.0)) - float(benchmark_weights.get(sym, 0.0)) for sym in names}


def _portfolio_exposures(panel, weights: Dict[str, float]) -> Optional[Dict[str, float]]:
    """The book's factor exposures, ``Xᵀw`` over the panel's exposure block.

    A weighted sum of exposure columns that were already built for the risk model -
    an aggregation of the book, not a second definition of what a factor is. Returns
    ``None`` when the run built no exposure block (no factors requested, or too few
    qualifying names), so a reader can tell "not measured" from "flat".
    """
    from tradeflow.data.features import EXPOSURE_PREFIX

    columns = [c for c in panel.columns if c.startswith(EXPOSURE_PREFIX)]
    if not columns or not weights:
        return None
    exposures: Dict[str, float] = {}
    for column in columns:
        values = panel.get(column)
        total = 0.0
        for symbol, weight in weights.items():
            value = values.get(symbol) if symbol in values.index else None
            if value is not None and not pd.isna(value):
                total += float(weight) * float(value)
        exposures[column[len(EXPOSURE_PREFIX) :]] = total
    return exposures


def run_conditional_risk_ab(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "demo_volume",
    benchmark: str = "SPY",
    neutralize_factors: Sequence[str] = (),
    target_te: float = 0.04,
    max_weight: float = 0.25,
    risk_model: str = "shrinkage",
    lookback_days: int = 365,
    timeframe: str = "1Day",
    horizon: int = 21,
    n_points: int = 12,
    capital: float = 1_000_000.0,
    holding_period_years: Optional[float] = None,
    conditional_method: str = "ewma",
    conditional_lambda: Optional[float] = None,
) -> Dict[str, Any]:
    """The net-of-cost A/B — the one that decides commercial adoption, not
    TE-tracking alone: walk ``[start, end]`` at spaced rebalances, constructing
    the SAME alpha book (``construct_portfolio``, same target_te, same cost
    model) against a conditional vs unconditional Σ, carrying each variant's
    weights forward to the next rebalance, and pricing the REALIZED forward
    return net of the real transaction cost (turnover cost annualized in the
    diagnostics, scaled down to this rebalance's holding period). A conditional
    Σ that tracks TE better but churns the book to death should — and, if the
    numbers say so, does — lose the net-IR comparison here. Read-only
    research-clock harness; no orders.
    """
    run_id = new_run_id()
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    holding_period = holding_period_years if holding_period_years is not None else horizon / periods_per_year

    bars = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, end, _window_days(start, end))
    bench = bars.get(benchmark)
    if bench is None or bench.empty:
        return {"run_id": run_id, "periods": 0, "note": "Insufficient data: need a benchmark series."}

    index = bench.index
    window = index[(index >= _to_ts(start, index)) & (index <= _to_ts(end, index))]
    points = [j for j in _rebalance_points(len(window), horizon, n_points) if j + horizon < len(window)]
    if len(points) < 2:
        return {"run_id": run_id, "periods": 0, "note": "Insufficient rebalances in window for the A/B."}

    variants = {"unconditional": None, "conditional": conditional_method}
    current_weights: Dict[str, Optional[Dict[str, float]]] = {k: None for k in variants}
    gross_returns: Dict[str, List[float]] = {k: [] for k in variants}
    net_returns: Dict[str, List[float]] = {k: [] for k in variants}
    turnovers: Dict[str, List[float]] = {k: [] for k in variants}
    predicted_tes: Dict[str, List[float]] = {k: [] for k in variants}

    for j in points:
        t, t_fwd = window[j], window[j + horizon]
        fwd = _forward_raw_return({s: bars[s] for s in symbols if s in bars}, t, t_fwd)
        for name, cond in variants.items():
            result = construct_portfolio(
                data_client,
                strategy,
                symbols,
                t.to_pydatetime(),
                source=source,
                scanner=scanner,
                target_te=target_te,
                max_weight=max_weight,
                benchmark=benchmark,
                neutralize_factors=neutralize_factors,
                risk_model=risk_model,
                lookback_days=lookback_days,
                timeframe=timeframe,
                capital=capital,
                current_weights=current_weights[name],
                holding_period_years=holding_period,
                cost_aware=True,
                conditional=cond,
                conditional_lambda=conditional_lambda,
            )
            if not result["feasible"]:
                continue
            new_weights = result["weights"]
            diag = result["diagnostics"]
            gross = float(sum(w * fwd.get(s, 0.0) for s, w in new_weights.items()))
            # The annualized one-way cost, scaled down to THIS rebalance's holding period.
            period_cost = float(diag.get("cost_drag", 0.0)) * holding_period
            gross_returns[name].append(gross)
            net_returns[name].append(gross - period_cost)
            turnovers[name].append(float(diag.get("turnover", 0.0)))
            predicted_tes[name].append(float(diag.get("predicted_tracking_error", 0.0)))
            current_weights[name] = new_weights

    rebalances_per_year = periods_per_year / horizon

    def _summary(name: str) -> Dict[str, Any]:
        net = np.array(net_returns[name])
        if len(net) < 2:
            return {"periods": int(len(net))}
        net_ir = float(np.mean(net) / np.std(net) * np.sqrt(rebalances_per_year)) if np.std(net) > 0 else 0.0
        return {
            "periods": int(len(net)),
            "mean_net_return_per_period": float(np.mean(net)),
            "mean_gross_return_per_period": float(np.mean(gross_returns[name])),
            "net_ir": net_ir,
            "realized_te": float(np.std(net) * np.sqrt(rebalances_per_year)),
            "mean_predicted_te": float(np.mean(predicted_tes[name])) if predicted_tes[name] else 0.0,
            "mean_turnover": float(np.mean(turnovers[name])) if turnovers[name] else 0.0,
        }

    summaries = {name: _summary(name) for name in variants}
    winner = max(variants, key=lambda k: summaries[k].get("net_ir", float("-inf")))

    return {
        "run_id": run_id,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "horizon_bars": horizon,
        "periods": len(points),
        "conditional_method": conditional_method,
        "summaries": _jsonable(summaries),
        "winner_net_ir": winner,
        "note": "SAME alphas/target_te/cost model, conditional vs unconditional Σ, weights "
        "carried forward rebalance to rebalance. 'winner_net_ir' picks by realized net "
        "IR (net of the real transaction cost), not by TE-tracking alone — churn that tracking-"
        "error-tracks-better-but-costs-more should lose, and this is where it would show.",
    }


def run_policy_ab(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "demo_volume",
    benchmark: str = "SPY",
    neutralize_factors: Sequence[str] = (),
    target_te: float = 0.04,
    max_weight: float = 0.25,
    risk_model: str = "shrinkage",
    lookback_days: int = 365,
    timeframe: str = "1Day",
    horizon: int = 21,
    n_points: int = 12,
    capital: float = 1_000_000.0,
    holding_period_years: Optional[float] = None,
    trade_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """The net-of-cost A/B that decides adoption: walk-forward the myopic policy
    vs the aim policy on the SAME alpha book, same target_te, same cost model,
    carrying each variant's weights forward rebalance to rebalance, and compare
    REALIZED net IR (turnover cost priced at each rebalance's actual holding
    period) — mirrors :func:`run_conditional_risk_ab` exactly, with 'myopic'/
    'aim' in place of 'unconditional'/'conditional'. Read-only research-clock
    harness; no orders. If the aim policy doesn't win here, that's a legitimate,
    complete outcome (ship nothing) — not a failure to keep iterating on.
    """
    run_id = new_run_id()
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    holding_period = holding_period_years if holding_period_years is not None else horizon / periods_per_year

    bars = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, end, _window_days(start, end))
    bench = bars.get(benchmark)
    if bench is None or bench.empty:
        return {"run_id": run_id, "periods": 0, "note": "Insufficient data: need a benchmark series."}

    index = bench.index
    window = index[(index >= _to_ts(start, index)) & (index <= _to_ts(end, index))]
    points = [j for j in _rebalance_points(len(window), horizon, n_points) if j + horizon < len(window)]
    if len(points) < 2:
        return {"run_id": run_id, "periods": 0, "note": "Insufficient rebalances in window for the A/B."}

    variants = {"myopic": None, "aim": "aim"}
    current_weights: Dict[str, Optional[Dict[str, float]]] = {k: None for k in variants}
    gross_returns: Dict[str, List[float]] = {k: [] for k in variants}
    net_returns: Dict[str, List[float]] = {k: [] for k in variants}
    turnovers: Dict[str, List[float]] = {k: [] for k in variants}
    predicted_tes: Dict[str, List[float]] = {k: [] for k in variants}

    for j in points:
        t, t_fwd = window[j], window[j + horizon]
        fwd = _forward_raw_return({s: bars[s] for s in symbols if s in bars}, t, t_fwd)
        for name, pol in variants.items():
            result = construct_portfolio(
                data_client,
                strategy,
                symbols,
                t.to_pydatetime(),
                source=source,
                scanner=scanner,
                target_te=target_te,
                max_weight=max_weight,
                benchmark=benchmark,
                neutralize_factors=neutralize_factors,
                risk_model=risk_model,
                lookback_days=lookback_days,
                timeframe=timeframe,
                capital=capital,
                current_weights=current_weights[name],
                holding_period_years=holding_period,
                cost_aware=True,
                policy=pol,
                trade_rate=trade_rate if pol == "aim" else None,
            )
            if not result["feasible"]:
                continue
            new_weights = result["weights"]
            diag = result["diagnostics"]
            gross = float(sum(w * fwd.get(s, 0.0) for s, w in new_weights.items()))
            period_cost = float(diag.get("cost_drag", 0.0)) * holding_period
            gross_returns[name].append(gross)
            net_returns[name].append(gross - period_cost)
            turnovers[name].append(float(diag.get("turnover", 0.0)))
            predicted_tes[name].append(float(diag.get("predicted_tracking_error", 0.0)))
            current_weights[name] = new_weights

    rebalances_per_year = periods_per_year / horizon

    def _summary(name: str) -> Dict[str, Any]:
        net = np.array(net_returns[name])
        if len(net) < 2:
            return {"periods": int(len(net))}
        net_ir = float(np.mean(net) / np.std(net) * np.sqrt(rebalances_per_year)) if np.std(net) > 0 else 0.0
        return {
            "periods": int(len(net)),
            "mean_net_return_per_period": float(np.mean(net)),
            "mean_gross_return_per_period": float(np.mean(gross_returns[name])),
            "net_ir": net_ir,
            "realized_te": float(np.std(net) * np.sqrt(rebalances_per_year)),
            "mean_predicted_te": float(np.mean(predicted_tes[name])) if predicted_tes[name] else 0.0,
            "mean_turnover": float(np.mean(turnovers[name])) if turnovers[name] else 0.0,
        }

    summaries = {name: _summary(name) for name in variants}
    winner = max(variants, key=lambda k: summaries[k].get("net_ir", float("-inf")))
    over_damped = (
        summaries["aim"].get("periods", 0) >= 2
        and summaries["myopic"].get("periods", 0) >= 2
        and summaries["aim"]["net_ir"] < summaries["myopic"]["net_ir"]
        and summaries["aim"]["mean_turnover"] < summaries["myopic"]["mean_turnover"]
    )

    return {
        "run_id": run_id,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "horizon_bars": horizon,
        "periods": len(points),
        "summaries": _jsonable(summaries),
        "winner_net_ir": winner,
        "over_damped": over_damped,
        "note": "SAME alphas/target_te/cost model, myopic vs aim policy, weights "
        "carried forward rebalance to rebalance. 'winner_net_ir' picks by realized net "
        "IR (net of the real transaction cost). 'over_damped' flags the double-damping "
        "failure mode: lower turnover AND lower net IR together mean "
        "the aim policy traded too little to capture what alpha there was, not that it "
        "improved anything.",
    }


def longshort_report(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    as_of: datetime,
    config: Optional[Dict[str, Any]] = None,
    source: str = "strategy",
    scanner: str = "demo_volume",
    target_te: float = 0.04,
    max_weight: float = 0.25,
    benchmark: str = "SPY",
    neutralize_factors: Sequence[str] = (),
    risk_model: str = "shrinkage",
    lookback_days: int = 365,
    timeframe: str = "1Day",
    capital: Optional[float] = None,
    holding_period_years: float = 1.0 / 12.0,
    cost_aware: bool = True,
    gross_leverage: float = 2.0,
    short_max_weight: float = 0.25,
) -> Dict[str, Any]:
    """The long-only price report: the SAME alphas, Σ, and costs solved
    ``long_only`` vs ``market_neutral``, so the difference is attributable to
    the constraint itself, not to a different universe or a different day's data.

    Reports the measured IR shrinkage (``IR_LO / IR_LS``) next to an illustrative
    reference line (``γ(N) = (53+N)^0.57``) - explicitly *not* a verified
    transcription of any published formula, just a comparison point - both
    transfer coefficients, and the long-only book's incidental **size exposure**
    (the "size" factor already in ``tradeflow/risk/exposures.py``, dotted with each
    book's weights) that a long/short book, free to short the small names
    long-only can only zero out, does not carry.
    """
    common = dict(
        source=source,
        scanner=scanner,
        target_te=target_te,
        max_weight=max_weight,
        benchmark=benchmark,
        neutralize_factors=neutralize_factors,
        risk_model=risk_model,
        lookback_days=lookback_days,
        timeframe=timeframe,
        capital=capital,
        holding_period_years=holding_period_years,
        cost_aware=cost_aware,
    )
    lo = construct_portfolio(data_client, strategy, symbols, as_of, book="long_only", **common)
    ls = construct_portfolio(
        data_client,
        strategy,
        symbols,
        as_of,
        book="market_neutral",
        gross_leverage=gross_leverage,
        short_max_weight=short_max_weight,
        **common,
    )
    if not lo["feasible"] or not ls["feasible"]:
        return {
            "as_of": as_of.isoformat(),
            "strategy": strategy,
            "feasible": False,
            "note": "Long-only and/or market-neutral solve was infeasible; see long_only/"
            "market_neutral for the binding constraint.",
            "long_only": lo,
            "market_neutral": ls,
        }

    ir_lo = lo["diagnostics"]["predicted_ir"]
    ir_ls = ls["diagnostics"]["predicted_ir"]
    shrinkage = (ir_lo / ir_ls) if ir_ls != 0 else 0.0
    n = max(lo["universe_size"], 1)
    gamma_n = (53.0 + n) ** 0.57
    reference_shrinkage = 1.0 - 1.0 / gamma_n

    size_exposure = _longshort_size_exposure(
        data_client, symbols, benchmark, as_of, lookback_days, timeframe, lo, ls
    )
    binding_fraction = _binding_fraction(lo["weights"], symbols)

    return {
        "as_of": as_of.isoformat(),
        "strategy": strategy,
        "feasible": True,
        "universe_size": lo["universe_size"],
        "ir_long_short": ir_ls,
        "ir_long_only": ir_lo,
        "shrinkage_measured": shrinkage,
        "shrinkage_reference_curve": reference_shrinkage,
        "transfer_coefficient_long_only": lo["diagnostics"]["transfer_coefficient"],
        "transfer_coefficient_long_short": ls["diagnostics"]["transfer_coefficient"],
        "size_exposure_long_only": size_exposure["long_only"],
        "size_exposure_long_short": size_exposure["long_short"],
        "binding_fraction": binding_fraction,
        "gross_leverage": ls["diagnostics"].get("gross_leverage"),
        "dollar_neutral_residual": ls["diagnostics"].get("dollar_neutral_residual"),
        "borrow_cost": ls["diagnostics"].get("borrow_cost"),
        "long_only": lo,
        "market_neutral": ls,
        "note": "shrinkage_reference_curve is an illustrative γ(N) reference line, "
        "not a verified transcription of any published formula - compare "
        "against shrinkage_measured, don't trust it as truth. binding_fraction is the "
        "share of the long-only universe pinned at zero weight (a proxy for the "
        "forced-underweight bound the long-only constraint imposes), not the exact "
        "|z|-mass figure.",
    }


def _longshort_size_exposure(data_client, symbols, benchmark, as_of, lookback_days, timeframe, lo, ls):
    """Each book's dot product with the cross-sectionally standardized size factor
    (``log(price·ADV)``, ``tradeflow/risk/exposures.py``) - the incidental size bias a
    long-only book picks up from being unable to short small, unattractive names.
    """
    from tradeflow.risk.exposures import build_factor_exposures

    bars = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, as_of, lookback_days)
    bench_frame = bars.get(benchmark)
    universe_bars = {s: bars[s] for s in symbols if s in bars}
    exposures = build_factor_exposures(universe_bars, bench_frame, factors=["size"])
    if exposures.empty:
        return {"long_only": None, "long_short": None}
    size = exposures["size"]

    def dot(weights: Dict[str, float]) -> float:
        common = size.index.intersection(list(weights))
        return float(sum(weights[s] * size[s] for s in common)) if len(common) else 0.0

    return {"long_only": dot(lo["weights"]), "long_short": dot(ls["weights"])}


def _binding_fraction(long_only_weights: Dict[str, float], symbols: List[str]) -> float:
    """Share of the universe the long-only solve holds at exactly zero - a proxy for
    names pinned at the long-only floor (``w_a = −w_B``, the underweight bound a
    long-only book can't relax). Not an exact "|z| mass" figure (that needs the
    per-name alpha z-score, which this report doesn't carry) - a documented
    simplification.
    """
    if not symbols:
        return 0.0
    pinned = sum(1 for s in symbols if long_only_weights.get(s, 0.0) <= 1e-9)
    return pinned / len(symbols)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
#: The bar feed carries no quotes, so the effective spread is proxied as this fraction
#: of the trailing daily high-low range (a rough liquidity signal), then clamped.
SPREAD_RANGE_FRACTION = 0.10
_COST_WINDOW = 20  # trailing bars for the ADV / vol / spread estimates (matches the backtest)


def _spread_proxy(frame, cost_model, window: int = _COST_WINDOW) -> float:
    """Per-name fractional spread from the trailing high-low range, clamped.

    Effective spread ≈ ``SPREAD_RANGE_FRACTION`` of the median daily range, floored at
    a fraction of the model default (so very liquid names can price below it) and capped
    at 2%. Falls back to the model default when OHLC is missing - an honest proxy, not a
    quote.
    """
    if not {"high", "low", "close"} <= set(frame.columns) or len(frame) < 2:
        return cost_model.default_spread
    rng = ((frame["high"] - frame["low"]) / frame["close"]).tail(window)
    proxy = float(rng.median()) * SPREAD_RANGE_FRACTION
    if not np.isfinite(proxy):
        return cost_model.default_spread
    return float(min(max(proxy, cost_model.default_spread * 0.2), 0.02))


def _cost_inputs(universe_bars, cost_model, window: int = _COST_WINDOW):
    """Per-name as-of liquidity context (spread proxy, ADV$, daily vol) for the solve.

    Trailing windows only, so nothing depends on post-``as_of`` bars (the same leakage
    discipline as the backtest's cost inputs and :func:`_capacity`).
    """
    from tradeflow.portfolio.optimizer import CostInputs

    spread, adv_dollar, daily_vol = {}, {}, {}
    for sym, frame in universe_bars.items():
        if frame is None or len(frame) < 2:
            continue
        price = float(frame["close"].iloc[-1])
        adv_shares = float(frame["volume"].tail(window).mean()) if "volume" in frame else 0.0
        spread[sym] = _spread_proxy(frame, cost_model, window)
        adv_dollar[sym] = price * adv_shares
        vol = float(frame["close"].pct_change().tail(window).std())
        daily_vol[sym] = vol if np.isfinite(vol) else 0.0
    return CostInputs(spread=spread, adv_dollar=adv_dollar, daily_vol=daily_vol)


def _capacity(weights, universe_bars, gross_alpha, holding_period_years) -> float:
    """Capital at which √-impact cost erases the gross alpha (net active return → 0).

    Cost as a fraction of capital grows ∝ √capital (square-root impact), so net alpha
    is monotone-decreasing in capital — solved by bisection. Returns 0 if the alpha
    can't even cover cost at tiny size.
    """
    from tradeflow.costs import ParametricCostModel, Trade

    model = ParametricCostModel()
    liquidity = {}
    for sym, w in weights.items():
        frame = universe_bars.get(sym)
        if frame is None or len(frame) < 2 or w == 0:
            continue
        liquidity[sym] = (
            w,
            float(frame["close"].iloc[-1]),
            float(frame["volume"].tail(20).mean()),
            float(frame["close"].pct_change().tail(20).std()),
        )
    if not liquidity or gross_alpha <= 0:
        return 0.0

    def net(capital: float) -> float:
        total = sum(
            model.cost(Trade(sym, w * capital / price, price, adv, vol)).total
            for sym, (w, price, adv, vol) in liquidity.items()
        )
        return gross_alpha - 2.0 * (total / capital) / max(holding_period_years, 1e-9)

    lo, hi = 1e3, 1e12
    if net(lo) <= 0:
        return lo
    if net(hi) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if net(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def _build_covariance(
    model, bars, benchmark_bars, periods_per_year, min_obs=60, conditional=None, conditional_lambda=None
):
    """Build a covariance RiskMatrix by model name (statistical estimator or factor).

    ``conditional`` (default ``None`` / off) conditions Σ's diagonal (or,
    for ``model='factor'``, both ``factor_cov`` and ``specific_var``) via an EWMA or
    HAR-lite per-name volatility forecast, holding the correlation structure fixed —
    see :mod:`tradeflow.risk.conditional`. Every caller of this helper reduces byte-for-byte
    to its unconditional behavior when ``conditional`` is left at the default.
    """
    from tradeflow.risk import RISK_MODELS, build_factor_risk_matrix, build_risk_matrix

    if model == "factor":
        return build_factor_risk_matrix(
            bars,
            benchmark_bars,
            periods_per_year,
            min_obs=min_obs,
            conditional=conditional,
            conditional_lambda=conditional_lambda,
        )
    return build_risk_matrix(
        RISK_MODELS[model](),
        bars,
        periods_per_year,
        min_obs=min_obs,
        conditional=conditional,
        conditional_lambda=conditional_lambda,
    )


def _resolve_benchmark_portfolio(benchmark_holdings, benchmark_premium, matrix):
    """Load ``w_B`` (restricted to Σ's covered universe) plus the reverse-optimization
    report, or ``(None, None)`` when no portfolio-level benchmark was requested -
    the no-op path that keeps ``construct_portfolio`` byte-for-byte unchanged
    without ``benchmark_holdings``.
    """
    if not benchmark_holdings:
        return None, None

    from tradeflow.portfolio.benchmark import (
        implied_returns,
        load_benchmark_weights,
        restrict_and_renormalize,
    )

    raw = load_benchmark_weights(benchmark_holdings, matrix.symbols)
    raw_total = sum(raw.values())
    restricted, coverage = restrict_and_renormalize(raw, matrix.symbols)
    consensus = implied_returns(restricted, matrix, benchmark_premium) if restricted else {}
    report = {
        "source": benchmark_holdings,
        "premium": benchmark_premium,
        "coverage": coverage,  # fraction of raw weight mass inside Σ's universe
        "uncovered_weight": max(0.0, 1.0 - coverage),
        "raw_weight_sum": raw_total,  # far from 1 => the file implied a cash position
        "consensus_returns": consensus,  # mu per name - print next to alpha
    }
    return (restricted or None), report


def _apply_bl_posterior(
    panel,
    alphas,
    matrix,
    as_of,
    benchmark_report,
    posterior,
    posterior_ic,
    posterior_t_eff,
    posterior_tau,
    scale_ic,
):
    """Blend ``alphas`` with the consensus prior via Black–Litterman.

    Reads ``panel.meta["shrink_chain"]`` (populated by ``refine_alpha`` upstream,
    where ``level_shrink`` stayed off - the raw, unshrunk alpha is exactly what BL's
    Ω needs) and appends the ``bl`` step so the IC-uncertainty haircut is auditably
    applied exactly once, here, not twice. Returns the new (Σ-universe-spanning)
    alpha list and the report section (per-name consensus/view/posterior/source
    table plus τ-sensitivity).
    """
    if posterior != "bl":
        raise ValueError(f"posterior must be 'bl' or None, got {posterior!r}")
    if posterior_t_eff is None:
        raise ValueError(
            "posterior_t_eff is required for posterior='bl' - tau is pinned to "
            "1/T_eff, never tuned; pass the effective_t a prior "
            "compute_information call measured for this strategy/window."
        )

    from tradeflow.portfolio.posterior import black_litterman_from_ic

    ic_bl = posterior_ic if posterior_ic is not None else scale_ic
    views = {a.symbol: a.alpha for a in alphas}
    bl = black_litterman_from_ic(views, matrix, ic_bl, posterior_t_eff, tau=posterior_tau)

    panel.meta.setdefault("shrink_chain", []).append(
        {
            "step": "bl",
            "owner": "bl",
            "ic": ic_bl,
            "t_eff": posterior_t_eff,
            "tau": bl.tau,
            "note": "IC-uncertainty owned here (Ω), not re-applied upstream - the "
            "refine step's level_shrink stayed off so the raw, unshrunk alpha feeds "
            "Ω, never both.",
        }
    )

    original_z = {a.symbol: a.raw_z for a in alphas}
    original_vol = {a.symbol: a.residual_vol for a in alphas}
    new_alphas = [
        Alpha(
            symbol=s,
            alpha=bl.mu_post[s],
            as_of=as_of,
            residual_vol=original_vol.get(s, float(np.sqrt(max(matrix.sigma[i, i], 0.0)))),
            ic=ic_bl,
            raw_z=original_z.get(s, 0.0),
        )
        for i, s in enumerate(matrix.symbols)
    ]

    consensus = (benchmark_report or {}).get("consensus_returns", {}) or {}
    per_name = [
        {
            "symbol": s,
            "consensus_pi": consensus.get(s),
            "view_q": bl.views.get(s),
            "posterior_mu": bl.mu_post[s],
            "source": bl.source[s],
        }
        for s in matrix.symbols
    ]
    report = {
        "method": "bl",
        "ic": ic_bl,
        "t_eff": posterior_t_eff,
        "tau": bl.tau,
        "tau_sensitivity": bl.tau_sensitivity,
        "per_name": per_name,
    }
    return new_alphas, report


def _scanner(scanner_name: str):
    """Instantiate a scanner from its declared defaults (for the scanner scorer)."""
    from tradeflow.scanners.symbol_scanner import SymbolScanner

    if scanner_name not in SymbolScanner.SCANNERS:
        raise ValueError(f"Unknown scanner '{scanner_name}'. Available: {SymbolScanner.available()}")
    cls = SymbolScanner.SCANNERS[scanner_name]
    sc = cls({p: spec["default"] for p, spec in cls.PARAM_RANGES.items()})
    sc.initialize()
    return sc


def _window_days(start: datetime, end: datetime) -> int:
    """Calendar days to fetch so the scan covers [start, end] with a warmup buffer."""
    return max((end - start).days + 90, 120)


def _to_ts(when: datetime, index: pd.Index) -> pd.Timestamp:
    """Localize a possibly-naive timestamp to a (possibly tz-aware) index's timezone."""
    ts = pd.Timestamp(when)
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        ts = ts.tz_localize(index.tz) if ts.tzinfo is None else ts.tz_convert(index.tz)
    return ts


def _rebalance_points(n_bars: int, horizon: int, n_points: int):
    """Evenly spaced rebalance indices leaving room for the forward horizon."""
    last = n_bars - horizon - 1
    warmup = 30
    if last <= warmup:
        return []
    return np.linspace(warmup, last, num=min(n_points, last - warmup), dtype=int)


def _raw_signal_history(universe_bars, scorer, n_points: int = 48, warmup: int = 30) -> pd.DataFrame:
    """Trailing time×name frame of the **raw** signal, for the Case test.

    Scores every name at ``n_points`` dates along the reference index using only bars
    ``≤ t`` (the same leakage discipline as the alpha cross-section), so each column is
    a name's raw-signal time series ``g_n(t)`` — the input :func:`refine.case_test`
    regresses ``Std_TS`` on ``ω``.
    """
    if not universe_bars:
        return pd.DataFrame()
    ref = max((f.index for f in universe_bars.values()), key=len)
    last = len(ref) - 1
    if last <= warmup:
        points = range(len(ref))
    else:
        points = np.linspace(warmup, last, num=min(n_points, last - warmup), dtype=int)

    rows: Dict[Any, Dict[str, float]] = {}
    for j in points:
        t = ref[j]
        row: Dict[str, float] = {}
        for sym, frame in universe_bars.items():
            hist = frame.loc[frame.index <= t]
            if len(hist) < 2:
                continue
            val = scorer(hist)
            if val is not None and not pd.isna(val):
                row[sym] = float(val)
        if row:
            rows[t] = row
    return pd.DataFrame(rows).T


def _run_case_test(universe_bars, scorer, residual_vol: pd.Series, price_derived: bool) -> Dict[str, Any]:
    """Case-1/Case-2 decision + the two candidate alphas' cross-sectional correlation.

    Runs :func:`refine.case_test` on the trailing raw-signal history, then reports how
    different the two scalings actually are at the latest cross-section (``corr`` of the
    Case-1 ``ω·z`` vector with the Case-2 ``c_g·z`` vector) — a correlation near 1 means
    the case choice barely matters here, near 0 means it matters a lot.
    """
    from tradeflow.alphas import refine

    history = _raw_signal_history(universe_bars, scorer)
    diag = refine.case_test(history, residual_vol, price_derived=price_derived)

    corr = float("nan")
    c_g = float("nan")
    if not history.empty:
        g = history.iloc[-1].dropna()
        if len(g) >= 2:
            z = refine.zscore(refine.winsorize(g))
            vol = residual_vol.reindex(z.index)
            c_g = refine.case_scale_factor(g, vol)
            scale = c_g if (c_g == c_g) else float(vol.mean())
            a1, a2 = vol * z, scale * z  # IC cancels in the correlation
            if a1.std() > 0 and a2.std() > 0:
                corr = float(a1.corr(a2))
    diag["candidate_correlation"] = corr
    diag["c_g"] = c_g
    return diag


def _alpha_cross_section(universe_bars, bench, scorer, periods_per_year, t, ctx) -> pd.Series:
    """The refined alpha per name as of ``t`` (bars ≤ t only)."""
    ub_t = {sym: f.loc[f.index <= t] for sym, f in universe_bars.items()}
    ub_t = {sym: f for sym, f in ub_t.items() if len(f) >= 2}
    bench_t = bench.loc[bench.index <= t]
    panel = FeaturePanel.for_universe(t, list(ub_t))
    add_risk_features(panel, ub_t, bench_t, periods_per_year)
    if ctx.neutralize_factors:
        add_factor_exposure_features(panel, ub_t, bench_t, ctx.neutralize_factors)
    add_score_feature(panel, scorer, ub_t)
    refine_alpha(panel, ctx)
    return pd.Series({a.symbol: a.alpha for a in panel_to_alphas(panel, ctx)})


def _forward_raw_return(universe_bars, t, t_fwd) -> pd.Series:
    """Realized raw return per name over ``(t, t_fwd]`` (no beta adjustment)."""
    out: Dict[str, float] = {}
    for sym, frame in universe_bars.items():
        close = frame["close"]
        if t in close.index and t_fwd in close.index:
            out[sym] = close.loc[t_fwd] / close.loc[t] - 1.0
    return pd.Series(out)


def _factor_attribution(weights: pd.Series, universe_bars, bench, t, t_fwd, periods_per_year):
    """Split the portfolio's realized return into (factor, specific) at one rebalance.

    Projects the realized raw-return cross-section onto the factor exposures known at
    ``t``; the factor part is ``w·fitted`` and the specific part ``w·(R − fitted)``, so
    the two sum to the portfolio's realized return exactly.
    """
    from tradeflow.risk.exposures import build_factor_exposures

    bars_t = {s: f.loc[f.index <= t] for s, f in universe_bars.items()}
    exposures = build_factor_exposures(bars_t, bench.loc[bench.index <= t])
    raw = _forward_raw_return(universe_bars, t, t_fwd)
    if exposures.empty or raw.dropna().empty:
        return None
    common = weights.index.intersection(exposures.index).intersection(raw.dropna().index)
    if len(common) < len(exposures.columns) + 1:
        return None
    return _factor_split(
        weights.loc[common].to_numpy(), exposures.loc[common].to_numpy(), raw.loc[common].to_numpy()
    )


def _factor_split(w: np.ndarray, x: np.ndarray, r: np.ndarray):
    """Split ``w·r`` into (factor, specific) by projecting returns ``r`` onto exposures ``x``.

    ``fitted = x·(xᵀx)⁻¹·xᵀ·r`` is the factor-explained return; the two parts sum to
    ``w·r`` exactly (the projection + its residual reconstruct ``r``).
    """
    fitted = x @ np.linalg.pinv(x.T @ x) @ x.T @ r
    return float(w @ fitted), float(w @ (r - fitted))


def _forward_residual_return(universe_bars, bench, t, t_fwd, indicators) -> pd.Series:
    """Realized residual return per name over ``(t, t_fwd]``: r − β·r_benchmark."""
    bench_close = bench["close"]
    if t not in bench_close.index or t_fwd not in bench_close.index:
        return pd.Series(dtype=float)
    bench_ret = bench_close.loc[t_fwd] / bench_close.loc[t] - 1.0
    out: Dict[str, float] = {}
    for sym, frame in universe_bars.items():
        close = frame["close"]
        if t not in close.index or t_fwd not in close.index:
            continue
        beta = indicators.calculate_beta(close.loc[close.index <= t], bench_close.loc[bench_close.index <= t])
        out[sym] = (close.loc[t_fwd] / close.loc[t] - 1.0) - beta * bench_ret
    return pd.Series(out)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        # JSON has no inf/nan; represent them as strings so round-trips don't crash.
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value
