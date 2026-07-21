"""Analysis services: scan, backtest, optimize, walk-forward, summarize bars.

Each function takes a data-only :class:`MarketDataClient`, runs an existing
engine/optimizer/walk-forward path, and returns a compact, JSON-serializable
dict. Large outputs (trade tables, full optimization grids) are written to an
artifact file and referenced by path - never inlined.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.alphas import (
    DEFAULT_IC,
    AlphaContext,
    panel_to_alphas,
    refine_alpha,
    scanner_scorer,
    signal_scorer,
    strategy_scorer,
)
from src.analytics import metrics as m
from src.data import (
    ClientBarSource,
    FeaturePanel,
    add_factor_exposure_features,
    add_risk_features,
    add_score_feature,
)
from src.engine.backtest import BacktestEngine
from src.marketdata.client import MarketDataClient
from src.marketdata.timeframe import Timeframe
from src.optimization.optimizer import ParameterOptimizer
from src.optimization.walk_forward import WalkForwardValidator
from src.services.audit import new_run_id
from src.services.registry import resolve_strategy_class

logger = logging.getLogger(__name__)

#: Where trade tables / optimization grids are written.
ARTIFACT_DIR = Path("logs") / "artifacts"

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
    from src.costs import ParametricCostModel

    if gross:
        return None
    return ParametricCostModel(
        commission_bps=commission_bps,
        impact_eta=impact_eta,
        participation_cap=participation_cap,
        annual_borrow_bps=borrow_bps,
    )


def run_scan(
    data_client: MarketDataClient,
    scanner: str,
    symbols: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a universe scanner; return the flagged ``(symbol, signal)`` pairs."""
    from src.scanners.symbol_scanner import SymbolScanner

    flagged = SymbolScanner(data_client, scanner, config).scan(symbols)
    return {
        "scanner": scanner,
        "candidates": list(symbols),
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
    commission_bps: float = 1.0,
    impact_eta: float = 0.3,
    participation_cap: float = 0.10,
    borrow_bps: float = 50.0,
) -> Dict[str, Any]:
    """Backtest a strategy; return the full metrics dict + a path to the trades CSV.

    Metrics are **net of transaction cost** by default (commission + half-spread +
    square-root impact); pass ``gross=True`` to disable cost for attribution. Trades
    are NOT inlined (could be thousands of rows); read the CSV if needed.
    """
    from src.services.sizing import build_beta_sizer

    run_id = new_run_id()
    strat = _strategy(strategy, config)
    sizer = build_beta_sizer(data_client, strat, symbols, benchmark, as_of=start) if beta_sizing else None
    cost_model = _build_cost_model(gross, commission_bps, impact_eta, participation_cap, borrow_bps)
    result = BacktestEngine(strat, data_client, sizer=sizer, cost_model=cost_model).run(
        symbols, start, end, capital
    )

    trades_csv = None
    if not result.trades.empty:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        trades_csv = str(ARTIFACT_DIR / f"backtest_{run_id}.csv")
        result.trades.to_csv(trades_csv, index=False)

    return {
        "run_id": run_id,
        "strategy": strategy,
        "symbols": list(symbols),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_capital": capital,
        "final_capital": result.final_capital,
        "gross": gross,
        "total_cost": result.total_cost,
        "gross_final_capital": result.gross_final_capital,
        "cost_drag_pct": (result.total_cost / capital * 100.0) if capital else 0.0,
        "metrics": _jsonable(result.metrics),
        "total_trades": int(len(result.trades)),
        "trades_csv": trades_csv,
        "resolved_config": _jsonable(result.strategy_config),
    }


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
) -> Dict[str, Any]:
    """Search a strategy's parameters IN-SAMPLE; return best params + top-N rows.

    WARNING for the caller: these are in-sample results from selecting the best of
    many configs - NOT evidence of edge. Validate with ``run_walk_forward`` before
    trusting any of this; ``best_score`` will almost always look good here.

    Net of transaction cost by default (commission + half-spread + square-root
    impact); pass ``gross=True`` to search gross returns instead - gross search
    reliably favors the highest-turnover config.
    """
    run_id = new_run_id()
    cls = resolve_strategy_class(strategy)
    cost_model = _build_cost_model(gross, commission_bps, impact_eta, participation_cap, borrow_bps)
    opt = ParameterOptimizer(cls, data_client, initial_capital=capital, seed=seed, cost_model=cost_model)
    if method == "grid":
        result = opt.grid_search(symbols, start, end, objective, max_evals=max_evals)
    elif method == "random":
        result = opt.random_search(symbols, start, end, objective, n_samples=max_evals)
    else:
        result = opt.optimize_bayesian(symbols, start, end, objective)

    results_csv = None
    total = len(result.results)
    top: List[Dict[str, Any]] = []
    if not result.results.empty:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        results_csv = str(ARTIFACT_DIR / f"optimize_{run_id}.csv")
        result.results.to_csv(results_csv, index=False)
        top = [_jsonable(row) for row in result.results.head(TOP_N).to_dict("records")]

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
        "top": top,
        "truncated": max(total - len(top), 0),
        "results_csv": results_csv,
        "seed": seed,
        "gross": gross,
        "note": "IN-SAMPLE. Selecting the best of many configs inflates these. "
        "Validate out-of-sample with run_walk_forward (it applies the Deflated Sharpe).",
    }


def run_walk_forward(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    mode: str = "anchored",
    n_folds: Optional[int] = 4,
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
) -> Dict[str, Any]:
    """Honest evaluation: optimize IS, score OOS across folds, gate the verdict.

    This is the advancement criterion - returns the OOS aggregate, efficiency,
    degradation, per-fold summary, holdout (if requested), the Deflated Sharpe
    (with n_trials across all folds), and the promotion-gate pass/fail + overall
    ``promotable``. ``include_pbo`` is expensive and defaults off.

    Net of transaction cost by default, in-sample and out - pass ``gross=True``
    to validate gross returns instead, which systematically promotes turnover
    the strategy could not afford live.
    """
    run_id = new_run_id()
    cls = resolve_strategy_class(strategy)
    cost_model = _build_cost_model(gross, commission_bps, impact_eta, participation_cap, borrow_bps)
    validator = WalkForwardValidator(
        cls, data_client, initial_capital=capital, seed=seed, gates=gates, cost_model=cost_model
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
    scanner: str = "volume",
    ic: float = DEFAULT_IC,
    benchmark: str = "SPY",
    neutralize: bool = False,
    neutralize_factors: Sequence[str] = (),
    lookback_days: int = 180,
    timeframe: Optional[str] = None,
    scaling: str = "case1",
    price_derived: bool = True,
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
    :func:`~src.alphas.refine.case_test` decide from trailing history
    (``price_derived`` is the base-rate default when the test can't decide). The case
    diagnostics — chosen case, R², both candidates' cross-sectional correlation — are
    echoed under ``case`` whenever the test runs so a wrong call is visible.
    """
    run_id = new_run_id()
    strat = _strategy(strategy, None)
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
    from src.alphas import combined_score, measure_signals, strategy_scorer

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
    scanner: str = "volume",
    benchmark: str = "SPY",
    neutralize: bool = True,
    neutralize_factors: Sequence[str] = (),
    ic_prior: float = DEFAULT_IC,
    horizon: int = 5,
    n_points: int = 24,
    n_trials: int = 1,
    timeframe: str = "1Day",
    risk_model: str = "shrinkage",
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
    from src.alphas import horizon as hz
    from src.alphas import refine
    from src.analytics import information as info
    from src.indicators import indicators

    run_id = new_run_id()
    strat = _strategy(strategy, None)
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
        "recommended_ic": stats["mean_ic"],  # feeds back into 005's scaling — a human applies it
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


def run_scaling_ab(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "volume",
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

    The regression-based :func:`~src.alphas.refine.case_test` is one cheap number; this
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
    from src.indicators import indicators

    return indicators


def compute_horizon(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "volume",
    benchmark: str = "SPY",
    neutralize: bool = True,
    neutralize_factors: Sequence[str] = (),
    ic_prior: float = DEFAULT_IC,
    max_lag: int = 10,
    n_points: int = 20,
    timeframe: str = "1Day",
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
    from src.alphas import horizon as hz
    from src.indicators import indicators

    run_id = new_run_id()
    strat = _strategy(strategy, None)
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
    from src.costs import ParametricCostModel

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
        "note": "δ is the per-period IC decay (HL = half-life). Rebalance near the "
        "cadence that maximizes IC·√(1/Δt); amortize cost over the half-life. The lagged "
        "blend is recommended only when it diversifies and its turnover cost is modest.",
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
) -> Dict[str, Any]:
    """Estimate the universe's covariance Σ and summarize its risk structure.

    Read-only research-clock flow: scans returns up to ``as_of`` (leakage-safe),
    estimates an annualized, well-conditioned Σ (``shrinkage`` = Ledoit–Wolf,
    ``factor`` = structural ``X F Xᵀ + Δ``, ``sample`` = raw), and returns a compact
    summary - shrinkage δ, condition number, mean correlation, equal-weight portfolio
    volatility, top risk contributors, and (factor model) the factor-vs-specific risk
    split. Σ itself is not inlined; this is the diagnostic the optimizer consumes.
    """
    from src.risk import COVARIANCE_MODELS
    from src.risk.factor import FactorRiskMatrix

    if model not in COVARIANCE_MODELS:
        raise ValueError(f"model must be one of {sorted(COVARIANCE_MODELS)}, got {model!r}")

    run_id = new_run_id()
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    fetched = ClientBarSource(data_client).scan([*symbols, benchmark], timeframe, as_of, lookback_days)
    bars = {s: fetched[s] for s in symbols if s in fetched}
    matrix = _build_covariance(model, bars, fetched.get(benchmark), periods_per_year, min_obs)

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
        "note": "Σ is annualized and kept invertible (shrinkage δ, or a structural "
        "factor model). Risk is not additive — correlated names are one bet. This is "
        "the denominator the portfolio optimizer divides alpha by.",
    }

    # The factor model makes risk attributable: split the equal-weight portfolio's
    # variance into common-factor risk and idiosyncratic (specific) risk.
    if isinstance(matrix, FactorRiskMatrix):
        total_var = matrix.variance(weights)
        factor_var = matrix.factor_variance(weights)
        result["factor_names"] = matrix.factor_names
        result["factor_risk_share"] = float(factor_var / total_var) if total_var > 0 else 0.0
        result["specific_risk_share"] = float(1.0 - factor_var / total_var) if total_var > 0 else 0.0

    return result


def construct_portfolio(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    as_of: datetime,
    source: str = "strategy",
    scanner: str = "volume",
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
) -> Dict[str, Any]:
    """Construct the utility-maximizing portfolio from alphas (005) and Σ (006).

    Read-only research-clock flow: scans the universe as of ``as_of``, builds
    benchmark-neutral alphas and an annualized covariance Σ, then maximizes
    ``αᵀw − λ·wᵀΣw − cost(w − w₀)`` over long-only, box-bounded, budgeted (and
    optionally cardinality-capped) weights, calibrating ``λ`` to ``target_te``.

    With ``cost_aware`` (default) the objective carries the transaction-cost term:
    name-specific linear turnover (commission + a high-low-range spread proxy) and,
    when ``capital`` is set, the √-impact term - so the optimizer trades a name's
    alpha against *that name's* cost and a no-trade band emerges from the cost itself.
    ``cost_aware=False`` recovers the cost-blind (gross) solve with an ex-post drag.
    Returns the proposed weights plus the Fundamental-Law report (IR*, predicted TE/IR,
    transfer coefficient, turnover, cost split). This is a **proposal** - no orders.
    """
    from src.costs import ParametricCostModel
    from src.portfolio.optimizer import MeanVarianceOptimizer

    run_id = new_run_id()
    strat = _strategy(strategy, None)
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
    matrix = _build_covariance(risk_model, universe_bars, bench_frame, periods_per_year)

    if not alphas or matrix is None:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "as_of": as_of.isoformat(),
            "feasible": False,
            "note": "Insufficient data for alphas and/or a covariance matrix.",
        }

    cost_model = ParametricCostModel()
    # Per-name, as-of liquidity context (spread proxy, ADV$, daily vol) priced by the
    # same cost model the backtest uses - the optimizer and backtest share one model.
    cost_inputs = _cost_inputs(universe_bars, cost_model) if cost_aware else None

    optimizer = MeanVarianceOptimizer(max_weight=max_weight, min_weight=min_weight, max_names=max_names)
    result = optimizer.optimize(
        alphas,
        matrix,
        target_te=target_te,
        current_weights=current_weights,
        cost_model=cost_model if cost_aware else None,
        cost_inputs=cost_inputs,
        capital=capital,
        holding_period_years=holding_period_years,
    )

    if result.feasible:
        if not result.diagnostics.get("cost_aware"):
            # Cost-blind (gross) objective: still report the ex-post drag so the net figure
            # is visible. Same convention as the cost-aware path - a round-trip haircut on
            # the book for the headline (matching capacity), one-way rebalance drag in detail.
            h = max(holding_period_years, 1e-9)
            rate = cost_model.turnover_cost_rate()
            expected = result.diagnostics["expected_active_return"]
            one_way = result.diagnostics["turnover"] * rate / h
            round_trip = 2.0 * rate * sum(result.weights.values()) / h
            result.diagnostics["cost_drag"] = one_way
            result.diagnostics["round_trip_cost"] = round_trip
            result.diagnostics["expected_active_return_net"] = expected - round_trip
            result.diagnostics["expected_active_return_net_oneway"] = expected - one_way
        # Capacity: the capital at which √-impact cost erases the gross alpha.
        result.diagnostics["capacity_capital"] = _capacity(
            result.weights, universe_bars, result.diagnostics["expected_active_return"], holding_period_years
        )

    holdings = []
    if result.feasible and capital:
        from src.utils.numeric import round_quantity

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

    return {
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
        "shrinkage": matrix.shrinkage,
        "weights": _jsonable(dict(sorted(result.weights.items(), key=lambda kv: kv[1], reverse=True))),
        "holdings": _jsonable(holdings),
        "diagnostics": _jsonable(result.diagnostics),
        "note": "PROPOSAL, not an order. Maximizes αᵀw − λ·wᵀΣw at the target tracking "
        "error; the transfer coefficient shows how much of IR* survives the constraints.",
    }


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
    from src.portfolio.optimizer import CostInputs

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
    from src.costs import ParametricCostModel, Trade

    model = ParametricCostModel()
    liquidity = {}
    for sym, w in weights.items():
        frame = universe_bars.get(sym)
        if frame is None or len(frame) < 2 or w <= 0:
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


def _build_covariance(model, bars, benchmark_bars, periods_per_year, min_obs=60):
    """Build a covariance RiskMatrix by model name (statistical estimator or factor)."""
    from src.risk import RISK_MODELS, build_factor_risk_matrix, build_risk_matrix

    if model == "factor":
        return build_factor_risk_matrix(bars, benchmark_bars, periods_per_year, min_obs=min_obs)
    return build_risk_matrix(RISK_MODELS[model](), bars, periods_per_year, min_obs=min_obs)


def _scanner(scanner_name: str):
    """Instantiate a scanner from its declared defaults (for the scanner scorer)."""
    from src.scanners.symbol_scanner import SymbolScanner

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
    from src.alphas import refine

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
    from src.risk.exposures import build_factor_exposures

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
