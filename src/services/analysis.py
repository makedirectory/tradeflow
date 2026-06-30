"""Analysis services: scan, backtest, optimize, walk-forward, summarize bars.

Each function takes a data-only :class:`MarketDataClient`, runs an existing
engine/optimizer/walk-forward path, and returns a compact, JSON-serializable
dict. Large outputs (trade tables, full optimization grids) are written to an
artifact file and referenced by path - never inlined.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.data import ClientBarSource, FeaturePanel, add_risk_features, add_score_feature
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


def _strategy(strategy_name: str, config: Optional[Dict[str, Any]] = None):
    """Instantiate a strategy from defaults, overlaid with ``config`` overrides."""
    cls = resolve_strategy_class(strategy_name)
    params = {name: spec["default"] for name, spec in cls.PARAM_RANGES.items() if "default" in spec}
    if config:
        params.update(config)
    return cls(params)


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
) -> Dict[str, Any]:
    """Backtest a strategy; return the full metrics dict + a path to the trades CSV.

    Trades are NOT inlined (could be thousands of rows); read the CSV if needed.
    """
    from src.services.sizing import build_beta_sizer

    run_id = new_run_id()
    strat = _strategy(strategy, config)
    sizer = build_beta_sizer(data_client, strat, symbols, benchmark, as_of=start) if beta_sizing else None
    result = BacktestEngine(strat, data_client, sizer=sizer).run(symbols, start, end, capital)

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
) -> Dict[str, Any]:
    """Search a strategy's parameters IN-SAMPLE; return best params + top-N rows.

    WARNING for the caller: these are in-sample results from selecting the best of
    many configs - NOT evidence of edge. Validate with ``run_walk_forward`` before
    trusting any of this; ``best_score`` will almost always look good here.
    """
    run_id = new_run_id()
    cls = resolve_strategy_class(strategy)
    opt = ParameterOptimizer(cls, data_client, initial_capital=capital, seed=seed)
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
) -> Dict[str, Any]:
    """Honest evaluation: optimize IS, score OOS across folds, gate the verdict.

    This is the advancement criterion - returns the OOS aggregate, efficiency,
    degradation, per-fold summary, holdout (if requested), the Deflated Sharpe
    (with n_trials across all folds), and the promotion-gate pass/fail + overall
    ``promotable``. ``include_pbo`` is expensive and defaults off.
    """
    run_id = new_run_id()
    cls = resolve_strategy_class(strategy)
    validator = WalkForwardValidator(cls, data_client, initial_capital=capital, seed=seed, gates=gates)
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

    Descriptive only. NOTE for the caller: choosing symbols by their realised
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
    lookback_days: int = 180,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn a per-name view into ranked residual-return alphas, via a feature panel.

    Read-only research-clock flow: scans the universe as of ``as_of`` (leakage-safe),
    assembles a :class:`FeaturePanel` (risk + score columns), refines it into a
    comparable annualised forecast (``alpha = sigma * IC * z``), and returns the
    ranked table. Produces no orders and saves no config.

    ``source`` selects the score column's origin: ``"strategy"`` uses the strategy's
    continuous conviction; ``"signal"`` uses its discrete BUY/SELL/HOLD as +1/-1/0;
    ``"scanner"`` uses the ``scanner``'s continuous signed strength.
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
    add_score_feature(panel, scorer(), universe_bars)

    context = AlphaContext(ic=ic, neutralize=neutralize)
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
        "universe_size": int(panel.get("score").notna().sum()) if panel.has("score") else 0,
        "low_confidence": bool(panel.meta.get("low_confidence")),
        "alphas": _jsonable(table),
        "note": "Alphas are residual-return FORECASTS, annualised, scaled by an "
        "ASSUMED IC (a prior until it is measured from realised outcomes). Relative sizing across "
        "names is correct regardless of IC; the absolute scale is only as good as it.",
    }


def compute_combined_alphas(
    data_client: MarketDataClient,
    signals: List[str],
    symbols: List[str],
    as_of: datetime,
    benchmark: str = "SPY",
    neutralize: bool = False,
    lookback_days: int = 365,
    timeframe: str = "1Day",
    horizon: int = 5,
    n_points: int = 12,
) -> Dict[str, Any]:
    """Combine several strategies' signals into one alpha by their IC and correlation.

    Read-only research-clock flow: measures each signal's IC and the signal
    correlation matrix over a trailing window (realised residual returns), shrinks the
    ICs by their estimation confidence, and combines them with GLS weights
    (``Ω⁻¹·IC``) so redundant signals split a weight rather than double-count. The
    combined score is scaled by the **measured** combined IC - replacing Spec 005's
    assumed per-signal scalar, never applied twice. Returns the ranked alpha table
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
    # The combined, measured, shrunk IC replaces the assumed scalar (no double-scaling).
    context = AlphaContext(ic=measurement.combined_ic, neutralize=neutralize)
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
        "universe_size": int(panel.get("score").notna().sum()) if panel.has("score") else 0,
        "low_confidence": bool(panel.meta.get("low_confidence")),
        "n_periods": measurement.n_periods,
        "combined_ic": measurement.combined_ic,
        "signal_ics": _jsonable(measurement.ics),
        "signal_shrunk_ics": _jsonable(measurement.shrunk_ics),
        "signal_weights": _jsonable(measurement.weights),
        "signal_correlation": _jsonable(measurement.correlation.to_dict()),
        "alphas": _jsonable(table),
        "note": "ICs and the signal correlation are MEASURED over the trailing window "
        "(not assumed) and shrunk by estimation confidence; redundant signals split a "
        "weight via Ω⁻¹. Measure on out-of-sample data for an honest combination.",
    }


def compute_risk(
    data_client: MarketDataClient,
    symbols: List[str],
    as_of: datetime,
    model: str = "shrinkage",
    lookback_days: int = 365,
    timeframe: str = "1Day",
    min_obs: int = 60,
) -> Dict[str, Any]:
    """Estimate the universe's covariance Σ and summarise its risk structure.

    Read-only research-clock flow: scans returns up to ``as_of`` (leakage-safe),
    estimates an annualised, well-conditioned Σ (Ledoit–Wolf shrinkage by default),
    and returns a compact summary - shrinkage δ, condition number, mean correlation,
    the equal-weight portfolio volatility, and the top risk contributors. Σ itself is
    not inlined; this is the diagnostic view the optimiser consumes programmatically.
    """
    from src.risk import RISK_MODELS, build_risk_matrix

    if model not in RISK_MODELS:
        raise ValueError(f"model must be one of {sorted(RISK_MODELS)}, got {model!r}")

    run_id = new_run_id()
    periods_per_year = Timeframe.parse(timeframe).periods_per_year()
    bars = ClientBarSource(data_client).scan(symbols, timeframe, as_of, lookback_days)
    matrix = build_risk_matrix(RISK_MODELS[model](), bars, periods_per_year, min_obs=min_obs)

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

    return {
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
        "note": "Σ is annualised and shrunk to stay invertible (δ is the shrinkage "
        "intensity; higher = noisier sample). Risk is not additive — correlated names "
        "are one bet. This is the denominator the portfolio optimiser divides alpha by.",
    }


def construct_portfolio(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    as_of: datetime,
    source: str = "strategy",
    scanner: str = "volume",
    target_te: float = 0.04,
    max_weight: float = 0.25,
    max_names: Optional[int] = None,
    benchmark: str = "SPY",
    neutralize: bool = True,
    risk_model: str = "shrinkage",
    lookback_days: int = 365,
    timeframe: str = "1Day",
    capital: Optional[float] = None,
    current_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Construct the utility-maximising portfolio from alphas (005) and Σ (006).

    Read-only research-clock flow: scans the universe as of ``as_of``, builds
    benchmark-neutral alphas and an annualised covariance Σ, then maximises
    ``αᵀw − λ·wᵀΣw`` over long-only, box-bounded, budgeted (and optionally
    cardinality-capped) weights, calibrating ``λ`` to ``target_te``. Returns the
    proposed weights plus the Fundamental-Law report (IR*, predicted TE/IR, transfer
    coefficient, turnover). This is a **proposal** - it places no orders.
    """
    from src.portfolio.optimizer import MeanVarianceOptimizer
    from src.risk import RISK_MODELS, build_risk_matrix

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
    add_score_feature(panel, scorer(), universe_bars)
    alpha_ctx = AlphaContext(ic=DEFAULT_IC, neutralize=neutralize)
    refine_alpha(panel, alpha_ctx)
    alphas = panel_to_alphas(panel, alpha_ctx)
    matrix = build_risk_matrix(RISK_MODELS[risk_model](), universe_bars, periods_per_year)

    if not alphas or matrix is None:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "as_of": as_of.isoformat(),
            "feasible": False,
            "note": "Insufficient data for alphas and/or a covariance matrix.",
        }

    optimizer = MeanVarianceOptimizer(max_weight=max_weight, max_names=max_names)
    result = optimizer.optimize(alphas, matrix, target_te=target_te, current_weights=current_weights)

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
        "note": "PROPOSAL, not an order. Maximises αᵀw − λ·wᵀΣw at the target tracking "
        "error; the transfer coefficient shows how much of IR* survives the constraints.",
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _scanner(scanner_name: str):
    """Instantiate a scanner from its declared defaults (for the scanner scorer)."""
    from src.scanners.symbol_scanner import SymbolScanner

    if scanner_name not in SymbolScanner.SCANNERS:
        raise ValueError(f"Unknown scanner '{scanner_name}'. Available: {SymbolScanner.available()}")
    cls = SymbolScanner.SCANNERS[scanner_name]
    sc = cls({p: spec["default"] for p, spec in cls.PARAM_RANGES.items()})
    sc.initialize()
    return sc


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
