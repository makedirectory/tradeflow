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

from src.alphas import DEFAULT_IC, AlphaContext, ScoreAlphaModel, SignalAlphaModel, scanner_scorer
from src.analytics import metrics as m
from src.engine.backtest import BacktestEngine
from src.indicators import indicators
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
    source: str = "signal",
    scanner: str = "volume",
    ic: float = DEFAULT_IC,
    benchmark: str = "SPY",
    neutralize: bool = False,
    lookback_days: int = 180,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn a strategy/scanner's per-name view into ranked residual-return alphas.

    Read-only research-clock flow: scores each name as of ``as_of``,
    scales the cross-section into comparable annualised-return forecasts via
    ``alpha = sigma * IC * z``, and returns the ranked table. Produces no orders and
    saves no config.

    ``source`` selects the raw-score origin: ``"signal"`` reads the strategy's
    discrete BUY/SELL/HOLD (via :class:`SignalAlphaModel`); ``"score"`` reads the
    ``scanner``'s continuous signed strength (via :class:`ScoreAlphaModel`).

    Leakage discipline: every frame is sliced to bars ``<= as_of`` before any
    computation, so the result is independent of whether later bars exist.
    """
    run_id = new_run_id()
    strat = _strategy(strategy, None)
    tf = timeframe or strat.config.get("timeframe", "1Day")
    periods_per_year = Timeframe.parse(tf).periods_per_year()

    # Fetch with end == as_of (defends leakage for real feeds) then slice defensively
    # to <= as_of (FakeMarketData ignores the window, so the slice is the real guard).
    start = as_of - timedelta(days=lookback_days)
    fetch = list(dict.fromkeys([*symbols, benchmark]))
    raw = data_client.get_bars(fetch, tf, start, as_of)
    bars = {sym: _slice_to_as_of(frame, as_of) for sym, frame in raw.items()}
    bars = {sym: frame for sym, frame in bars.items() if not frame.empty}

    bench_frame = bars.get(benchmark)
    universe = {sym: frame for sym, frame in bars.items() if sym in symbols}

    # Build the scoring model and produce raw scores.
    if source == "score":
        model = ScoreAlphaModel(scanner_scorer(_scanner(scanner)))
    elif source == "signal":
        model = SignalAlphaModel(strat)
    else:
        raise ValueError(f"source must be 'signal' or 'score', got {source!r}")
    scores = model.raw_scores(universe, as_of)
    raw_by_symbol = {sc.symbol: sc.score for sc in scores}

    # Residual volatility (annualised) + beta per scored name.
    residual_vol: Dict[str, float] = {}
    betas: Dict[str, float] = {}
    benchmark_available = bench_frame is not None and not bench_frame.empty
    for sc in scores:
        close = universe[sc.symbol]["close"]
        if benchmark_available:
            beta = indicators.calculate_beta(close, bench_frame["close"])
            vol = indicators.calculate_residual_volatility(
                close, bench_frame["close"], beta, periods_per_year
            )
        else:
            # No benchmark series: fall back to total volatility, beta unknown (=1).
            beta = 1.0
            vol = m.annualized_volatility(close.pct_change().dropna(), int(round(periods_per_year)))
        betas[sc.symbol] = beta
        residual_vol[sc.symbol] = vol

    exposures = {sym: {"beta": betas[sym]} for sym in betas} if neutralize else None
    context = AlphaContext(residual_vol=residual_vol, ic=ic, neutralize=neutralize, exposures=exposures)
    alphas = model.alphas(scores, context)

    table = sorted(
        (
            {
                "symbol": a.symbol,
                "raw_score": raw_by_symbol.get(a.symbol, 0.0),
                "z": a.raw_z,
                "beta": betas.get(a.symbol, 1.0),
                "residual_vol": a.residual_vol,
                "alpha": a.alpha,
            }
            for a in alphas
        ),
        key=lambda row: row["alpha"],
        reverse=True,
    )

    return {
        "run_id": run_id,
        "strategy": strategy,
        "source": source,
        "scanner": scanner if source == "score" else None,
        "as_of": as_of.isoformat(),
        "timeframe": tf,
        "ic": ic,
        "benchmark": benchmark,
        "benchmark_available": benchmark_available,
        "neutralize": neutralize,
        "universe_size": len(scores),
        "low_confidence": context.low_confidence,
        "alphas": _jsonable(table),
        "note": "Alphas are residual-return FORECASTS, annualised, scaled by an "
        "ASSUMED IC (a prior until it is measured from realised outcomes). Relative sizing across "
        "names is correct regardless of IC; the absolute scale is only as good as it.",
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _scanner(scanner_name: str):
    """Instantiate a scanner from its declared defaults (for ScoreAlphaModel)."""
    from src.scanners.symbol_scanner import SymbolScanner

    if scanner_name not in SymbolScanner.SCANNERS:
        raise ValueError(f"Unknown scanner '{scanner_name}'. Available: {SymbolScanner.available()}")
    cls = SymbolScanner.SCANNERS[scanner_name]
    sc = cls({p: spec["default"] for p, spec in cls.PARAM_RANGES.items()})
    sc.initialize()
    return sc


def _slice_to_as_of(frame: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Return only bars at or before ``as_of`` (the leakage guard).

    Handles a tz-aware bar index against a possibly-naive ``as_of`` by localising
    the cutoff to the frame's timezone.
    """
    if frame is None or frame.empty:
        return frame
    ts = pd.Timestamp(as_of)
    idx = frame.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        ts = ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)
    return frame.loc[idx <= ts]


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
