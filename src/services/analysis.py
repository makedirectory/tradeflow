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
    Alpha,
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
    """The bootstrap-skill report (spec 023 §3.3): this config's OWN zero-alpha
    bootstrap p, always shown next to the FAMILY p from White's Reality Check
    over every OOS return series the 026 trial store has recorded for
    ``(strategy, universe, accounting)`` — replacing the Deflated Sharpe's
    assumed ``E[max]``/effective-trial-count with the actual trials.

    Call this AFTER the current trial has been journaled (``journal_trial``), so
    the family query below includes it — the whole point of the family test is
    to ask "is this trial's result still notable once every trial this campaign
    has tried is priced in," which requires this trial to already be one of them.

    Best-effort on the trial store, exactly like every other trial-store
    touchpoint (spec 026): a store-open failure (or too few trials with a usable
    stored return series) degrades to "own p only" — it never blocks the caller,
    and the report says so rather than silently omitting the family half.
    Own and family are always returned together (never one alone) — a great own
    p and a terrible family p is exactly the selection-luck signature (spec 023
    hidden factor 6).
    """
    from src.analytics import bootstrap as boot
    from src.analytics.metrics import TRADING_DAYS_PER_YEAR

    if oos_returns is None or len(oos_returns) < 8:
        return {
            "available": False,
            "note": "Not enough OOS periods for a bootstrap (need >= 8 daily observations).",
        }

    own = boot.bootstrap_null(
        oos_returns.to_numpy(), B=B, block_length=block_length, seed=seed, periods_per_year=TRADING_DAYS_PER_YEAR
    )

    family: Dict[str, Any] = {"available": False}
    try:
        from src.engine.backtest import ACCOUNTING_VERSION
        from src.store.trials import DEFAULT_JOURNAL_PATH, TrialStore, db_path_for_journal

        jpath = Path(journal_path) if journal_path else DEFAULT_JOURNAL_PATH
        acct = accounting if accounting is not None else ACCOUNTING_VERSION
        with TrialStore(db_path_for_journal(jpath), journal_path=jpath) as store:
            panel = store.returns_panel(strategy, symbols, acct, min_overlap=min_overlap)
        if panel["n_used"] >= 2:
            matrix = np.array(panel["matrix"], dtype=float)
            fam = boot.reality_check(
                matrix, B=B, block_length=block_length, seed=seed,
                periods_per_year=TRADING_DAYS_PER_YEAR, trial_ids=panel["trial_ids"],
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
        return "significant both individually and as the family's best — the strongest verdict this test gives."
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


def compute_attribution(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "volume",
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
    research-integrity guardrails 009/``compute_information`` apply to ICs.

    Read-only research-clock diagnostic (spec 019). Mirrors ``compute_information``'s
    pattern exactly: at sampled rebalances it rebuilds a leakage-safe cross-section
    (bars strictly ``<= t``) - alpha (for the paper active book), risk-factor
    exposures, and a per-period covariance Σ(t) (for the canonical Σ-implied beta,
    spec 017's "one β, everywhere") - then pairs it with the forward realized return
    over ``(t, t_fwd]``. There is no persisted weights/exposure history to consume
    (see the module-level deviation note next to ``compute_information``); this
    recomputes on the fly, the same as that function already does for alpha/IC.

    Each rebalance's active return is split, by an exact regression identity
    (:func:`src.analytics.attribution.attribute_period`), into: the systematic
    benchmark-timing bucket (``β_a(t)·r_B(t)``, further decomposed in aggregate
    into expected/surprise/timing per G&K 17.25-17.27), each risk factor
    (market/momentum/volatility/size), the strategy's own alpha as a signal column
    (plus any additional ``signals`` - other strategies' combined scores, so a
    013 ``--combine`` weight can be checked against its realized counterpart), and
    a specific (stock-picking) remainder. Every attributed t-stat uses a
    Bayesian-blended risk (17A.12: short samples lean on the risk model instead of
    a wild few-point sample SD) and the whole ranked table is deflated by the same
    multiple-testing inflation ``compute_information`` applies to ICs - ranking ~8
    attributed rows and quoting the best is exactly 009's trap, replayed here.

    ``conditional`` (spec 024, default ``None`` / off) threads an EWMA/HAR-conditioned
    Σ(t) into the per-period covariance this function already rebuilds at every
    sampled rebalance; when set, the report adds ``te_by_regime`` — predicted TE
    (from Σ(t)) vs a realized-return-dispersion proxy, bucketed by the benchmark's
    own trailing realized-vol tercile as of each rebalance (spec §3.3's regime
    split) — the number that answers "does the tracking-error budget actually hold
    across vol regimes." This runs ONE Σ choice per call (conditional or not, not
    both side by side); the net-of-cost conditional-vs-unconditional comparison
    lives in ``run_conditional_risk_ab``.

    ``bootstrap_skill`` (spec 023, default off) adds a nonparametric OWN p-value
    next to the parametric ``SE{IR}≈1/√Y`` verdict: a stationary block bootstrap
    of ``r_active_series`` under the imposed null (demeaned by its own estimated
    alpha), reported as ``bootstrap`` in the result and folded into ``verdict``.
    This is the *own* test only (a single track record, not a trial family) - the
    family Reality Check needs the 026 trial store's stored trials and lives on
    ``run_walk_forward``'s ``--bootstrap-skill`` instead.
    """
    from src.analytics import attribution as attr
    from src.analytics import information as info
    from src.portfolio.benchmark import load_benchmark_weights, restrict_and_renormalize
    from src.risk.exposures import FACTOR_NAMES, build_factor_exposures

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
        # regime label spec 024 §3.3 wants for the predicted-vs-realized TE split.
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
        bench_vol_series.append(trailing_vol)

    periods = len(r_active_series)
    if periods < 5:
        return {
            "run_id": run_id,
            "strategy": strategy,
            "periods": periods,
            "note": "Insufficient overlapping history for attribution (need >= 5 rebalances "
            "with a buildable Σ, benchmark weights, and factor exposures).",
        }

    # T_eff-honest T0 (hidden factor 7): the risk model's own min_obs, converted from
    # bars to this attribution's rebalance-period units.
    t0 = attr.prior_weight_t0(min_obs, horizon)
    psi2_bar = float(np.mean(psi2_series)) if psi2_series else 0.0
    n_rows = len(risk_names) + len(signal_names) + 2  # + timing + specific
    sigma2_prior_per_row = psi2_bar / n_rows if n_rows else 0.0

    rows: Dict[str, Any] = {}
    mu_b_period = benchmark_premium * horizon / periods_per_year
    split = attr.systematic_split(beta_a_series, r_bench_series, mu_b_period)
    rows["beta_expected"] = {"total": split["expected"], "note": "not skill (assumed premium x mean active beta)"}
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
        "distinguishable from luck" if abs(total_active_ir) / max(total_ir_se, 1e-9) >= 2 else
        "NOT distinguishable from luck"
    )
    if bootstrap_skill:
        from src.analytics import bootstrap as boot

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
        "the row's own realized variance (17A.12); a ranked table of "
        f"{n_rows} rows is a multiple-testing family - P(any |t|>2 in {n_trials} trials) = "
        f"{inflation:.2f} - so quoting the single best row (here: {best_row}) without that "
        "context is exactly the trap 009 guards ICs against. Cumulative active return is "
        "ΠR_P - ΠR_B, never Π(1+r_active); cumulation.delta_cp is the honest leftover from "
        "top-down chain-linking the per-period split, reported not hidden. te_by_regime "
        "buckets rebalances by the benchmark's own trailing realized-vol tercile and shows "
        "predicted TE (from Σ(t)) next to a realized-dispersion proxy per bucket (spec 024) — "
        "the number that says whether the tracking-error budget holds through a stress regime.",
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
    psi2_series: List[float], r_active_series: List[float], bench_vol_series: List[float], rebalances_per_year: float
) -> Dict[str, Any]:
    """Spec 024 §3.3: bucket rebalances by the benchmark's trailing realized-vol
    tercile (ex-post labels, report-time only — no look-ahead in the model itself)
    and compare, per bucket, the **predicted** TE (``sqrt(mean psi2)``, from the
    per-period Σ(t) already built for attribution) against a **realized**
    dispersion proxy (``std(r_active)·sqrt(rebalances_per_year)``) — the number
    that answers whether the tracking-error budget holds through a stress regime,
    or breaches it the way an unconditional Σ mechanically must (spec's own
    motivation, §1).
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
        realized_te = (
            float(np.std(r_active[mask], ddof=1) * np.sqrt(rebalances_per_year)) if n > 1 else 0.0
        )
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
    from src.alphas import refine

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

    ``conditional`` (spec 024, default ``None`` / **off** — the MZ/QLIKE evidence
    gate hasn't cleared this repo's own data yet, see
    ``specs/complete/024-conditional-risk.md``) conditions Σ_t's volatilities via an
    EWMA (``"ewma"``) or HAR-lite (``"har"``) per-name forecast, holding the
    correlation structure fixed. When set, the report adds ``sigma_regime`` — the
    current conditional/unconditional vol ratio per name, the "how stressed is the
    book right now" diagnostic (008/016 consume Σ_t transparently; this is the
    number a human reads).
    """
    from src.risk import COVARIANCE_MODELS
    from src.risk.factor import FactorRiskMatrix

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
    """The MZ/QLIKE evidence gate (spec 024 §4 hidden factor 8, §6): per name AND
    pooled across the universe, compare EWMA / HAR / unconditional (expanding
    trailing) one-bar-ahead variance forecasts against realized ``r²`` — Mincer–
    Zarnowitz (``b`` near 1 is well-calibrated) and QLIKE (lower is better), split
    by realized-vol tercile. **This is the gate that decides whether
    ``--conditional`` is worth turning on for this repo's own data** — not a
    preference; see the "as-built" notes in
    ``specs/complete/024-conditional-risk.md`` for what it found. Read-only, no
    orders, no feedback into any model.
    """
    from src.risk.conditional import evaluate_vol_forecasts, mincer_zarnowitz, qlike_loss

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
            returns, min_obs=min_obs, n_points=n_points, lambda_=conditional_lambda, periods_per_year=periods_per_year
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
        pooled[method] = {"mincer_zarnowitz": mincer_zarnowitz(realized_arr, arr), "qlike": qlike_loss(realized_arr, arr)}

    ranked = sorted(pooled.items(), key=lambda kv: kv[1]["qlike"])
    best_method = ranked[0][0]
    uncond = pooled["unconditional"]
    best = pooled[best_method]
    # Both prongs, per the spec's own framing (§4 hidden factor 8) — QLIKE improvement
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
        "note": "The evidence gate (spec 024 §4 hidden factor 8): QLIKE lower is better, "
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

    ``benchmark_holdings`` (spec 017) makes the benchmark a **portfolio** (``w_B``)
    rather than the ``benchmark`` return series above (which stays a beta/vol
    regression input, orthogonal to this): ``"equal"`` for uniform weight over the
    Σ-covered universe, or a ``symbol,weight`` CSV/JSON holdings file. Tracking
    error, alpha neutralization, and the transfer coefficient all move into active
    space (``w_a = w − w_B``); ``benchmark_premium`` (``μ_B``, an assumed annual
    benchmark excess return) drives the reverse-optimization report (the consensus
    returns for which ``w_B`` is itself optimal). Without ``benchmark_holdings``
    this is a no-op - every quantity reduces byte-for-byte to the cash-relative
    (pre-017) behavior.

    ``book="market_neutral"`` (spec 018) relaxes the long-only box to
    ``[−short_max_weight, max_weight]`` and the budget to ``Σw=0``; ``gross_leverage``
    (``‖w‖₁ ≤ L``) is then mandatory - see
    :meth:`~src.portfolio.optimizer.MeanVarianceOptimizer.optimize`. Borrow carry on
    the short book is priced automatically from the cost model's flat default when
    ``cost_aware``; a per-name override belongs in a future ``CostInputs.borrow`` feed
    (v1 has no per-name borrow-rate source at the service layer). ``book="long_only"``
    (the default) is unaffected.

    Returns the proposed weights plus the Fundamental-Law report (IR*, predicted TE/IR,
    transfer coefficient, turnover, cost split). This is a **proposal** - no orders.

    ``conditional`` (spec 024, default ``None`` / **off** — see
    ``specs/complete/024-conditional-risk.md`` for the evidence-gate finding that
    decided the default) conditions Σ's volatilities (EWMA/HAR) before the solve, so
    ``target_te`` is measured against *current* risk, not the trailing-window
    average — the whole point being that the optimizer sells into a vol spike to
    hold the TE budget (spec's own hidden factor 1: mechanically correct, but it
    pays 016's real transaction cost to do it; see ``sigma_regime`` in the
    diagnostics for how stressed Σ_t is relative to the unconditional estimate).

    ``posterior="bl"`` (spec 021, default ``None`` / off until validated OOS)
    blends the refined alphas with 017's consensus prior via Black–Litterman
    before the solve: names with no signal get a real, Σ-propagated posterior
    (G&K 11.25) instead of being excluded outright, and view confidence is tied to
    ``posterior_ic``/``posterior_t_eff`` rather than baked only into magnitude.
    ``posterior_t_eff`` is required when set (τ is pinned to ``1/T_eff``, never
    tuned — pass the ``effective_t`` a prior ``compute_information`` call
    measured); ``posterior_ic`` defaults to the same assumed IC the refinement
    step used; ``posterior_tau`` overrides the pinned τ. The report gains a
    ``posterior`` section (per-name consensus/view/posterior/source table, plus
    τ-sensitivity) and ``shrink_chain`` gains a ``bl`` step - the IC-uncertainty
    haircut moves from the refine step (which stays unshrunk for this path,
    avoiding a double-shrink) into Ω.
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

    cost_model = ParametricCostModel()
    # Per-name, as-of liquidity context (spread proxy, ADV$, daily vol) priced by the
    # same cost model the backtest uses - the optimizer and backtest share one model.
    cost_inputs = _cost_inputs(universe_bars, cost_model) if cost_aware else None

    benchmark_weights, benchmark_report = _resolve_benchmark_portfolio(
        benchmark_holdings, benchmark_premium, matrix
    )

    posterior_report = None
    if posterior is not None:
        alphas, posterior_report = _apply_bl_posterior(
            panel, alphas, matrix, as_of, benchmark_report, posterior, posterior_ic, posterior_t_eff,
            posterior_tau, alpha_ctx.ic,
        )

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
            # Value-added identity (G&K eq. 5.12-adjacent): SR_P² ≈ SR_B² + IR² -
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
        "shrinkage": matrix.shrinkage,
        "weights": _jsonable(dict(sorted(result.weights.items(), key=lambda kv: kv[1], reverse=True))),
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
    return out


def run_conditional_risk_ab(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    source: str = "strategy",
    scanner: str = "volume",
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
    """The net-of-cost A/B (spec 024 §3.2/§6) — the one that decides commercial
    adoption, not TE-tracking alone: walk ``[start, end]`` at spaced rebalances,
    constructing the SAME alpha book (``construct_portfolio``, same target_te, same
    cost model) against a conditional vs unconditional Σ, carrying each variant's
    weights forward to the next rebalance, and pricing the REALIZED forward return
    net of 016's actual cost (turnover cost annualized in the diagnostics, scaled
    down to this rebalance's holding period). A conditional Σ that tracks TE better
    but churns the book to death should — and, if the numbers say so, does — lose
    the net-IR comparison here. Read-only research-clock harness; no orders.
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
        "IR (net of 016's actual cost), not by TE-tracking alone — churn that tracking-"
        "error-tracks-better-but-costs-more should lose, and this is where it would show.",
    }


def longshort_report(
    data_client: MarketDataClient,
    strategy: str,
    symbols: List[str],
    as_of: datetime,
    source: str = "strategy",
    scanner: str = "volume",
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
    """The long-only price report (Spec 018 §3.2): the SAME alphas, Σ, and costs
    solved ``long_only`` vs ``market_neutral``, so the difference is attributable to
    the constraint itself, not to a different universe or a different day's data.

    Reports the measured IR shrinkage (``IR_LO / IR_LS``) next to G&K's ``(15.13)``-
    style reference line (``γ(N) = (53+N)^0.57``) - explicitly *not* a verified
    transcription of the textbook formula, just an illustrative comparison point, per
    the spec's own framing - both transfer coefficients, and the long-only book's
    incidental **size exposure** (the "size" factor already in
    ``src/risk/exposures.py``, dotted with each book's weights) that a long/short
    book, free to short the small names long-only can only zero out, does not carry.
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
        "shrinkage_reference_gk": reference_shrinkage,
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
        "note": "shrinkage_reference_gk is an illustrative G&K (15.13)-style γ(N) "
        "reference line, not a verified transcription of the exact formula - compare "
        "against shrinkage_measured, don't trust it as truth. binding_fraction is the "
        "share of the long-only universe pinned at zero weight (a proxy for the "
        "forced-underweight bound this spec relaxes), not the exact |z|-mass figure.",
    }


def _longshort_size_exposure(data_client, symbols, benchmark, as_of, lookback_days, timeframe, lo, ls):
    """Each book's dot product with the cross-sectionally standardized size factor
    (``log(price·ADV)``, ``src/risk/exposures.py``) - the incidental size bias a
    long-only book picks up from being unable to short small, unattractive names
    (spec 018 §2 goal, §6 "size bias" test).
    """
    from src.risk.exposures import build_factor_exposures

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
    names pinned at the long-only floor (``w_a = −w_B``, spec 017's underweight
    bound). Not the spec's exact "|z| mass" figure (that needs the per-name alpha
    z-score, which this report doesn't carry) - a documented simplification.
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

    ``conditional`` (spec 024, default ``None`` / off) conditions Σ's diagonal (or,
    for ``model='factor'``, both ``factor_cov`` and ``specific_var``) via an EWMA or
    HAR-lite per-name volatility forecast, holding the correlation structure fixed —
    see :mod:`src.risk.conditional`. Every caller of this helper reduces byte-for-byte
    to its pre-024 behavior when ``conditional`` is left at the default.
    """
    from src.risk import RISK_MODELS, build_factor_risk_matrix, build_risk_matrix

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
    without ``benchmark_holdings`` (spec 017).
    """
    if not benchmark_holdings:
        return None, None

    from src.portfolio.benchmark import implied_returns, load_benchmark_weights, restrict_and_renormalize

    raw = load_benchmark_weights(benchmark_holdings, matrix.symbols)
    raw_total = sum(raw.values())
    restricted, coverage = restrict_and_renormalize(raw, matrix.symbols)
    consensus = implied_returns(restricted, matrix, benchmark_premium) if restricted else {}
    report = {
        "source": benchmark_holdings,
        "premium": benchmark_premium,
        "coverage": coverage,  # fraction of raw weight mass inside Σ's universe
        "uncovered_weight": max(0.0, 1.0 - coverage),
        "raw_weight_sum": raw_total,  # far from 1 => the file implied a cash position (§4.6)
        "consensus_returns": consensus,  # μ per name - print next to α (§3.2 corollary)
    }
    return (restricted or None), report


def _apply_bl_posterior(
    panel, alphas, matrix, as_of, benchmark_report, posterior, posterior_ic, posterior_t_eff, posterior_tau, scale_ic
):
    """Spec 021: blend ``alphas`` with 017's consensus prior via Black–Litterman.

    Reads ``panel.meta["shrink_chain"]`` (populated by ``refine_alpha`` upstream,
    where ``level_shrink`` stayed off - the raw, unshrunk alpha is exactly what BL's
    Ω needs, spec 021 §4 hidden factor 1) and appends the ``bl`` step so the
    IC-uncertainty haircut is auditably applied exactly once, here, not twice.
    Returns the new (Σ-universe-spanning) alpha list and the report section
    (per-name consensus/view/posterior/source table plus τ-sensitivity).
    """
    if posterior != "bl":
        raise ValueError(f"posterior must be 'bl' or None, got {posterior!r}")
    if posterior_t_eff is None:
        raise ValueError(
            "posterior_t_eff is required for posterior='bl' - tau is pinned to "
            "1/T_eff (spec 021 §3.1), never tuned; pass the effective_t a prior "
            "compute_information call measured for this strategy/window."
        )

    from src.portfolio.posterior import black_litterman_from_ic

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
            "Ω, never both (spec 021 §4 hidden factor 1).",
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
