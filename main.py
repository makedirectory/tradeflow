"""Command-line entry point.

A thin adapter that wires the layers together per command - all the real work
lives in ``src/`` (and the shared service core in ``src/services/``, so the CLI,
the MCP server, and the research agent run the same code):

    demo         run the whole pipeline on synthetic data (no keys, no network)
    scan         run the universe scanner and print flagged symbols
    backtest     scan universe -> BacktestEngine -> performance report
    live         scan universe -> LiveEngine -> LiveTrader (paper/live orders)
    optimize     search a strategy's parameters by backtest objective
    allocate     weight a portfolio across scanned symbols (OR-Tools)
    alphas       rank a universe by continuous alpha (residual-return forecast)
    risk         estimate the universe covariance Σ and summarize its risk structure
    info         information report: IC, breadth, predicted-vs-realized IR
    horizon      measure alpha decay / half-life; recommend cadence + lagged blend
    walkforward  out-of-sample validation with promotion gates
    mcp          serve TradeFlow to an agent over MCP (read-only)
    research     autonomous, offline research loop -> shortlist of configs

Run ``python main.py <command> --help`` for options, or use the Makefile targets
for preconfigured combos (``make demo``, ``make backtest``, ...).
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Optional

from src.services.registry import STRATEGIES
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# A reasonable default candidate list for the scanner to filter.
DEFAULT_UNIVERSE = ["NVDA", "RIVN", "NFLX", "META", "BAC", "MS", "TSLA", "GS", "AMD", "AAPL"]


# ---------------------------------------------------------------------------- #
# Wiring
# ---------------------------------------------------------------------------- #
def build_data_and_broker():
    """Construct the Alpaca-backed broker and market-data client from settings."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    from src.brokers.alpaca.broker import AlpacaBroker
    from src.brokers.alpaca.market_data import AlpacaMarketData
    from src.marketdata.client import MarketDataClient
    from src.settings import load_settings

    settings = load_settings()
    trading_client = TradingClient(
        api_key=settings.alpaca_key,
        secret_key=settings.alpaca_secret,
        paper=settings.paper_trade,
    )
    historical = StockHistoricalDataClient(settings.alpaca_key, settings.alpaca_secret)

    broker = AlpacaBroker(trading_client, settings.alpaca_key, settings.alpaca_secret, settings.paper_trade)
    data_client = MarketDataClient(AlpacaMarketData(historical, settings.alpaca_key, settings.alpaca_secret))
    return broker, data_client


def resolve_universe(data_client, scanner_name: Optional[str], candidates: List[str]) -> List[str]:
    """Filter ``candidates`` through the scanner, falling back to them if none flag.

    Delegates to the shared service core so the CLI and MCP server use one path.
    """
    from src.services.data import resolve_universe as _resolve

    return _resolve(data_client, scanner_name, candidates)


# ---------------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------------- #
def cmd_backtest(args) -> None:
    from src.analytics.reporting import log_backtest_report
    from src.costs import ParametricCostModel
    from src.engine.backtest import BacktestEngine
    from src.services.sizing import build_beta_sizer

    _, data_client = build_data_and_broker()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    strategy = STRATEGIES[args.strategy].create_with_defaults()

    sizer = None
    if args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark, as_of=args.start)

    # Metrics are net of transaction cost by default; --gross disables the charge.
    cost_model = (
        None
        if args.gross
        else ParametricCostModel(
            commission_bps=args.commission_bps, impact_eta=args.impact_eta, annual_borrow_bps=args.borrow_bps
        )
    )
    result = BacktestEngine(strategy, data_client, sizer=sizer, cost_model=cost_model).run(
        universe, args.start, args.end, args.capital
    )
    log_backtest_report(result.metrics, result.initial_capital, result.final_capital)
    if not args.gross and result.total_cost:
        print(
            f"Transaction cost: ${result.total_cost:,.2f} "
            f"({result.total_cost / result.initial_capital * 100:.2f}% of capital); "
            f"gross final ${result.gross_final_capital:,.2f}"
        )

    if getattr(args, "chart", None):
        from src.analytics.charts import render_backtest_chart

        try:
            path = render_backtest_chart(result, args.chart, title=f"{args.strategy} — backtest")
            print(f"Chart saved to {path}")
        except RuntimeError as exc:  # matplotlib (viz extra) not installed
            print(f"Chart skipped: {exc}")


def cmd_scan(args) -> None:
    from src.scanners.symbol_scanner import SymbolScanner

    _, data_client = build_data_and_broker()
    flagged = SymbolScanner(data_client, args.scanner).scan(args.symbols)
    if not flagged:
        print("No symbols flagged.")
        return
    print(f"{'SYMBOL':10}SIGNAL")
    for symbol, signal in flagged:
        print(f"{symbol:10}{signal}")


def cmd_allocate(args) -> None:
    if getattr(args, "objective", "weights") == "utility":
        _allocate_utility(args)
        return

    from src.scanners.symbol_scanner import SymbolScanner
    from src.services.sizing import allocate_portfolio

    _, data_client = build_data_and_broker()
    scanner = SymbolScanner(data_client, args.scanner)
    flagged = [symbol for symbol, _ in scanner.scan(args.symbols)]
    if not flagged:
        print("No symbols flagged; nothing to allocate.")
        return

    allocations = allocate_portfolio(
        data_client, flagged, scanner.timeframe, args.capital, args.max_positions, args.max_weight
    )
    if not allocations:
        print("Solver produced no allocation (no positively-scored candidates).")
        return

    print(f"{'SYMBOL':10}{'WEIGHT':>8}{'DOLLARS':>14}{'SHARES':>10}")
    for a in allocations:
        print(f"{a.symbol:10}{a.weight:>7.1%}{a.dollars:>14,.2f}{a.shares:>10.0f}")


def _allocate_utility(args) -> None:
    """Mean-variance portfolio construction (alpha + Σ) — a read-only proposal."""
    from src.services.analysis import construct_portfolio

    _, data_client = build_data_and_broker()
    result = construct_portfolio(
        data_client,
        args.strategy,
        args.symbols,
        as_of=args.as_of,
        source=args.source,
        scanner=args.scanner,
        target_te=args.target_te,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        max_names=args.max_names,
        benchmark=args.benchmark,
        capital=args.capital,
    )
    if not result["feasible"]:
        print(f"Infeasible: {result.get('binding_constraint') or result.get('note')}")
        return

    d = result["diagnostics"]
    print(f"\nPortfolio for '{args.strategy}' as of {args.as_of:%Y-%m-%d} (target TE {args.target_te:.0%})")
    print(
        f"  IR* {d['ir_star']:.2f}  predicted TE {d['predicted_tracking_error']:.1%}  "
        f"predicted IR {d['predicted_ir']:.2f}  transfer coef {d['transfer_coefficient']:.2f}  "
        f"turnover {d['turnover']:.1%}"
    )
    if "capacity_capital" in d:
        print(
            f"  cost drag {d['cost_drag']:.2%}/yr  capacity ≈ ${d['capacity_capital']:,.0f} "
            f"(where √-impact erases the alpha)"
        )
    print(f"\n{'SYMBOL':10}{'WEIGHT':>8}" + (f"{'DOLLARS':>14}{'SHARES':>10}" if result["holdings"] else ""))
    if result["holdings"]:
        for h in result["holdings"]:
            print(f"{h['symbol']:10}{h['weight']:>7.1%}{h['dollars']:>14,.2f}{h['shares']:>10.0f}")
    else:
        for sym, w in result["weights"].items():
            print(f"{sym:10}{w:>7.1%}")


def cmd_optimize(args) -> None:
    from src.optimization.optimizer import ParameterOptimizer

    _, data_client = build_data_and_broker()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    optimizer = ParameterOptimizer(STRATEGIES[args.strategy], data_client, initial_capital=args.capital)

    if args.method == "grid":
        result = optimizer.grid_search(
            universe, args.start, args.end, args.objective, max_evals=args.max_evals
        )
    elif args.method == "random":
        result = optimizer.random_search(
            universe, args.start, args.end, args.objective, n_samples=args.max_evals
        )
    else:  # bayesian
        result = optimizer.optimize_bayesian(universe, args.start, args.end, args.objective)

    print(f"\nBest {result.objective}: {result.best_score:.4f}")
    print(f"Best parameters: {result.best_params}")
    if not result.results.empty:
        result.results.to_csv(args.output, index=False)
        print(f"Full results written to {args.output}")


def cmd_walkforward(args) -> None:
    from src.optimization.config_store import build_provenance, save_config
    from src.optimization.walk_forward import WalkForwardValidator

    _, data_client = build_data_and_broker()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    validator = WalkForwardValidator(
        STRATEGIES[args.strategy], data_client, initial_capital=args.capital, seed=args.seed
    )
    result = validator.run(
        universe,
        args.start,
        args.end,
        mode=args.mode,
        n_folds=args.folds,
        train_days=args.train_days,
        test_days=args.test_days,
        embargo_days=args.embargo_days,
        holdout_days=args.holdout_days,
        method=args.method,
        objective=args.objective,
        max_evals=args.max_evals,
        pbo=args.pbo,
        monte_carlo=args.monte_carlo,
        parameter_sensitivity=args.param_sensitivity,
        leakage_probe=args.leakage_probe,
    )
    _print_walkforward(result, args.objective)

    if getattr(args, "chart", None):
        from src.analytics.charts import render_walkforward_chart

        try:
            path = render_walkforward_chart(result, args.chart, title=f"{args.strategy} — walk-forward")
            print(f"Chart saved to {path}")
        except RuntimeError as exc:  # matplotlib (viz extra) not installed
            print(f"Chart skipped: {exc}")

    if args.results_csv:
        rows = [
            {
                "fold": fr.fold.index,
                **{f"is_{k}": v for k, v in fr.is_metrics.items()},
                **{f"oos_{k}": v for k, v in fr.oos_metrics.items()},
                "oos_trades": fr.oos_trades,
            }
            for fr in result.folds
        ]
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.results_csv, index=False)
        print(f"\nPer-fold results written to {args.results_csv}")

    if args.save_config and result.folds:
        chosen = result.holdout_params or result.folds[-1].is_best_params
        provenance = build_provenance(
            method=args.method,
            objective=args.objective,
            windows={
                "start": args.start,
                "end": args.end,
                "mode": args.mode,
                "folds": len(result.folds),
                "holdout_days": args.holdout_days,
                "embargo_days": args.embargo_days,
            },
            oos_metrics=result.oos_aggregate,
            n_trials=result.n_trials_total,
            seed=args.seed,
        )
        path = save_config(
            args.save_config,
            strategy=args.strategy,
            scanner=args.scanner,
            params=chosen,
            provenance=provenance,
        )
        print(f"Chosen config saved to {path} (a human promotes it to live; nothing auto-flips)")


def _print_walkforward(result, objective: str) -> None:
    print("\n=== Walk-Forward Validation ===")
    print(
        f"{'FOLD':>4} {'IS ' + objective:>16} {'OOS ' + objective:>16} {'OOS Sharpe':>12} "
        f"{'OOS PF':>8} {'OOS trades':>11}"
    )
    for fr in result.folds:
        print(
            f"{fr.fold.index:>4} {fr.is_metrics.get(objective, 0):>16.3f} "
            f"{fr.oos_metrics.get(objective, 0):>16.3f} {fr.oos_metrics.get('sharpe_ratio', 0):>12.3f} "
            f"{fr.oos_metrics.get('profit_factor', 0):>8.2f} {fr.oos_trades:>11}"
        )

    agg = result.oos_aggregate
    print("\n--- OOS aggregate (concatenated trades) ---")
    print(
        f"  Sharpe {agg.get('sharpe_ratio', 0):.3f}  CAGR {agg.get('cagr', 0):.2f}%  "
        f"MaxDD {agg.get('max_drawdown', 0):.2f}%  PF {agg.get('profit_factor', 0):.2f}  "
        f"DSR {agg.get('deflated_sharpe_ratio', 0):.3f}  trades {agg.get('total_trades', 0)}"
    )
    print(
        f"  Efficiency (OOS/IS {objective}): {result.efficiency:.3f}  trials total: {result.n_trials_total}"
    )
    if result.degradation:
        deg = "  ".join(f"{k} {v:+.3f}" for k, v in result.degradation.items())
        print(f"  Degradation (IS-OOS): {deg}")
    if result.holdout is not None:
        print("\n--- Holdout (scored once) ---")
        print(
            f"  Sharpe {result.holdout.get('sharpe_ratio', 0):.3f}  "
            f"CAGR {result.holdout.get('cagr', 0):.2f}%  trades {result.holdout.get('total_trades', 0)}"
        )
    if result.pbo is not None:
        print(f"\nPBO (prob. of backtest overfitting): {result.pbo:.2f}")
    if result.monte_carlo:
        mc = result.monte_carlo
        print(
            f"Monte Carlo OOS Sharpe p05/p50: {mc.get('sharpe_p05', 0):.3f} / {mc.get('sharpe_p50', 0):.3f}"
        )

    report = result.gate_report()
    print("\n--- Promotion gates ---")
    for name, check in report["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {check['value']} (threshold {check['threshold']})")
    verdict = "PROMOTABLE" if report["promotable"] else "NOT promotable"
    median_sharpe = result.median_oos("sharpe_ratio")
    print(
        f"\nVerdict: {verdict} — OOS Sharpe {median_sharpe:.2f}, efficiency "
        f"{result.median_efficiency():.2f}, {result.total_oos_trades()} OOS trades, "
        f"DSR {agg.get('deflated_sharpe_ratio', 0):.2f}"
    )


def cmd_research(args) -> None:
    """Run the autonomous research loop (opt-in; needs the ``ai`` extra).

    Offline research clock only: proposes hypotheses, validates them out-of-sample
    via walk-forward, and writes a shortlist of provenance-stamped candidate
    configs for a human to review. Never touches live trading.
    """
    import importlib.util

    # Each provider pulls in a different optional dependency (Ollama needs none).
    required = {"anthropic": "anthropic", "openai": "openai"}.get(args.provider)
    if required and importlib.util.find_spec(required) is None:
        extra = {"anthropic": "ai", "openai": "openai"}[args.provider]
        sys.exit(
            f"Provider '{args.provider}' needs the '{extra}' extra. Install it:\n    uv sync --extra {extra}"
        )

    from src.research.agent import ResearchAgent, ResearchConfig
    from src.research.proposer import build_proposer
    from src.services.data import build_data_client

    data_client = build_data_client()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    cfg = ResearchConfig(
        goal=args.goal,
        mode=args.mode,
        n_folds=args.folds,
        embargo_days=args.embargo_days,
        holdout_days=args.holdout_days,
        method=args.method,
        objective=args.objective,
        max_evals=args.max_evals,
        capital=args.capital,
        max_trials=args.max_trials,
        max_dry_rounds=args.max_dry_rounds,
        max_tokens=args.max_tokens,
        shortlist_size=args.shortlist_size,
        allow_code_gen=args.allow_code_gen,
    )
    proposer = build_proposer(args.provider, args.model)
    agent = ResearchAgent(args.strategy, data_client, proposer, cfg, seed=args.seed)
    result = agent.run(universe, args.start, args.end)

    print(f"\n=== Research session {agent.session_id} ===")
    print(
        f"Stopped: {result.stopped_reason} after {result.rounds} rounds, "
        f"{result.n_trials_total} cumulative trials"
    )
    print(f"Holdout (scored once): {result.holdout_window}")
    if not result.shortlist:
        print("No candidate cleared the promotion gates.")
    for c in result.shortlist:
        oos = c.oos_metrics.get("sharpe_ratio", 0.0)
        hold = (c.holdout_metrics or {}).get("sharpe_ratio", 0.0)
        print(f"  [{c.id}] OOS Sharpe {oos:.2f} | holdout Sharpe {hold:.2f} | {c.saved_path}")
        print(f"        hypothesis: {c.hypothesis[:100]}")
    print(f"\nJournal: {result.journal_path}  (a human reviews and promotes; nothing is live)")


def cmd_alphas(args) -> None:
    """Print the ranked alpha table (residual-return forecasts) for a universe.

    Read-only research-clock flow: scores each name as of --as-of, scales the
    cross-section into comparable annualised-return forecasts, and ranks them.
    Produces no orders and saves no config.
    """
    from src.services.analysis import compute_alphas, compute_combined_alphas

    _, data_client = build_data_and_broker()

    if args.combine:
        _print_combined_alphas(
            compute_combined_alphas(
                data_client,
                args.combine,
                args.symbols,
                as_of=args.as_of,
                benchmark=args.benchmark,
                neutralize=args.neutralize,
                lookback_days=args.lookback_days,
            ),
            args,
        )
        return

    result = compute_alphas(
        data_client,
        args.strategy,
        args.symbols,
        as_of=args.as_of,
        source=args.source,
        scanner=args.scanner,
        ic=args.ic,
        benchmark=args.benchmark,
        neutralize=args.neutralize,
        lookback_days=args.lookback_days,
    )

    src_desc = {
        "strategy": f"'{args.strategy}' score",
        "signal": f"'{args.strategy}' signal",
        "scanner": f"scanner '{args.scanner}' strength",
    }[args.source]
    print(f"\nAlphas from {src_desc} as of {args.as_of:%Y-%m-%d} (IC={args.ic}, benchmark={args.benchmark})")
    if not result["benchmark_available"]:
        print("  ! benchmark unavailable — residual vol falls back to total volatility")
    if result["low_confidence"]:
        print(f"  ! thin universe ({result['universe_size']} names) — demean-only, low confidence")
    if not result["alphas"]:
        print("No scorable names.")
        return

    print(f"\n{'SYMBOL':10}{'SCORE':>10}{'Z':>9}{'BETA':>8}{'RESID_VOL':>11}{'ALPHA':>10}")
    for row in result["alphas"]:
        print(
            f"{row['symbol']:10}{row['score']:>10.3f}{row['z']:>9.2f}{row['beta']:>8.2f}"
            f"{row['residual_vol']:>10.1%}{row['alpha']:>10.2%}"
        )


def _print_combined_alphas(result, args) -> None:
    """Print the multi-signal combination: per-signal weights/ICs + the combined alphas."""
    if not result.get("universe_size"):
        print(result.get("note", "No combined alphas produced."))
        return

    print(f"\nCombined alpha from {', '.join(result['signals'])} as of {args.as_of:%Y-%m-%d}")
    print(f"  measured over {result['n_periods']} rebalances  |  combined IC {result['combined_ic']:.4f}")
    print(f"\n{'SIGNAL':16}{'IC':>9}{'SHRUNK':>9}{'WEIGHT':>9}")
    for sig in result["signals"]:
        print(
            f"{sig:16}{result['signal_ics'][sig]:>9.4f}{result['signal_shrunk_ics'][sig]:>9.4f}"
            f"{result['signal_weights'][sig]:>9.3f}"
        )
    if result["low_confidence"]:
        print(f"\n  ! thin universe ({result['universe_size']} names) — demean-only, low confidence")
    print(f"\n{'SYMBOL':10}{'SCORE':>10}{'Z':>9}{'BETA':>8}{'RESID_VOL':>11}{'ALPHA':>10}")
    for row in result["alphas"]:
        print(
            f"{row['symbol']:10}{row['score']:>10.3f}{row['z']:>9.2f}{row['beta']:>8.2f}"
            f"{row['residual_vol']:>10.1%}{row['alpha']:>10.2%}"
        )


def cmd_risk(args) -> None:
    """Print the universe's covariance risk summary (read-only).

    Estimates an annualized, well-conditioned Σ as of --as-of and reports the
    shrinkage intensity, conditioning, mean correlation, equal-weight portfolio
    volatility, and the top risk contributors. Produces no orders.
    """
    from src.services.analysis import compute_risk

    _, data_client = build_data_and_broker()
    result = compute_risk(
        data_client,
        args.symbols,
        as_of=args.as_of,
        model=args.model,
        benchmark=args.benchmark,
        lookback_days=args.lookback_days,
        timeframe=args.timeframe,
    )
    if not result.get("universe_size"):
        print(result.get("note", "No risk matrix produced."))
        return

    delta = result["shrinkage"]
    delta_str = f"  shrinkage δ {delta:.3f}" if delta is not None else ""
    print(f"\nRisk model '{result['model']}' as of {args.as_of:%Y-%m-%d} ({result['timeframe']} returns)")
    print(f"  names {result['universe_size']}{delta_str}")
    print(
        f"  condition number {result['condition_number']:.1f}  PD {result['positive_definite']}  "
        f"mean corr {result['mean_correlation']:.2f}  eq-weight vol {result['equal_weight_volatility']:.1%}"
    )
    if "factor_risk_share" in result:
        print(
            f"  risk split: {result['factor_risk_share']:.0%} factor / "
            f"{result['specific_risk_share']:.0%} specific  (factors: {', '.join(result['factor_names'])})"
        )
    print(f"\n{'SYMBOL':10}{'VOL':>9}{'RISK CONTRIB':>14}")
    for row in result["top_risk_contributors"]:
        print(f"{row['symbol']:10}{row['volatility']:>8.1%}{row['risk_contribution']:>14.2%}")


def cmd_info(args) -> None:
    """Print the information report: IC, breadth, and the predicted-vs-realized IR.

    Read-only research diagnostic: measures the strategy's information coefficient and
    effective breadth over [start, end] and reconciles predicted IR with realized,
    surfacing the research-integrity guardrails. Produces no orders.
    """
    from src.services.analysis import compute_information

    _, data_client = build_data_and_broker()
    r = compute_information(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        horizon=args.horizon,
        n_trials=args.n_trials,
    )
    if not r.get("periods"):
        print(r.get("note", "No information report produced."))
        return

    print(f"\nInformation report: '{args.strategy}' {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}")
    print(f"  measured over {r['periods']} rebalances (horizon {r['horizon_bars']} bars)")
    flag = "  ⚠ low sample" if r["low_sample"] else ""
    print(f"  IC mean {r['mean_ic']:+.4f}  t-stat {r['ic_tstat']:+.2f}  rank-IC {r['rank_ic']:+.4f}{flag}")
    print(
        f"  breadth: {r['breadth_effective']:.0f} effective vs {r['breadth_naive']:.0f} naive "
        f"(ρ̄ {r['rho_bar']:.2f}, {r['n_names']} names)"
    )
    print(
        f"  IR: predicted {r['predicted_ir']:+.2f}  realized {r['realized_ir']:+.2f} "
        f"± {r['ir_standard_error']:.2f} (SE)"
    )
    print(
        f"  guardrails: P(any |t|>2 in {r['n_trials']} trials) = {r['multiple_testing_inflation']:.2f}"
        + ("  | ⚠ realized IR > 2 — suspect a bug/leak" if r["sanity_ceiling_breached"] else "")
    )
    print(
        f"  attribution: factor {r['factor_return']:+.4f} / specific {r['specific_return']:+.4f} "
        f"per rebalance (factor tilts vs name selection)"
    )
    verdict = "distinguishable from luck" if abs(r["ic_tstat"]) >= 2 else "NOT distinguishable from luck"
    print(f"\n  Verdict: skill is {verdict} (IC t-stat {r['ic_tstat']:+.2f}).")
    if abs(r["ic_tstat"]) >= 2:
        print(f"  Recommended IC for alpha scaling (a human applies it): {r['recommended_ic']:.4f}")


def cmd_horizon(args) -> None:
    """Print the alpha-decay curve, half-life, recommended cadence, and lagged blend.

    Read-only research diagnostic: measures how fast the signal's IC decays and turns
    that into a rebalance cadence and a current/lagged blend. Produces no orders.
    """
    from src.services.analysis import compute_horizon

    _, data_client = build_data_and_broker()
    r = compute_horizon(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        max_lag=args.max_lag,
        timeframe=args.timeframe,
    )
    if not r.get("ic_by_lag"):
        print(r.get("note", "No horizon report produced."))
        return

    print(f"\nInformation horizon: '{args.strategy}' {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}")
    print("  IC by lag: " + "  ".join(f"{n}:{ic:+.3f}" for n, ic in sorted(r["ic_by_lag"].items())))
    hl = r["half_life"]
    hl_str = f"{hl:.1f} periods" if hl == hl and hl != float("inf") else "∞ (no decay detected)"
    print(f"  decay δ {r['decay_delta']:.3f}  half-life {hl_str}  fit R² {r['decay_r_squared']:.2f}")
    print(
        f"  recommended cadence: every {r['recommended_cadence']} periods  "
        f"(best return horizon ≈ {r['peak_return_horizon']:.1f})"
    )
    print(
        f"  lagged blend [{r['blend_regime']}]: w_now {r['blend_weight_now']:+.2f}  "
        f"w_lagged {r['blend_weight_lagged']:+.2f}  (signal autocorr ρ {r['signal_autocorrelation']:.2f})"
    )


def cmd_mcp(args) -> None:
    """Serve TradeFlow over MCP (stdio). Opt-in; requires the ``mcp`` extra.

    Live trading is intentionally not exposed: the server builds
    only a data client, so it cannot place orders.
    """
    try:
        from src.mcp.server import serve
    except ImportError:
        sys.exit("The MCP server needs the 'mcp' extra. Install it:\n    uv sync --extra mcp")
    serve()


def cmd_live(args) -> None:
    from src.engine.live import LiveEngine
    from src.execution.live_trader import LiveTrader
    from src.services.sizing import build_beta_sizer, build_portfolio_weight_sizer

    broker, data_client = build_data_and_broker()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    strategy = STRATEGIES[args.strategy].create_with_defaults()

    sizer = None
    if args.portfolio:
        account = broker.get_account()
        equity = account.equity if account else 100_000.0
        sizer = build_portfolio_weight_sizer(
            data_client, equity, universe, "1Day", args.max_positions, args.max_weight
        )
        if sizer is not None:
            universe = sizer.symbols  # trade only the funded names
    elif args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark)

    engine = LiveEngine(strategy, data_client, LiveTrader(broker, strategy, sizer=sizer))
    try:
        asyncio.run(engine.start(universe))
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down live engine.")


def cmd_demo(args) -> None:
    """Run the whole pipeline on synthetic data - no Alpaca keys, no network.

    The point isn't the numbers; it's the *shape* of the workflow. We backtest
    every registered strategy (in-sample, where everything looks plausible), then
    walk-forward one of them out-of-sample and let the promotion gates deliver the
    verdict. The data is a seeded random walk with no real edge - so a healthy run
    ends in "NOT promotable", which is exactly the honesty the engine exists for.
    """
    from src.engine.backtest import BacktestEngine
    from src.marketdata.client import MarketDataClient
    from src.marketdata.synthetic import SyntheticMarketData
    from src.marketdata.timeframe import Timeframe
    from src.optimization.walk_forward import WalkForwardValidator

    data_client = MarketDataClient(SyntheticMarketData(seed=args.seed))
    symbols = ["SYNW", "SYNX", "SYNY", "SYNZ"]
    anchor = datetime(2024, 12, 31)

    def window_for(timeframe: str) -> tuple:
        """A window wide enough for the timeframe (intraday needs fewer days)."""
        unit = Timeframe.parse(timeframe).unit
        days = 60 if unit in ("min", "hour") else 730
        return anchor - timedelta(days=days), anchor

    print("\n" + "=" * 70)
    print("  TradeFlow demo — synthetic data, no API keys, no network")
    print("  (a seeded random walk: realistic-looking, no actual edge)")
    print("=" * 70)

    print("\n1) In-sample backtest of every registered strategy")
    print("   In-sample, almost anything looks tradeable. That's the trap.\n")
    print(f"   {'STRATEGY':18}{'RETURN':>10}{'SHARPE':>9}{'TRADES':>8}")
    print(f"   {'-' * 43}")
    chart_result = None  # the wf strategy's in-sample BacktestResult, for the chart
    wf_strategy = "ma_crossover"
    for name, cls in STRATEGIES.items():
        try:
            strategy = cls.create_with_defaults()
            start, end = window_for(strategy.config["timeframe"])
            result = BacktestEngine(strategy, data_client).run(symbols, start, end, 100_000.0)
            if name == wf_strategy:
                chart_result = result
            m = result.metrics
            print(
                f"   {name:18}{m.get('total_return', 0.0):>9.2f}%"
                f"{m.get('sharpe_ratio', 0.0):>9.2f}{int(m.get('total_trades', 0)):>8}"
            )
        except Exception as exc:  # noqa: BLE001 - demo should never hard-crash
            print(f"   {name:18}{'(skipped: ' + str(exc)[:30] + ')':>30}")

    print(f"\n2) Walk-forward validation of '{wf_strategy}' (the honest scorecard)")
    print("   Optimize in-sample, score out-of-sample across folds, then gate it.\n")
    start, end = window_for(STRATEGIES[wf_strategy].create_with_defaults().config["timeframe"])
    wf_result = WalkForwardValidator(
        STRATEGIES[wf_strategy], data_client, initial_capital=100_000.0, seed=args.seed
    ).run(
        symbols,
        start,
        end,
        n_folds=3,
        holdout_days=90,
        method="grid",
        objective="sharpe_ratio",
        max_evals=12,
    )
    report = wf_result.gate_report()

    print(
        f"   OOS Sharpe (median): {wf_result.median_oos('sharpe_ratio'):.2f}   "
        f"efficiency (OOS/IS): {wf_result.median_efficiency():.2f}   "
        f"OOS trades: {wf_result.total_oos_trades()}"
    )
    print("\n   Promotion gates:")
    for gate_name, check in report["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"     [{mark}] {gate_name}: {check['value']} (threshold {check['threshold']})")
    verdict = "PROMOTABLE" if report["promotable"] else "NOT promotable"
    print(f"\n   Verdict: {verdict}")
    print(
        "\n   No edge in a random walk → the gates refuse to promote it. That refusal\n"
        "   is the product. Point TradeFlow at real data with `make backtest` (add\n"
        "   your Alpaca paper keys to .env first — see .env.example).\n"
    )

    if getattr(args, "chart", None):
        if chart_result is None:
            print("   Chart skipped: no backtest result was captured.")
        else:
            from src.analytics.charts import render_demo_summary

            try:
                path = render_demo_summary(
                    chart_result, wf_result, args.chart, strategy=wf_strategy, seed=args.seed
                )
                print(f"   Saved demo chart → {path}\n")
            except RuntimeError as exc:  # matplotlib (viz extra) not installed
                print(f"   Chart skipped: {exc}\n")


# ---------------------------------------------------------------------------- #
# Argument parsing
# ---------------------------------------------------------------------------- #
def _symbols(value: str) -> List[str]:
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TradeFlow — a broker-agnostic trading engine that's refreshingly "
        "honest about how hard making money actually is"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p, *, with_dates: bool) -> None:
        p.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
        p.add_argument("--scanner", default="volume", help="Universe scanner ('none' to skip)")
        p.add_argument(
            "--symbols", type=_symbols, default=DEFAULT_UNIVERSE, help="Comma-separated candidate symbols"
        )
        if with_dates:
            p.add_argument("--start", type=_date, default=datetime.now() - timedelta(days=30))
            p.add_argument("--end", type=_date, default=datetime.now())
            p.add_argument("--capital", type=float, default=100_000.0)

    bt = subparsers.add_parser("backtest", help="Run a historical backtest (did the idea ever work?)")
    add_common(bt, with_dates=True)
    bt.add_argument(
        "--beta-sizing",
        dest="beta_sizing",
        action="store_true",
        help="Scale position sizing inversely by each symbol's beta",
    )
    bt.add_argument("--benchmark", default="SPY", help="Benchmark symbol for beta")
    bt.add_argument(
        "--gross",
        action="store_true",
        help="Report GROSS metrics (disable transaction cost; net is the default)",
    )
    bt.add_argument("--commission-bps", dest="commission_bps", type=float, default=1.0)
    bt.add_argument(
        "--impact-eta", dest="impact_eta", type=float, default=0.3, help="Market-impact coefficient"
    )
    bt.add_argument(
        "--borrow-bps", dest="borrow_bps", type=float, default=50.0, help="Annual short borrow rate (bps)"
    )
    bt.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="render the equity curve + metrics to an image (needs the viz extra: matplotlib)",
    )
    bt.set_defaults(func=cmd_backtest)

    live = subparsers.add_parser("live", help="Run live/paper trading (paper by default, for your own good)")
    add_common(live, with_dates=False)
    live.add_argument(
        "--portfolio",
        action="store_true",
        help="Size positions by OR-Tools portfolio weights instead of per-trade risk",
    )
    live.add_argument(
        "--beta-sizing",
        dest="beta_sizing",
        action="store_true",
        help="Scale position sizing inversely by each symbol's beta",
    )
    live.add_argument("--benchmark", default="SPY", help="Benchmark symbol for beta")
    live.add_argument("--max-positions", dest="max_positions", type=int, default=5)
    live.add_argument("--max-weight", dest="max_weight", type=float, default=0.25)
    live.set_defaults(func=cmd_live)

    scan = subparsers.add_parser("scan", help="Run the universe scanner only")
    scan.add_argument("--scanner", default="volume")
    scan.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    scan.set_defaults(func=cmd_scan)

    alloc = subparsers.add_parser("allocate", help="Weight a portfolio over scanned symbols")
    alloc.add_argument("--scanner", default="volume")
    alloc.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    alloc.add_argument("--capital", type=float, default=100_000.0)
    alloc.add_argument("--max-positions", dest="max_positions", type=int, default=5)
    alloc.add_argument("--max-weight", dest="max_weight", type=float, default=0.25)
    alloc.add_argument(
        "--objective",
        choices=["weights", "utility"],
        default="weights",
        help="'weights' = trailing-return scalar sizing (OR-Tools); "
        "'utility' = mean-variance construction from alpha + Σ (a research proposal)",
    )
    alloc.add_argument(
        "--strategy", choices=STRATEGIES, default="volume_spike", help="Alpha source (utility)"
    )
    alloc.add_argument(
        "--source",
        choices=["strategy", "signal", "scanner"],
        default="strategy",
        help="Score origin (utility)",
    )
    alloc.add_argument(
        "--as-of", dest="as_of", type=_date, default=datetime.now(), help="Rebalance date (utility)"
    )
    alloc.add_argument(
        "--target-te", dest="target_te", type=float, default=0.04, help="Target tracking error"
    )
    alloc.add_argument(
        "--max-names", dest="max_names", type=int, default=None, help="Cardinality cap (utility)"
    )
    alloc.add_argument(
        "--min-weight",
        dest="min_weight",
        type=float,
        default=0.0,
        help="Dust floor: min weight if held (utility)",
    )
    alloc.add_argument("--benchmark", default="SPY", help="Benchmark for residual vol / beta (utility)")
    alloc.set_defaults(func=cmd_allocate)

    opt = subparsers.add_parser(
        "optimize", help="Tune strategy parameters via backtest (in-sample — trust nothing yet)"
    )
    add_common(opt, with_dates=True)
    opt.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    opt.add_argument("--objective", default="sharpe_ratio")
    opt.add_argument("--max-evals", dest="max_evals", type=int, default=50)
    opt.add_argument("--output", default="optimization_results.csv")
    opt.set_defaults(func=cmd_optimize)

    wf = subparsers.add_parser(
        "walkforward",
        help="Out-of-sample validation: optimize IS, score OOS, across folds (the honest scorecard)",
    )
    add_common(wf, with_dates=True)
    wf.add_argument("--mode", choices=["anchored", "rolling"], default="anchored")
    wf.add_argument("--folds", type=int, default=None, help="Number of folds (or use --train/--test-days)")
    wf.add_argument("--train-days", dest="train_days", type=int, default=None)
    wf.add_argument("--test-days", dest="test_days", type=int, default=None)
    wf.add_argument(
        "--embargo-days",
        dest="embargo_days",
        type=int,
        default=None,
        help="IS->OOS gap; defaults to required lookback in calendar days",
    )
    wf.add_argument("--holdout-days", dest="holdout_days", type=int, default=0)
    wf.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    wf.add_argument("--objective", default="sharpe_ratio")
    wf.add_argument("--max-evals", dest="max_evals", type=int, default=50)
    wf.add_argument("--seed", type=int, default=42)
    wf.add_argument("--pbo", action="store_true", help="Estimate probability of backtest overfitting")
    wf.add_argument(
        "--monte-carlo",
        dest="monte_carlo",
        action="store_true",
        help="Block-bootstrap the OOS trade sequence",
    )
    wf.add_argument(
        "--param-sensitivity",
        dest="param_sensitivity",
        action="store_true",
        help="Perturb chosen params +-10% and re-test",
    )
    wf.add_argument(
        "--leakage-probe",
        dest="leakage_probe",
        action="store_true",
        help="Shift the data feed forward to detect future-data leakage",
    )
    wf.add_argument("--results-csv", dest="results_csv", default=None, help="Write per-fold table to CSV")
    wf.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="render the verdict + promotion-gate scorecard to an image (needs the viz extra)",
    )
    wf.add_argument(
        "--save-config",
        dest="save_config",
        default=None,
        help="Save the chosen config (with provenance) to this path",
    )
    wf.set_defaults(func=cmd_walkforward)

    alphas = subparsers.add_parser(
        "alphas",
        help="Rank a universe by continuous alpha (residual-return forecast) — read-only",
    )
    alphas.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    alphas.add_argument("--scanner", default="volume", help="Scanner used as the --source score metric")
    alphas.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    alphas.add_argument(
        "--as-of", dest="as_of", type=_date, default=datetime.now(), help="Rebalance date (YYYY-MM-DD)"
    )
    alphas.add_argument(
        "--source",
        choices=["strategy", "signal", "scanner"],
        default="strategy",
        help="Score origin: 'strategy' = continuous conviction; 'signal' = BUY/SELL/HOLD; "
        "'scanner' = scanner strength",
    )
    alphas.add_argument("--ic", type=float, default=0.03, help="Assumed information coefficient")
    alphas.add_argument(
        "--combine",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=None,
        help="Combine several strategies' signals into one alpha (comma-separated, "
        "e.g. volume_spike,ma_crossover,mean_reversion). Measures + shrinks their ICs.",
    )
    alphas.add_argument("--benchmark", default="SPY", help="Benchmark for residual vol / beta")
    alphas.add_argument(
        "--neutralize",
        action="store_true",
        help="Make alphas beta-neutral (regress out benchmark beta)",
    )
    alphas.add_argument("--lookback-days", dest="lookback_days", type=int, default=180)
    alphas.set_defaults(func=cmd_alphas)

    info = subparsers.add_parser(
        "info",
        help="Information report: measure IC, breadth, and predicted-vs-realized IR — read-only",
    )
    info.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    info.add_argument(
        "--source", choices=["strategy", "signal", "scanner"], default="strategy", help="Alpha score origin"
    )
    info.add_argument("--scanner", default="volume", help="Scanner used when --source scanner")
    info.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    info.add_argument("--start", type=_date, default=datetime.now() - timedelta(days=365))
    info.add_argument("--end", type=_date, default=datetime.now())
    info.add_argument("--benchmark", default="SPY", help="Benchmark for residual returns")
    info.add_argument("--horizon", type=int, default=5, help="Forward-return horizon in bars")
    info.add_argument(
        "--n-trials",
        dest="n_trials",
        type=int,
        default=1,
        help="Configs tried (for multiple-testing inflation)",
    )
    info.set_defaults(func=cmd_info)

    hz = subparsers.add_parser(
        "horizon",
        help="Measure alpha decay / half-life and recommend rebalance cadence + blend — read-only",
    )
    hz.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    hz.add_argument("--source", choices=["strategy", "signal", "scanner"], default="strategy")
    hz.add_argument("--scanner", default="volume")
    hz.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    hz.add_argument("--start", type=_date, default=datetime.now() - timedelta(days=365))
    hz.add_argument("--end", type=_date, default=datetime.now())
    hz.add_argument("--benchmark", default="SPY")
    hz.add_argument(
        "--max-lag", dest="max_lag", type=int, default=10, help="Largest lag (periods) to measure"
    )
    hz.add_argument("--timeframe", default="1Day")
    hz.set_defaults(func=cmd_horizon)

    risk = subparsers.add_parser(
        "risk",
        help="Estimate the universe covariance Σ and summarise its risk structure — read-only",
    )
    risk.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    risk.add_argument(
        "--as-of", dest="as_of", type=_date, default=datetime.now(), help="Rebalance date (YYYY-MM-DD)"
    )
    risk.add_argument(
        "--model",
        choices=["shrinkage", "sample", "factor"],
        default="shrinkage",
        help="Covariance model: shrinkage (Ledoit–Wolf), sample (raw), or factor (XFXᵀ+Δ)",
    )
    risk.add_argument("--benchmark", default="SPY", help="Benchmark for the market factor / beta")
    risk.add_argument("--timeframe", default="1Day", help="Bar timeframe for returns")
    risk.add_argument("--lookback-days", dest="lookback_days", type=int, default=365)
    risk.set_defaults(func=cmd_risk)

    mcp = subparsers.add_parser(
        "mcp", help="Serve TradeFlow over MCP for an agent (opt-in; needs the 'mcp' extra)"
    )
    mcp.set_defaults(func=cmd_mcp)

    res = subparsers.add_parser(
        "research", help="Autonomous research loop -> shortlist of validated configs (needs 'ai' extra)"
    )
    add_common(res, with_dates=True)
    res.add_argument("--goal", default="Improve out-of-sample Sharpe without raising max drawdown")
    res.add_argument("--mode", choices=["anchored", "rolling"], default="anchored")
    res.add_argument("--folds", type=int, default=4)
    res.add_argument("--embargo-days", dest="embargo_days", type=int, default=None)
    res.add_argument("--holdout-days", dest="holdout_days", type=int, default=60)
    res.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    res.add_argument("--objective", default="sharpe_ratio")
    res.add_argument("--max-evals", dest="max_evals", type=int, default=25)
    res.add_argument("--max-trials", dest="max_trials", type=int, default=10)
    res.add_argument("--max-dry-rounds", dest="max_dry_rounds", type=int, default=3)
    res.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    res.add_argument("--shortlist-size", dest="shortlist_size", type=int, default=3)
    res.add_argument(
        "--allow-code-gen",
        dest="allow_code_gen",
        action="store_true",
        help="Permit agent-authored strategy code (validated in the sandbox)",
    )
    res.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama"],
        default="anthropic",
        help="LLM provider for the proposer ('ollama' runs locally, no API key)",
    )
    res.add_argument(
        "--model",
        default=None,
        help="Model id (defaults per provider, e.g. claude-opus-4-8 / gpt-4o / llama3.1)",
    )
    res.add_argument("--seed", type=int, default=42)
    res.set_defaults(func=cmd_research)

    demo = subparsers.add_parser(
        "demo", help="Run the full pipeline on synthetic data (no API keys, no network)"
    )
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="also render the demo as an image to PATH (needs the viz extra: matplotlib)",
    )
    demo.set_defaults(func=cmd_demo)

    return parser


def main() -> None:
    from src.settings import SettingsError

    setup_logging()
    args = build_parser().parse_args()
    try:
        args.func(args)
    except SettingsError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
