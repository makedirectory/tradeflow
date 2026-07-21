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
    trials       inspect the trial store: campaign n_trials, recent trials, drift check

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


def build_cost_model(args):
    """Build the parametric cost model from ``--gross``/``--commission-bps``/etc.,
    or ``None`` for ``--gross``.

    Shared by backtest/optimize/walkforward so a search or validation prices trades
    the same way a live backtest does. Gross-by-default optimization silently
    favors the highest-turnover config (see ``ParameterOptimizer``'s and
    ``WalkForwardValidator``'s own docstrings), so this must reach every
    research-clock entrypoint that can run a search or a validation, not just
    ``backtest``.
    """
    from src.costs import ParametricCostModel

    if args.gross:
        return None
    return ParametricCostModel(
        commission_bps=args.commission_bps, impact_eta=args.impact_eta, annual_borrow_bps=args.borrow_bps
    )


# ---------------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------------- #
def cmd_backtest(args) -> None:
    from src.analytics.reporting import log_backtest_report
    from src.engine.backtest import BacktestEngine
    from src.services.sizing import build_beta_sizer

    _, data_client = build_data_and_broker()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    strategy = STRATEGIES[args.strategy].create_with_defaults()

    sizer = None
    if args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark, as_of=args.start)

    # Metrics are net of transaction cost by default; --gross disables the charge.
    cost_model = build_cost_model(args)
    result = BacktestEngine(strategy, data_client, sizer=sizer, cost_model=cost_model).run(
        universe, args.start, args.end, args.capital
    )

    if not args.no_journal:
        from src.services.audit import journal_trial

        # One backtest is one evaluated config = one trial. Record only the tunable
        # params (config also carries timeframe/limits/lookback, which are not knobs).
        tunable = {k: strategy.config[k] for k in strategy.PARAM_RANGES if k in strategy.config}
        journal_trial(
            "backtest",
            strategy=args.strategy,
            symbols=universe,
            start=args.start,
            end=args.end,
            params=tunable,
            metrics=result.metrics,
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
        neutralize_factors=args.neutralize_factors,
        capital=args.capital,
        holding_period_years=args.holding_period,
        cost_aware=not args.gross_objective,
    )
    if not result["feasible"]:
        print(f"Infeasible: {result.get('binding_constraint') or result.get('note')}")
        return

    d = result["diagnostics"]
    mode = "cost-aware" if d.get("cost_aware") else "gross (cost-blind)"
    print(
        f"\nPortfolio for '{args.strategy}' as of {args.as_of:%Y-%m-%d} "
        f"(target TE {args.target_te:.0%}, {mode})"
    )
    print(
        f"  IR* {d['ir_star']:.2f}  predicted TE {d['predicted_tracking_error']:.1%}  "
        f"predicted IR {d['predicted_ir']:.2f}  transfer coef {d['transfer_coefficient']:.2f}  "
        f"turnover {d['turnover']:.1%}"
    )
    if "round_trip_cost" in d:
        # Headline: the conservative round-trip haircut (enter + exit, amortized) - the
        # same cost model as capacity, so the two agree.
        print(
            f"  net active return {d['expected_active_return_net']:.2%}/yr (round-trip)  "
            f"= gross {d['expected_active_return']:.2%} − round-trip cost {d['round_trip_cost']:.2%}"
        )
        # Detail: the one-way cost of this rebalance's turnover (prior reporting).
        if d.get("cost_aware"):
            print(
                f"    this rebalance: turnover cost {d['cost_drag']:.2%}/yr one-way "
                f"(linear {d['linear_cost']:.2%} + √-impact {d['impact_cost']:.2%}); "
                f"one-way net {d['expected_active_return_net_oneway']:.2%}"
            )
        else:
            print(
                f"    this rebalance: turnover cost {d['cost_drag']:.2%}/yr one-way; "
                f"one-way net {d['expected_active_return_net_oneway']:.2%}"
            )
    if "capacity_capital" in d:
        print(f"  capacity ≈ ${d['capacity_capital']:,.0f} (where √-impact erases the alpha)")
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
    # Net of transaction cost by default; --gross searches on gross returns, which
    # reliably favors the highest-turnover config.
    optimizer = ParameterOptimizer(
        STRATEGIES[args.strategy],
        data_client,
        initial_capital=args.capital,
        cost_model=build_cost_model(args),
    )

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

    if not args.no_journal and not result.results.empty:
        from src.services.audit import journal_trial

        # Each evaluated config is a distinct trial — a 50-point search is 50 trials,
        # not one. Recording them per-config is what makes a campaign-level Deflated
        # Sharpe honest (spec 026); the search columns are the params, the rest metrics.
        searchable = optimizer.space.searchable
        defaults = optimizer.space.defaults
        for row in result.results.to_dict("records"):
            searched = {k: row[k] for k in searchable if k in row}
            metrics = {k: v for k, v in row.items() if k not in searchable}
            journal_trial(
                "optimize",
                strategy=args.strategy,
                symbols=universe,
                start=args.start,
                end=args.end,
                params={**defaults, **searched},
                metrics=metrics,
                objective=args.objective,
            )

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
    # Net of transaction cost by default; --gross validates on gross returns, which
    # systematically promotes turnover the strategy could not afford live.
    validator = WalkForwardValidator(
        STRATEGIES[args.strategy],
        data_client,
        initial_capital=args.capital,
        seed=args.seed,
        cost_model=build_cost_model(args),
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

    if not args.no_journal and result.folds:
        from src.services.audit import journal_trial

        # One walk-forward is one *validated* config — the OOS aggregate is the
        # headline. The many IS-optimization configs it evaluated internally are
        # already counted in n_trials_total (which the DSR used); record that count
        # rather than one row per inner config, matching how the agent journals.
        chosen = result.holdout_params or result.folds[-1].is_best_params
        report = result.gate_report()
        journal_trial(
            "walkforward",
            strategy=args.strategy,
            symbols=universe,
            start=args.start,
            end=args.end,
            params=dict(chosen),
            metrics=result.oos_aggregate,
            objective=args.objective,
            extra={
                "n_trials": result.n_trials_total,
                "promotable": report["promotable"],
                "efficiency": result.median_efficiency(),
            },
        )

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
    proposer = build_proposer(args.provider, args.model, allow_code_gen=args.allow_code_gen)
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


def _journal_alpha(args, strategy_label: str, source: str, result: dict) -> None:
    """Record one alpha run as a read-only trial.

    An alpha run is a point-in-time forecast, not a backtest — it has no Sharpe to
    deflate, so it is journaled under ``kind="alpha"`` for dedup and future IC /
    bootstrap work (specs 009, 023), and a multiple-testing count skips this kind.
    The window collapses to the single as-of date.
    """
    if args.no_journal:
        return
    from src.services.audit import journal_trial

    knobs = {
        "source": source,
        "ic": getattr(args, "ic", None),
        "scaling": getattr(args, "scaling", None),
        "neutralize": args.neutralize,
        "neutralize_factors": list(args.neutralize_factors or []),
        "lookback_days": args.lookback_days,
    }
    journal_trial(
        "alpha",
        strategy=strategy_label,
        symbols=args.symbols,
        start=args.as_of,
        end=args.as_of,
        params={k: v for k, v in knobs.items() if v is not None},
        metrics={},  # a forecast has no realized-edge metric
        extra={
            "universe_size": result.get("universe_size", len(result.get("alphas", []))),
            "benchmark_available": result.get("benchmark_available"),
            "low_confidence": result.get("low_confidence"),
        },
    )


def cmd_alphas(args) -> None:
    """Print the ranked alpha table (residual-return forecasts) for a universe.

    Read-only research-clock flow: scores each name as of --as-of, scales the
    cross-section into comparable annualized-return forecasts, and ranks them.
    Produces no orders and saves no config.
    """
    from src.services.analysis import compute_alphas, compute_combined_alphas

    _, data_client = build_data_and_broker()

    if args.combine:
        combined = compute_combined_alphas(
            data_client,
            args.combine,
            args.symbols,
            as_of=args.as_of,
            benchmark=args.benchmark,
            neutralize=args.neutralize,
            neutralize_factors=args.neutralize_factors,
            lookback_days=args.lookback_days,
        )
        _journal_alpha(args, "+".join(args.combine), "combine", combined)
        _print_combined_alphas(combined, args)
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
        neutralize_factors=args.neutralize_factors,
        lookback_days=args.lookback_days,
        scaling=args.scaling,
    )

    _journal_alpha(args, args.strategy, args.source, result)

    src_desc = {
        "strategy": f"'{args.strategy}' score",
        "signal": f"'{args.strategy}' signal",
        "scanner": f"scanner '{args.scanner}' strength",
    }[args.source]
    print(f"\nAlphas from {src_desc} as of {args.as_of:%Y-%m-%d} (IC={args.ic}, benchmark={args.benchmark})")
    _print_neutralization(result)
    _print_case(result)
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


def _print_neutralization(result) -> None:
    """One honest line about what the alphas were actually regressed against.

    Reports what was *applied* (from the refinement's meta), not what was requested —
    and warns when a requested factor exposure was unavailable, so "factor-neutral"
    is never claimed for un-neutralized output.
    """
    requested = result.get("neutralize_factors") or []
    applied = result.get("neutralized_against") or []
    if applied:
        print(f"  neutralized against: {', '.join(applied)}")
    missing = [f for f in requested if f not in applied]
    if missing:
        print(
            f"  ! requested factor(s) NOT neutralized (exposures unavailable — "
            f"insufficient history?): {', '.join(missing)}"
        )


def _print_case(result) -> None:
    """One line on the chosen scaling and the Case test, when it ran."""
    case = result.get("case")
    if not case:
        if result.get("scaling") and result["scaling"] != "case1":
            print(f"  scaling: {result['scaling']}")
        return
    amb = " (ambiguous → base rate)" if case.get("ambiguous") else ""
    print(
        f"  scaling: {result['scaling']} — Case {case['case']}{amb} "
        f"(R²={case['r_squared']:.2f}, t={case['t_stat']:+.1f}, "
        f"candidate corr {case.get('candidate_correlation', float('nan')):.2f})"
    )


def _print_combined_alphas(result, args) -> None:
    """Print the multi-signal combination: per-signal weights/ICs + the combined alphas."""
    if not result.get("universe_size"):
        print(result.get("note", "No combined alphas produced."))
        return

    print(f"\nCombined alpha from {', '.join(result['signals'])} as of {args.as_of:%Y-%m-%d}")
    _print_neutralization(result)
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

    if getattr(args, "scaling_ab", False):
        _print_scaling_ab(data_client, args)
        return

    r = compute_information(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
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
    # IC-uncertainty level shrink: how much of the measured IC level survives its own
    # estimation error, with an overlap-honest effective T.
    if "level_shrink_factor" in r:
        print(
            f"  level shrink: keep {r['level_shrink_factor']:.0%} of the naive level "
            f"(T_eff {r['effective_t']:.1f}, IC {r['recommended_ic']:+.4f})"
        )
    _print_bucket_diagnostic(r.get("risk_bucket_diagnostic"))
    verdict = "distinguishable from luck" if abs(r["ic_tstat"]) >= 2 else "NOT distinguishable from luck"
    print(f"\n  Verdict: skill is {verdict} (IC t-stat {r['ic_tstat']:+.2f}).")
    if abs(r["ic_tstat"]) >= 2:
        print(f"  Recommended IC for alpha scaling (a human applies it): {r['recommended_ic']:.4f}")
        print(
            f"  After the IC-uncertainty level shrink, deploy IC ≈ "
            f"{r['recommended_ic'] * r.get('level_shrink_factor', 1.0):.4f}"
        )


def _print_bucket_diagnostic(diag) -> None:
    """The equal-risk-contribution check, when it engaged."""
    if not diag or not diag.get("engaged"):
        if diag and diag.get("reason"):
            print(f"  risk buckets: suppressed — {diag['reason']}")
        return
    marker = "⚠ tilt" if diag["tilt_detected"] else "ok"
    print(
        f"  risk buckets ({diag['n_buckets']}, by residual vol): {marker} — "
        f"variance-share gradient {diag['variance_share_gradient']:+.2f} "
        f"vs band ±{diag['sampling_band']:.2f}"
    )
    if diag["tilt_detected"]:
        print(f"    → {diag['verdict']}")


def _print_scaling_ab(data_client, args) -> None:
    """Walk-forward A/B of the two scalings (the `--scaling-ab` research mode)."""
    from src.services.analysis import run_scaling_ab

    r = run_scaling_ab(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
        horizon=args.horizon,
    )
    if not r.get("periods"):
        print(r.get("note", "No scaling A/B produced."))
        return
    print(f"\nScaling A/B: '{args.strategy}' {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}")
    print(f"  measured over {r['periods']} rebalances (horizon {r['horizon_bars']} bars)")
    print(f"  Case 1 (σ·IC·z) realized IR: {r['case1_realized_ir']:+.2f}")
    print(f"  Case 2 (IC·c_g·z) realized IR: {r['case2_realized_ir']:+.2f}")
    agree = "agree" if r["agree"] else "DISAGREE"
    print(f"  regression picks {r['regression_pick']}, A/B picks {r['ab_pick']} — {agree}")
    print("  (compare the IR gap to its standard-error band before acting — a small gap is noise)")


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
        neutralize_factors=args.neutralize_factors,
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
    rec = "recommended" if r["blend_recommended"] else "not worth the turnover cost"
    print(
        f"  lagged blend [{r['blend_regime']}]: w_now {r['blend_weight_now']:+.2f}  "
        f"w_lagged {r['blend_weight_lagged']:+.2f}  (ρ {r['signal_autocorrelation']:.2f}, "
        f"cost {r['blend_annual_cost']:.2%}/yr → {rec})"
    )


def cmd_trials(args) -> None:
    """Inspect the trial store (spec 026): the queryable index over the research
    journal that lets a campaign-level Deflated Sharpe count every config you've
    ever tried, not just the ones from this process.
    """
    from src.store.trials import DEFAULT_JOURNAL_PATH, TrialStore

    # `query` has no --journal flag; status/rebuild do. Passing it here too means
    # a schema-mismatch rebuild (which can fire inside the constructor) replays
    # the journal the user actually pointed at, not the global default.
    with TrialStore(args.db, journal_path=getattr(args, "journal", None)) as store:
        if args.trials_command == "status":
            info = store.status(args.journal)
            print(f"DB      : {info['db_path']}")
            print(f"Journal : {info['journal_path']}")
            print(f"Schema  : v{info['schema_version']}")
            print(f"Rows    : {info['rows']}")
            print(f"Journal : {info['journal_lines']} lines ({info['journal_trial_lines']} trials)")
            if info["orphaned_rows"]:
                print(f"Orphaned: {info['orphaned_rows']} row(s) with no strategy (session_start missing)")
            if info["drift"]:
                print(
                    "\nDRIFT DETECTED — rows undercount journaled trials, or rows are orphaned. Run `trials rebuild`."
                )
            else:
                print("\nOK — no drift detected.")
            return

        if args.trials_command == "rebuild":
            journal = args.journal or DEFAULT_JOURNAL_PATH
            stats = store.rebuild(args.journal)
            print(
                f"Rebuilt {stats['rows']} trial rows from {stats['journal_lines']} journal lines ({journal})."
            )
            return

        # query — defaults to the current accounting version (spec 026 §4.6);
        # --all-accounting opts into a listing that spans versions.
        rows = store.query(
            strategy=args.strategy,
            kind=args.kind,
            accounting=args.accounting,
            all_accounting=args.all_accounting,
            limit=args.limit,
        )
        if not rows:
            print("No trials matched.")
        else:
            print(
                f"{'ID':14}{'KIND':12}{'STRATEGY':16}{'OOS SHARPE':>11}{'DSR':>7}{'PROMO':>7}{'ACCT':>6}  TS"
            )
            for r in rows:
                oos = r["oos_sharpe"] if r["oos_sharpe"] is not None else 0.0
                dsr = r["deflated_sharpe"] if r["deflated_sharpe"] is not None else 0.0
                promo = "-" if r["promotable"] is None else ("yes" if r["promotable"] else "no")
                print(
                    f"{r['id']:14}{r['kind']:12}{(r['strategy'] or '')[:16]:16}"
                    f"{oos:>11.3f}{dsr:>7.3f}{promo:>7}{r['accounting']:>6}  {(r['ts'] or '')[:19]}"
                )

        if args.strategy and args.symbols:
            from src.engine.backtest import ACCOUNTING_VERSION

            accounting = args.accounting if args.accounting is not None else ACCOUNTING_VERSION
            n = store.family_count(args.strategy, args.symbols, accounting)
            print(
                f"\nCampaign n_trials for '{args.strategy}' over {', '.join(args.symbols)} "
                f"(accounting v{accounting}): {n}"
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


def cmd_demo_agent(args) -> None:
    """Narrate one research session end-to-end on live Alpaca data.

    The arc the guardrails produce: an AI proposes a strategy, the sandbox admits
    or rejects it, survivors are validated out-of-sample by walk-forward, the
    promotion gates deliver a verdict, and whatever is left is scored once on a
    holdout reserved before the search began. Nothing here can place an order.

    ``--provider replay`` (default) replays a curated proposal set, so the demo is
    deterministic and needs no LLM key. Any other provider drives a live model
    through the identical loop.
    """
    from src.costs import ParametricCostModel
    from src.research.agent import ResearchAgent, ResearchConfig
    from src.research.demo_proposals import DEMO_PROPOSALS
    from src.research.proposer import FixedProposer, build_proposer
    from src.services.data import build_data_client

    rule = "=" * 74
    print(f"\n{rule}")
    print("  TradeFlow demo-agent — AI proposal → sandbox → walk-forward → verdict")
    print("  Live Alpaca market data · research clock only · no order path exists")
    print(rule)

    end = args.end or (datetime.now() - timedelta(days=2))
    start = args.start or (end - timedelta(days=args.lookback_days))

    if args.provider == "replay":
        proposer = FixedProposer(DEMO_PROPOSALS)
        source = f"replayed proposal set ({len(DEMO_PROPOSALS)} proposals, deterministic)"
    else:
        proposer = build_proposer(args.provider, args.model, allow_code_gen=True)
        source = f"live model — {args.provider}/{proposer.client.model}"

    # Costs are charged on every simulated fill, in-sample and out. Validating on
    # gross returns systematically promotes turnover a strategy could not afford.
    cost_model = None if args.no_costs else ParametricCostModel()

    data_client = build_data_client()
    symbols = args.symbols

    state = {"round": 0}

    def narrate(event: str, payload: dict) -> None:
        if event == "session_start":
            research = payload["research_window"]
            holdout = payload["holdout_window"]
            print(f"\n  Proposals from : {source}")
            print(f"  Universe       : {', '.join(symbols)}  (bars fetched live from Alpaca)")
            costs = (
                "gross returns (--no-costs)"
                if cost_model is None
                else "commission + half-spread + sqrt impact, charged per fill"
            )
            print(f"  Costs          : {costs}")
            print(f"  Base strategy  : {payload['strategy']}")
            print(f"  Goal           : {payload['goal']}")
            print(f"\n  Research window: {research['start'][:10]} → {research['end'][:10]}")
            print(f"  Sacred holdout : {holdout['start'][:10]} → {holdout['end'][:10]}")
            print("                   reserved now, never searched, scored once at the end")
            return

        if event == "reject":
            state["round"] += 1
            print(f"\n  ── Round {state['round']} " + "─" * 52)
            hypothesis = payload.get("hypothesis") or "(no hypothesis supplied)"
            print(f"     Hypothesis  {_wrap_field(hypothesis)}")
            print(f"     Sandbox     REJECTED — {payload['reason']}")
            print("                 ↳ no bars loaded, no backtest run, no trial consumed")
            return

        if event == "trial":
            state["round"] += 1
            print(f"\n  ── Round {state['round']} " + "─" * 52)
            kind = "new strategy implementation" if payload["kind"] == "code" else "parameter config"
            print(f"     Proposal    [{payload['kind']}] {kind}")
            print(f"     Hypothesis  {_wrap_field(payload['hypothesis'])}")
            if payload["kind"] == "code":
                print("     Sandbox     ADMITTED — imports clean, contract valid, params within cap")
            print(
                f"     Walk-forward  in-sample Sharpe {payload['is_sharpe']:>6.2f}"
                f"   →   out-of-sample {payload['oos_sharpe']:>6.2f}"
                f"   (efficiency {payload['efficiency']:.2f})"
            )
            print(f"     Multiple-testing correction applied over {payload['n_trials_cumulative']} trials")
            print("     Promotion gates:")
            for name, check in payload["gate_report"]["checks"].items():
                mark = "PASS" if check["passed"] else "FAIL"
                print(
                    f"       [{mark}] {name:<24} {check['value']:>10.2f}   threshold {check['threshold']:.2f}"
                )
            if payload["advanced"]:
                print("     Verdict     PROMOTABLE — enters the shortlist")
            elif payload["promotable"]:
                print("     Verdict     passes gates but does not beat the incumbent — discarded")
            else:
                print("     Verdict     NOT promotable — discarded")
            return

        if event == "holdout_score":
            metrics = payload["holdout_metrics"] or {}
            print(f"\n  Holdout score (first and only look) — candidate {payload['candidate']}")
            print(f"     Sharpe {metrics.get('sharpe_ratio', 0.0):.2f}   config → {payload['saved']}")

    cfg = ResearchConfig(
        goal=args.goal,
        n_folds=args.folds,
        holdout_days=args.holdout_days,
        max_evals=args.max_evals,
        max_trials=len(DEMO_PROPOSALS) if args.provider == "replay" else args.max_trials,
        # The replay is built so the early rounds are *meant* to be rejected; the
        # default dryness stop would end the session before the last proposal ran.
        max_dry_rounds=len(DEMO_PROPOSALS) if args.provider == "replay" else args.max_dry_rounds,
        capital=args.capital,
        allow_code_gen=True,
        cost_model=cost_model,
    )
    agent = ResearchAgent(args.strategy, data_client, proposer, cfg, seed=args.seed, observer=narrate)
    result = agent.run(symbols, start, end)

    print(f"\n{rule}")
    print(f"  Session {agent.session_id} — stopped: {result.stopped_reason}")
    print(f"  {result.rounds} rounds, {result.n_trials_total} cumulative trials")
    print(rule)

    if not result.shortlist:
        print("\n  Shortlist: EMPTY — nothing cleared the gates.")
        print("\n  This is the intended outcome, not a failed run. Every proposal was either")
        print("  rejected before evaluation or failed out-of-sample. An agent that cannot")
        print("  produce a losing strategy quietly is an agent you cannot trust with a")
        print("  winning one.")
    else:
        print(f"\n  Shortlist: {len(result.shortlist)} candidate(s) for HUMAN review")
        for c in result.shortlist:
            oos = c.oos_metrics.get("sharpe_ratio", 0.0)
            hold = (c.holdout_metrics or {}).get("sharpe_ratio", 0.0)
            print(f"    [{c.id}] OOS Sharpe {oos:.2f} | holdout Sharpe {hold:.2f}")
            print(f"          {c.saved_path}")
        print("\n  These are provenance-stamped config files on disk. Nothing is live.")

    print("\n  What did NOT happen — structurally, not by policy:")
    print("    · No order was placed. This command builds a data client only.")
    print("    · PAPER_TRADE was never read, let alone flipped.")
    print("    · No model output reached the trade clock. Promotion is a human step.")
    print(f"\n  Full audit trail (every proposal, rejection, and gate): {result.journal_path}\n")


def _wrap_field(text: str, width: int = 58, indent: int = 17) -> str:
    """Wrap a long narration field under a fixed label column."""
    import textwrap

    lines = textwrap.wrap(text.strip(), width=width) or [""]
    pad = " " * indent
    return f"\n{pad}".join(lines)


# ---------------------------------------------------------------------------- #
# Argument parsing
# ---------------------------------------------------------------------------- #
def _symbols(value: str) -> List[str]:
    return [s.strip().upper() for s in value.split(",") if s.strip()]


#: The bare-flag default for --neutralize-factors: the risk-control factors.
#: Momentum is deliberately excluded — a momentum tilt is a return bet the alphas
#: may intend; regress it out explicitly if that's the goal.
DEFAULT_NEUTRAL_FACTORS = ["market", "volatility", "size"]


def _factors(value: str) -> List[str]:
    from src.risk.exposures import FACTOR_NAMES

    factors = [s.strip().lower() for s in value.split(",") if s.strip()]
    unknown = [f for f in factors if f not in FACTOR_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown factor(s) {', '.join(unknown)}; available: {', '.join(FACTOR_NAMES)}"
        )
    return factors


def _add_neutralize_factors_flag(parser, note: str = "") -> None:
    parser.add_argument(
        "--neutralize-factors",
        dest="neutralize_factors",
        type=_factors,
        nargs="?",
        const=DEFAULT_NEUTRAL_FACTORS,
        default=[],
        help="Factor-neutral alphas: regress out these risk-model exposures "
        f"(comma-separated; bare flag = {','.join(DEFAULT_NEUTRAL_FACTORS)} — momentum kept){note}",
    )


def _add_cost_flags(parser) -> None:
    """--gross/--commission-bps/--impact-eta/--borrow-bps, shared by every command
    that can price a fill (backtest/optimize/walkforward) so a search or
    validation is never silently gross by omission."""
    parser.add_argument(
        "--gross",
        action="store_true",
        help="Use GROSS returns (disable transaction cost; net is the default)",
    )
    parser.add_argument("--commission-bps", dest="commission_bps", type=float, default=1.0)
    parser.add_argument(
        "--impact-eta", dest="impact_eta", type=float, default=0.3, help="Market-impact coefficient"
    )
    parser.add_argument(
        "--borrow-bps", dest="borrow_bps", type=float, default=50.0, help="Annual short borrow rate (bps)"
    )


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

    def add_no_journal(p) -> None:
        # Trials are journaled so a campaign-level Deflated Sharpe can count them
        # (spec 026). Opt out for throwaway/reproducibility runs you don't want
        # inflating the multiple-testing total.
        p.add_argument(
            "--no-journal",
            dest="no_journal",
            action="store_true",
            help="Do not record this run's trial(s) in the research journal",
        )

    bt = subparsers.add_parser("backtest", help="Run a historical backtest (did the idea ever work?)")
    add_common(bt, with_dates=True)
    bt.add_argument(
        "--beta-sizing",
        dest="beta_sizing",
        action="store_true",
        help="Scale position sizing inversely by each symbol's beta",
    )
    bt.add_argument("--benchmark", default="SPY", help="Benchmark symbol for beta")
    _add_cost_flags(bt)
    bt.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="render the equity curve + metrics to an image (needs the viz extra: matplotlib)",
    )
    add_no_journal(bt)
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
    _add_neutralize_factors_flag(alloc, note=" (utility)")
    alloc.add_argument(
        "--gross-objective",
        dest="gross_objective",
        action="store_true",
        help="Cost-blind solve (utility): drop 007's cost from the objective, report it ex-post only. "
        "Default is cost-aware (name-specific turnover + √-impact in the objective).",
    )
    alloc.add_argument(
        "--holding-period",
        dest="holding_period",
        type=float,
        default=1.0 / 12.0,
        help="Expected holding period in years, to annualize the in-objective cost (utility; default 1/12)",
    )
    alloc.set_defaults(func=cmd_allocate)

    opt = subparsers.add_parser(
        "optimize", help="Tune strategy parameters via backtest (in-sample — trust nothing yet)"
    )
    add_common(opt, with_dates=True)
    opt.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    opt.add_argument("--objective", default="sharpe_ratio")
    opt.add_argument("--max-evals", dest="max_evals", type=int, default=50)
    opt.add_argument("--output", default="optimization_results.csv")
    _add_cost_flags(opt)
    add_no_journal(opt)
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
    _add_cost_flags(wf)
    add_no_journal(wf)
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
    _add_neutralize_factors_flag(alphas)
    alphas.add_argument("--lookback-days", dest="lookback_days", type=int, default=180)
    alphas.add_argument(
        "--scaling",
        choices=["case1", "case2", "auto"],
        default="case1",
        help="Per-name scaling: 'case1' = σ·IC·z (default); 'case2' = IC·c_g·z "
        "(no per-name vol multiply); 'auto' = let a Std_TS-vs-ω regression decide",
    )
    add_no_journal(alphas)
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
    info.add_argument(
        "--scaling-ab",
        dest="scaling_ab",
        action="store_true",
        help="Research mode: walk-forward the realized IR under Case-1 vs "
        "Case-2 scaling and compare against the regression's pick",
    )
    _add_neutralize_factors_flag(info, note="; measure the alpha you deploy")
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
    _add_neutralize_factors_flag(hz, note="; measure the alpha you deploy")
    hz.set_defaults(func=cmd_horizon)

    risk = subparsers.add_parser(
        "risk",
        help="Estimate the universe covariance Σ and summarize its risk structure — read-only",
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

    def _add_db_flag(p) -> None:
        p.add_argument("--db", default=None, help="Trial store DB path (default: logs/trials.db)")

    trials = subparsers.add_parser(
        "trials", help="Inspect the trial store — the queryable index over the research journal (spec 026)"
    )
    trials_sub = trials.add_subparsers(dest="trials_command", required=True)

    t_status = trials_sub.add_parser("status", help="Row/journal-line counts and a drift check")
    _add_db_flag(t_status)
    t_status.add_argument(
        "--journal", default=None, help="Journal path (default: logs/research_journal.jsonl)"
    )

    t_rebuild = trials_sub.add_parser(
        "rebuild", help="Rebuild the index from the journal (derived — safe to delete the DB)"
    )
    _add_db_flag(t_rebuild)
    t_rebuild.add_argument(
        "--journal", default=None, help="Journal path (default: logs/research_journal.jsonl)"
    )

    t_query = trials_sub.add_parser("query", help="List recent trials, and a campaign's n_trials")
    _add_db_flag(t_query)
    t_query.add_argument("--strategy", default=None)
    t_query.add_argument(
        "--kind", default=None, choices=["backtest", "optimize", "walkforward", "alpha", "research"]
    )
    t_query.add_argument(
        "--symbols",
        type=_symbols,
        default=None,
        help="Universe — with --strategy, also prints campaign n_trials",
    )
    t_query.add_argument(
        "--accounting", type=int, default=None, help="Filter to one accounting version (default: current)"
    )
    t_query.add_argument(
        "--all-accounting",
        dest="all_accounting",
        action="store_true",
        help="Show every accounting version, not just current — read the ACCOUNTING column "
        "before comparing rows across versions",
    )
    t_query.add_argument("--limit", type=int, default=20)

    trials.set_defaults(func=cmd_trials)

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

    dagent = subparsers.add_parser(
        "demo-agent",
        help="Narrate an AI research session on live data: proposal -> sandbox -> walk-forward -> verdict",
    )
    dagent.add_argument("--symbols", type=_symbols, default=["NVDA", "AAPL", "META", "AMD", "MSFT"])
    dagent.add_argument("--strategy", default="ma_crossover")
    dagent.add_argument("--start", type=_date, default=None)
    dagent.add_argument("--end", type=_date, default=None)
    dagent.add_argument(
        "--lookback-days",
        dest="lookback_days",
        type=int,
        default=1095,
        help="History to research over when --start is omitted (default: 3 years)",
    )
    dagent.add_argument("--goal", default="Improve out-of-sample Sharpe without raising max drawdown")
    dagent.add_argument("--folds", type=int, default=4)
    dagent.add_argument("--holdout-days", dest="holdout_days", type=int, default=90)
    dagent.add_argument("--max-evals", dest="max_evals", type=int, default=25)
    dagent.add_argument("--max-trials", dest="max_trials", type=int, default=6)
    dagent.add_argument("--max-dry-rounds", dest="max_dry_rounds", type=int, default=3)
    dagent.add_argument("--capital", type=float, default=100_000.0)
    dagent.add_argument(
        "--provider",
        choices=["replay", "anthropic", "openai", "ollama"],
        default="replay",
        help="'replay' (default) needs no API key and is deterministic; anything else drives a live model",
    )
    dagent.add_argument("--model", default=None, help="Model id (ignored when --provider replay)")
    dagent.add_argument("--seed", type=int, default=42)
    dagent.add_argument(
        "--no-costs",
        dest="no_costs",
        action="store_true",
        help="Validate on gross returns (diagnostic only; costs are charged by default)",
    )
    dagent.set_defaults(func=cmd_demo_agent)

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
