"""Command-line entry point.

Wires the layers together for four workflows and nothing more - all the real
work lives in ``src/``:

    backtest   scan universe -> BacktestEngine -> performance report
    live       scan universe -> LiveEngine -> LiveTrader (paper/live orders)
    scan       run the universe scanner and print flagged symbols
    optimize   search a strategy's parameters by backtest objective

Run ``python main.py <command> --help`` for options, or use the Makefile targets
for preconfigured combos (``make backtest``, ``make live``, ...).
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
def _load_config():
    try:
        import config
    except ModuleNotFoundError:
        sys.exit("config.py not found. Copy config_example.py to config.py and add your Alpaca keys.")
    return config


def build_data_and_broker():
    """Construct the Alpaca-backed broker and market-data client from config."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    from src.brokers.alpaca.broker import AlpacaBroker
    from src.brokers.alpaca.market_data import AlpacaMarketData
    from src.marketdata.client import MarketDataClient

    config = _load_config()
    trading_client = TradingClient(
        api_key=config.APCA_API_KEY_ID,
        secret_key=config.APCA_API_SECRET_KEY,
        paper=config.PAPER_TRADE,
    )
    historical = StockHistoricalDataClient(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)

    broker = AlpacaBroker(
        trading_client, config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, config.PAPER_TRADE
    )
    data_client = MarketDataClient(
        AlpacaMarketData(historical, config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)
    )
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
    from src.engine.backtest import BacktestEngine

    _, data_client = build_data_and_broker()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    strategy = STRATEGIES[args.strategy].create_with_defaults()

    sizer = None
    if args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark, as_of=args.start)

    result = BacktestEngine(strategy, data_client, sizer=sizer).run(
        universe, args.start, args.end, args.capital
    )
    log_backtest_report(result.metrics, result.initial_capital, result.final_capital)


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


def score_candidates(data_client, symbols, timeframe, lookback_days=90):
    """Score symbols by trailing return over a recent window -> [Candidate].

    Trailing return is a simple, transparent factor; swap in momentum /
    inverse-vol / signal strength here. Shared by `allocate` and portfolio-sized
    `live`.
    """
    from src.portfolio.allocator import Candidate

    end = datetime.now()
    bars = data_client.get_bars(symbols, timeframe, end - timedelta(days=lookback_days), end)
    candidates = []
    for symbol in symbols:
        frame = bars.get(symbol)
        if frame is None or len(frame) < 2:
            continue
        trailing_return = frame["close"].iloc[-1] / frame["close"].iloc[0] - 1
        candidates.append(Candidate(symbol, score=max(trailing_return, 0.0), price=frame["close"].iloc[-1]))
    return candidates


def allocate_portfolio(data_client, symbols, timeframe, capital, max_positions, max_weight):
    """Scan-agnostic allocation: score `symbols`, solve weights via OR-Tools."""
    from src.portfolio.allocator import PortfolioAllocator

    candidates = score_candidates(data_client, symbols, timeframe)
    allocator = PortfolioAllocator(max_positions=max_positions, max_weight=max_weight)
    return allocator.allocate(candidates, capital)


def compute_betas(data_client, symbols, benchmark="SPY", lookback_days=90, as_of=None):
    """Beta of each symbol vs a benchmark over a trailing daily window.

    For backtests, pass ``as_of=start`` so betas use only data *before* the
    backtest window (no look-ahead).
    """
    from src.indicators.indicators import calculate_beta

    end = as_of or datetime.now()
    bars = data_client.get_bars([benchmark, *symbols], "1Day", end - timedelta(days=lookback_days), end)
    benchmark_bars = bars.get(benchmark)
    if benchmark_bars is None or benchmark_bars.empty:
        logger.warning("No %s data; beta sizing will fall back to neutral beta", benchmark)
        return {}
    return {
        symbol: calculate_beta(bars[symbol]["close"], benchmark_bars["close"])
        for symbol in symbols
        if symbol in bars and not bars[symbol].empty
    }


def build_beta_sizer(data_client, strategy, symbols, benchmark, as_of=None):
    """Construct a BetaSizer (neutral for any symbol whose beta can't be computed)."""
    from src.execution.sizing import BetaSizer

    betas = compute_betas(data_client, symbols, benchmark, as_of=as_of)
    if betas:
        logger.info("Beta sizing: %s", {s: round(b, 2) for s, b in betas.items()})
    return BetaSizer(strategy, betas)


def cmd_allocate(args) -> None:
    from src.scanners.symbol_scanner import SymbolScanner

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
        universe, args.start, args.end,
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

    if args.results_csv:
        rows = [{"fold": fr.fold.index, **{f"is_{k}": v for k, v in fr.is_metrics.items()},
                 **{f"oos_{k}": v for k, v in fr.oos_metrics.items()}, "oos_trades": fr.oos_trades}
                for fr in result.folds]
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.results_csv, index=False)
        print(f"\nPer-fold results written to {args.results_csv}")

    if args.save_config and result.folds:
        chosen = result.holdout_params or result.folds[-1].is_best_params
        provenance = build_provenance(
            method=args.method, objective=args.objective,
            windows={"start": args.start, "end": args.end, "mode": args.mode,
                     "folds": len(result.folds), "holdout_days": args.holdout_days,
                     "embargo_days": args.embargo_days},
            oos_metrics=result.oos_aggregate, n_trials=result.n_trials_total, seed=args.seed,
        )
        path = save_config(args.save_config, strategy=args.strategy, scanner=args.scanner,
                           params=chosen, provenance=provenance)
        print(f"Chosen config saved to {path} (a human promotes it to live; nothing auto-flips)")


def _print_walkforward(result, objective: str) -> None:
    print("\n=== Walk-Forward Validation ===")
    print(f"{'FOLD':>4} {'IS '+objective:>16} {'OOS '+objective:>16} {'OOS Sharpe':>12} "
          f"{'OOS PF':>8} {'OOS trades':>11}")
    for fr in result.folds:
        print(f"{fr.fold.index:>4} {fr.is_metrics.get(objective, 0):>16.3f} "
              f"{fr.oos_metrics.get(objective, 0):>16.3f} {fr.oos_metrics.get('sharpe_ratio', 0):>12.3f} "
              f"{fr.oos_metrics.get('profit_factor', 0):>8.2f} {fr.oos_trades:>11}")

    agg = result.oos_aggregate
    print("\n--- OOS aggregate (concatenated trades) ---")
    print(f"  Sharpe {agg.get('sharpe_ratio', 0):.3f}  CAGR {agg.get('cagr', 0):.2f}%  "
          f"MaxDD {agg.get('max_drawdown', 0):.2f}%  PF {agg.get('profit_factor', 0):.2f}  "
          f"DSR {agg.get('deflated_sharpe_ratio', 0):.3f}  trades {agg.get('total_trades', 0)}")
    print(f"  Efficiency (OOS/IS {objective}): {result.efficiency:.3f}  "
          f"trials total: {result.n_trials_total}")
    if result.degradation:
        deg = "  ".join(f"{k} {v:+.3f}" for k, v in result.degradation.items())
        print(f"  Degradation (IS-OOS): {deg}")
    if result.holdout is not None:
        print(f"\n--- Holdout (scored once) ---")
        print(f"  Sharpe {result.holdout.get('sharpe_ratio', 0):.3f}  "
              f"CAGR {result.holdout.get('cagr', 0):.2f}%  trades {result.holdout.get('total_trades', 0)}")
    if result.pbo is not None:
        print(f"\nPBO (prob. of backtest overfitting): {result.pbo:.2f}")
    if result.monte_carlo:
        mc = result.monte_carlo
        print(f"Monte Carlo OOS Sharpe p05/p50: {mc.get('sharpe_p05', 0):.3f} / {mc.get('sharpe_p50', 0):.3f}")

    report = result.gate_report()
    print("\n--- Promotion gates ---")
    for name, check in report["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {check['value']} (threshold {check['threshold']})")
    verdict = "PROMOTABLE" if report["promotable"] else "NOT promotable"
    median_sharpe = result.median_oos("sharpe_ratio")
    print(f"\nVerdict: {verdict} — OOS Sharpe {median_sharpe:.2f}, efficiency "
          f"{result.median_efficiency():.2f}, {result.total_oos_trades()} OOS trades, "
          f"DSR {agg.get('deflated_sharpe_ratio', 0):.2f}")


def cmd_mcp(args) -> None:
    """Serve TradeFlow over MCP (stdio). Opt-in; requires the ``mcp`` extra.

    Live trading is intentionally not exposed (Spec 003 §4): the server builds
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

    broker, data_client = build_data_and_broker()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    strategy = STRATEGIES[args.strategy].create_with_defaults()

    sizer = None
    if args.portfolio:
        sizer = _portfolio_sizer(broker, data_client, universe, args)
        if sizer is not None:
            universe = sizer.symbols  # trade only the funded names
    elif args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark)

    engine = LiveEngine(strategy, data_client, LiveTrader(broker, strategy, sizer=sizer))
    try:
        asyncio.run(engine.start(universe))
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down live engine.")


def _portfolio_sizer(broker, data_client, universe, args):
    """Build a PortfolioWeightSizer by allocating capital across the universe.

    Returns None (so LiveTrader falls back to risk-based sizing) if OR-Tools is
    missing or the allocator funds nothing.
    """
    from src.execution.sizing import PortfolioWeightSizer

    account = broker.get_account()
    capital = account.equity if account else 100_000.0
    try:
        allocations = allocate_portfolio(
            data_client, universe, "1Day", capital, args.max_positions, args.max_weight
        )
    except RuntimeError as exc:  # OR-Tools not installed
        logger.warning("%s\nFalling back to risk-based sizing.", exc)
        return None

    weights = {a.symbol: a.weight for a in allocations}
    if not weights:
        logger.warning("Portfolio allocator funded nothing; using risk-based sizing.")
        return None

    logger.info("Portfolio-weighted live sizing: %s", {s: round(w, 3) for s, w in weights.items()})
    sizer = PortfolioWeightSizer(weights)
    sizer.symbols = list(weights)  # convenience for the caller
    return sizer


# ---------------------------------------------------------------------------- #
# Argument parsing
# ---------------------------------------------------------------------------- #
def _symbols(value: str) -> List[str]:
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TradeFlow — a broker-agnostic trading engine")
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

    bt = subparsers.add_parser("backtest", help="Run a historical backtest")
    add_common(bt, with_dates=True)
    bt.add_argument(
        "--beta-sizing",
        dest="beta_sizing",
        action="store_true",
        help="Scale position sizing inversely by each symbol's beta",
    )
    bt.add_argument("--benchmark", default="SPY", help="Benchmark symbol for beta")
    bt.set_defaults(func=cmd_backtest)

    live = subparsers.add_parser("live", help="Run live/paper trading")
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

    alloc = subparsers.add_parser("allocate", help="Weight a portfolio over scanned symbols (OR-Tools)")
    alloc.add_argument("--scanner", default="volume")
    alloc.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    alloc.add_argument("--capital", type=float, default=100_000.0)
    alloc.add_argument("--max-positions", dest="max_positions", type=int, default=5)
    alloc.add_argument("--max-weight", dest="max_weight", type=float, default=0.25)
    alloc.set_defaults(func=cmd_allocate)

    opt = subparsers.add_parser("optimize", help="Tune strategy parameters via backtest")
    add_common(opt, with_dates=True)
    opt.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    opt.add_argument("--objective", default="sharpe_ratio")
    opt.add_argument("--max-evals", dest="max_evals", type=int, default=50)
    opt.add_argument("--output", default="optimization_results.csv")
    opt.set_defaults(func=cmd_optimize)

    wf = subparsers.add_parser(
        "walkforward", help="Out-of-sample validation: optimize IS, score OOS, across folds"
    )
    add_common(wf, with_dates=True)
    wf.add_argument("--mode", choices=["anchored", "rolling"], default="anchored")
    wf.add_argument("--folds", type=int, default=None, help="Number of folds (or use --train/--test-days)")
    wf.add_argument("--train-days", dest="train_days", type=int, default=None)
    wf.add_argument("--test-days", dest="test_days", type=int, default=None)
    wf.add_argument(
        "--embargo-days", dest="embargo_days", type=int, default=None,
        help="IS->OOS gap; defaults to required lookback in calendar days",
    )
    wf.add_argument("--holdout-days", dest="holdout_days", type=int, default=0)
    wf.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    wf.add_argument("--objective", default="sharpe_ratio")
    wf.add_argument("--max-evals", dest="max_evals", type=int, default=50)
    wf.add_argument("--seed", type=int, default=42)
    wf.add_argument("--pbo", action="store_true", help="Estimate probability of backtest overfitting")
    wf.add_argument("--monte-carlo", dest="monte_carlo", action="store_true",
                    help="Block-bootstrap the OOS trade sequence")
    wf.add_argument("--param-sensitivity", dest="param_sensitivity", action="store_true",
                    help="Perturb chosen params +-10% and re-test")
    wf.add_argument("--leakage-probe", dest="leakage_probe", action="store_true",
                    help="Shift the data feed forward to detect future-data leakage")
    wf.add_argument("--results-csv", dest="results_csv", default=None, help="Write per-fold table to CSV")
    wf.add_argument("--save-config", dest="save_config", default=None,
                    help="Save the chosen config (with provenance) to this path")
    wf.set_defaults(func=cmd_walkforward)

    mcp = subparsers.add_parser(
        "mcp", help="Serve TradeFlow over MCP for an agent (opt-in; needs the 'mcp' extra)"
    )
    mcp.set_defaults(func=cmd_mcp)

    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
