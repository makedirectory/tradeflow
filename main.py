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

from src.strategies.volume_spike import VolumeSpikeStrategy
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Registry of trading strategies exposed on the CLI.
STRATEGIES = {"volume_spike": VolumeSpikeStrategy}

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
    """Filter ``candidates`` through the scanner, falling back to them if none flag."""
    if not scanner_name or scanner_name == "none":
        return candidates

    from src.scanners.symbol_scanner import SymbolScanner

    flagged = SymbolScanner(data_client, scanner_name).scan(candidates)
    universe = [symbol for symbol, _ in flagged]
    if not universe:
        logger.warning("Scanner flagged no symbols; falling back to the candidate list")
        return candidates
    logger.info("Trading universe from scanner: %s", universe)
    return universe


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

    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
