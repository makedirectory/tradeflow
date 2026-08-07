"""Position-sizing & portfolio-allocation services.

The shared business logic for turning a universe + market data into position
sizes: trailing-return scoring, OR-Tools portfolio weights, and beta-scaled
sizing. It lives here - not in the CLI - so the CLI, the MCP server, and the
research agent all reach it through one path, the same "reuse the core" rule the
rest of ``services/`` follows. The CLI is just another adapter; it shouldn't own
domain logic any more than the MCP server does.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from tradeflow.execution.sizing import BetaSizer, PortfolioWeightSizer
from tradeflow.indicators.indicators import calculate_beta
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.portfolio.allocator import Candidate
from tradeflow.strategies.base import Strategy

logger = logging.getLogger(__name__)


def score_candidates(
    data_client: MarketDataClient,
    symbols: List[str],
    timeframe: str,
    lookback_days: int = 90,
) -> List[Candidate]:
    """Score symbols by trailing return over a recent window -> ``[Candidate]``.

    Trailing return is a simple, transparent factor; swap in momentum /
    inverse-vol / signal strength here. Shared by ``allocate`` and portfolio-sized
    ``live``.
    """
    end = datetime.now()
    bars = data_client.get_bars(symbols, timeframe, end - timedelta(days=lookback_days), end)
    candidates: List[Candidate] = []
    for symbol in symbols:
        frame = bars.get(symbol)
        if frame is None or len(frame) < 2:
            continue
        trailing_return = frame["close"].iloc[-1] / frame["close"].iloc[0] - 1
        candidates.append(Candidate(symbol, score=max(trailing_return, 0.0), price=frame["close"].iloc[-1]))
    return candidates


def allocate_portfolio(
    data_client: MarketDataClient,
    symbols: List[str],
    timeframe: str,
    capital: float,
    max_positions: int,
    max_weight: float,
):
    """Scan-agnostic allocation: score ``symbols``, solve weights via OR-Tools."""
    from tradeflow.portfolio.allocator import PortfolioAllocator

    candidates = score_candidates(data_client, symbols, timeframe)
    allocator = PortfolioAllocator(max_positions=max_positions, max_weight=max_weight)
    return allocator.allocate(candidates, capital)


def build_portfolio_weight_sizer(
    data_client: MarketDataClient,
    equity: float,
    symbols: List[str],
    timeframe: str,
    max_positions: int,
    max_weight: float,
) -> Optional[PortfolioWeightSizer]:
    """Allocate capital across ``symbols`` and wrap the weights in a sizer.

    Returns ``None`` (so the caller falls back to risk-based sizing) if OR-Tools
    is missing or the allocator funds nothing. The returned sizer exposes its
    funded ``symbols`` for convenience.
    """
    try:
        allocations = allocate_portfolio(data_client, symbols, timeframe, equity, max_positions, max_weight)
    except RuntimeError as exc:  # OR-Tools not installed
        logger.warning("%s\nFalling back to risk-based sizing.", exc)
        return None

    weights = {a.symbol: a.weight for a in allocations}
    if not weights:
        logger.warning("Portfolio allocator funded nothing; using risk-based sizing.")
        return None

    logger.info("Portfolio-weighted sizing: %s", {s: round(w, 3) for s, w in weights.items()})
    sizer = PortfolioWeightSizer(weights)
    sizer.symbols = list(weights)  # convenience for the caller
    return sizer


def compute_betas(
    data_client: MarketDataClient,
    symbols: List[str],
    benchmark: str = "SPY",
    lookback_days: int = 90,
    as_of: Optional[datetime] = None,
) -> Dict[str, float]:
    """Beta of each symbol vs a benchmark over a trailing daily window.

    For backtests, pass ``as_of=start`` so betas use only data *before* the
    backtest window (no look-ahead). Returns ``{}`` if the benchmark is missing,
    which leaves :class:`BetaSizer` to fall back to neutral betas.
    """
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


def build_beta_sizer(
    data_client: MarketDataClient,
    strategy: Strategy,
    symbols: List[str],
    benchmark: str = "SPY",
    as_of: Optional[datetime] = None,
) -> BetaSizer:
    """Construct a :class:`BetaSizer` (neutral for any symbol with no computable beta)."""
    betas = compute_betas(data_client, symbols, benchmark, as_of=as_of)
    if betas:
        logger.info("Beta sizing: %s", {s: round(b, 2) for s, b in betas.items()})
    return BetaSizer(strategy, betas)
