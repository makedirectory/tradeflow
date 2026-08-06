"""Data-client construction and universe resolution (the read-only data path).

The MCP server and research agent must be *structurally* incapable
of trading, so this builds **only** a :class:`MarketDataClient` over Alpaca's
historical-data client - never a ``TradingClient`` or broker. That absence is the
safety model.
"""

import logging
from typing import Any, List, Optional

from tradeflow.marketdata.client import MarketDataClient

logger = logging.getLogger(__name__)


def build_data_client(
    cache: bool = False, offline: bool = False, cache_dir: Optional[Any] = None
) -> MarketDataClient:
    """Construct the Alpaca-backed historical data client from settings.

    Deliberately constructs no trading client / broker: the
    process that calls this cannot place orders.

    ``cache``/``offline`` opt into the persistent bar cache
    (:class:`~tradeflow.store.bars.CachedMarketData`): ``cache`` wraps the Alpaca
    provider so repeated requests reuse previously-fetched bars; ``offline``
    additionally forbids any network call (a request touching an uncached range
    raises rather than falling through to Alpaca), and implies ``cache`` on its
    own. Default behavior (neither flag) is unchanged - a plain Alpaca provider.
    """
    from tradeflow.brokers.alpaca.factory import build_market_data
    from tradeflow.settings import load_settings

    settings = load_settings()
    provider = build_market_data(settings.alpaca_key, settings.alpaca_secret)
    if cache or offline:
        from tradeflow.store.bars import CachedMarketData

        provider = CachedMarketData(provider, cache_dir=cache_dir, offline=offline)
    return MarketDataClient(provider)


def resolve_universe(
    data_client: MarketDataClient, scanner_name: Optional[str], candidates: List[str]
) -> List[str]:
    """Filter ``candidates`` through a scanner, falling back to them if none flag."""
    if not scanner_name or scanner_name == "none":
        return candidates

    from tradeflow.scanners.symbol_scanner import SymbolScanner

    flagged = SymbolScanner(data_client, scanner_name).scan(candidates)
    universe = [symbol for symbol, _ in flagged]
    if not universe:
        logger.warning("Scanner flagged no symbols; falling back to the candidate list")
        return candidates
    logger.info("Trading universe from scanner: %s", universe)
    return universe
