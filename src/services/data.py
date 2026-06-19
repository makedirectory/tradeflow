"""Data-client construction and universe resolution (the read-only data path).

The MCP server (Spec 003 §4) and research agent must be *structurally* incapable
of trading, so this builds **only** a :class:`MarketDataClient` over Alpaca's
historical-data client - never a ``TradingClient`` or broker. That absence is the
safety model.
"""

import logging
from typing import List, Optional

from src.marketdata.client import MarketDataClient

logger = logging.getLogger(__name__)


def build_data_client() -> MarketDataClient:
    """Construct the Alpaca-backed historical data client from ``config.py``.

    Deliberately constructs no trading client / broker (Spec 003 §5.6): the
    process that calls this cannot place orders.
    """
    import sys

    try:
        import config
    except ModuleNotFoundError:
        sys.exit("config.py not found. Copy config_example.py to config.py and add your Alpaca keys.")

    from alpaca.data.historical import StockHistoricalDataClient

    from src.brokers.alpaca.market_data import AlpacaMarketData

    historical = StockHistoricalDataClient(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)
    return MarketDataClient(
        AlpacaMarketData(historical, config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)
    )


def resolve_universe(
    data_client: MarketDataClient, scanner_name: Optional[str], candidates: List[str]
) -> List[str]:
    """Filter ``candidates`` through a scanner, falling back to them if none flag."""
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
