"""Data-client construction and universe resolution (the read-only data path).

The MCP server and research agent must be *structurally* incapable
of trading, so this builds **only** a :class:`MarketDataClient` over Alpaca's
historical-data client - never a ``TradingClient`` or broker. That absence is the
safety model.
"""

import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple

from tradeflow.marketdata.client import MarketDataClient

logger = logging.getLogger(__name__)


def build_data_client(
    cache: bool = False,
    offline: bool = False,
    cache_dir: Optional[Any] = None,
    feed: Optional[str] = None,
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

    ``feed`` pins the Alpaca market-data feed; ``None`` (the default) resolves from
    ``ALPACA_DATA_FEED`` and otherwise leaves the SDK's defaults alone.
    """
    from tradeflow.brokers.alpaca.factory import build_market_data
    from tradeflow.settings import data_feed, load_settings

    settings = load_settings()
    provider = build_market_data(settings.alpaca_key, settings.alpaca_secret, feed=feed or data_feed())
    if cache or offline:
        from tradeflow.store.bars import CachedMarketData

        provider = CachedMarketData(provider, cache_dir=cache_dir, offline=offline)
    return MarketDataClient(provider)


def resolve_universe(
    data_client: MarketDataClient,
    scanner_name: Optional[str],
    candidates: List[str],
    as_of: Optional[datetime] = None,
    *,
    then_reads: Optional[Tuple[str, datetime, datetime]] = None,
) -> List[str]:
    """Filter ``candidates`` through a scanner, falling back to them if none flag.

    ``then_reads`` is the ``(timeframe, start, end)`` the caller reads next for the
    universe this returns. Offline, when the scan itself cannot run from the cache, it
    widens the miss to cover that read too, so one warm fixes the command rather than
    one for the scan and a second for the run.
    """
    if not scanner_name or scanner_name == "none":
        return candidates

    from tradeflow.scanners.symbol_scanner import SymbolScanner

    scanner = SymbolScanner(data_client, scanner_name)
    if then_reads is not None:
        _refuse_unscannable(data_client, scanner, candidates, as_of, then_reads)
    flagged = scanner.scan(candidates, as_of=as_of)
    universe = [symbol for symbol, _ in flagged]
    if not universe:
        logger.warning("Scanner flagged no symbols; falling back to the candidate list")
        return candidates
    logger.info("Trading universe from scanner: %s", universe)
    return universe


def _refuse_unscannable(
    data_client: MarketDataClient,
    scanner: Any,
    candidates: List[str],
    as_of: Optional[datetime],
    then_reads: Tuple[str, datetime, datetime],
) -> None:
    """Offline, raise one miss for the scan and the run when the scan cannot run.

    Only when the scan window is short: if the scan can run, the universe it picks is
    checked on its own afterwards, and asking for every candidate over the run window
    would warm names the scanner then drops.
    """
    from tradeflow.store.bars import CachedMarketData

    provider = getattr(data_client, "provider", None)
    if not isinstance(provider, CachedMarketData) or not provider.offline:
        return
    scan_tf, scan_start, scan_end = scanner.scan_window(as_of)
    if provider.is_cached(candidates, scan_tf, scan_start, scan_end):
        return
    timeframe, start, end = then_reads
    reads = [(candidates, scan_tf, scan_start, scan_end)]
    if str(timeframe) == str(scan_tf):
        reads.append((candidates, timeframe, start, end))
    provider.refuse_uncached_reads(
        reads,
        why=(
            "The scanner picks the universe from these bars, so which names the run keeps "
            "is not known yet; the window covers the scan and the run for every candidate."
        )
        if len(reads) > 1
        else None,
    )
