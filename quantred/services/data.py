"""Data-client construction and universe resolution (the read-only data path).

The MCP server and research agent must be *structurally* incapable
of trading, so this builds **only** a :class:`MarketDataClient` over Alpaca's
historical-data client - never a ``TradingClient`` or broker. That absence is the
safety model.
"""

import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple

from quantred.marketdata.base import MarketDataProvider
from quantred.marketdata.client import MarketDataClient

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
    (:class:`~quantred.store.bars.CachedMarketData`): ``cache`` wraps the Alpaca
    provider so repeated requests reuse previously-fetched bars; ``offline``
    additionally forbids any network call (a request touching an uncached range
    raises rather than falling through to Alpaca), and implies ``cache`` on its
    own and needs no credentials. Default behavior (neither flag) is a plain Alpaca
    provider.

    ``feed`` pins the Alpaca market-data feed; ``None`` (the default) resolves from
    ``ALPACA_DATA_FEED`` and otherwise leaves the SDK's defaults alone.
    """
    from quantred.brokers.alpaca.factory import build_market_data
    from quantred.settings import data_feed, load_settings

    if offline:
        from quantred.store.bars import CachedMarketData

        # A cache-only run never reaches the vendor, so it needs no keys: demanding
        # them hid "these bars are not cached" behind a credentials wall, and a
        # warmed cache was unusable on a machine without them.
        return MarketDataClient(CachedMarketData(_NoVendor(), cache_dir=cache_dir, offline=True))
    settings = load_settings()
    provider = build_market_data(settings.alpaca_key, settings.alpaca_secret, feed=feed or data_feed())
    if cache:
        from quantred.store.bars import CachedMarketData

        provider = CachedMarketData(provider, cache_dir=cache_dir)
    return MarketDataClient(provider)


class _NoVendor(MarketDataProvider):
    """The upstream of a cache-only client: there is no vendor behind it.

    ``CachedMarketData(offline=True)`` refuses an uncached read before it would ask its
    upstream, so nothing should reach this; if something does, it fails loudly rather
    than becoming a network call nobody asked for.
    """

    def get_bars(self, symbols, timeframe, start, end):
        raise RuntimeError("--offline: an uncached read reached the vendor")

    async def stream_bars(self, symbols, handler):
        raise RuntimeError("--offline: live streaming needs a network connection")

    def supports_streaming(self) -> bool:
        return False


def vintage_stamp(data_client, universe: List[str], timeframe: Any, start: Any, end: Any) -> Optional[str]:
    """The bar cache's data-vintage stamp for this exact fetch, or ``None`` when the
    client is not cache-backed.

    Part of a run's dedup identity whenever the cache is in play, so a run over bars
    that have since changed is not answered from the old result. One definition for
    both surfaces: the CLI folded it and the service did not, so once the MCP server
    could read the cache (`mcp --offline`) the same run over the two surfaces could never
    match and journaled twice. Calling it ensures the window is cached (it warms the
    cache as a side effect), which is why callers compute it once, up front.
    """
    from quantred.store.bars import CachedMarketData

    provider = getattr(data_client, "provider", None)
    if not isinstance(provider, CachedMarketData):
        return None
    return provider.vintage_stamp(universe, timeframe, start, end)


#: ``(symbols, timeframe, start, end)``: the arguments of one ``get_bars`` call.
BarRead = Tuple[List[str], Any, datetime, datetime]


def resolve_universe(
    data_client: MarketDataClient,
    scanner_name: Optional[str],
    candidates: List[str],
    as_of: Optional[datetime] = None,
    *,
    then_reads: Optional[List[BarRead]] = None,
) -> List[str]:
    """Filter ``candidates`` through a scanner, falling back to them if none flag.

    ``then_reads`` lists the reads the caller makes next, written as if the scanner kept
    every candidate. Offline, when the scan itself cannot run from the cache, the miss
    covers that read too, so one warm fixes the command rather than one for the scan
    and a second for the run.
    """
    if not scanner_name or scanner_name == "none":
        return candidates

    from quantred.scanners.symbol_scanner import SymbolScanner

    scanner = SymbolScanner(data_client, scanner_name)
    if then_reads is not None:
        scan_tf, scan_start, scan_end = scanner.scan_window(as_of)
        refuse_uncached_together(
            data_client,
            (candidates, scan_tf, scan_start, scan_end),
            list(then_reads),
            why=(
                "The scanner picks the universe from these bars, so which names the run keeps "
                "is not known yet; the command covers the scan and the run for every candidate."
            ),
        )
    flagged = scanner.scan(candidates, as_of=as_of)
    universe = [symbol for symbol, _ in flagged]
    if not universe:
        logger.warning("Scanner flagged no symbols; falling back to the candidate list")
        return candidates
    logger.info("Trading universe from scanner: %s", universe)
    return universe


def refuse_uncached_together(
    data_client: MarketDataClient, first: BarRead, rest: List[BarRead], *, why: Optional[str] = None
) -> None:
    """Offline, when ``first`` cannot be served from the cache, raise one miss for it and ``rest``.

    A command that reads in stages otherwise learns of each stage only after warming
    the one before, a rerun per stage. Nothing is raised while ``first`` is cached,
    because whether the later reads happen at all can turn on it — a scan picks the
    names, a memoized result reads nothing more — and asking for them then would warm
    data the run may never touch; their own read refuses if it comes to that. ``why``
    is printed only when ``rest`` widened the miss.
    """
    from quantred.marketdata.timeframe import Timeframe
    from quantred.store.bars import CachedMarketData

    provider = getattr(data_client, "provider", None)
    if not isinstance(provider, CachedMarketData) or not provider.offline:
        return
    if provider.is_cached(*first):
        return

    def key(timeframe: Any) -> str:
        return str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)

    def adds(read: BarRead) -> bool:
        symbols, _, start, end = read
        return bool(set(symbols) - set(first[0])) or start < first[2] or end > first[3]

    # A warm command takes one timeframe, so a read on another is left to its own miss.
    widened = [r for r in rest if key(r[1]) == key(first[1]) and adds(r) and not provider.is_cached(*r)]
    provider.refuse_uncached_reads([first, *widened], why=why if widened else None)
