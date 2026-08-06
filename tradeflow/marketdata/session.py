"""A session-scoped, in-memory bar cache.

A composite research command runs several analysis steps against one universe and
one window. Each step fetches its own bars, so the same request is issued many
times over a single run - the universe scan, the alpha panel, the covariance
lookback, and the information sampler all want the same frames.

:class:`SessionBarCache` wraps a provider and serves a repeated request from
memory instead of re-issuing it. It is deliberately *exact*: a response is reused
only for an identical ``(symbols, timeframe, start, end)`` request, so the frames
a caller sees are byte-identical to the ones the underlying provider would have
returned. It never slices a wider fetch down to a narrower window - that would
make the wrapper's output depend on request order, which is precisely the kind of
silent divergence a research result cannot afford.

Scope is one command, not one process: there is no eviction and no persistence.
For durable, cross-run caching see the Parquet-backed bar cache
(:class:`~tradeflow.store.bars.CachedMarketData`), which this composes with rather than
replaces - wrapping a cache-backed client keeps the cache's own behavior (gap
fill, vintage stamping) intact underneath.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from tradeflow.marketdata.base import BarHandler, MarketDataProvider
from tradeflow.marketdata.timeframe import Timeframe

logger = logging.getLogger(__name__)

#: The cache key for one fetch: the exact symbols, timeframe, and window asked for.
_Key = Tuple[Tuple[str, ...], str, str, str]


def _stamp(when: Any) -> str:
    """A stable string for a window bound (``None`` included) so it can key a dict."""
    iso = getattr(when, "isoformat", None)
    return iso() if callable(iso) else str(when)


class SessionBarCache(MarketDataProvider):
    """Memoize ``get_bars`` for the lifetime of one composite command.

    ``fetches`` counts requests that reached the wrapped provider; ``requests``
    counts every call this wrapper received. The difference is the work saved,
    and both are reported in a composite run's provenance so "one shared fetch"
    is a measured claim rather than a promise.
    """

    def __init__(self, provider: MarketDataProvider):
        self._provider = provider
        self._frames: Dict[_Key, Dict[str, pd.DataFrame]] = {}
        self.fetches = 0
        self.requests = 0

    @property
    def provider(self) -> MarketDataProvider:
        """The wrapped provider - so a caller can still detect what is underneath."""
        return self._provider

    def get_bars(
        self,
        symbols: List[str],
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Serve ``{symbol: OHLCV}`` from memory when this exact request was already
        made in this session, else fetch it and remember the response."""
        self.requests += 1
        key = (tuple(symbols), str(timeframe), _stamp(start), _stamp(end))
        cached = self._frames.get(key)
        if cached is None:
            self.fetches += 1
            cached = self._provider.get_bars(symbols, timeframe, start, end)
            self._frames[key] = cached
        # Copy on the way out: callers (the feature panel, the engine) mutate the
        # frames they receive, and a cache that hands out its own objects would let
        # one step's edits leak into the next step's inputs.
        return {symbol: frame.copy() for symbol, frame in cached.items()}

    async def stream_bars(self, symbols: List[str], handler: BarHandler) -> None:
        """Streaming is passed straight through - there is nothing to memoize."""
        await self._provider.stream_bars(symbols, handler)

    def supports_streaming(self) -> bool:
        return self._provider.supports_streaming()

    def stats(self) -> Dict[str, int]:
        """Request/fetch counts for a run's provenance block."""
        return {"requests": self.requests, "fetches": self.fetches, "distinct": len(self._frames)}


def session_client(data_client) -> Tuple[Any, Optional[SessionBarCache]]:
    """Wrap ``data_client``'s provider in a :class:`SessionBarCache`.

    Returns ``(client, cache)`` - the client to hand to every step of a composite
    run, and the cache itself for its stats. If the client exposes no provider
    (an unusual double or a bare stub), it is returned unchanged with ``None``
    stats rather than failing: sharing a fetch is an optimization, never a
    correctness requirement.
    """
    from tradeflow.marketdata.client import MarketDataClient

    provider = getattr(data_client, "provider", None)
    if provider is None:
        return data_client, None
    cache = SessionBarCache(provider)
    return MarketDataClient(cache), cache
