"""High-level market-data access - the one part of the pipeline that, mercifully,
just returns bars and doesn't have opinions about whether you'll make money.

A thin orchestration layer over a :class:`MarketDataProvider`. It owns the
friendly timeframe-string API and the live-stream entry point, and is the object
the engine talks to - so the engine never depends on a concrete data vendor.
"""

import logging
from datetime import datetime
from typing import Dict, List, Union

import pandas as pd

from tradeflow.marketdata.base import BarHandler, MarketDataProvider
from tradeflow.marketdata.timeframe import Timeframe

logger = logging.getLogger(__name__)

TimeframeLike = Union[str, Timeframe]


class MarketDataClient:
    """Fetches historical bars and dispatches live bars via an injected provider."""

    def __init__(self, provider: MarketDataProvider):
        self._provider = provider

    @property
    def provider(self) -> MarketDataProvider:
        """The underlying provider - lets a caller detect e.g. a cache wrapper
        without reaching into a private attribute."""
        return self._provider

    def get_bars(
        self,
        symbols: List[str],
        timeframe: TimeframeLike,
        start: datetime,
        end: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Return ``{symbol: OHLCV DataFrame}`` for the requested window."""
        tf = self._coerce_timeframe(timeframe)
        data = self._provider.get_bars(symbols, tf, start, end)
        logger.info("Fetched bars for %d/%d symbols (%s)", len(data), len(symbols), tf)
        return data

    async def stream(self, symbols: List[str], handler: BarHandler) -> None:
        """Stream live bars for ``symbols``, invoking ``handler`` per bar."""
        if not self._provider.supports_streaming():
            raise RuntimeError("The configured market-data provider does not support streaming")
        await self._provider.stream_bars(symbols, handler)

    def supports_streaming(self) -> bool:
        return self._provider.supports_streaming()

    @staticmethod
    def _coerce_timeframe(timeframe: TimeframeLike) -> Timeframe:
        return timeframe if isinstance(timeframe, Timeframe) else Timeframe.parse(timeframe)
