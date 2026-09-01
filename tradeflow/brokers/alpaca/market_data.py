"""Alpaca implementation of the :class:`MarketDataProvider` interface.

Along with :mod:`tradeflow.brokers.alpaca.factory`, the only place where ``alpaca-py``
market-data types are imported. Converts
the project's :class:`Timeframe` into Alpaca's ``TimeFrame`` and normalizes the
returned bars into per-symbol, NY-localized OHLCV frames.
"""

import inspect
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from tradeflow.marketdata.base import BarEvent, BarHandler, MarketDataProvider
from tradeflow.marketdata.timeframe import DAY, HOUR, MINUTE, WEEK, Timeframe
from tradeflow.utils.streaming import run_with_reconnect
from tradeflow.utils.timeutils import localize_index_to_new_york

logger = logging.getLogger(__name__)

# Map our canonical timeframe units onto Alpaca's TimeFrameUnit.
_UNIT_TO_ALPACA = {
    MINUTE: TimeFrameUnit.Minute,
    HOUR: TimeFrameUnit.Hour,
    DAY: TimeFrameUnit.Day,
    WEEK: TimeFrameUnit.Week,
}


class AlpacaMarketData(MarketDataProvider):
    """Historical + live equity bars from Alpaca."""

    def __init__(
        self,
        historical_client: StockHistoricalDataClient,
        api_key: str,
        api_secret: str,
        base_reconnect_delay: float = 5.0,
        max_reconnect_delay: float = 60.0,
        feed: Optional[str] = None,
    ):
        self._historical = historical_client
        self._api_key = api_key
        self._api_secret = api_secret
        # One feed for both halves when pinned. Left unset the SDK's own defaults
        # apply, and those disagree with each other: historical resolves to the full
        # consolidated tape while the stream defaults to a single venue. An account
        # entitled to one and not the other then warms up on nothing and streams
        # fine, which reads as an empty market rather than as a wrong feed.
        self._feed = feed
        self._base_reconnect_delay = base_reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay

    def get_bars(
        self, symbols: List[str], timeframe: Timeframe, start: datetime, end: datetime
    ) -> Dict[str, pd.DataFrame]:
        if not symbols:
            return {}

        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=self._to_alpaca_timeframe(timeframe),
            start=start,
            end=end,
            adjustment="split",
            **({"feed": DataFeed(self._feed)} if self._feed else {}),
        )

        try:
            combined = self._historical.get_stock_bars(request).df
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch bars for %s: %s", symbols, exc)
            return {}

        if combined is None or combined.empty:
            return {}

        return self._split_by_symbol(combined)

    async def stream_bars(self, symbols: List[str], handler: BarHandler) -> None:
        """Stream live bars for ``symbols``, reconnecting on failure.

        Live WebSockets drop. Each attempt creates a *fresh* stream (the SDK can
        be left in a bad state after an error), subscribes the monitored symbols,
        and runs until it errors; on error we back off (capped exponential) and
        reconnect. Cancellation (e.g. Ctrl-C) breaks out cleanly.
        """
        on_bar = self._make_bar_callback(handler)

        async def connect() -> None:
            stream = self._new_stream()
            try:
                for symbol in symbols:
                    stream.subscribe_bars(on_bar, symbol)
                logger.info("Subscribed to live bars for %d symbols: %s", len(symbols), symbols)
                # _run_forever is the awaitable entry point (run() wraps it in
                # asyncio.run, which we can't use inside an existing loop).
                await stream._run_forever()
            finally:
                await self._safe_stop(stream)

        await run_with_reconnect(
            "market-data",
            connect,
            base_delay=self._base_reconnect_delay,
            max_delay=self._max_reconnect_delay,
        )

    def _make_bar_callback(self, handler: BarHandler):
        """Wrap a project BarHandler as an async Alpaca bar callback."""

        async def _on_alpaca_bar(bar) -> None:
            event = BarEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            result = handler(event)
            if inspect.isawaitable(result):
                await result

        return _on_alpaca_bar

    def _new_stream(self) -> StockDataStream:
        """Create a fresh stream (isolated for testability)."""
        if self._feed:
            return StockDataStream(self._api_key, self._api_secret, feed=DataFeed(self._feed))
        return StockDataStream(self._api_key, self._api_secret)

    @staticmethod
    async def _safe_stop(stream) -> None:
        """Best-effort stream shutdown; never raises."""
        try:
            result = stream.stop()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - cleanup must not mask the real error
            pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_alpaca_timeframe(timeframe: Timeframe) -> TimeFrame:
        return TimeFrame(timeframe.amount, _UNIT_TO_ALPACA[timeframe.unit])

    @staticmethod
    def _split_by_symbol(combined: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split Alpaca's (symbol, timestamp) MultiIndex frame into per-symbol frames."""
        result: Dict[str, pd.DataFrame] = {}
        for symbol in combined.index.get_level_values(0).unique():
            frame = combined.xs(symbol, level=0).copy()
            result[symbol] = localize_index_to_new_york(frame)
        return result
