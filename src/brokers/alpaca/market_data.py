"""Alpaca implementation of the :class:`MarketDataProvider` interface.

The single place where ``alpaca-py`` market-data types are imported. Converts
the project's :class:`Timeframe` into Alpaca's ``TimeFrame`` and normalises the
returned bars into per-symbol, NY-localised OHLCV frames.
"""

import inspect
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.marketdata.base import BarEvent, BarHandler, MarketDataProvider
from src.marketdata.timeframe import DAY, HOUR, MINUTE, WEEK, Timeframe
from src.utils.timeutils import localize_index_to_new_york

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

    def __init__(self, historical_client: StockHistoricalDataClient, api_key: str, api_secret: str):
        self._historical = historical_client
        self._api_key = api_key
        self._api_secret = api_secret

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
        stream = StockDataStream(self._api_key, self._api_secret)

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

        for symbol in symbols:
            stream.subscribe_bars(_on_alpaca_bar, symbol)

        logger.info("Subscribed to live bars for %d symbols", len(symbols))
        # _run_forever is the awaitable entry point; run() wraps it in asyncio.run
        # which we cannot use from inside an existing event loop.
        await stream._run_forever()

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
