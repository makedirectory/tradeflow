"""Market-data provider abstraction.

Splitting *data* from *trading* keeps the two concerns independent: a deployment
can pull bars from one vendor and route orders to another. Implementations
return plain pandas OHLCV frames so strategies and the engine never see vendor
types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Union

import pandas as pd

from tradeflow.marketdata.timeframe import Timeframe

#: Columns every OHLCV frame returned by a provider must contain.
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class BarEvent:
    """A single bar pushed by a live stream."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


#: A bar handler may be sync or async; the provider awaits it if needed.
BarHandler = Callable[[BarEvent], Union[None, Awaitable[None]]]


class MarketDataProvider(ABC):
    """Interface every market-data source must implement."""

    @abstractmethod
    def get_bars(
        self,
        symbols: List[str],
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical bars.

        Returns a mapping of ``symbol -> OHLCV DataFrame`` indexed by a tz-aware
        timestamp. Symbols with no data are omitted.
        """

    @abstractmethod
    async def stream_bars(self, symbols: List[str], handler: BarHandler) -> None:
        """Subscribe to live bars and invoke ``handler`` for each, until canceled."""

    def supports_streaming(self) -> bool:
        """Whether this provider can stream live bars. Override if not."""
        return True
