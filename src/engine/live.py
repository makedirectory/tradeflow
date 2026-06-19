"""Live trading engine.

Mirrors the backtest pipeline for real time:

    warm-up history (marketdata) -> stream bars -> signals (strategy)
        -> orders (execution)

The engine wires the layers and owns the bar->signal->order loop; it contains no
indicator math, no order-placement detail, and no vendor specifics.
"""

import logging
from datetime import datetime, timedelta
from typing import List

from src.execution.live_trader import LiveTrader
from src.marketdata.base import BarEvent
from src.marketdata.client import MarketDataClient
from src.marketdata.timeframe import DAY, HOUR, MINUTE, WEEK, Timeframe
from src.strategies import signals
from src.strategies.base import Strategy
from src.utils.timeutils import NEW_YORK

logger = logging.getLogger(__name__)

# Extra history fetched beyond the bare lookback, to cover non-trading gaps.
_WARMUP_BUFFER = 2

_UNIT_TO_TIMEDELTA = {
    MINUTE: lambda n: timedelta(minutes=n),
    HOUR: lambda n: timedelta(hours=n),
    DAY: lambda n: timedelta(days=n),
    WEEK: lambda n: timedelta(weeks=n),
}


class LiveEngine:
    """Streams live bars into a strategy and routes signals to execution."""

    def __init__(self, strategy: Strategy, data_client: MarketDataClient, live_trader: LiveTrader):
        self.strategy = strategy
        self.data_client = data_client
        self.live_trader = live_trader

    async def start(self, symbols: List[str]) -> None:
        """Warm up indicators with history, then stream live bars until cancelled."""
        self.strategy.initialize()
        self._warm_up(symbols)

        logger.info("Starting live stream for %d symbols", len(symbols))
        await self.data_client.stream(symbols, self._on_bar)

    def _warm_up(self, symbols: List[str]) -> None:
        """Seed each symbol's rolling buffer so indicators are valid on bar one."""
        timeframe = Timeframe.parse(self.strategy.config["timeframe"])
        periods = self.strategy.config.get("required_lookback_periods", 50)
        start = self._lookback_start(timeframe, periods)
        end = datetime.now(NEW_YORK)

        history = self.data_client.get_bars(symbols, timeframe, start, end)
        for symbol, bars in history.items():
            if not bars.empty:
                self.strategy.warm_up(symbol, self.strategy.process_data(bars))
                logger.info("Warmed up %s with %d bars", symbol, len(bars))

    def _on_bar(self, event: BarEvent) -> None:
        """Per-bar callback: update the strategy and act on any emitted signal."""
        signal = self.strategy.process_real_time_data(
            event.symbol, event.close, event.volume, event.timestamp
        )
        if signal and signal != signals.HOLD:
            logger.info("Signal %s for %s @ $%.4f", signal, event.symbol, event.close)
            self.live_trader.handle_signal(event.symbol, signal, event.close)

    @staticmethod
    def _lookback_start(timeframe: Timeframe, periods: int) -> datetime:
        units = periods * timeframe.amount * _WARMUP_BUFFER
        delta = _UNIT_TO_TIMEDELTA[timeframe.unit](units)
        return datetime.now(NEW_YORK) - delta
