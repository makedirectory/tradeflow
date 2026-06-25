"""Deterministic synthetic market data for the offline ``demo``.

A :class:`MarketDataProvider` that fabricates bars from a seeded random walk, so
the whole pipeline (backtest, optimize, walk-forward) runs with **no Alpaca keys
and no network**. Unlike the test fakes, it honours the requested ``[start, end]``
window and timeframe, so date-based machinery like walk-forward fold splitting
works exactly as it does on real data.

By construction the series has **no real edge** - it's a drift-free random walk.
That's deliberate: the demo's punchline is watching honest, out-of-sample
evaluation refuse to promote noise. If a strategy looks great here in-sample and
then fails the walk-forward gates, that's the system working, not a bug.
"""

import math
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.marketdata.base import BarHandler, MarketDataProvider
from src.marketdata.timeframe import Timeframe
from src.utils.timeutils import NEW_YORK


def _symbol_seed(symbol: str, base_seed: int) -> int:
    """A stable per-symbol seed (independent of Python's hash randomisation)."""
    return base_seed + sum(ord(ch) for ch in symbol)


class SyntheticMarketData(MarketDataProvider):
    """Serves seeded random-walk OHLCV across the requested window and timeframe."""

    def __init__(self, seed: int = 42, annual_drift: float = 0.0, annual_vol: float = 0.35):
        self._seed = seed
        self._annual_drift = annual_drift
        self._annual_vol = annual_vol

    def get_bars(
        self, symbols: List[str], timeframe: Timeframe, start: datetime, end: datetime
    ) -> Dict[str, pd.DataFrame]:
        index = pd.date_range(start=start, end=end, freq=timeframe.to_pandas_offset(), tz=NEW_YORK)
        if len(index) == 0:
            return {}

        periods_per_year = timeframe.periods_per_year()
        sigma = self._annual_vol / math.sqrt(periods_per_year)
        mu = self._annual_drift / periods_per_year

        return {symbol: self._make_frame(symbol, index, mu, sigma) for symbol in symbols}

    def _make_frame(self, symbol: str, index: pd.DatetimeIndex, mu: float, sigma: float) -> pd.DataFrame:
        rng = np.random.default_rng(_symbol_seed(symbol, self._seed))
        n = len(index)

        close = 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
        open_ = close * (1 + rng.normal(0, sigma / 2, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma / 2, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma / 2, n)))

        volume = rng.integers(100_000, 500_000, n).astype(float)
        volume[rng.choice(n, size=max(n // 20, 1), replace=False)] *= 5  # occasional spikes

        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )

    async def stream_bars(self, symbols: List[str], handler: BarHandler) -> None:  # pragma: no cover
        raise NotImplementedError("SyntheticMarketData is for offline backtests, not streaming")

    def supports_streaming(self) -> bool:
        return False
