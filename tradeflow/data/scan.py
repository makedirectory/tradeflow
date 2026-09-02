"""The data-access seam - one place to scan bars point-in-time.

Every cross-sectional research flow (alphas, and soon risk / portfolio /
information analysis) needs the same thing: the bars for a universe *as of* a
rebalance timestamp, with nothing after it. This module is that single seam, and
the single home of the leakage guard (``<= as_of``) - so no call site re-implements
slicing and accidentally lets a future bar through.

Today it is pandas-backed over :class:`MarketDataClient`. The ``scan(universe,
timeframe, as_of, lookback)`` signature is deliberately the same contract an
out-of-core Arrow/Polars/DuckDB source would implement, so growing the storage
tier is a new adapter behind this interface, not a rewrite of the layers above.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Protocol, runtime_checkable

import pandas as pd

from tradeflow.marketdata.client import MarketDataClient, TimeframeLike


@runtime_checkable
class BarSource(Protocol):
    """A point-in-time bar source: bars for a universe, never past ``as_of``."""

    def scan(
        self, universe: List[str], timeframe: TimeframeLike, as_of: datetime, lookback_days: int
    ) -> Dict[str, pd.DataFrame]:
        """Return ``{symbol: OHLCV}`` over ``(as_of - lookback, as_of]`` per symbol."""
        ...


class ClientBarSource:
    """A :class:`BarSource` backed by the existing :class:`MarketDataClient`."""

    def __init__(self, data_client: MarketDataClient):
        self._client = data_client

    def scan(
        self,
        universe: List[str],
        timeframe: TimeframeLike,
        as_of: datetime,
        lookback_days: int = 365,
    ) -> Dict[str, pd.DataFrame]:
        # Fetch with end == as_of (defends leakage for real feeds), then slice
        # defensively (synthetic/fake feeds ignore the window, so the slice is the
        # real guard). De-dup the universe so a benchmark listed twice is fetched once.
        start = as_of - timedelta(days=lookback_days)
        raw = self._client.get_bars(list(dict.fromkeys(universe)), timeframe, start, as_of)
        out: Dict[str, pd.DataFrame] = {}
        for symbol, frame in raw.items():
            sliced = slice_to_as_of(frame, as_of)
            if sliced is not None and not sliced.empty:
                out[symbol] = sliced
        return out


def slice_to_as_of(frame: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Return only bars at or before ``as_of`` - the leakage guard.

    Aligns the cutoff to the frame in **both** directions. Handling only the aware
    index / naive cutoff pair was enough while callers passed naive cutoffs, and
    stopped being enough the moment the scanner began localizing its window
    unconditionally: an aware cutoff against a naive index raised ``Invalid
    comparison between dtype=datetime64[ns] and Timestamp`` on any provider whose
    bar timestamps carry no zone - a frame this guard had always accepted.

    Being the single seam means owning the whole coercion, not half of it. A naive
    bar index is taken as UTC, the assumption
    :func:`~tradeflow.utils.timeutils.localize_index_to_new_york` already makes about
    a provider's timestamps.
    """
    if frame is None or frame.empty:
        return frame
    ts = pd.Timestamp(as_of)
    idx = frame.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            ts = ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)
        elif ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
    return frame.loc[idx <= ts]
