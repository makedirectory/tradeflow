"""Timezone helpers.

US equity sessions are reasoned about in America/New_York, so all market-data
timestamps are normalized to that zone in one place.
"""

import pandas as pd
import pytz

#: The exchange timezone used throughout the project.
NEW_YORK = pytz.timezone("America/New_York")


def localize_index_to_new_york(data: pd.DataFrame) -> pd.DataFrame:
    """Return ``data`` with a tz-aware DatetimeIndex in New York time.

    Naive indexes are assumed to be UTC (Alpaca returns UTC), then converted.
    """
    if data.empty:
        return data

    index = data.index
    if index.tz is None:
        index = pd.to_datetime(index, utc=True)
    data = data.copy()
    data.index = index.tz_convert(NEW_YORK)
    return data


def match_index_tz(timestamp, index: pd.Index):
    """Return ``timestamp`` made comparable with ``index``.

    A live feed is free to serialize one bar naive and the next tz-aware, and the
    warm-up history is always localized. Mixing the two in a single index makes every
    later comparison raise ``TypeError`` — which, inside a callback that swallows
    exceptions, presents as a strategy that has simply stopped emitting signals.

    Naive input is assumed to be UTC, matching how bars are read everywhere else, and
    then converted to whatever the index already uses. An empty or non-datetime index
    imposes nothing, so the timestamp is returned untouched.
    """
    ts = pd.Timestamp(timestamp)
    index_tz = getattr(index, "tz", None)
    if index_tz is None:
        # A naive index: strip any zone rather than poison it with an aware value.
        return ts.tz_convert(None) if ts.tz is not None else ts
    ts = ts.tz_localize("UTC") if ts.tz is None else ts
    return ts.tz_convert(index_tz)
