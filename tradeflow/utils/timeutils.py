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
