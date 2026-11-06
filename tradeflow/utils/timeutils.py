"""Timezone helpers.

US equity sessions are reasoned about in America/New_York, so all market-data
timestamps are normalized to that zone in one place.
"""

from datetime import datetime

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


def parse_market_date(value: str) -> datetime:
    """Parse a date or ISO datetime into a New York instant.

    **One date contract.** ``2026-08-22`` means that market date in the exchange's own
    zone, everywhere in this program and over every surface. That is a domain decision:
    someone typing a date at a trading tool means the session, not UTC midnight.

    Leaving the value naive and letting each consumer localize it is what produced the
    bug this replaced. ``cache warm --end 2026-08-22`` recorded coverage through
    00:00Z (the store reads a naive datetime as UTC) while ``scan --as-of 2026-08-22``
    asked for 04:00Z (the scanner reads one as New York), so a cache holding exactly
    the right daily bar reported a four-hour hole and refused an offline run whose
    data was complete. Two readings of one string is a semantic fork; it surfaced in
    the cache first and would have surfaced next in reports, or in live/backtest
    parity.

    Returning an *aware* value is what stops the contract re-forking: a naive datetime
    carries no zone to disagree about because it carries no zone at all.

    The CLI and the MCP server both call this. They used to parse separately — the MCP
    tool left the value naive — and the window's text is part of a trial's dedup
    identity, so one request over the two surfaces journaled two trials and never found
    the other's memoized answer.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    if parsed.tzinfo is None:
        return NEW_YORK.localize(parsed)
    return parsed.astimezone(NEW_YORK)
