"""The bar cache: a persistent OHLCV cache behind ``MarketDataProvider``.

Two pieces, mirroring the derived/rebuildable-index philosophy ``src/store/trials.py``
already established for the research journal:

- :class:`BarCoverage` — a small SQLite index of *which* ``(symbol, timeframe)``
  windows have actually been fetched from upstream. Derived, safe to delete: it can
  be rebuilt (approximately) from the Parquet store it indexes.
- :class:`CachedMarketData` — a :class:`~src.marketdata.base.MarketDataProvider` that
  wraps an upstream provider with :class:`~src.data.store.ParquetBarStore`
  (spec 011's columnar substrate) and a :class:`BarCoverage`, so ``get_bars`` becomes
  *check cache, fetch only the missing sub-ranges, write back, return*. A run can pin
  to cache-only (``offline=True``) for a byte-reproducible, network-free result.

Bars are never assumed present just because a date range was requested before -
weekends/holidays genuinely have no bars, so "covered" is tracked independently of
"has rows": every fetch, even an empty one, is recorded, or a market-closed gap would
be re-queried forever.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.marketdata.base import BarHandler, MarketDataProvider
from src.marketdata.client import TimeframeLike
from src.marketdata.timeframe import Timeframe

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.data.store import ParquetBarStore

logger = logging.getLogger(__name__)

#: Default cache location (gitignored, alongside logs/).
DEFAULT_CACHE_ROOT = Path("cache") / "bars"
DEFAULT_COVERAGE_DB = Path("cache") / "bars_coverage.db"

#: "Read everything" sentinels for the read-merge-write path: ParquetBarStore.scan()
#: takes an (as_of, lookback_days) window, not an arbitrary [start, end] range, so a
#: full-history read uses a lookback and an as_of far past anything real data can
#: reach - the same idiom tests/test_store.py already uses with lookback_days=10_000,
#: just wide enough to never truncate a merge.
_FAR_FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)
_FULL_HISTORY_LOOKBACK_DAYS = 100_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetches (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol      TEXT NOT NULL,
  timeframe   TEXT NOT NULL,
  start       TEXT NOT NULL,
  end         TEXT NOT NULL,
  adjustment  TEXT NOT NULL,
  fetched_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fetches_symbol ON fetches(symbol, timeframe);
"""


class CacheMiss(RuntimeError):
    """Raised in ``--offline`` mode when a requested window isn't fully cached."""

    def __init__(self, symbol: str, timeframe: str, gaps: List[Tuple[datetime, datetime]]):
        self.symbol = symbol
        self.timeframe = timeframe
        self.gaps = gaps
        ranges = ", ".join(f"{s.isoformat()}..{e.isoformat()}" for s, e in gaps)
        super().__init__(
            f"--offline: {symbol} ({timeframe}) is missing cached bars for {ranges}. "
            f"Run `python main.py cache warm --symbols {symbol} --timeframe {timeframe}` first."
        )


def _to_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _iso(ts: pd.Timestamp) -> str:
    return ts.isoformat()


def _normalize_index_utc(frame: pd.DataFrame) -> pd.DataFrame:
    """Localize/convert a bar frame's index to UTC - upstream providers may
    return NY-localized bars (see ``AlpacaMarketData``); normalizing before any
    comparison/concat avoids relying on pandas to reconcile mixed timezones."""
    idx = frame.index
    if idx.tz is None:
        frame = frame.tz_localize("UTC")
    elif str(idx.tz) != "UTC":
        frame = frame.tz_convert("UTC")
    return frame


class BarCoverage:
    """A derived SQLite index of which ``(symbol, timeframe)`` windows have been
    fetched from upstream. Safe to delete: :meth:`rebuild` reconstructs it
    (approximately) from a :class:`~src.data.store.ParquetBarStore`."""

    def __init__(self, db_path: Optional[Any] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_COVERAGE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BarCoverage":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def record_fetch(
        self,
        symbol: str,
        timeframe: str,
        start: Any,
        end: Any,
        adjustment: str,
        fetched_at: Optional[Any] = None,
    ) -> None:
        """Record that ``[start, end]`` was queried from upstream for ``symbol`` -
        even when the response was empty (a weekend must never look "not yet
        checked" and get re-queried forever)."""
        stamp = _to_utc(fetched_at) if fetched_at is not None else pd.Timestamp.now(tz="UTC")
        self._conn.execute(
            "INSERT INTO fetches (symbol, timeframe, start, end, adjustment, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (symbol, timeframe, _iso(_to_utc(start)), _iso(_to_utc(end)), adjustment, _iso(stamp)),
        )
        self._conn.commit()

    def forget(self, symbol: str, timeframe: str) -> None:
        """Delete all fetch rows for ``(symbol, timeframe)`` - used by ``refresh()``
        so a corporate-action re-fetch is never treated as "already covered"."""
        self._conn.execute("DELETE FROM fetches WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def covered_intervals(self, symbol: str, timeframe: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Merge-on-read: the stored fetch rows for ``(symbol, timeframe)``, sorted
        and coalesced into non-overlapping, non-touching intervals."""
        rows = self._conn.execute(
            "SELECT start, end FROM fetches WHERE symbol = ? AND timeframe = ? ORDER BY start",
            (symbol, timeframe),
        ).fetchall()
        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for row in rows:
            s, e = _to_utc(row["start"]), _to_utc(row["end"])
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def gaps(
        self, symbol: str, timeframe: str, start: Any, end: Any
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """The sub-ranges of ``[start, end]`` not yet covered by a recorded fetch.

        Cached ``[b, c]``, requested ``[a, d]`` -> ``[(a, b), (c, d)]`` - only the
        genuinely missing pieces, never a full refetch of an already-cached window.
        """
        start, end = _to_utc(start), _to_utc(end)
        if end <= start:
            return []
        covered = self.covered_intervals(symbol, timeframe)
        out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cursor = start
        for s, e in covered:
            if e <= cursor or s >= end:
                continue
            if s > cursor:
                out.append((cursor, min(s, end)))
            cursor = max(cursor, e)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
        return out

    def vintage(self, symbols: Iterable[str], timeframe: str, start: Any, end: Any) -> Optional[str]:
        """The data-vintage stamp for this exact window: ``max(fetched_at)`` over
        every recorded fetch overlapping ``[start, end]``, across ``symbols``.

        Deterministic and stable across repeated warm/offline calls to the same
        window (no new fetch happens -> the same rows cover it -> the same stamp);
        it only changes when :meth:`~CachedMarketData.refresh` forces a real
        re-fetch. ``None`` when nothing has been recorded yet for this window.
        """
        start, end = _to_utc(start), _to_utc(end)
        latest: Optional[pd.Timestamp] = None
        for symbol in dict.fromkeys(symbols):
            rows = self._conn.execute(
                "SELECT start, end, fetched_at FROM fetches WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            ).fetchall()
            for row in rows:
                s, e = _to_utc(row["start"]), _to_utc(row["end"])
                if e <= start or s >= end:
                    continue
                stamp = _to_utc(row["fetched_at"])
                if latest is None or stamp > latest:
                    latest = stamp
        return _iso(latest) if latest is not None else None

    # ------------------------------------------------------------------ #
    # Derived-store precedent: rebuild + drift check
    # ------------------------------------------------------------------ #
    def rebuild(
        self, bar_store: "ParquetBarStore", symbols: Iterable[str], timeframe: str, adjustment: str
    ) -> Dict[str, int]:
        """Reconstruct coverage from the Parquet store alone.

        An approximation, not an exact replay (the Parquet store has no record of
        *when* something was fetched, or of a queried-but-empty range): each
        symbol's coverage becomes a single interval spanning
        ``[min(ts), max(ts)]`` of whatever is on disk, stamped with the rebuild
        time. This assumes no internal fetch gap inside that span - true unless a
        corporate-action refresh carved one out - the same honesty tradeoff
        ``TrialStore.rebuild()`` already accepts for its own derived index.
        """
        self._conn.execute("DELETE FROM fetches")
        self._conn.commit()
        n_symbols = 0
        for symbol in dict.fromkeys(symbols):
            scanned = bar_store.scan(
                [symbol], timeframe, _FAR_FUTURE, lookback_days=_FULL_HISTORY_LOOKBACK_DAYS
            )
            frame = scanned.get(symbol)
            if frame is None or frame.empty:
                continue
            self.record_fetch(symbol, str(timeframe), frame.index.min(), frame.index.max(), adjustment)
            n_symbols += 1
        return {"symbols": n_symbols}

    def status(self, bar_store: Optional["ParquetBarStore"] = None) -> Dict[str, Any]:
        """Row counts, plus (when ``bar_store`` is given) a drift check: does the
        Parquet store actually still hold data for what coverage claims is cached?
        """
        rows = self._conn.execute(
            "SELECT symbol, timeframe, MIN(start) AS lo, MAX(end) AS hi, MAX(fetched_at) AS last_fetch "
            "FROM fetches GROUP BY symbol, timeframe ORDER BY symbol, timeframe"
        ).fetchall()
        entries = [dict(r) for r in rows]
        drift: List[str] = []
        if bar_store is not None:
            for entry in entries:
                scanned = bar_store.scan(
                    [entry["symbol"]],
                    entry["timeframe"],
                    _FAR_FUTURE,
                    lookback_days=_FULL_HISTORY_LOOKBACK_DAYS,
                )
                if entry["symbol"] not in scanned or scanned[entry["symbol"]].empty:
                    drift.append(f"{entry['symbol']} ({entry['timeframe']})")
        return {"db_path": str(self.db_path), "entries": entries, "drift": drift}


class CachedMarketData(MarketDataProvider):
    """A :class:`MarketDataProvider` that transparently caches OHLCV bars on the
    Parquet/DuckDB substrate (spec 011), fetching only missing date ranges.

    ``offline=True`` forbids any network call: a request touching an uncached
    range raises :class:`CacheMiss` instead of falling through to ``upstream`` -
    the byte-reproducible, cache-only mode.
    """

    def __init__(
        self,
        upstream: MarketDataProvider,
        cache_dir: Optional[Any] = None,
        *,
        coverage_db: Optional[Any] = None,
        offline: bool = False,
        adjustment: str = "split",
    ):
        from src.data.store import ParquetBarStore

        self.upstream = upstream
        self.offline = offline
        self.adjustment = adjustment
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_ROOT
        self.store = ParquetBarStore(self.cache_dir)
        self.coverage = BarCoverage(
            coverage_db or self.cache_dir.parent / f"{self.cache_dir.name}_coverage.db"
        )

    def close(self) -> None:
        self.coverage.close()

    def __enter__(self) -> "CachedMarketData":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # MarketDataProvider
    # ------------------------------------------------------------------ #
    def get_bars(
        self, symbols: List[str], timeframe: TimeframeLike, start: datetime, end: datetime
    ) -> Dict[str, pd.DataFrame]:
        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        start_ts, end_ts = _to_utc(start), _to_utc(end)
        result: Dict[str, pd.DataFrame] = {}
        for symbol in dict.fromkeys(symbols):
            self._fill_gaps(symbol, tf, start_ts, end_ts)
            scanned = self.store.scan(
                [symbol], tf, end_ts.to_pydatetime(), lookback_days=self._lookback_days(start_ts, end_ts)
            )
            frame = scanned.get(symbol)
            if frame is not None and not frame.empty:
                frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
                if not frame.empty:
                    result[symbol] = frame
        return result

    async def stream_bars(self, symbols: List[str], handler: BarHandler) -> None:
        if self.offline:
            raise RuntimeError("--offline: live streaming needs a network connection")
        await self.upstream.stream_bars(symbols, handler)

    def supports_streaming(self) -> bool:
        return False if self.offline else self.upstream.supports_streaming()

    # ------------------------------------------------------------------ #
    # Gap-fill (the core of the cache)
    # ------------------------------------------------------------------ #
    def _fill_gaps(self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
        gaps = self.coverage.gaps(symbol, timeframe, start, end)
        if not gaps:
            return
        if self.offline:
            raise CacheMiss(symbol, timeframe, gaps)

        fetched_frames: List[pd.DataFrame] = []
        for gap_start, gap_end in gaps:
            fetched = self.upstream.get_bars(
                [symbol], timeframe, gap_start.to_pydatetime(), gap_end.to_pydatetime()
            )
            frame = fetched.get(symbol)
            if frame is not None and not frame.empty:
                frame = _normalize_index_utc(frame)
                # Defend against an upstream that returns more than asked (e.g. a
                # fake/test provider that ignores the window): only what falls
                # inside this exact gap is authoritative for what we mark covered.
                sliced = frame.loc[(frame.index >= gap_start) & (frame.index <= gap_end)]
                if not sliced.empty:
                    fetched_frames.append(sliced)
            self.coverage.record_fetch(symbol, timeframe, gap_start, gap_end, self.adjustment)

        if fetched_frames:
            self._merge_write(symbol, timeframe, fetched_frames)

    def _merge_write(self, symbol: str, timeframe: str, new_frames: List[pd.DataFrame]) -> None:
        """Read the symbol's full existing history, merge in the newly-fetched
        frames, and write the union back.

        Required because ``ParquetBarStore.write()`` replaces a symbol's *entire*
        subtree on every call (by design - no stale year partitions survive a
        rewrite); writing only the new gap would silently delete every other
        year already cached for this symbol.
        """
        existing = self.store.scan(
            [symbol], timeframe, _FAR_FUTURE, lookback_days=_FULL_HISTORY_LOOKBACK_DAYS
        )
        parts = [existing[symbol]] if symbol in existing and not existing[symbol].empty else []
        parts.extend(new_frames)
        if not parts:
            return
        merged = pd.concat(parts)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.store.write({symbol: merged}, timeframe)

    @staticmethod
    def _lookback_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
        return max(int((end - start) / pd.Timedelta(days=1)) + 2, 1)

    # ------------------------------------------------------------------ #
    # CLI-facing helpers
    # ------------------------------------------------------------------ #
    def vintage_stamp(
        self, symbols: List[str], timeframe: TimeframeLike, start: datetime, end: datetime
    ) -> Optional[str]:
        """Ensure ``[start, end]`` is cached for ``symbols`` (gap-filling exactly as
        :meth:`get_bars` would), then return the data-vintage stamp for that
        window - suitable for folding into a trial's dedup key."""
        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        start_ts, end_ts = _to_utc(start), _to_utc(end)
        for symbol in dict.fromkeys(symbols):
            self._fill_gaps(symbol, tf, start_ts, end_ts)
        return self.coverage.vintage(symbols, tf, start_ts, end_ts)

    def warm(
        self, symbols: List[str], timeframe: TimeframeLike, start: datetime, end: datetime
    ) -> Dict[str, Any]:
        """Explicit prefetch for ``cache warm``: fetch and cache ``[start, end]``
        for ``symbols``, returning a summary of what was already covered vs.
        newly fetched."""
        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        start_ts, end_ts = _to_utc(start), _to_utc(end)
        summary: Dict[str, Any] = {}
        for symbol in dict.fromkeys(symbols):
            gaps_before = self.coverage.gaps(symbol, tf, start_ts, end_ts)
            self._fill_gaps(symbol, tf, start_ts, end_ts)
            summary[symbol] = {"already_cached": not gaps_before, "gaps_fetched": len(gaps_before)}
        return summary

    def refresh(
        self,
        symbols: List[str],
        timeframe: TimeframeLike,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Invalidate and re-fetch ``symbols`` - the corporate-action lever: a
        split/dividend backfill since the original fetch is never silently
        served stale. Re-fetches ``[start, end]`` if given, otherwise the
        symbol's previously-known covered extent (or is a no-op if neither is
        available)."""
        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        summary: Dict[str, Any] = {}
        for symbol in dict.fromkeys(symbols):
            window = self._resolve_refresh_window(symbol, tf, start, end)
            self.store.delete_symbol(symbol, tf)
            self.coverage.forget(symbol, tf)
            if window is None:
                summary[symbol] = {
                    "refreshed": False,
                    "reason": "no prior coverage and no --start/--end given",
                }
                continue
            self._fill_gaps(symbol, tf, window[0], window[1])
            summary[symbol] = {"refreshed": True, "start": _iso(window[0]), "end": _iso(window[1])}
        return summary

    def _resolve_refresh_window(
        self, symbol: str, timeframe: str, start: Optional[datetime], end: Optional[datetime]
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        if start is not None and end is not None:
            return _to_utc(start), _to_utc(end)
        covered = self.coverage.covered_intervals(symbol, timeframe)
        if not covered:
            return None
        return covered[0][0], covered[-1][1]

    def status(
        self, symbols: Optional[List[str]] = None, timeframe: Optional[TimeframeLike] = None
    ) -> Dict[str, Any]:
        """Coverage summary for ``cache status`` - all cached symbols/timeframes,
        or a filtered subset, plus a drift check against the Parquet store."""
        info = self.coverage.status(self.store)
        entries = info["entries"]
        if symbols is not None:
            wanted = set(symbols)
            entries = [e for e in entries if e["symbol"] in wanted]
        if timeframe is not None:
            tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
            entries = [e for e in entries if e["timeframe"] == tf]
        return {
            "db_path": info["db_path"],
            "cache_dir": str(self.cache_dir),
            "entries": entries,
            "drift": info["drift"],
        }
