"""Out-of-core bar storage: a Parquet/Arrow store behind the same ``scan()`` seam.

The load-bearing assumption everywhere else is that an in-memory pandas frame is the
currency between layers — fine for a few hundred symbols of daily bars, fatal for the
volumes a real system sees. This module is the first storage tier that scales past
RAM: bars live in **partitioned Parquet**, and a scan reads them through Arrow with
**predicate/projection pushdown** — the ``as_of`` filter is applied at the storage
layer, so a point-in-time scan physically never reads rows after ``as_of``.

Crucially it implements the *same* :class:`~src.data.scan.BarSource` contract as the
in-memory :class:`~src.data.scan.ClientBarSource`, so it is a **drop-in**: the alpha,
risk, and portfolio layers above never learn where the data physically lives. pandas
appears only at the edge (the returned per-symbol frames), never as the cross-layer
currency. Requires the ``store`` extra (``pyarrow``).
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import pandas as pd

from src.marketdata.client import TimeframeLike
from src.marketdata.timeframe import Timeframe

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl

#: The enforced bar schema columns (the Arrow boundary contract).
BAR_COLUMNS = ["ts", "symbol", "open", "high", "low", "close", "volume"]


class ParquetBarStore:
    """A point-in-time :class:`BarSource` backed by symbol/date-partitioned Parquet.

    Layout: ``<root>/<timeframe>/symbol=<SYM>/year=<YYYY>/part.parquet``. One timeframe
    per subtree, then a Hive ``symbol=…/year=…`` partitioning that serves both access
    patterns (a symbol's full history, and a date window across symbols). A scan prunes
    to the **partition files that overlap the date window** (file-level pruning) and
    then pushes the exact ``as_of`` window into the Parquet reader (row-group pruning),
    so a one-year scan of a decade-deep store physically opens ~one year of files.

    Year is the partition granularity: coarse enough that daily/most-intraday data
    keeps a sane file count, fine enough that a typical lookback window touches one or
    two partitions. Writing a symbol replaces its whole subtree (all years).
    """

    def __init__(self, root: str):
        self.root = Path(root)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def write(self, bars: Dict[str, pd.DataFrame], timeframe: TimeframeLike = "1Day") -> None:
        """Persist ``{symbol: OHLCV}`` to the store (replacing each symbol's subtree).

        Each symbol's bars are split into ``year=<YYYY>`` partition files (by the bar's
        UTC year, the same basis the scan window prunes on). The symbol's existing
        subtree is removed first so a rewrite never leaves a stale year partition behind.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        for symbol, frame in bars.items():
            if frame is None or frame.empty:
                continue
            records = self._to_records(symbol, frame)
            sdir = self._symbol_dir(tf, symbol)
            if sdir.exists():
                shutil.rmtree(sdir)  # full replace: no stale year partitions survive
            for year, group in records.groupby(records["ts"].dt.year):
                part = sdir / f"year={int(year)}"
                part.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pandas(group, preserve_index=False)
                pq.write_table(table, part / "part.parquet")

    # ------------------------------------------------------------------ #
    # Scan (the BarSource contract)
    # ------------------------------------------------------------------ #
    def scan(
        self, universe: List[str], timeframe: TimeframeLike, as_of: datetime, lookback_days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """Return ``{symbol: OHLCV}`` over ``(as_of - lookback, as_of]`` — pushed down.

        Only the ``year=`` partition files overlapping the window are opened (file-level
        pruning), and the timestamp window is then a Parquet filter, so rows outside it
        (in particular any bar *after* ``as_of``) are never read into memory.
        """
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        if not (self.root / tf).exists():
            return {}

        start, end = self._window(as_of, lookback_days)
        out: Dict[str, pd.DataFrame] = {}
        for symbol in dict.fromkeys(universe):
            files = self._window_files(tf, symbol, start, end)
            if not files:
                continue
            dataset = ds.dataset(files, format="parquet")
            # Predicate pushdown: only ts in (start, end] is read off disk.
            table = dataset.to_table(filter=(pc.field("ts") > start) & (pc.field("ts") <= end))
            if table.num_rows:
                out[symbol] = self._from_records(table.to_pandas())
        return out

    # ------------------------------------------------------------------ #
    # Partition geometry (symbol=…/year=… Hive layout)
    # ------------------------------------------------------------------ #
    def _tf_root(self, timeframe: TimeframeLike) -> Path:
        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        return self.root / tf

    def _symbol_dir(self, tf: str, symbol: str) -> Path:
        return self.root / tf / f"symbol={symbol}"

    @staticmethod
    def _window(as_of: datetime, lookback_days: int) -> tuple:
        """Normalise ``as_of`` to UTC and return the ``(start, end]`` window."""
        end = pd.Timestamp(as_of)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        return end - timedelta(days=lookback_days), end

    def _year_files(self, sdir: Path, lo_year: Optional[int], hi_year: Optional[int]) -> List[str]:
        """``part.parquet`` paths in ``sdir``'s ``year=`` partitions within ``[lo, hi]``."""
        if not sdir.exists():
            return []
        files = []
        for ydir in sorted(sdir.glob("year=*")):
            try:
                year = int(ydir.name.split("=", 1)[1])
            except (IndexError, ValueError):
                continue
            if (lo_year is None or year >= lo_year) and (hi_year is None or year <= hi_year):
                part = ydir / "part.parquet"
                if part.exists():
                    files.append(str(part))
        return files

    def _window_files(self, tf: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
        """The partition files for ``symbol`` overlapping ``[start.year, end.year]``."""
        return self._year_files(self._symbol_dir(tf, symbol), start.year, end.year)

    def partition_paths(
        self,
        universe: Sequence[str],
        timeframe: TimeframeLike = "1Day",
        as_of: Optional[datetime] = None,
        lookback_days: int = 365,
    ) -> List[str]:
        """The Parquet partition files backing ``universe`` (existing partitions only).

        The file list a DuckDB/Polars set-based query (:func:`src.data.compute.sql_query`)
        or :meth:`scan_lazy` reads over. With ``as_of`` given the list is pruned to the
        ``year=`` partitions overlapping ``(as_of - lookback, as_of]`` (file-level
        date pruning); without it, every year partition for the symbols is returned.
        """
        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        if as_of is None:
            lo_year = hi_year = None
        else:
            start, end = self._window(as_of, lookback_days)
            lo_year, hi_year = start.year, end.year
        paths: List[str] = []
        for symbol in dict.fromkeys(universe):
            paths.extend(self._year_files(self._symbol_dir(tf, symbol), lo_year, hi_year))
        return paths

    def scan_lazy(
        self,
        universe: List[str],
        timeframe: TimeframeLike,
        as_of: datetime,
        lookback_days: int = 365,
        columns: Optional[Sequence[str]] = None,
    ) -> "pl.LazyFrame":
        """Return a **lazy** long-format panel for ``universe`` over ``(as_of - lookback, as_of]``.

        The compute-tier counterpart to :meth:`scan`: instead of materialising
        per-symbol pandas frames, it returns a Polars :class:`~polars.LazyFrame` over
        the Parquet partitions with the ``as_of`` window pushed into the reader as a
        predicate (rows after ``as_of`` are physically never read) and an optional
        column projection pushed down too. ``as_of`` may be naive (treated as UTC) or
        tz-aware; the window matches the eager :meth:`scan` exactly (strict ``>`` on the
        lower bound, ``<=`` on ``as_of``). Nothing executes until a terminal collect
        (:func:`src.data.edges.collect_streaming`). The leaf scan + pushdown stream; a
        downstream ``over(...)``/sort buffers (see :mod:`src.data.compute` on what is
        genuinely bounded-memory). Requires the ``store`` extra (``polars``).

        An empty universe yields an empty but **schema-carrying** LazyFrame, so a
        window helper composed onto it resolves its source column and returns an empty
        frame rather than raising (the lazy analogue of :meth:`scan` returning ``{}``).
        """
        import polars as pl

        schema = {
            "ts": pl.Datetime("us", "UTC"),
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }

        # File-level date pruning: only year partitions overlapping the window.
        paths = self.partition_paths(universe, timeframe, as_of=as_of, lookback_days=lookback_days)
        if not paths:
            empty = pl.DataFrame(schema=schema).lazy()
            if columns is not None:
                empty = empty.select(list(dict.fromkeys(["ts", "symbol", *columns])))
            return empty

        start, end = self._window(as_of, lookback_days)
        # hive_partitioning=false: `symbol` is a real column in each file; don't also
        # synthesise it from the `symbol=…` directory (that collides).
        lf = pl.scan_parquet(paths, hive_partitioning=False)
        # Predicate pushdown: the window (in particular the as_of upper bound) is
        # pushed into the Parquet SCAN, so post-as_of rows are never read.
        lf = lf.filter((pl.col("ts") > pl.lit(start)) & (pl.col("ts") <= pl.lit(end)))
        if columns is not None:
            keep = list(dict.fromkeys(["ts", "symbol", *columns]))
            lf = lf.select(keep)
        return lf

    # ------------------------------------------------------------------ #
    # The pandas edge (the only sanctioned crossings)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_records(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        """OHLCV frame (ts index) → a flat, schema-typed records frame for Parquet."""
        records = frame.reset_index()
        records = records.rename(columns={records.columns[0]: "ts"})
        ts = pd.to_datetime(records["ts"], utc=True)
        records["ts"] = ts
        records["symbol"] = symbol
        for col in ("open", "high", "low", "close", "volume"):
            records[col] = records[col].astype("float64")
        return records[BAR_COLUMNS]

    @staticmethod
    def _from_records(records: pd.DataFrame) -> pd.DataFrame:
        """Records frame → an OHLCV frame indexed by a sorted UTC timestamp."""
        frame = records.sort_values("ts").set_index("ts")
        frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
        return frame[["open", "high", "low", "close", "volume"]]
