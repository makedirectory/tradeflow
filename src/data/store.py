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
    """A point-in-time :class:`BarSource` backed by symbol-partitioned Parquet.

    Layout: ``<root>/<timeframe>/symbol=<SYM>/part.parquet``. One timeframe per
    subtree; the scan prunes by symbol partition and pushes the ``as_of`` window down
    to the Parquet reader.
    """

    def __init__(self, root: str):
        self.root = Path(root)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def write(self, bars: Dict[str, pd.DataFrame], timeframe: TimeframeLike = "1Day") -> None:
        """Persist ``{symbol: OHLCV}`` to the store (overwriting each symbol's partition)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        for symbol, frame in bars.items():
            if frame is None or frame.empty:
                continue
            table = pa.Table.from_pandas(self._to_records(symbol, frame), preserve_index=False)
            part = self.root / tf / f"symbol={symbol}"
            part.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, part / "part.parquet")

    # ------------------------------------------------------------------ #
    # Scan (the BarSource contract)
    # ------------------------------------------------------------------ #
    def scan(
        self, universe: List[str], timeframe: TimeframeLike, as_of: datetime, lookback_days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """Return ``{symbol: OHLCV}`` over ``(as_of - lookback, as_of]`` — pushed down.

        The timestamp window is a Parquet filter, so rows outside it (in particular any
        bar *after* ``as_of``) are never read into memory.
        """
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        root = self.root / tf
        if not root.exists():
            return {}

        end = pd.Timestamp(as_of)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        start = end - timedelta(days=lookback_days)
        out: Dict[str, pd.DataFrame] = {}
        for symbol in dict.fromkeys(universe):
            part = root / f"symbol={symbol}"
            if not part.exists():
                continue
            dataset = ds.dataset(part, format="parquet")
            # Predicate pushdown: only ts in (start, end] is read off disk.
            table = dataset.to_table(filter=(pc.field("ts") > start) & (pc.field("ts") <= end))
            if table.num_rows:
                out[symbol] = self._from_records(table.to_pandas())
        return out

    # ------------------------------------------------------------------ #
    # Lazy scan (the out-of-core compute seam, spec 015)
    # ------------------------------------------------------------------ #
    def _tf_root(self, timeframe: TimeframeLike) -> Path:
        tf = str(Timeframe.parse(timeframe) if isinstance(timeframe, str) else timeframe)
        return self.root / tf

    def partition_paths(self, universe: Sequence[str], timeframe: TimeframeLike = "1Day") -> List[str]:
        """The Parquet file paths backing ``universe`` (existing partitions only).

        The glob a DuckDB/Polars set-based query (:func:`src.data.compute.sql_query`)
        reads over — one file per symbol partition that actually exists.
        """
        root = self._tf_root(timeframe)
        paths = []
        for symbol in dict.fromkeys(universe):
            part = root / f"symbol={symbol}" / "part.parquet"
            if part.exists():
                paths.append(str(part))
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

        paths = self.partition_paths(universe, timeframe)
        if not paths:
            empty = pl.DataFrame(schema=schema).lazy()
            if columns is not None:
                empty = empty.select(list(dict.fromkeys(["ts", "symbol", *columns])))
            return empty

        end = pd.Timestamp(as_of)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        start = end - timedelta(days=lookback_days)
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
