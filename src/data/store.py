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
from typing import Dict, List

import pandas as pd

from src.marketdata.client import TimeframeLike
from src.marketdata.timeframe import Timeframe

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
