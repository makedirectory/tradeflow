"""The pandas edge — the *only* sanctioned crossings between the lazy columnar core
and pandas.

The lazy compute model (spec 015) keeps the panel in Arrow/Polars and never
materializes the whole thing into pandas. But pandas is still the right currency at
two narrow places: a **per-symbol leaf** where the data is provably small (the
indicator math the strategy already speaks), and the **final report** a human reads.

Routing every crossing through this module makes the edge *greppable*: a
``to_pandas`` on a full panel is a bug, not a convenience, and a reviewer can find
every place the column store collapses to row-major pandas by searching for these
two functions. Nothing else in the codebase should call ``.to_pandas()`` /
``pl.from_pandas`` on panel-wide data directly.

Requires the ``store`` extra (``polars``).
"""

from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl


def to_pandas(
    frame: Union["pl.DataFrame", "pl.LazyFrame"],
    *,
    index: Optional[str] = None,
    streaming: bool = True,
) -> pd.DataFrame:
    """Collapse a Polars frame to pandas — the sanctioned core→edge crossing.

    Accepts either an eager :class:`polars.DataFrame` or a :class:`polars.LazyFrame`;
    a lazy frame is collected first (with the streaming engine by default, so a large
    plan materializes in bounded-memory chunks rather than all at once). Pass
    ``index`` to set a column as the resulting pandas index (e.g. ``"ts"`` for a
    per-symbol leaf, ``"symbol"`` for a one-timestamp cross-section).

    This is a deliberate narrowing point: call it at a small per-symbol leaf or the
    final report, never on the full multi-symbol panel.
    """
    import polars as pl

    if isinstance(frame, pl.LazyFrame):
        frame = collect_streaming(frame) if streaming else frame.collect()
    pdf = frame.to_pandas()
    if index is not None:
        if index not in pdf.columns:
            raise KeyError(f"index column {index!r} not in frame; columns are {list(pdf.columns)}")
        pdf = pdf.set_index(index)
    return pdf


def from_pandas(df: pd.DataFrame, *, include_index: bool = True) -> "pl.DataFrame":
    """Lift a pandas frame into Polars — the sanctioned edge→core crossing.

    Used to bring a per-symbol leaf (or a provider frame) back into the columnar
    core. By default the pandas index is materialized as a column; pass
    ``include_index=False`` to drop it. A provider/OHLCV frame is typically indexed by
    an *unnamed* ``DatetimeIndex`` — Polars would materialize that as a column literally
    named ``"None"``, so we name it first: a ``DatetimeIndex`` becomes ``ts`` (the
    promise that a time index survives as a real ``ts`` column), any other unnamed
    index becomes ``index``.
    """
    import polars as pl

    if include_index and df.index.name is None:
        df = df.rename_axis("ts" if isinstance(df.index, pd.DatetimeIndex) else "index")
    return pl.from_pandas(df, include_index=include_index)


def collect_streaming(lazy: "pl.LazyFrame") -> "pl.DataFrame":
    """Materialize a :class:`polars.LazyFrame` via the **streaming** engine.

    The streaming engine processes the query plan in bounded-memory chunks, so a plan
    over a larger-than-RAM panel collects without ever holding the whole result. This
    is the one terminal the lazy helpers in :mod:`src.data.compute` are meant to end
    on; it lives here (next to ``to_pandas``) because materializing *is* the edge.
    """
    return lazy.collect(engine="streaming")
