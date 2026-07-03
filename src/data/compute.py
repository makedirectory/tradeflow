"""Lazy, out-of-core compute over the bar panel (spec 015 — Polars · DuckDB).

Spec 011 delivered the *storage* half of out-of-core (a Parquet/Arrow ``BarSource``
behind ``scan()``). This module is the *compute* half: the hot, panel-wide
operations expressed as a **lazy** plan, rather than eagerly materializing a ``T×N``
pandas panel into RAM.

Two engines, sharing Arrow so it isn't either/or:

- **Polars ``LazyFrame``** for dataframe-style transforms — rolling-window indicators
  become window expressions ``.over("symbol", order_by="ts")``, and the
  cross-sectional z-score / winsorize / rank become ``.over("ts")`` expressions.
  Nothing materializes until a terminal :func:`~src.data.edges.collect_streaming`.
- **DuckDB SQL** (:func:`sql_query`) for set-based work over the Parquet store
  (aggregations, joins) — its out-of-core engine streams the scan too.

Three disciplines the spec calls out and this module keeps:

1. **Pushdown survives the plan.** A lazy scan reads only the columns/row-ranges it
   needs and the ``as_of`` predicate is pushed into the Parquet reader — never a
   post-materialize filter (see :meth:`ParquetBarStore.scan_lazy`).
2. **Determinism under multithreading.** Polars/DuckDB parallelize; float reduction
   order can vary. The cross-sectional reductions here are order-invariant
   (mean/std/quantile), and :data:`CANONICAL_SORT` pins a stable output order so
   repeated runs are byte-identical.
3. **The pandas edge stays narrow.** Every crossing back to pandas routes through
   :mod:`src.data.edges`; this module returns Polars frames / numpy, never pandas.

**A note on what is genuinely bounded-memory.** The leaf Parquet scan (with its
pushed-down predicate/projection) streams, and :func:`streaming_covariance` is bounded
by the universe size, not by history length. But in the current Polars engine a
``.over(...)`` window groupby and a top-level :func:`sort_canonical` are *buffering*
operators — they materialize their input — so a pipeline that ends on a cross-sectional
``over("ts")`` plus a canonical sort has peak memory ~ the panel it sorts, not O(chunk).
The honest out-of-core wins today are the streaming scan, the as-of pushdown, and the
streaming covariance accumulator; the set-based DuckDB path (which can spill) is the
route to a genuinely bounded cross-sectional reduction. The window *math* is correct
either way; only the memory profile of the buffering stages is not O(1) in history.

The legacy pandas implementations (:mod:`src.indicators.indicators`,
:mod:`src.alphas.refine`, :mod:`src.risk.sample`) are kept as the equivalence
oracle — the tests prove these lazy ports match them within tolerance.

Requires the ``store`` extra (``polars``; ``duckdb`` for :func:`sql_query`).
"""

from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl

#: Column names of the long-format bar panel (matches the Parquet store schema).
TS = "ts"
SYMBOL = "symbol"

#: Canonical output ordering. Pinning an explicit sort makes a multithreaded collect
#: byte-identical run to run (hidden factor 2 — determinism).
CANONICAL_SORT = [TS, SYMBOL]


def sort_canonical(lf: "pl.LazyFrame") -> "pl.LazyFrame":
    """Pin the deterministic ``(ts, symbol)`` output order on a lazy plan."""
    return lf.sort(CANONICAL_SORT)


# --------------------------------------------------------------------------- #
# Time-series indicators — Polars window expressions, partitioned per symbol.
# Pandas (src/indicators) stays the per-symbol leaf and the equivalence oracle.
# --------------------------------------------------------------------------- #
def _ordered_over(expr: "pl.Expr", by: str, order: str = TS) -> "pl.Expr":
    """A window expression evaluated per ``by`` group, **sorted by ``order`` within it**.

    A rolling/lagged indicator is meaningless unless the window sees rows in time
    order, and the on-disk/scan order is not guaranteed to be ts-ascending (multiple
    row groups, concatenated files, an unsorted write). Passing ``order_by`` makes the
    window sort *inside* the group, so the result is independent of physical row order
    — matching the eager :meth:`ParquetBarStore.scan` path, which sorts each symbol by
    ts. (A post-hoc :func:`sort_canonical` cannot fix this: it reorders the finished
    output, not the sequence the window was computed over.)
    """
    return expr.over(by, order_by=order)


def with_sma(
    lf: "pl.LazyFrame", period: int, *, src: str = "close", out: str = "sma", by: str = SYMBOL
) -> "pl.LazyFrame":
    """Add a simple moving average column, per symbol (matches ``calculate_sma``).

    Polars ``rolling_mean`` defaults ``min_periods == window_size``, so the head is
    null until the window fills — exactly pandas' ``rolling(period).mean()``. The
    window is ts-ordered within each symbol (see :func:`_ordered_over`).
    """
    import polars as pl

    return lf.with_columns(_ordered_over(pl.col(src).rolling_mean(window_size=period), by).alias(out))


def with_ema(
    lf: "pl.LazyFrame", period: int, *, src: str = "close", out: str = "ema", by: str = SYMBOL
) -> "pl.LazyFrame":
    """Add an exponential moving average column, per symbol (matches ``calculate_ema``).

    Uses ``adjust=False`` to match pandas' ``ewm(span=period, adjust=False).mean()``
    (recursive form seeded on the first observation), ts-ordered within each symbol.

    Equivalence holds for a gap-free source. An *interior* null/NaN between two valid
    bars diverges from pandas: pandas (``ignore_na=False``) decays the weight across the
    gap, Polars re-weights as if the missing bar were absent. A clean bar series has no
    interior gaps (the only nulls are the leading warm-up, which agree); feed
    gap-filled input if a source can drop interior bars.
    """
    import polars as pl

    return lf.with_columns(_ordered_over(pl.col(src).ewm_mean(span=period, adjust=False), by).alias(out))


def with_returns(
    lf: "pl.LazyFrame", *, src: str = "close", out: str = "ret", by: str = SYMBOL
) -> "pl.LazyFrame":
    """Add a one-bar simple return column, per symbol (matches ``close.pct_change()``).

    ts-ordered within each symbol. Note ``pct_change`` off a zero/again-zero price is a
    genuine NaN/inf (``0/0``, ``x/0``) just as in pandas; the cross-sectional helpers
    below treat such non-finite values as missing.
    """
    import polars as pl

    return lf.with_columns(_ordered_over(pl.col(src).pct_change(), by).alias(out))


# --------------------------------------------------------------------------- #
# Cross-sectional refinement — Polars ``.over("ts")`` expressions on the panel.
# src/alphas/refine.py stays the equivalence oracle.
#
# Missing-data contract: the pandas oracle (refine.py) runs on ``.dropna()`` — it
# skips non-finite names and standardizes/ranks the rest. Polars reductions skip
# *null* but PROPAGATE NaN/inf, which would poison the whole cross-section (one NaN
# score → every name's z collapses to 0; a NaN gets handed the top rank). So each
# helper first maps non-finite to null via ``_finite`` — then a single contaminated
# name no longer destroys the rest of the cross-section, matching the oracle.
# --------------------------------------------------------------------------- #
def _finite(src: str) -> "pl.Expr":
    """``src`` with non-finite values (NaN/±inf) mapped to null, so reductions skip them."""
    import polars as pl

    col = pl.col(src)
    return pl.when(col.is_finite()).then(col).otherwise(None)


def cross_sectional_zscore(
    lf: "pl.LazyFrame", *, src: str = "score", out: str = "z", by: str = TS
) -> "pl.LazyFrame":
    """Cross-sectional z-score ``(x - mean) / std`` per timestamp (matches ``refine.zscore``).

    Uses the population std (``ddof=0``) so the cross-section has unit dispersion.
    Non-finite names are skipped (null out) exactly as the oracle's ``dropna``; a
    degenerate cross-section (zero std across the finite names) maps to all-zeros — a
    universe with no spread expresses no view, the same contract as the pandas oracle.
    """
    import polars as pl

    x = _finite(src)
    mean = x.mean().over(by)
    std = x.std(ddof=0).over(by)
    z = (
        pl.when(x.is_null())  # a missing/non-finite name stays missing (oracle: NaN)
        .then(None)
        .when((std == 0) | std.is_null())  # no spread among finite names → no view
        .then(pl.lit(0.0))
        .otherwise((x - mean) / std)
    )
    return lf.with_columns(z.alias(out))


def cross_sectional_winsorize(
    lf: "pl.LazyFrame",
    *,
    src: str = "score",
    out: Optional[str] = None,
    lower: float = 0.025,
    upper: float = 0.975,
    by: str = TS,
) -> "pl.LazyFrame":
    """Clip ``src`` to its ``[lower, upper]`` cross-sectional quantiles (matches ``refine.winsorize``).

    Quantiles use linear interpolation to match pandas' ``Series.quantile`` default and
    are taken over the finite names only (non-finite stay missing). Writes back into
    ``src`` unless ``out`` is given.
    """

    out = out or src
    x = _finite(src)
    lo = x.quantile(lower, interpolation="linear").over(by)
    hi = x.quantile(upper, interpolation="linear").over(by)
    return lf.with_columns(x.clip(lower_bound=lo, upper_bound=hi).alias(out))


def cross_sectional_rank(
    lf: "pl.LazyFrame", *, src: str = "score", out: str = "rank", by: str = TS
) -> "pl.LazyFrame":
    """Add a cross-sectional rank per timestamp (1 = smallest), average ties.

    ``method="average"`` matches pandas' ``Series.rank()`` default. Non-finite names are
    null (Polars ranks null as null, like pandas), so a missing score is never handed a
    tradable rank.
    """

    return lf.with_columns(_finite(src).rank(method="average").over(by).alias(out))


def cross_sectional_demean(
    lf: "pl.LazyFrame", *, src: str = "score", out: str = "z", by: str = TS
) -> "pl.LazyFrame":
    """Subtract the cross-sectional mean per timestamp (matches ``refine.demean``).

    Mean is taken over the finite names; non-finite names stay missing.
    """

    x = _finite(src)
    return lf.with_columns((x - x.mean().over(by)).alias(out))


# --------------------------------------------------------------------------- #
# Streaming covariance — accumulate cross-products by streaming the return panel
# instead of materializing the full T×N matrix. src/risk/sample.SampleCovariance
# is the equivalence oracle (this reproduces its population MLE estimate).
# --------------------------------------------------------------------------- #
def streaming_covariance(
    return_chunks: Iterable[np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Per-bar covariance accumulated over chunks of an aligned return panel.

    Each chunk is a ``rows × N`` block of an already-aligned (complete-case) return
    panel — the same panel :func:`src.risk.base.build_return_panel` produces, but fed
    in row-chunks so no more than one chunk plus the ``N×N`` accumulator is ever in
    memory. Memory is bounded by the **universe size**, not the length of history —
    the point of an out-of-core covariance.

    Accumulates the uncentered sum ``s = Σ xₜ`` and second-moment ``S₂ = Σ xₜxₜᵀ``,
    then returns the population covariance ``Σ = S₂/n − (s/n)(s/n)ᵀ``. The result
    matches the single-pass estimate to machine epsilon regardless of how the rows are
    split (a tested invariant; not byte-identical, since float summation order varies
    with chunking). The uncentered form is *algebraically* identical to centering first,
    and *numerically* matches :class:`~src.risk.sample.SampleCovariance` to machine
    epsilon for the inputs it is contracted on — **return-scale, near-zero-mean** rows
    (what :func:`src.risk.base.build_return_panel` produces from ``pct_change``). It is
    NOT for price levels: when the mean dwarfs the spread (|mean| ≫ std), ``S₂/n`` and
    ``(s/n)(s/n)ᵀ`` are two large near-equal numbers and their difference loses
    precision (catastrophic cancellation). Feed returns, not levels.

    Returns ``(Σ_per_bar, n_observations)``; with fewer than two observations Σ is the
    zero matrix (mirroring the oracle's degenerate-case contract).
    """
    s: Optional[np.ndarray] = None
    s2: Optional[np.ndarray] = None
    count = 0
    n_cols = 0

    for chunk in return_chunks:
        x = np.asarray(chunk, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            continue
        if s is None:
            n_cols = x.shape[1]
            s = np.zeros(n_cols)
            s2 = np.zeros((n_cols, n_cols))
        elif x.shape[1] != n_cols:
            raise ValueError(f"chunk has {x.shape[1]} columns, expected {n_cols}")
        s += x.sum(axis=0)
        s2 += x.T @ x
        count += x.shape[0]

    if count < 2 or s is None:
        return np.zeros((n_cols, n_cols)), count
    mean = s / count
    cov = s2 / count - np.outer(mean, mean)
    return cov, count


def iter_row_chunks(matrix: np.ndarray, chunk_rows: int) -> Iterable[np.ndarray]:
    """Yield row-blocks of ``matrix`` of at most ``chunk_rows`` rows.

    The bridge from a materialized return panel to :func:`streaming_covariance`: lets
    a caller feed an in-memory panel in bounded slices, or a streaming reader hand
    chunks straight through.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    for start in range(0, len(matrix), chunk_rows):
        yield matrix[start : start + chunk_rows]


# --------------------------------------------------------------------------- #
# DuckDB SQL path — set-based work over the Parquet store, out-of-core.
# --------------------------------------------------------------------------- #
def sql_query(
    parquet_glob: str | Sequence[str],
    sql: str,
    *,
    params: Optional[Sequence] = None,
    view: str = "bars",
) -> "pl.DataFrame":
    """Run a DuckDB SQL query over Parquet, returned as a Polars frame (zero-copy Arrow).

    Registers ``parquet_glob`` (a path/glob or a list of paths) as a view named
    ``view`` (default ``bars``) so the query reads ``FROM bars``. DuckDB's reader does
    its own projection/predicate pushdown and streams the scan, so an aggregation over
    a multi-file store never materializes the whole thing. The result returns as
    Polars (shared Arrow) — collapse it to pandas only at the edge via
    :func:`src.data.edges.to_pandas`.

    ``params`` binds DuckDB ``?`` placeholders (e.g. an ``as_of`` cutoff), keeping the
    point-in-time filter inside the pushed-down scan rather than a post-filter.
    """
    import duckdb

    paths: List[str] = [parquet_glob] if isinstance(parquet_glob, str) else list(parquet_glob)
    con = duckdb.connect()
    try:
        # hive_partitioning=false: the store embeds `symbol` as a real column, so we
        # must not also synthesize it from the `symbol=…` path (that would collide).
        # Register the scan as a relation (CREATE VIEW can't take a prepared param).
        con.register(view, con.read_parquet(paths, hive_partitioning=False))
        arrow = con.execute(sql, list(params) if params is not None else None).to_arrow_table()
    finally:
        con.close()

    import polars as pl

    return pl.from_arrow(arrow)
