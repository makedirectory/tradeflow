"""Tests for lazy out-of-core compute (spec 015 — Polars · DuckDB).

Skipped automatically unless the ``store`` extra (pyarrow + polars + duckdb) is
installed. These prove the lazy ports are *faithful*: each migrated op is checked
against its legacy pandas implementation (the equivalence oracle), plus the spec's
cross-cutting properties — as-of pushdown, Arrow round-trip, determinism, streaming
== eager, and bounded-memory accumulation.
"""

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("polars")
pytest.importorskip("duckdb")

import polars as pl  # noqa: E402

from src.alphas import refine  # noqa: E402
from src.data import (  # noqa: E402
    ParquetBarStore,  # noqa: E402
    compute,
    edges,
)
from src.indicators.indicators import calculate_ema, calculate_sma  # noqa: E402
from src.risk.sample import SampleCovariance  # noqa: E402
from tests.fakes import make_ohlcv  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


def _store(tmp_path, n=300):
    bars = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate(SYMBOLS)}
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    as_of = max(b.index[-1] for b in bars.values()).to_pydatetime()
    return store, bars, as_of


def _series_for(frame: pl.DataFrame, symbol: str, col: str) -> np.ndarray:
    sub = frame.filter(pl.col("symbol") == symbol).sort("ts")
    return sub[col].to_numpy()


# --------------------------------------------------------------------------- #
# Time-series indicator equivalence (lazy Polars window == pandas oracle)
# --------------------------------------------------------------------------- #
def test_sma_matches_pandas_oracle(tmp_path):
    store, bars, as_of = _store(tmp_path)
    lf = compute.with_sma(store.scan_lazy(SYMBOLS, "1Day", as_of, 10_000), period=20)
    out = edges.collect_streaming(compute.sort_canonical(lf))
    for s in SYMBOLS:
        oracle = calculate_sma(bars[s]["close"], 20).to_numpy()
        np.testing.assert_allclose(_series_for(out, s, "sma"), oracle, equal_nan=True)


def test_ema_matches_pandas_oracle(tmp_path):
    store, bars, as_of = _store(tmp_path)
    lf = compute.with_ema(store.scan_lazy(SYMBOLS, "1Day", as_of, 10_000), period=12)
    out = edges.collect_streaming(compute.sort_canonical(lf))
    for s in SYMBOLS:
        oracle = calculate_ema(bars[s]["close"], 12).to_numpy()
        np.testing.assert_allclose(_series_for(out, s, "ema"), oracle, equal_nan=True)


def test_returns_match_pandas_oracle(tmp_path):
    store, bars, as_of = _store(tmp_path)
    lf = compute.with_returns(store.scan_lazy(SYMBOLS, "1Day", as_of, 10_000))
    out = edges.collect_streaming(compute.sort_canonical(lf))
    for s in SYMBOLS:
        oracle = bars[s]["close"].pct_change().to_numpy()
        np.testing.assert_allclose(_series_for(out, s, "ret"), oracle, equal_nan=True)


# --------------------------------------------------------------------------- #
# Cross-sectional refinement equivalence (lazy over("ts") == refine.py oracle)
# --------------------------------------------------------------------------- #
def _panel_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(5):
        for sym in SYMBOLS:
            rows.append({"ts": day, "symbol": sym, "score": float(rng.normal())})
    return pl.DataFrame(rows)


def test_cross_sectional_zscore_matches_refine(tmp_path):
    df = _panel_frame()
    out = compute.cross_sectional_zscore(df.lazy()).collect()
    for day in range(5):
        got = out.filter(pl.col("ts") == day).sort("symbol")["z"].to_numpy()
        oracle = refine.zscore(df.filter(pl.col("ts") == day).sort("symbol")["score"].to_pandas()).to_numpy()
        np.testing.assert_allclose(got, oracle, atol=1e-12)


def test_cross_sectional_zscore_degenerate_is_zero():
    # A cross-section with no spread expresses no view → all zeros, not NaN (matches refine).
    df = pl.DataFrame({"ts": [0, 0, 0], "symbol": ["A", "B", "C"], "score": [5.0, 5.0, 5.0]})
    out = compute.cross_sectional_zscore(df.lazy()).collect()
    assert out["z"].to_list() == [0.0, 0.0, 0.0]


def test_cross_sectional_winsorize_matches_refine():
    df = pl.DataFrame({"ts": [0] * 6, "symbol": list("ABCDEF"), "score": [1.0, 2, 3, 4, 5, 100]})
    out = compute.cross_sectional_winsorize(df.lazy(), lower=0.1, upper=0.9).collect()
    oracle = refine.winsorize(df["score"].to_pandas(), 0.1, 0.9).to_numpy()
    np.testing.assert_allclose(out.sort("symbol")["score"].to_numpy(), oracle, atol=1e-12)


def test_cross_sectional_rank_matches_pandas():
    df = pl.DataFrame({"ts": [0] * 4, "symbol": list("ABCD"), "score": [3.0, 1.0, 1.0, 9.0]})
    out = compute.cross_sectional_rank(df.lazy()).collect()
    oracle = df["score"].to_pandas().rank().to_numpy()  # average ties, ascending
    np.testing.assert_allclose(out.sort("symbol")["rank"].to_numpy(), oracle)


# --------------------------------------------------------------------------- #
# Streaming covariance (accumulated == SampleCovariance oracle, chunk-invariant)
# --------------------------------------------------------------------------- #
def test_streaming_covariance_matches_sample_oracle():
    import pandas as pd

    rng = np.random.default_rng(7)
    panel = rng.normal(size=(250, 5))  # T×N return matrix
    sigma, n = compute.streaming_covariance(compute.iter_row_chunks(panel, chunk_rows=37))
    oracle, _ = SampleCovariance().estimate(pd.DataFrame(panel))
    assert n == 250
    np.testing.assert_allclose(sigma, oracle, atol=1e-12)


def test_streaming_covariance_is_chunk_invariant():
    rng = np.random.default_rng(11)
    panel = rng.normal(size=(200, 4))
    whole, _ = compute.streaming_covariance([panel])
    for size in (1, 7, 50, 199, 1000):
        chunked, _ = compute.streaming_covariance(compute.iter_row_chunks(panel, size))
        np.testing.assert_allclose(chunked, whole, atol=1e-12)


def test_streaming_covariance_bounded_memory_via_generator():
    # The generator yields chunks lazily — the full T×N matrix never exists in memory,
    # only one chunk plus the N×N accumulator. We assert the result still matches a
    # reference accumulation, proving the out-of-core path is faithful.
    import pandas as pd

    N, CHUNK, NCHUNKS = 6, 64, 200  # 12,800 rows, never materialized together
    seeds = range(NCHUNKS)

    def gen():
        for sd in seeds:
            yield np.random.default_rng(sd).normal(size=(CHUNK, N))

    sigma, n = compute.streaming_covariance(gen())
    assert n == CHUNK * NCHUNKS
    # Reference: same blocks concatenated once (only valid because the test is small).
    full = np.vstack([np.random.default_rng(sd).normal(size=(CHUNK, N)) for sd in seeds])
    oracle, _ = SampleCovariance().estimate(pd.DataFrame(full))
    np.testing.assert_allclose(sigma, oracle, atol=1e-10)


def test_streaming_covariance_degenerate():
    sigma, n = compute.streaming_covariance([np.zeros((1, 3))])
    assert n == 1
    assert sigma.shape == (3, 3) and not sigma.any()


# --------------------------------------------------------------------------- #
# As-of pushdown: a lazy scan physically never reads rows after as_of
# --------------------------------------------------------------------------- #
def test_scan_lazy_pushes_as_of_into_parquet_scan(tmp_path):
    store, bars, _ = _store(tmp_path)
    cutoff = bars["AAA"].index[150]
    lf = store.scan_lazy(SYMBOLS, "1Day", cutoff.to_pydatetime(), 10_000)
    # The predicate is pushed into the SCAN node, not applied after materialize.
    plan = lf.explain()
    assert "SELECTION" in plan and "ts" in plan
    out = edges.collect_streaming(lf)
    assert out["ts"].max() <= cutoff.tz_convert("UTC")  # no future bar was read


def test_scan_lazy_projection_pushdown(tmp_path):
    store, _, as_of = _store(tmp_path)
    lf = store.scan_lazy(SYMBOLS, "1Day", as_of, 10_000, columns=["close"])
    out = edges.collect_streaming(lf)
    assert set(out.columns) == {"ts", "symbol", "close"}  # only projected columns read


def test_scan_lazy_empty_universe_is_empty(tmp_path):
    store, _, as_of = _store(tmp_path)
    lf = store.scan_lazy(["NOPE"], "1Day", as_of, 10_000)
    assert edges.collect_streaming(lf).is_empty()


# --------------------------------------------------------------------------- #
# Arrow round-trip: provider → store → polars → pandas edge preserves everything
# --------------------------------------------------------------------------- #
def test_arrow_roundtrip_preserves_values_dtypes_tz_order(tmp_path):
    store, bars, as_of = _store(tmp_path)
    lf = store.scan_lazy(["AAA"], "1Day", as_of, 10_000)
    pdf = edges.to_pandas(compute.sort_canonical(lf), index="ts")

    original = bars["AAA"]
    assert str(pdf.index.tz) == "UTC"  # tz-aware UTC survived the crossing
    assert pdf.index.is_monotonic_increasing  # order preserved
    np.testing.assert_allclose(pdf["close"].to_numpy(), original["close"].to_numpy())
    for col in ("open", "high", "low", "close", "volume"):
        assert pdf[col].dtype == np.float64  # no float64 → object demotion


def test_from_pandas_edge_roundtrips():
    store_pdf = make_ohlcv(n=20, seed=0, freq="1D")
    store_pdf.index.name = "ts"  # name the index so it survives as a named column
    back = edges.to_pandas(edges.from_pandas(store_pdf, include_index=True), index="ts")
    np.testing.assert_allclose(back["close"].to_numpy(), store_pdf["close"].to_numpy())


# --------------------------------------------------------------------------- #
# Determinism + streaming == eager
# --------------------------------------------------------------------------- #
def test_streaming_equals_eager_collect(tmp_path):
    # Same plan, two engines (streaming vs in-memory). Row order and ints/strings are
    # identical; float reductions can differ at machine epsilon by reduction order, so
    # the numeric columns are compared within tolerance (byte-identity is the
    # *same-engine repeated-run* property, asserted separately below).
    store, _, as_of = _store(tmp_path)
    lf = compute.sort_canonical(
        compute.cross_sectional_zscore(
            compute.with_returns(store.scan_lazy(SYMBOLS, "1Day", as_of, 10_000)),
            src="ret",
        )
    )
    streamed = edges.collect_streaming(lf)
    eager = lf.collect()
    assert streamed["symbol"].equals(eager["symbol"])
    assert streamed["ts"].equals(eager["ts"])
    for col in ("ret", "z"):
        np.testing.assert_allclose(
            streamed[col].to_numpy(), eager[col].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_deterministic_across_repeated_runs(tmp_path):
    store, _, as_of = _store(tmp_path)
    lf = compute.sort_canonical(compute.with_sma(store.scan_lazy(SYMBOLS, "1Day", as_of, 10_000), 10))
    first = edges.collect_streaming(lf)
    for _ in range(3):
        assert edges.collect_streaming(lf).equals(first)  # byte-identical run to run


# --------------------------------------------------------------------------- #
# DuckDB set-based path over the Parquet store
# --------------------------------------------------------------------------- #
def test_sql_query_aggregation_matches_pandas(tmp_path):
    store, bars, _ = _store(tmp_path)
    paths = store.partition_paths(SYMBOLS, "1Day")
    got = compute.sql_query(
        paths, "SELECT symbol, count(*) AS n, avg(close) AS m FROM bars GROUP BY symbol ORDER BY symbol"
    )
    got_pd = edges.to_pandas(got, index="symbol")
    for s in SYMBOLS:
        assert int(got_pd.loc[s, "n"]) == len(bars[s])
        assert got_pd.loc[s, "m"] == pytest.approx(float(bars[s]["close"].mean()))


def test_sql_query_as_of_param_pushdown(tmp_path):
    store, bars, _ = _store(tmp_path)
    paths = store.partition_paths(["AAA"], "1Day")
    cutoff = bars["AAA"].index[100].tz_convert("UTC").to_pydatetime()
    got = compute.sql_query(
        paths, "SELECT count(*) AS n, max(ts) AS last FROM bars WHERE ts <= ?", params=[cutoff]
    )
    assert got["last"][0] <= cutoff  # the point-in-time cutoff held inside the scan


def test_sql_query_empty_result_has_schema(tmp_path):
    store, _, _ = _store(tmp_path)
    paths = store.partition_paths(["AAA"], "1Day")
    got = compute.sql_query(paths, "SELECT * FROM bars WHERE close < -1")  # matches nothing
    assert got.height == 0 and "close" in got.columns


# --------------------------------------------------------------------------- #
# Regression tests for the spec-015 adversarial review.
# Each guards a confirmed finding the original suite missed.
# --------------------------------------------------------------------------- #
def _shuffled_store(tmp_path, n=40, seed=3):
    # A single symbol whose bars are written OUT of ts order on disk (write does not
    # sort), to prove the lazy window helpers no longer trust physical row order.
    import pandas as pd

    idx = pd.date_range("2024-01-02", periods=n, freq="1D", tz="UTC")
    close = np.arange(n, dtype=float) + 10
    frame = pd.DataFrame({c: close for c in ("open", "high", "low", "close", "volume")}, index=idx)
    store = ParquetBarStore(tmp_path)
    store.write({"AAA": frame.sample(frac=1.0, random_state=seed)}, timeframe="1Day")
    return store, frame, idx[-1].to_pydatetime()


def test_indicators_correct_on_unsorted_partition(tmp_path):
    # Finding lazy-window-unsorted-ts / indicators-no-ts-sort: rolling/ewm/returns must
    # match the ts-sorted pandas oracle even when the partition is shuffled on disk.
    store, frame, as_of = _shuffled_store(tmp_path)
    lf = store.scan_lazy(["AAA"], "1Day", as_of, 10_000)
    lf = compute.with_returns(compute.with_ema(compute.with_sma(lf, 5), 12))
    out = edges.collect_streaming(compute.sort_canonical(lf))
    np.testing.assert_allclose(
        out["sma"].to_numpy(), calculate_sma(frame["close"], 5).to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(
        out["ema"].to_numpy(), calculate_ema(frame["close"], 12).to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(out["ret"].to_numpy(), frame["close"].pct_change().to_numpy(), equal_nan=True)


def _xs(scores):
    """A one-timestamp cross-section panel from a list of (possibly non-finite) scores."""
    syms = [f"S{i}" for i in range(len(scores))]
    return pl.DataFrame({"ts": [0] * len(scores), "symbol": syms, "score": [float(s) for s in scores]})


def test_zscore_nan_does_not_poison_cross_section():
    # Finding zscore-nan-poisons-cross-section: one NaN must not zero the whole bar.
    import pandas as pd

    df = _xs([1.0, 2.0, float("nan"), 4.0])
    got = compute.cross_sectional_zscore(df.lazy()).collect().sort("symbol")["z"].to_numpy()
    oracle = refine.zscore(pd.Series([1.0, 2.0, np.nan, 4.0])).to_numpy()
    np.testing.assert_allclose(got, oracle, atol=1e-12, equal_nan=True)


def test_zscore_inf_treated_as_missing():
    import pandas as pd

    df = _xs([1.0, float("inf"), 3.0])
    got = compute.cross_sectional_zscore(df.lazy()).collect().sort("symbol")["z"].to_numpy()
    # inf -> missing, the two finite names standardize around their own mean/std.
    oracle = refine.zscore(pd.Series([1.0, np.nan, 3.0])).to_numpy()
    np.testing.assert_allclose(got, oracle, atol=1e-12, equal_nan=True)


def test_rank_nan_stays_missing_not_top():
    # Finding rank-treats-nan-as-largest: a NaN must not be handed a tradable rank.
    import pandas as pd

    df = _xs([3.0, float("nan"), 1.0, 1.0, 9.0])
    got = compute.cross_sectional_rank(df.lazy()).collect().sort("symbol")["rank"].to_numpy()
    oracle = pd.Series([3.0, np.nan, 1.0, 1.0, 9.0]).rank().to_numpy()
    np.testing.assert_allclose(got, oracle, equal_nan=True)


def test_winsorize_skips_nan_like_oracle():
    import pandas as pd

    vals = [1.0, 2.0, float("nan"), 4.0, 5.0, 100.0]
    df = _xs(vals)
    got = compute.cross_sectional_winsorize(df.lazy(), lower=0.1, upper=0.9).collect().sort("symbol")["score"]
    oracle = refine.winsorize(pd.Series(vals), 0.1, 0.9).to_numpy()
    np.testing.assert_allclose(got.to_numpy(), oracle, atol=1e-12, equal_nan=True)


def test_demean_matches_refine_including_nan():
    # cross_sectional_demean had no test at all (coverage gap) and must skip non-finite.
    import pandas as pd

    for vals in ([1.0, 2.0, 3.0, 6.0], [1.0, float("nan"), 3.0]):
        df = _xs(vals)
        got = compute.cross_sectional_demean(df.lazy()).collect().sort("symbol")["z"].to_numpy()
        oracle = refine.demean(pd.Series(vals)).to_numpy()
        np.testing.assert_allclose(got, oracle, atol=1e-12, equal_nan=True)


def test_returns_then_zscore_survives_zero_price():
    # End-to-end: a 0/0 return is a real NaN; it must not flatten the rest of the bar.
    import pandas as pd

    # Three symbols at two timestamps; symbol B has a flat-zero price → NaN return at t1.
    rows = []
    closes = {"A": [10.0, 11.0], "B": [0.0, 0.0], "C": [10.0, 15.0]}
    ts = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    for sym, cs in closes.items():
        for t, c in zip(ts, cs):
            rows.append({"ts": t, "symbol": sym, "close": c})
    df = pl.DataFrame(rows)
    lf = compute.cross_sectional_zscore(compute.with_returns(df.lazy(), src="close"), src="ret", by="ts")
    out = edges.collect_streaming(compute.sort_canonical(lf)).filter(pl.col("ts") == ts[1]).sort("symbol")
    z = out["z"].to_list()
    # A and C have real, differing returns → a real long/short view; B (NaN) stays null.
    assert z[0] is not None and z[2] is not None and z[0] != z[2]
    assert z[1] is None


def test_scan_lazy_naive_as_of_localises_utc(tmp_path):
    # Finding coverage gap: a naive (tz-less) as_of must behave like the eager scan().
    import datetime as dt

    store, bars, _ = _store(tmp_path)
    naive = bars["AAA"].index[100].tz_convert("UTC").tz_localize(None).to_pydatetime()
    assert naive.tzinfo is None
    lazy_rows = edges.collect_streaming(store.scan_lazy(["AAA"], "1Day", naive, 10_000)).height
    eager_rows = len(store.scan(["AAA"], "1Day", naive, 10_000)["AAA"])
    assert lazy_rows == eager_rows  # lazy and eager agree on the localized window

    # And a non-UTC tz-aware as_of resolves to the same instant.
    ny = bars["AAA"].index[100].tz_convert("America/New_York").to_pydatetime()
    assert isinstance(ny, dt.datetime) and ny.tzinfo is not None
    assert edges.collect_streaming(store.scan_lazy(["AAA"], "1Day", ny, 10_000)).height == eager_rows


def test_scan_lazy_window_matches_eager_scan(tmp_path):
    # The lazy and eager BarSource paths must return identical row sets (same window).
    store, bars, _ = _store(tmp_path)
    cutoff = bars["AAA"].index[150]
    as_of = cutoff.to_pydatetime()
    lazy = edges.collect_streaming(store.scan_lazy(["AAA"], "1Day", as_of, 90)).sort("ts")
    eager = store.scan(["AAA"], "1Day", as_of, 90)["AAA"]
    assert lazy.height == len(eager)
    np.testing.assert_allclose(lazy["close"].to_numpy(), eager["close"].to_numpy())


def test_scan_lazy_empty_universe_supports_pipeline(tmp_path):
    # Finding lazy-empty-universe-no-schema: a window helper on an empty-universe scan
    # must yield an empty (correctly-columned) frame, not ColumnNotFoundError.
    store, _, as_of = _store(tmp_path)
    lf = compute.with_returns(store.scan_lazy(["NOPE"], "1Day", as_of, 10_000))
    out = edges.collect_streaming(lf)
    assert out.is_empty() and {"ts", "symbol", "close", "ret"} <= set(out.columns)


def test_from_pandas_unnamed_datetime_index_becomes_ts():
    # Finding from-pandas-unnamed-index-string-None: an unnamed DatetimeIndex must
    # cross as a real `ts` column, not the string "None".
    df = make_ohlcv(n=8, seed=0, freq="1D")  # index.name is None
    assert df.index.name is None
    lifted = edges.from_pandas(df, include_index=True)
    assert "ts" in lifted.columns and "None" not in lifted.columns
    back = edges.to_pandas(lifted, index="ts")
    np.testing.assert_allclose(back["close"].to_numpy(), df["close"].to_numpy())


def test_to_pandas_missing_index_raises_clear_error():
    with pytest.raises(KeyError, match="not in frame"):
        edges.to_pandas(pl.DataFrame({"a": [1]}), index="ts")


def test_streaming_covariance_bounded_memory_tracemalloc():
    # Finding no-bounded-memory-test: the accumulator's peak memory must be bounded by
    # one chunk + the N×N accumulator, NOT by total history. Feed a generator so the
    # full T×N panel never exists, and assert peak allocation stays near chunk-size.
    import tracemalloc

    N, CHUNK, NCHUNKS = 8, 256, 400  # 102,400 rows would be ~6.5 MB if materialized
    full_bytes = N * CHUNK * NCHUNKS * 8

    def gen():
        for sd in range(NCHUNKS):
            yield np.random.default_rng(sd).normal(size=(CHUNK, N))

    tracemalloc.start()
    _, n = compute.streaming_covariance(gen())
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert n == CHUNK * NCHUNKS
    # Peak should be a small multiple of one chunk, far below the full panel.
    assert peak < full_bytes // 4
