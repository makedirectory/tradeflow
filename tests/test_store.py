"""Tests for the out-of-core Parquet bar store.

Skipped automatically if the optional ``store`` extra (pyarrow) isn't installed.
"""

import pytest

pytest.importorskip("pyarrow")

from datetime import datetime, timezone  # noqa: E402

import pandas as pd  # noqa: E402

from src.data import BarSource, ClientBarSource, ParquetBarStore  # noqa: E402
from src.marketdata.client import MarketDataClient  # noqa: E402
from src.marketdata.timeframe import Timeframe  # noqa: E402
from tests.fakes import DictMarketData, make_ohlcv  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC"]

#: The store normalizes the timeframe to its canonical directory name (e.g. "1day"),
#: which differs in case from the "1Day" people type. Derive it rather than hardcode
#: the string so raw-path assertions match the on-disk layout on case-sensitive
#: filesystems (Linux CI), not just case-insensitive ones (default macOS).
TF_DIR = str(Timeframe.parse("1Day"))


def _bars():
    return {s: make_ohlcv(n=300, seed=i, freq="1D") for i, s in enumerate(SYMBOLS)}


def _multiyear_bars(start="2018-01-01", years=6):
    # Daily bars spanning several calendar years, so the year= partitions are exercised.
    idx = pd.date_range(start, periods=365 * years, freq="1D", tz="UTC")
    close = pd.Series(range(len(idx)), index=idx, dtype=float) + 100.0
    frame = pd.DataFrame({c: close for c in ("open", "high", "low", "close", "volume")}, index=idx)
    return {"AAA": frame}


def test_is_a_bar_source(tmp_path):
    store = ParquetBarStore(tmp_path)
    assert isinstance(store, BarSource)  # duck-typed drop-in for ClientBarSource


def test_roundtrip_preserves_values(tmp_path):
    bars = _bars()
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")

    as_of = bars["AAA"].index[-1].to_pydatetime()
    scanned = store.scan(SYMBOLS, "1Day", as_of, lookback_days=10_000)
    assert set(scanned) == set(SYMBOLS)
    for s in SYMBOLS:
        original = bars[s]
        got = scanned[s]
        assert len(got) == len(original)
        assert got["close"].to_numpy() == pytest.approx(original["close"].to_numpy())
        assert list(got.columns) == ["open", "high", "low", "close", "volume"]


def test_as_of_is_pushed_down_no_future_rows(tmp_path):
    bars = _bars()
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")

    cutoff = bars["AAA"].index[150]
    as_of = cutoff.to_pydatetime()
    scanned = store.scan(SYMBOLS, "1Day", as_of, lookback_days=10_000)
    for s, frame in scanned.items():
        assert frame.index.max() <= cutoff.tz_convert("UTC")  # nothing after as_of was read


def test_drop_in_equivalent_to_client_bar_source(tmp_path):
    # The same data, read two ways (in-memory client vs Parquet store), agrees — the
    # storage tier swaps behind the scan() seam without the layers above noticing.
    bars = _bars()
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    client_source = ClientBarSource(MarketDataClient(DictMarketData(bars)))

    cutoff = bars["AAA"].index[200]
    as_of = cutoff.to_pydatetime()
    from_store = store.scan(SYMBOLS, "1Day", as_of, lookback_days=10_000)
    from_client = client_source.scan(SYMBOLS, "1Day", as_of, lookback_days=10_000)

    assert set(from_store) == set(from_client)
    for s in SYMBOLS:
        assert from_store[s]["close"].to_numpy() == pytest.approx(from_client[s]["close"].to_numpy())


def test_scan_window_limits_rows(tmp_path):
    bars = _bars()
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    as_of = bars["AAA"].index[-1].to_pydatetime()
    # A short lookback returns materially fewer rows than the full history.
    short = store.scan(["AAA"], "1Day", as_of, lookback_days=30)
    full = store.scan(["AAA"], "1Day", as_of, lookback_days=10_000)
    assert len(short["AAA"]) < len(full["AAA"])


def test_missing_symbol_is_skipped(tmp_path):
    store = ParquetBarStore(tmp_path)
    store.write({"AAA": make_ohlcv(n=100, seed=0, freq="1D")}, timeframe="1Day")
    scanned = store.scan(["AAA", "ZZZ"], "1Day", datetime(2025, 1, 1), lookback_days=10_000)
    assert set(scanned) == {"AAA"}


def test_streaming_backtest_matches_batch(tmp_path):
    # Streaming one symbol at a time from the store is equivalent to the in-memory
    # batch backtest on the same data — bounded memory, identical result.
    from src.engine.backtest import BacktestEngine
    from src.strategies.ma_crossover import MovingAverageCrossoverStrategy

    syms = ["AAA", "BBB", "CCC"]
    bars = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate(syms)}
    end = bars["AAA"].index[-1].to_pydatetime()
    start = bars["AAA"].index[0].to_pydatetime()

    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")

    batch = BacktestEngine(
        MovingAverageCrossoverStrategy.create_with_defaults(), MarketDataClient(DictMarketData(bars))
    ).run(syms, start, end, 100_000.0)
    streamed = BacktestEngine(MovingAverageCrossoverStrategy.create_with_defaults(), None).run_streaming(
        store, syms, start, end, 100_000.0
    )
    assert streamed.final_capital == pytest.approx(batch.final_capital)
    assert len(streamed.trades) == len(batch.trades)


# --------------------------------------------------------------------------- #
# Date-partitioned layout (symbol=…/year=…) and file-level date pruning
# --------------------------------------------------------------------------- #
def test_write_creates_year_partitions(tmp_path):
    store = ParquetBarStore(tmp_path)
    store.write(_multiyear_bars(start="2018-01-01", years=6), timeframe="1Day")
    years = sorted(p.name for p in (tmp_path / TF_DIR / "symbol=AAA").glob("year=*"))
    assert years == [f"year={y}" for y in range(2018, 2024)]  # one partition per calendar year


def test_windowed_scan_prunes_partition_files(tmp_path):
    store = ParquetBarStore(tmp_path)
    store.write(_multiyear_bars(start="2018-01-01", years=6), timeframe="1Day")
    as_of = datetime(2021, 6, 1, tzinfo=timezone.utc)

    all_files = store.partition_paths(["AAA"], "1Day")  # every year
    pruned = store.partition_paths(["AAA"], "1Day", as_of=as_of, lookback_days=200)
    assert len(all_files) == 6
    # A ~200-day window ending mid-2021 spans only 2020–2021 → 2 partition files read.
    assert len(pruned) < len(all_files)
    assert all(("year=2020" in p or "year=2021" in p) for p in pruned)


def test_windowed_scan_data_correct_across_years(tmp_path):
    # File-level pruning must not drop in-window rows: the scan still returns exactly
    # the bars in (as_of - lookback, as_of], spanning the year boundary.
    store = ParquetBarStore(tmp_path)
    bars = _multiyear_bars(start="2018-01-01", years=6)
    store.write(bars, timeframe="1Day")
    as_of = datetime(2021, 3, 1, tzinfo=timezone.utc)
    scanned = store.scan(["AAA"], "1Day", as_of, lookback_days=120)["AAA"]
    original = bars["AAA"]
    expected = original[
        (original.index > (pd.Timestamp(as_of) - pd.Timedelta(days=120)))
        & (original.index <= pd.Timestamp(as_of))
    ]
    assert len(scanned) == len(expected)
    assert scanned.index.min().year == 2020 and scanned.index.max().year == 2021  # crosses the boundary
    assert scanned["close"].to_numpy() == pytest.approx(expected["close"].to_numpy())


def test_rewrite_replaces_stale_year_partitions(tmp_path):
    # Re-writing a symbol with a shorter history must not leave stale year partitions.
    store = ParquetBarStore(tmp_path)
    store.write(_multiyear_bars(start="2018-01-01", years=6), timeframe="1Day")
    store.write(_multiyear_bars(start="2022-01-01", years=1), timeframe="1Day")  # only 2022
    years = sorted(p.name for p in (tmp_path / TF_DIR / "symbol=AAA").glob("year=*"))
    assert years == ["year=2022"]
