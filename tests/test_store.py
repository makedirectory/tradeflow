"""Tests for the out-of-core Parquet bar store (spec 011).

Skipped automatically if the optional ``store`` extra (pyarrow) isn't installed.
"""

import pytest

pytest.importorskip("pyarrow")

from datetime import datetime  # noqa: E402

from src.data import BarSource, ClientBarSource, ParquetBarStore  # noqa: E402
from src.marketdata.client import MarketDataClient  # noqa: E402
from tests.fakes import DictMarketData, make_ohlcv  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC"]


def _bars():
    return {s: make_ohlcv(n=300, seed=i, freq="1D") for i, s in enumerate(SYMBOLS)}


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
