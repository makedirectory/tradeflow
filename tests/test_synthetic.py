"""Tests for the offline SyntheticMarketData provider used by `make demo`."""

from datetime import datetime

from tradeflow.marketdata.synthetic import SyntheticMarketData
from tradeflow.marketdata.timeframe import Timeframe

_DAILY = Timeframe.parse("1Day")


def _bars(symbol="X", seed=42, start=datetime(2024, 1, 1), end=datetime(2024, 3, 1)):
    return SyntheticMarketData(seed=seed).get_bars([symbol], _DAILY, start, end)[symbol]


def test_honours_window_and_has_ohlcv_columns():
    df = _bars()
    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None  # tz-aware, like real provider output


def test_ohlc_relationships_hold():
    df = _bars()
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()


def test_is_deterministic_for_a_seed():
    assert _bars(seed=7)["close"].equals(_bars(seed=7)["close"])


def test_distinct_symbols_get_distinct_series():
    bars = SyntheticMarketData(seed=1).get_bars(
        ["AAA", "BBB"], _DAILY, datetime(2024, 1, 1), datetime(2024, 3, 1)
    )
    assert not bars["AAA"]["close"].equals(bars["BBB"]["close"])


def test_intraday_timeframe_produces_more_bars_than_daily():
    start, end = datetime(2024, 1, 1), datetime(2024, 1, 10)
    provider = SyntheticMarketData(seed=3)
    daily = provider.get_bars(["X"], _DAILY, start, end)["X"]
    intraday = provider.get_bars(["X"], Timeframe.parse("5Min"), start, end)["X"]
    assert len(intraday) > len(daily)


def test_does_not_support_streaming():
    assert SyntheticMarketData().supports_streaming() is False
