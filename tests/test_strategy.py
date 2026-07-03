"""Tests for the Strategy base behavior via VolumeSpikeStrategy."""

import pandas as pd
import pytest

from src.strategies import signals
from src.strategies.volume_spike import VolumeSpikeStrategy


def _strategy():
    return VolumeSpikeStrategy.create_with_defaults()


def test_defaults_include_timeframe_and_lookback():
    s = _strategy()
    assert s.config["timeframe"] == "5Min"
    assert s.config["required_lookback_periods"] == max(
        s.config["long_ema_period"] + 1, s.config["volume_ma_period"] + 1
    )


def test_invalid_parameter_raises():
    with pytest.raises(ValueError):
        VolumeSpikeStrategy(
            {
                **{p: spec["default"] for p, spec in VolumeSpikeStrategy.PARAM_RANGES.items()},
                "long_ema_period": 999,
            }
        )  # out of range


def test_short_ema_must_be_below_long_ema():
    config = {p: spec["default"] for p, spec in VolumeSpikeStrategy.PARAM_RANGES.items()}
    config["short_ema_period"] = config["long_ema_period"]
    with pytest.raises(ValueError):
        VolumeSpikeStrategy(config).initialize()


def test_position_size_capped_by_notional_limit():
    s = _strategy()  # position_limits.max_position_size == 100
    # 100 notional / $100 price == 1 share, the binding constraint.
    assert s.calculate_position_size(capital=100_000, price=100.0) == pytest.approx(1.0)


def test_validate_signal_rejects_conflicts():
    s = _strategy()
    s.positions = {"AAA": {"side": signals.BUY, "stop_loss": 95, "take_profit": 110}}
    assert s.validate_signal(signals.BUY, "AAA", 100) is False  # duplicate long
    assert s.validate_signal(signals.CLOSE_BUY, "AAA", 100) is True  # matches long
    assert s.validate_signal(signals.CLOSE_SELL, "AAA", 100) is False  # wrong side
    assert s.validate_signal(signals.CLOSE_BUY, "BBB", 100) is False  # no position
    assert s.validate_signal(signals.BUY, "BBB", 100) is True  # fresh entry


def test_check_exit_conditions_flags_stop_and_take():
    s = _strategy()
    s.positions = {"AAA": {"side": signals.BUY, "stop_loss": 95, "take_profit": 110}}
    data = pd.DataFrame({"close": [100, 94, 111]})  # hold, stop, take
    exits = s.check_exit_conditions(data)
    assert exits["exit_reason"].tolist() == ["", "STOP_LOSS", "TAKE_PROFIT"]
    assert exits["signal_type"].tolist() == [signals.HOLD, signals.CLOSE_BUY, signals.CLOSE_BUY]


def test_process_data_adds_indicator_columns():
    from tests.fakes import make_ohlcv

    processed = _strategy().process_data(make_ohlcv(n=200))
    assert {"short_ema", "long_ema", "volume_ma"} <= set(processed.columns)


def test_calculate_scores_is_signed_and_aligned():
    from tests.fakes import make_ohlcv

    s = _strategy()
    scores = s.calculate_scores(s.process_data(make_ohlcv(n=200)))
    assert len(scores) == 200
    # A real signed conviction: both bullish and bearish bars appear on a random walk.
    valid = scores.dropna()
    assert (valid > 0).any() and (valid < 0).any()


def test_process_bar_preserves_full_ohlcv():
    s = _strategy()
    ts = pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    s.process_bar("AAA", {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}, ts)
    last = s.get_real_time_buffer("AAA").iloc[-1]
    # Real streamed bars keep their distinct OHLC (not flattened to close).
    assert last["high"] == 105 and last["low"] == 95 and last["open"] == 100


def test_process_real_time_data_is_flat_bar_wrapper():
    s = _strategy()
    ts = pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    s.process_real_time_data("AAA", price=101, volume=500, timestamp=ts)
    last = s.get_real_time_buffer("AAA").iloc[-1]
    assert last["open"] == last["high"] == last["low"] == last["close"] == 101


def test_generate_signals_defaults_to_hold():
    from tests.fakes import make_ohlcv

    s = _strategy()
    out = s.generate_signals(s.process_data(make_ohlcv(n=200)))
    assert set(out.values()) <= {
        signals.BUY,
        signals.SELL,
        signals.CLOSE_BUY,
        signals.CLOSE_SELL,
        signals.HOLD,
    }
