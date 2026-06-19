"""Scanner tests: volume scanner signal logic and base-class behaviour."""

import pandas as pd
import pytest

from src.scanners.base import SCANNER_BUY, SCANNER_HOLD, SCANNER_SELL
from src.scanners.volume_scanner import VolumeScannerStrategy


def _scanner(**overrides):
    config = {p: spec["default"] for p, spec in VolumeScannerStrategy.PARAM_RANGES.items()}
    config.update(overrides)
    return VolumeScannerStrategy(config)


def test_validate_config_fills_defaults():
    scanner = VolumeScannerStrategy({})
    assert scanner.config["volume_threshold"] == VolumeScannerStrategy.PARAM_RANGES["volume_threshold"]["default"]


def test_validate_config_rejects_out_of_range():
    with pytest.raises(ValueError):
        VolumeScannerStrategy({"volume_ma_period": 9999})


def test_process_data_adds_columns():
    df = pd.DataFrame({
        "open": [100] * 6, "high": [101] * 6, "low": [99] * 6,
        "close": [100.5] * 6, "volume": [1e6] * 6,
    })
    processed = _scanner(volume_ma_period=5).process_data(df)
    assert {"volume_ratio", "price_change"} <= set(processed.columns)


def test_volume_spike_flags_buy_on_up_bar():
    # Five flat bars, then a big up-bar with a clear price move on high volume.
    df = pd.DataFrame({
        "open": [100] * 6,
        "high": [101, 101, 101, 101, 101, 110],
        "low": [99] * 6,
        "close": [100, 100, 100, 100, 100, 108],     # +8% on the last bar
        "volume": [1e6, 1e6, 1e6, 1e6, 1e6, 1e7],    # 10x volume spike
    })
    scanner = _scanner(volume_ma_period=5, volume_threshold=1.5,
                       price_change_threshold=0.5, min_volume=100_000)
    signals_df = scanner.generate_signals_df(scanner.process_data(df))
    assert scanner.latest_signal(signals_df) == SCANNER_BUY


def test_latest_signal_hold_when_quiet():
    df = pd.DataFrame({
        "open": [100] * 6, "high": [100.1] * 6, "low": [99.9] * 6,
        "close": [100] * 6, "volume": [1e6] * 6,
    })
    scanner = _scanner(volume_ma_period=5)
    signals_df = scanner.generate_signals_df(scanner.process_data(df))
    assert scanner.latest_signal(signals_df) == SCANNER_HOLD


def test_evaluate_forward_returns_metric_keys():
    idx = pd.date_range("2024-01-02", periods=6, freq="D")
    signals_df = pd.DataFrame({"signal": [SCANNER_BUY, SCANNER_HOLD, SCANNER_HOLD,
                                          SCANNER_HOLD, SCANNER_HOLD, SCANNER_HOLD]}, index=idx)
    forward = pd.DataFrame({"close": [100, 102, 104, 103, 105, 106]}, index=idx)
    metrics = _scanner().evaluate_forward(signals_df, forward, hold_bars=3)
    assert {"hit_rate", "avg_return", "total_signals", "sharpe_ratio", "profit_factor"} <= metrics.keys()
