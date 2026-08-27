"""Scanner tests: volume scanner signal logic and base-class behavior."""

from datetime import datetime

import pandas as pd
import pytest

from tests.fakes import DictMarketData, FakeMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.scanners.base import SCANNER_BUY, SCANNER_HOLD
from tradeflow.scanners.symbol_scanner import SymbolScanner
from tradeflow.scanners.volume_scanner import VolumeScannerStrategy
from tradeflow.services import analysis
from tradeflow.utils.timeutils import NEW_YORK


def _scanner(**overrides):
    config = {p: spec["default"] for p, spec in VolumeScannerStrategy.PARAM_RANGES.items()}
    config.update(overrides)
    return VolumeScannerStrategy(config)


def test_validate_config_fills_defaults():
    scanner = VolumeScannerStrategy({})
    assert (
        scanner.config["volume_threshold"]
        == VolumeScannerStrategy.PARAM_RANGES["volume_threshold"]["default"]
    )


def test_validate_config_rejects_out_of_range():
    with pytest.raises(ValueError):
        VolumeScannerStrategy({"volume_ma_period": 9999})


def test_process_data_adds_columns():
    df = pd.DataFrame(
        {
            "open": [100] * 6,
            "high": [101] * 6,
            "low": [99] * 6,
            "close": [100.5] * 6,
            "volume": [1e6] * 6,
        }
    )
    processed = _scanner(volume_ma_period=5).process_data(df)
    assert {"volume_ratio", "price_change"} <= set(processed.columns)


def test_volume_spike_flags_buy_on_up_bar():
    # Five flat bars, then a big up-bar with a clear price move on high volume.
    df = pd.DataFrame(
        {
            "open": [100] * 6,
            "high": [101, 101, 101, 101, 101, 110],
            "low": [99] * 6,
            "close": [100, 100, 100, 100, 100, 108],  # +8% on the last bar
            "volume": [1e6, 1e6, 1e6, 1e6, 1e6, 1e7],  # 10x volume spike
        }
    )
    scanner = _scanner(
        volume_ma_period=5, volume_threshold=1.5, price_change_threshold=0.5, min_volume=100_000
    )
    signals_df = scanner.generate_signals_df(scanner.process_data(df))
    assert scanner.latest_signal(signals_df) == SCANNER_BUY


def test_latest_signal_hold_when_quiet():
    df = pd.DataFrame(
        {
            "open": [100] * 6,
            "high": [100.1] * 6,
            "low": [99.9] * 6,
            "close": [100] * 6,
            "volume": [1e6] * 6,
        }
    )
    scanner = _scanner(volume_ma_period=5)
    signals_df = scanner.generate_signals_df(scanner.process_data(df))
    assert scanner.latest_signal(signals_df) == SCANNER_HOLD


def test_evaluate_forward_returns_metric_keys():
    idx = pd.date_range("2024-01-02", periods=6, freq="D")
    signals_df = pd.DataFrame(
        {"signal": [SCANNER_BUY, SCANNER_HOLD, SCANNER_HOLD, SCANNER_HOLD, SCANNER_HOLD, SCANNER_HOLD]},
        index=idx,
    )
    forward = pd.DataFrame({"close": [100, 102, 104, 103, 105, 106]}, index=idx)
    metrics = _scanner().evaluate_forward(signals_df, forward, hold_bars=3)
    assert {"hit_rate", "avg_return", "total_signals", "sharpe_ratio", "profit_factor"} <= metrics.keys()


def test_symbol_scanner_can_scan_at_a_historical_as_of():
    scanner = SymbolScanner(MarketDataClient(FakeMarketData(["AAA"], n=60, freq="1D")), "volume")
    as_of = NEW_YORK.localize(datetime(2024, 6, 1, 16, 0))

    start, end = scanner._scan_window("1Day", as_of=as_of)

    assert end == as_of
    assert start < as_of


def test_symbol_scanner_localizes_naive_historical_as_of():
    scanner = SymbolScanner(MarketDataClient(FakeMarketData(["AAA"], n=60, freq="1D")), "volume")
    as_of = datetime(2024, 6, 1, 16, 0)

    _, end = scanner._scan_window("1Day", as_of=as_of)

    assert end == NEW_YORK.localize(as_of)


def test_symbol_scanner_ignores_bars_after_as_of():
    index = pd.date_range("2024-01-02 16:00", periods=16, freq="D", tz=NEW_YORK)
    frame = pd.DataFrame(
        {
            "open": [100.0] * 16,
            "high": [101.0] * 16,
            "low": [99.0] * 16,
            "close": [100.0] * 15 + [110.0],
            "volume": [1_000_000.0] * 15 + [12_000_000.0],
        },
        index=index,
    )
    as_of = index[-2].to_pydatetime()
    scanner = SymbolScanner(MarketDataClient(DictMarketData({"AAA": frame})), "volume")

    flagged = scanner.scan(["AAA"], as_of=as_of)

    assert flagged == []


def test_run_scan_reports_the_clock_it_resolved_at_not_the_one_requested():
    """The payload echoed the caller's argument.

    A naive `2024-06-01` came back as a bare date while the scan had actually run at
    `2024-06-01` New York, and an omitted `as_of` came back as null rather than as the
    wall-clock now it resolved to. A selection clock reported differently from the one
    applied is worse than none, because it reads as provenance.
    """
    from tradeflow.scanners.symbol_scanner import resolve_scan_clock

    client = MarketDataClient(FakeMarketData(["AAA"], n=60, freq="1D"))
    as_of = datetime(2024, 6, 1)

    result = analysis.run_scan(client, "volume", ["AAA"], as_of=as_of)

    assert result["as_of"] == resolve_scan_clock(as_of).isoformat()
    assert result["as_of"].endswith("-04:00")  # localized, not echoed back naive


def test_run_scan_names_its_clock_even_when_none_was_asked_for():
    """ "Now" is a selection clock like any other; reporting null for it hides which
    universe the scan actually saw."""
    client = MarketDataClient(FakeMarketData(["AAA"], n=60, freq="1D"))

    result = analysis.run_scan(client, "volume", ["AAA"])

    assert result["as_of"] is not None
