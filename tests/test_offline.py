"""Offline end-to-end tests.

These run the full stack against synthetic data (no network/keys/alpaca), which
is possible precisely because every layer depends on the broker/data-provider
abstractions rather than a concrete vendor.
"""

from datetime import datetime

import pandas as pd
import pytest

from src.analytics import metrics
from src.engine.backtest import BacktestEngine
from src.indicators import indicators
from src.marketdata.client import MarketDataClient
from src.optimization.optimizer import ParameterOptimizer
from src.scanners.symbol_scanner import SymbolScanner
from src.strategies.volume_spike import VolumeSpikeStrategy
from tests.fakes import FakeMarketData, make_ohlcv

SYMBOLS = ["AAA", "BBB", "CCC"]
START, END = datetime(2024, 1, 2), datetime(2024, 2, 1)


# --- indicators -------------------------------------------------------------
def test_rsi_bounds():
    rsi = indicators.calculate_rsi(make_ohlcv()["close"], period=14).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_volume_spike_is_boolean():
    df = make_ohlcv()
    spikes = indicators.calculate_volume_spike(df["volume"], df["close"])
    assert spikes.dtype == bool and len(spikes) == len(df)


def test_metric_primitives_handle_degenerate_input():
    assert metrics.sharpe_ratio([]) == 0.0
    assert metrics.max_drawdown([]) == 0.0
    assert metrics.profit_factor(pd.Series([1.0, 2.0])) == float("inf")


# --- backtest engine --------------------------------------------------------
def test_backtest_produces_metrics_and_consistent_capital():
    data_client = MarketDataClient(FakeMarketData(SYMBOLS))
    strategy = VolumeSpikeStrategy.create_with_defaults()

    result = BacktestEngine(strategy, data_client).run(SYMBOLS, START, END, 100_000)

    assert {"total_return", "sharpe_ratio", "max_drawdown", "total_trades"} <= result.metrics.keys()
    expected_final = 100_000 + (result.trades["pnl"].sum() if not result.trades.empty else 0.0)
    assert result.final_capital == pytest.approx(expected_final)
    assert result.equity_curve[0] == 100_000


def test_backtest_metrics_complete_and_json_serializable():
    """Every declared metric key is present and the dict survives JSON round-trip."""
    import json

    from src.analytics.performance import FLAG_KEYS, METRIC_KEYS

    data_client = MarketDataClient(FakeMarketData(SYMBOLS))
    strategy = VolumeSpikeStrategy.create_with_defaults()
    result = BacktestEngine(strategy, data_client).run(SYMBOLS, START, END, 100_000)

    for key in (*METRIC_KEYS, *FLAG_KEYS):
        assert key in result.metrics, f"missing metric: {key}"

    # JSON-serializable matters for the planned MCP server (Spec 003).
    restored = json.loads(json.dumps(result.metrics))
    assert restored["total_trades"] == result.metrics["total_trades"]


def test_empty_metrics_has_all_keys():
    from src.analytics.performance import FLAG_KEYS, METRIC_KEYS, empty_metrics

    empty = empty_metrics()
    for key in (*METRIC_KEYS, *FLAG_KEYS):
        assert key in empty


def test_mae_mfe_tracked_on_trades():
    data_client = MarketDataClient(FakeMarketData(SYMBOLS))
    strategy = VolumeSpikeStrategy.create_with_defaults()
    result = BacktestEngine(strategy, data_client).run(SYMBOLS, START, END, 100_000)
    if not result.trades.empty:
        assert {"mae_pct", "mfe_pct"} <= set(result.trades.columns)
        assert (result.trades["mae_pct"] >= 0).all()
        assert (result.trades["mfe_pct"] >= 0).all()


# --- scanner ----------------------------------------------------------------
def test_scanner_returns_actionable_signals():
    data_client = MarketDataClient(FakeMarketData(SYMBOLS, freq="1D"))
    flagged = SymbolScanner(data_client, "volume").scan(SYMBOLS, timeframe="1Day")
    assert isinstance(flagged, list)
    for symbol, signal in flagged:
        assert symbol in SYMBOLS
        assert signal in ("SCANNER_BUY", "SCANNER_SELL")


# --- optimizer --------------------------------------------------------------
def test_grid_search_returns_best_params():
    data_client = MarketDataClient(FakeMarketData(SYMBOLS))
    optimizer = ParameterOptimizer(VolumeSpikeStrategy, data_client, initial_capital=100_000)

    result = optimizer.grid_search(SYMBOLS, START, END, objective="sharpe_ratio", max_evals=5)

    assert not result.results.empty
    assert result.objective == "sharpe_ratio"
    assert set(result.best_params) <= set(VolumeSpikeStrategy.PARAM_RANGES)
