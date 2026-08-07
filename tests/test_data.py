"""Tests for the cross-sectional data substrate: scan seam + FeaturePanel + producers."""

from datetime import datetime

import numpy as np
import pandas as pd

from tests.fakes import DictMarketData, make_ohlcv
from tradeflow.data import ClientBarSource, FeaturePanel, add_risk_features, add_score_feature
from tradeflow.data.scan import slice_to_as_of
from tradeflow.marketdata.client import MarketDataClient

SYMBOLS = ["AAA", "BBB", "SPY"]


def _client():
    data = {s: make_ohlcv(n=200, seed=i, freq="1D") for i, s in enumerate(SYMBOLS)}
    return MarketDataClient(DictMarketData(data)), data


# --- scan seam ---------------------------------------------------------------
def test_scan_never_returns_bars_after_as_of():
    client, data = _client()
    cutoff = data["AAA"].index[120]
    as_of = cutoff.to_pydatetime()

    bars = ClientBarSource(client).scan(SYMBOLS, "1Day", as_of, lookback_days=365)
    for symbol, frame in bars.items():
        assert frame.index.max() <= cutoff, f"{symbol} leaked a post-as_of bar"


def test_slice_to_as_of_handles_naive_cutoff_on_tz_index():
    frame = make_ohlcv(n=50, seed=0, freq="1D")  # tz-aware index
    cutoff = frame.index[20]
    sliced = slice_to_as_of(frame, cutoff.to_pydatetime().replace(tzinfo=None))
    assert sliced.index.max() <= cutoff
    assert len(sliced) == 21


# --- panel -------------------------------------------------------------------
def test_panel_set_aligns_to_universe_and_get_roundtrips():
    panel = FeaturePanel.for_universe(datetime(2024, 6, 1), ["AAA", "BBB"])
    panel.set("score", {"AAA": 1.0})  # BBB missing -> NaN, not an error
    assert panel.get("score")["AAA"] == 1.0
    assert pd.isna(panel.get("score")["BBB"])
    assert panel.symbols == ["AAA", "BBB"]
    assert "score" in panel.columns


# --- producers ---------------------------------------------------------------
def test_add_risk_features_sets_beta_and_residual_vol():
    _, data = _client()
    bars = {s: data[s] for s in ("AAA", "BBB")}
    panel = FeaturePanel.for_universe(datetime(2024, 6, 1), ["AAA", "BBB"])
    add_risk_features(panel, bars, data["SPY"], periods_per_year=252.0)

    assert panel.meta["benchmark_available"] is True
    assert panel.has("beta") and panel.has("residual_vol")
    assert (panel.get("residual_vol").dropna() >= 0).all()


def test_add_risk_features_falls_back_without_benchmark():
    _, data = _client()
    bars = {s: data[s] for s in ("AAA", "BBB")}
    panel = FeaturePanel.for_universe(datetime(2024, 6, 1), ["AAA", "BBB"])
    add_risk_features(panel, bars, benchmark_bars=None, periods_per_year=252.0)

    assert panel.meta["benchmark_available"] is False
    assert (panel.get("beta") == 1.0).all()  # beta unknown -> 1.0


def test_add_score_feature_applies_scorer():
    _, data = _client()
    bars = {s: data[s] for s in ("AAA", "BBB")}
    panel = FeaturePanel.for_universe(datetime(2024, 6, 1), ["AAA", "BBB"])
    add_score_feature(panel, lambda frame: float(frame["close"].iloc[-1]), bars)

    assert panel.get("score")["AAA"] == float(np.float64(data["AAA"]["close"].iloc[-1]))
