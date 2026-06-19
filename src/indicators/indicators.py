"""Technical indicators implemented with pure pandas/numpy.

This module is the single source of truth for indicator math in the project.
It deliberately uses **no compiled dependencies** (no TA-Lib, no tulipy): every
function here is plain pandas/numpy so the project installs with `pip install`
alone and the Docker image needs no build toolchain.

All functions are pure: they take Series/DataFrames in and return new
Series/DataFrames out, leaving the inputs untouched.
"""

import numpy as np
import pandas as pd


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-style smoothing via rolling mean).

    Args:
        data: Price series (typically close).
        period: Lookback period.

    Returns:
        RSI values in the range [0, 100].
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        period: Lookback period.

    Returns:
        ATR series.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_volume_spike(
    volume: pd.Series,
    price: pd.Series,
    volume_ma_period: int = 20,
    volume_threshold: float = 2.0,
    price_change_threshold: float = 0.02,
) -> pd.Series:
    """Detect volume spikes that coincide with meaningful price movement.

    A spike requires both (a) volume above ``volume_threshold`` times its moving
    average and (b) an absolute one-bar price change above
    ``price_change_threshold`` (expressed as a fraction, e.g. 0.02 == 2%).

    Args:
        volume: Volume series.
        price: Price series (typically close).
        volume_ma_period: Window for the volume moving average baseline.
        volume_threshold: Multiplier over the volume MA that defines a spike.
        price_change_threshold: Minimum absolute fractional price change.

    Returns:
        Boolean series, True where a qualifying spike occurred.
    """
    volume_ma = volume.rolling(window=volume_ma_period).mean()
    price_change = price.pct_change().abs()

    spikes = (volume > volume_ma * volume_threshold) & (price_change > price_change_threshold)
    # pct_change/rolling produce NaN at the head; treat those as "no spike".
    return spikes.fillna(False)
