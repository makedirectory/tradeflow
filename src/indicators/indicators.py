"""Technical indicators implemented with pure pandas/numpy - the squiggly lines we
all pretend predict the future.

This module is the single source of truth for indicator math in the project.
It deliberately uses **no compiled dependencies** (no TA-Lib, no tulipy): every
function here is plain pandas/numpy so the project installs with `pip install`
alone and the Docker image needs no build toolchain.

All functions are pure: they take Series/DataFrames in and return new
Series/DataFrames out, leaving the inputs untouched.
"""

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


def calculate_beta(symbol_close: pd.Series, benchmark_close: pd.Series) -> float:
    """Beta of a symbol's returns against a benchmark's returns.

    Beta = cov(symbol, benchmark) / var(benchmark), computed on aligned
    period-over-period returns. Returns 1.0 (market-neutral) for degenerate input
    (too few overlapping points, or a flat benchmark).

    Args:
        symbol_close: Close-price series for the symbol.
        benchmark_close: Close-price series for the benchmark (e.g. SPY).

    Returns:
        The beta coefficient.
    """
    prices = pd.concat([symbol_close, benchmark_close], axis=1, keys=["symbol", "benchmark"]).dropna()
    if len(prices) < 3:
        return 1.0

    returns = prices.pct_change().dropna()
    benchmark_var = returns["benchmark"].var()
    if benchmark_var == 0 or returns.empty:
        return 1.0

    covariance = returns["symbol"].cov(returns["benchmark"])
    return float(covariance / benchmark_var)


def calculate_residual_volatility(
    symbol_close: pd.Series,
    benchmark_close: pd.Series,
    beta: float,
    periods_per_year: float = 252.0,
) -> float:
    """Annualised volatility of a symbol's *residual* returns vs a benchmark.

    Residual return strips out the part of the move that beta to the benchmark
    explains: ``r_resid = r_symbol - beta * r_benchmark``. Its annualised std is the
    ``sigma`` the alpha-scaling identity (alpha = sigma * IC * z) needs - the risk
    of the bet that *isn't* just market exposure.

    Computed on aligned period-over-period returns. Returns 0.0 for degenerate
    input (too few overlapping points). ``periods_per_year`` annualises to the
    sampling frequency of the series passed in (daily bars -> 252).

    Args:
        symbol_close: Close-price series for the symbol.
        benchmark_close: Close-price series for the benchmark (e.g. SPY).
        beta: The symbol's beta to the benchmark (see :func:`calculate_beta`).
        periods_per_year: Annualisation factor for the bar frequency.

    Returns:
        Annualised residual volatility (a fraction, e.g. 0.20 == 20%/yr).
    """
    prices = pd.concat([symbol_close, benchmark_close], axis=1, keys=["symbol", "benchmark"]).dropna()
    if len(prices) < 3:
        return 0.0

    returns = prices.pct_change().dropna()
    if returns.empty:
        return 0.0

    residual = returns["symbol"] - beta * returns["benchmark"]
    std = residual.std()
    if not (std == std):  # NaN guard
        return 0.0
    return float(std * (periods_per_year**0.5))


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
