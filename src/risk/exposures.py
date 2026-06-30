"""Factor exposures - the matrix X of each name's loading on each factor.

A structural risk model decomposes returns as ``r = X f + u``: common factor
exposures ``X`` times factor returns ``f``, plus idiosyncratic ``u``. The starter
factor set is computable from price/volume alone (no fundamentals feed):

- **market** — beta to the benchmark.
- **momentum** — trailing return, skipping the most recent month (the classic 12-1).
- **volatility** — trailing realized volatility.
- **size** — a liquidity proxy, ``log(price · ADV)`` (dollar volume).

Exposures are **cross-sectionally standardized** (z-scored across names) so the
factors are on a comparable scale and the factor-return regression is well-posed.
"""

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.indicators import indicators

FACTOR_NAMES = ["market", "momentum", "volatility", "size"]


def build_factor_exposures(
    bars: Dict[str, pd.DataFrame],
    benchmark_bars: Optional[pd.DataFrame],
    *,
    momentum_window: int = 126,
    momentum_skip: int = 21,
    vol_window: int = 60,
    as_of: Optional[datetime] = None,
) -> pd.DataFrame:
    """Build the cross-sectionally standardized exposure matrix ``X`` (symbols × factors).

    Names without enough history for every factor are dropped (the cross-section just
    has fewer names). Returns an empty frame if fewer than two names qualify.
    """
    bench_close = benchmark_bars["close"] if benchmark_bars is not None and not benchmark_bars.empty else None
    rows: Dict[str, Dict[str, float]] = {}
    for symbol, frame in bars.items():
        if frame is None or len(frame) < momentum_window + momentum_skip + 1:
            continue
        close = frame["close"]
        returns = close.pct_change().dropna()
        beta = indicators.calculate_beta(close, bench_close) if bench_close is not None else 1.0
        momentum = close.iloc[-1 - momentum_skip] / close.iloc[-1 - momentum_skip - momentum_window] - 1.0
        volatility = float(returns.tail(vol_window).std())
        dollar_volume = float(close.iloc[-1] * frame["volume"].tail(vol_window).mean())
        size = np.log(dollar_volume) if dollar_volume > 0 else np.nan
        rows[symbol] = {"market": beta, "momentum": momentum, "volatility": volatility, "size": size}

    frame = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FACTOR_NAMES).dropna()
    if len(frame) < 2:
        return frame.iloc[0:0]
    # Cross-sectional z-score per factor (unit dispersion, mean 0).
    std = frame.std(ddof=0).replace(0.0, 1.0)
    return (frame - frame.mean()) / std
