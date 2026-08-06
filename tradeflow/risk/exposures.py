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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tradeflow.indicators import indicators

FACTOR_NAMES = ["market", "momentum", "volatility", "size"]


def build_factor_exposures(
    bars: Dict[str, pd.DataFrame],
    benchmark_bars: Optional[pd.DataFrame],
    *,
    momentum_window: int = 126,
    momentum_skip: int = 21,
    vol_window: int = 60,
    as_of: Optional[datetime] = None,
    factors: Optional[List[str]] = None,
    betas: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build the cross-sectionally standardized exposure matrix ``X`` (symbols × factors).

    Names without enough history for every requested factor are dropped (the
    cross-section just has fewer names). Returns an empty frame if fewer than two
    names qualify. ``factors`` selects a subset of :data:`FACTOR_NAMES` (default:
    all); the history requirement adapts — momentum needs the longest window, so a
    subset without it keeps names a full build would drop. ``betas`` supplies
    precomputed per-name betas for the market factor (a Series by symbol), skipping
    the per-name regression when the caller already ran it.
    """
    wanted = list(FACTOR_NAMES) if factors is None else list(factors)
    unknown = [f for f in wanted if f not in FACTOR_NAMES]
    if unknown:
        raise ValueError(f"unknown factors {unknown}; available: {FACTOR_NAMES}")
    if not wanted:
        return pd.DataFrame()

    min_bars = momentum_window + momentum_skip + 1 if "momentum" in wanted else vol_window + 1
    bench_close = benchmark_bars["close"] if benchmark_bars is not None and not benchmark_bars.empty else None
    rows: Dict[str, Dict[str, float]] = {}
    for symbol, frame in bars.items():
        if frame is None or len(frame) < min_bars:
            continue
        close = frame["close"]
        row: Dict[str, float] = {}
        if "market" in wanted:
            known = betas.get(symbol) if betas is not None else None
            if known is not None and known == known:  # not None, not NaN
                row["market"] = float(known)
            elif bench_close is not None:
                row["market"] = indicators.calculate_beta(close, bench_close)
            else:
                row["market"] = 1.0
        if "momentum" in wanted:
            row["momentum"] = (
                close.iloc[-1 - momentum_skip] / close.iloc[-1 - momentum_skip - momentum_window] - 1.0
            )
        if "volatility" in wanted:
            returns = close.tail(vol_window + 1).pct_change().dropna()
            row["volatility"] = float(returns.std())
        if "size" in wanted:
            dollar_volume = float(close.iloc[-1] * frame["volume"].tail(vol_window).mean())
            row["size"] = np.log(dollar_volume) if dollar_volume > 0 else np.nan
        rows[symbol] = row

    frame = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=wanted).dropna()
    if len(frame) < 2:
        return frame.iloc[0:0]
    # Cross-sectional z-score per factor (unit dispersion, mean 0).
    std = frame.std(ddof=0).replace(0.0, 1.0)
    return (frame - frame.mean()) / std
