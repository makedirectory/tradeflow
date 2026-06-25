"""Feature producers - populate a :class:`FeaturePanel`'s columns from scanned bars.

Each producer is the cross-sectional analog of an indicator: it reads the
point-in-time bars for the universe and writes one or more feature columns. Today
there are two - a risk producer (beta + residual volatility, the seed of a future
risk model) and a generic score producer (apply any per-name scorer). New
producers (factor exposures, liquidity, transaction-cost params) slot in the same
way without the consumers above changing.
"""

from typing import Callable, Dict, Optional

import pandas as pd

from src.analytics import metrics as m
from src.data.panel import FeaturePanel
from src.indicators import indicators

#: A scorer reads one symbol's (as-of-sliced) bars and returns a continuous score.
Scorer = Callable[[pd.DataFrame], float]


def add_risk_features(
    panel: FeaturePanel,
    bars: Dict[str, pd.DataFrame],
    benchmark_bars: Optional[pd.DataFrame],
    periods_per_year: float,
    default_residual_vol: float = 0.20,
) -> FeaturePanel:
    """Write ``beta`` and ``residual_vol`` columns (annualised) for each name.

    Residual volatility strips the benchmark-explained part of each return; it is
    the ``sigma`` the alpha-scaling identity needs. With no benchmark series
    available it falls back to total volatility (beta unknown, set to 1.0) and
    records ``benchmark_available=False`` in the panel meta.
    """
    available = benchmark_bars is not None and not benchmark_bars.empty
    betas: Dict[str, float] = {}
    vols: Dict[str, float] = {}
    for symbol in panel.symbols:
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            continue
        close = frame["close"]
        if available:
            beta = indicators.calculate_beta(close, benchmark_bars["close"])
            vol = indicators.calculate_residual_volatility(
                close, benchmark_bars["close"], beta, periods_per_year
            )
        else:
            beta = 1.0
            vol = m.annualized_volatility(close.pct_change().dropna(), int(round(periods_per_year)))
        betas[symbol] = beta
        vols[symbol] = vol or default_residual_vol

    panel.set("beta", betas)
    panel.set("residual_vol", vols)
    panel.meta["benchmark_available"] = available
    return panel


def add_score_feature(
    panel: FeaturePanel,
    scorer: Scorer,
    bars: Dict[str, pd.DataFrame],
    column: str = "score",
) -> FeaturePanel:
    """Write a continuous ``score`` column by applying ``scorer`` to each name's bars."""
    scores: Dict[str, float] = {}
    for symbol in panel.symbols:
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            continue
        value = scorer(frame)
        if value is not None and not pd.isna(value):
            scores[symbol] = float(value)
    panel.set(column, scores)
    return panel
