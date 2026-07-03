"""Feature producers - populate a :class:`FeaturePanel`'s columns from scanned bars.

Each producer is the cross-sectional analog of an indicator: it reads the
point-in-time bars for the universe and writes one or more feature columns. Three
today - a risk producer (beta + residual volatility), a factor-exposure producer
(the risk model's standardized exposures, for factor-neutral alphas), and a
generic score producer (apply any per-name scorer). New producers (liquidity,
transaction-cost params) slot in the same way without the consumers changing.
"""

from typing import Callable, Dict, Optional, Sequence

import pandas as pd

from src.analytics import metrics as m
from src.data.panel import FeaturePanel
from src.indicators import indicators

#: A scorer reads one symbol's (as-of-sliced) bars and returns a continuous score.
Scorer = Callable[[pd.DataFrame], float]

#: Panel column prefix for factor-exposure features (``exp_market``, ``exp_size``, ...).
EXPOSURE_PREFIX = "exp_"


def add_risk_features(
    panel: FeaturePanel,
    bars: Dict[str, pd.DataFrame],
    benchmark_bars: Optional[pd.DataFrame],
    periods_per_year: float,
    default_residual_vol: float = 0.20,
) -> FeaturePanel:
    """Write ``beta`` and ``residual_vol`` columns (annualized) for each name.

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


def add_factor_exposure_features(
    panel: FeaturePanel,
    bars: Dict[str, pd.DataFrame],
    benchmark_bars: Optional[pd.DataFrame],
    factors: Sequence[str],
) -> FeaturePanel:
    """Write standardized risk-model factor exposures as ``exp_<factor>`` columns.

    The columns come from the same exposure builder the factor risk model uses
    (:func:`src.risk.exposures.build_factor_exposures`), so "factor-neutral" in the
    alpha pipeline means neutral to the *risk model's* factors — one definition,
    both places. The market factor reuses the panel's ``beta`` column when present
    (same regression, already run by :func:`add_risk_features`). If the build
    qualifies fewer than two names, **no columns are written** — the refinement then
    falls back to plain-beta neutralization rather than silently doing nothing.
    Names missing a single factor get a NaN exposure (mean-imputed downstream).
    """
    from src.risk.exposures import build_factor_exposures

    if not list(factors):
        return panel
    betas = panel.get("beta") if panel.has("beta") else None
    frame = build_factor_exposures(bars, benchmark_bars, factors=list(factors), betas=betas)
    if frame.empty:
        return panel
    for name in frame.columns:
        panel.set(f"{EXPOSURE_PREFIX}{name}", frame[name])
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
