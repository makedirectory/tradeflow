"""Alpha types and the refinement that turns a score column into a forecast.

An *alpha* is a forecast of **residual return** (return in excess of what beta to
the benchmark explains), annualised and on the same scale for every name, so it is
directly comparable across symbols and directly consumable by a mean-variance
optimiser.

The refinement runs over a :class:`~src.data.panel.FeaturePanel`: it reads the
cross-section's ``score`` and ``residual_vol`` columns (and ``beta`` when
neutralising) and writes ``z`` and ``alpha`` columns back. One implementation, one
place - whatever produced the score (a strategy, a scanner, a combination of
signals) flows through the same pipeline.

Research clock only: it forecasts, it never reads a realised forward return, it
places no orders.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd

from src.alphas import refine
from src.data.panel import FeaturePanel

#: Conservative default information coefficient. The absolute scale of alphas is
#: only as good as this prior until IC is measured from realised outcomes; the
#: *relative* sizing across names is correct regardless (IC is a common scalar).
DEFAULT_IC = 0.03

#: Below this many names the cross-sectional z-score and winsorize quantiles are
#: unstable, so the pipeline falls back to demean-only and flags low confidence.
DEFAULT_MIN_UNIVERSE = 10


@dataclass
class Alpha:
    """A refined residual-return forecast: annualised, benchmark-relative."""

    symbol: str
    alpha: float  # expected annualised residual return, e.g. 0.04 = +4%/yr
    as_of: datetime
    residual_vol: float  # sigma used in scaling (annualised), for audit
    ic: float  # information coefficient used in scaling
    raw_z: float  # standardised score that produced it


@dataclass
class AlphaContext:
    """Knobs for the refinement (everything that isn't a panel column)."""

    ic: float = DEFAULT_IC
    default_residual_vol: float = 0.20  # fallback when a name has no risk estimate
    neutralize: bool = False  # regress out the benchmark-beta exposure
    winsorize_limits: tuple = (0.025, 0.975)
    alpha_cap_std: float = 3.0
    min_universe: int = DEFAULT_MIN_UNIVERSE


def refine_alpha(
    panel: FeaturePanel,
    context: AlphaContext,
    score_col: str = "score",
) -> FeaturePanel:
    """Refine a panel's ``score`` column into ``z`` and ``alpha`` columns.

    Cross-sectional at this one rebalance: winsorize, z-score, optionally neutralize
    against the ``beta`` exposure, scale by ``sigma * IC``, then cap. On a thin
    universe (``< context.min_universe`` scored names) winsorize and the unit-std
    scaling are skipped in favour of demean-only, and ``panel.meta['low_confidence']``
    is set. Reads ``residual_vol`` for ``sigma`` (falling back to the context default).
    """
    if not panel.has(score_col):
        panel.meta["low_confidence"] = False
        return panel

    s = panel.get(score_col).dropna()
    if s.empty:
        panel.meta["low_confidence"] = False
        return panel

    thin = len(s) < context.min_universe
    panel.meta["low_confidence"] = thin

    # 1-2. Winsorize then standardise. On thin universes the quantiles and the
    #      cross-sectional std are unreliable, so demean-only (no scaling).
    if thin:
        z = refine.demean(s)
    else:
        z = refine.zscore(refine.winsorize(s, *context.winsorize_limits))

    # 3. Optional benchmark-beta neutralisation (regress z on the beta exposure).
    if context.neutralize and not thin and panel.has("beta"):
        exposures = panel.get("beta").reindex(z.index).to_frame("beta")
        z = refine.neutralize(z, exposures)

    # 4. Scale to a residual-return forecast via alpha_i = sigma_i * IC * z_i.
    if panel.has("residual_vol"):
        vol = panel.get("residual_vol").reindex(z.index).fillna(context.default_residual_vol)
    else:
        vol = pd.Series(context.default_residual_vol, index=z.index)
    alpha = refine.scale_to_alpha(z, vol, context.ic)

    # 5. Final sanity cap (meaningless on a thin, unscaled vector).
    if not thin:
        alpha = refine.cap(alpha, context.alpha_cap_std)

    panel.set("z", z)
    panel.set("alpha", alpha)
    return panel


def panel_to_alphas(panel: FeaturePanel, context: AlphaContext) -> List[Alpha]:
    """Export the refined panel rows as :class:`Alpha` records (alpha-ranked)."""
    if not panel.has("alpha"):
        return []
    rows = []
    for symbol in panel.symbols:
        alpha = panel.get("alpha").get(symbol)
        if alpha is None or pd.isna(alpha):
            continue
        z = panel.get("z").get(symbol) if panel.has("z") else float("nan")
        vol = (
            panel.get("residual_vol").get(symbol)
            if panel.has("residual_vol")
            else context.default_residual_vol
        )
        rows.append(
            Alpha(
                symbol=symbol,
                alpha=float(alpha),
                as_of=panel.as_of,
                residual_vol=float(vol) if vol == vol else context.default_residual_vol,
                ic=context.ic,
                raw_z=float(z) if z == z else 0.0,
            )
        )
    return sorted(rows, key=lambda a: a.alpha, reverse=True)
