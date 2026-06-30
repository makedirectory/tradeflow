"""Continuous alphas: turn a score column into a comparable residual-return forecast.

The central active-management move: convert a raw, arbitrary-scale signal into an
*alpha* - a forecast of residual return in annualised-return units, the same for
every name - so views can be pooled, ranked, and sized by a mean-variance
optimiser. Research-clock only: alphas forecast, they never trade.

The refinement runs over a :class:`~src.data.panel.FeaturePanel`: a scorer
(:mod:`src.alphas.scorers`) fills the ``score`` column, :func:`refine_alpha` adds
``z`` and ``alpha`` via the identity ``alpha = sigma * IC * z``, and
:func:`panel_to_alphas` exports the ranked :class:`Alpha` rows.
"""

from src.alphas.base import (
    DEFAULT_IC,
    DEFAULT_MIN_UNIVERSE,
    Alpha,
    AlphaContext,
    panel_to_alphas,
    refine_alpha,
)
from src.alphas.combine import (
    SignalMeasurement,
    combination_weights,
    combine_scores,
    combined_ic,
    combined_score,
    effective_ic,
    measure_signals,
    shrink_ic,
)
from src.alphas.scorers import Scorer, scanner_scorer, signal_scorer, strategy_scorer

__all__ = [
    "Alpha",
    "AlphaContext",
    "refine_alpha",
    "panel_to_alphas",
    "Scorer",
    "strategy_scorer",
    "signal_scorer",
    "scanner_scorer",
    "DEFAULT_IC",
    "DEFAULT_MIN_UNIVERSE",
    "SignalMeasurement",
    "measure_signals",
    "combined_score",
    "combination_weights",
    "combined_ic",
    "effective_ic",
    "shrink_ic",
    "combine_scores",
]
