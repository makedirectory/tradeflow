"""Scorers - the atomic "score one name's bars" functions that feed the panel.

A scorer is just ``Callable[[pd.DataFrame], float]``: given one symbol's as-of-sliced
bars, return a continuous conviction. The panel's score producer applies a scorer
across the universe to fill the ``score`` column, which the refinement then turns
into an alpha. Keeping the source of the score behind this one-line contract is
what lets a strategy, a scanner, or (later) a combination of signals all flow
through the same alpha pipeline.

- :func:`strategy_scorer` - a strategy's own continuous conviction (its score).
  The natural, richest source: it's the same number the trade clock's signal is
  derived from.
- :func:`signal_scorer` - a strategy's *discrete* direction as +1/-1/0. A lossy
  bucketing of the score; useful when only the direction is trusted.
- :func:`scanner_scorer` - a scanner's signed ``signal_strength``.
"""

from typing import Callable

import pandas as pd

from src.scanners.base import SCANNER_BUY, SCANNER_SELL, ScannerStrategy
from src.strategies import signals
from src.strategies.base import Strategy

Scorer = Callable[[pd.DataFrame], float]

#: Map a discrete trade signal to a directional score. Exit signals carry no
#: cross-sectional view (they're about an existing position), so they score 0.
_SIGNAL_SCORE = {
    signals.BUY: 1.0,
    signals.SELL: -1.0,
    signals.HOLD: 0.0,
    signals.CLOSE_BUY: 0.0,
    signals.CLOSE_SELL: 0.0,
}


def strategy_scorer(strategy: Strategy) -> Scorer:
    """Score a name by the strategy's continuous conviction on its last bar."""

    def score(frame: pd.DataFrame) -> float:
        processed = strategy.process_data(frame)
        if processed.empty:
            return 0.0
        scores = strategy.calculate_scores(processed)
        if scores.empty:
            return 0.0
        value = scores.iloc[-1]
        return 0.0 if pd.isna(value) else float(value)

    return score


def signal_scorer(strategy: Strategy) -> Scorer:
    """Score a name by the strategy's discrete direction (+1 BUY / -1 SELL / 0)."""

    def score(frame: pd.DataFrame) -> float:
        processed = strategy.process_data(frame)
        if processed.empty:
            return 0.0
        signal = strategy._latest_signal(strategy.generate_signals(processed))
        return _SIGNAL_SCORE.get(signal, 0.0)

    return score


def scanner_scorer(scanner: ScannerStrategy) -> Scorer:
    """Score a name by the scanner's ``signal_strength``, signed by its direction."""

    def score(frame: pd.DataFrame) -> float:
        processed = scanner.process_data(frame)
        if processed.empty:
            return 0.0
        signals_df = scanner.generate_signals_df(processed)
        if signals_df.empty or "signal_strength" not in signals_df:
            return 0.0
        last = signals_df.iloc[-1]
        strength = float(last.get("signal_strength", 0.0) or 0.0)
        signal = last.get("signal")
        if signal == SCANNER_BUY:
            return strength
        if signal == SCANNER_SELL:
            return -strength
        return 0.0

    return score
