"""SignalAlphaModel - derive a raw score from a strategy's discrete signal.

A :class:`~src.strategies.base.Strategy` emits ``BUY`` / ``SELL`` / ``HOLD`` for the
trade clock. That direction *is* the strategy's per-name view; this model reads it
as a raw conviction score (``BUY -> +1``, ``SELL -> -1``, everything else ``0``) so
the strategy's signal can be pooled, ranked, and scaled into a comparable alpha
without touching the strategy or the live order path.

This is the bridge for strategies that only express direction. A strategy with a
genuinely continuous conviction is better served by :mod:`src.alphas.from_score`.
"""

from datetime import datetime
from typing import Dict

import pandas as pd

from src.alphas.base import AlphaModel, RawScore
from src.strategies import signals
from src.strategies.base import Strategy

#: Map a discrete trade signal onto a directional raw score. Exit signals carry no
#: cross-sectional view (they're about an existing position, not attractiveness),
#: so they score 0 - the same as HOLD.
_SIGNAL_SCORE = {
    signals.BUY: 1.0,
    signals.SELL: -1.0,
    signals.HOLD: 0.0,
    signals.CLOSE_BUY: 0.0,
    signals.CLOSE_SELL: 0.0,
}


class SignalAlphaModel(AlphaModel):
    """Wrap a strategy: its latest discrete signal becomes a +/-1/0 raw score."""

    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def raw_scores(self, bars: Dict[str, pd.DataFrame], as_of: datetime) -> list[RawScore]:
        """Score each name by the strategy's signal on its last bar <= ``as_of``."""
        scores: list[RawScore] = []
        for symbol, frame in bars.items():
            if frame is None or frame.empty:
                continue
            processed = self.strategy.process_data(frame)
            if processed.empty:
                continue
            signal = self.strategy._latest_signal(self.strategy.generate_signals(processed))
            scores.append(RawScore(symbol=symbol, score=_SIGNAL_SCORE.get(signal, 0.0), as_of=as_of))
        return scores
