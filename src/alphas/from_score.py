"""ScoreAlphaModel - wrap an already-continuous per-name score.

Some sources express conviction directly as a number, not a discrete signal: a
scanner's ``signal_strength``, a momentum z, an inverse-vol weight. This model
takes a ``scorer`` callable that reads one name's OHLCV frame and returns that
continuous score, then feeds it through the shared refinement pipeline. The raw
scale is irrelevant - the z-score makes scores comparable across names.

:func:`scanner_scorer` builds a scorer from a :class:`ScannerStrategy`, signing the
scanner's (non-negative) ``signal_strength`` by its scan direction so a SELL flag
becomes a negative conviction.
"""

from datetime import datetime
from typing import Callable, Dict

import pandas as pd

from src.alphas.base import AlphaModel, RawScore
from src.scanners.base import SCANNER_BUY, SCANNER_SELL, ScannerStrategy

#: A scorer reads one symbol's (as-of-sliced) frame and returns a continuous score.
Scorer = Callable[[pd.DataFrame], float]


class ScoreAlphaModel(AlphaModel):
    """Wrap a continuous per-name scorer into the alpha pipeline."""

    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    def raw_scores(self, bars: Dict[str, pd.DataFrame], as_of: datetime) -> list[RawScore]:
        scores: list[RawScore] = []
        for symbol, frame in bars.items():
            if frame is None or frame.empty:
                continue
            value = self.scorer(frame)
            if value is None or pd.isna(value):
                continue
            scores.append(RawScore(symbol=symbol, score=float(value), as_of=as_of))
        return scores


def scanner_scorer(scanner: ScannerStrategy) -> Scorer:
    """Build a scorer from a scanner: signed ``signal_strength`` on the last bar.

    The scanner's strength is non-negative; we sign it by the latest scan signal
    (BUY -> positive, SELL -> negative, otherwise 0) so the resulting score carries
    direction as well as magnitude.
    """

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
