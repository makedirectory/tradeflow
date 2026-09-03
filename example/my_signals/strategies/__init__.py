"""Strategies this pack contributes.

Re-exported here so the entry points in ``pyproject.toml`` can name a stable import
path — ``my_signals.strategies:BreakoutStrategy`` — while the file a class lives in
stays free to move. One strategy per module keeps a growing pack readable.
"""

from my_signals.strategies.breakout import BreakoutStrategy
from my_signals.strategies.pairs_reversion import PairsReversionStrategy

__all__ = ["BreakoutStrategy", "PairsReversionStrategy"]
