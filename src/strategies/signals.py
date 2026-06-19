"""The signal vocabulary shared by strategies, the engine, and execution.

Signals are plain strings so they can live directly in pandas objects and be
compared cheaply. Keeping them in one module avoids the classic bug where one
layer emits ``"buy"`` and another checks for ``"BUY"``.
"""

from typing import FrozenSet

# Entry signals
BUY = "BUY"
SELL = "SELL"

# Explicit position-closing signals (emitted by exit logic / strategies)
CLOSE_BUY = "CLOSE_BUY"
CLOSE_SELL = "CLOSE_SELL"

# No action
HOLD = "HOLD"

#: Signals that open a new position.
ENTRY_SIGNALS: FrozenSet[str] = frozenset({BUY, SELL})

#: Signals that close an existing position.
EXIT_SIGNALS: FrozenSet[str] = frozenset({CLOSE_BUY, CLOSE_SELL})

#: Every actionable (non-HOLD) signal.
ACTIONABLE_SIGNALS: FrozenSet[str] = ENTRY_SIGNALS | EXIT_SIGNALS
