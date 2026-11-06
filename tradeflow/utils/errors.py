"""Refusals shared across layers.

Here rather than in a service because the optimization layer refuses runs too, and it
must not import the services above it.
"""

from datetime import datetime
from typing import Sequence


class RunRefused(ValueError):
    """A run refused before it read or recorded anything, with what to change.

    A ``ValueError`` so every caller that already handles one keeps working; its own
    type so the CLI can end on the message rather than a stack trace.
    """


def no_bars_refusal(symbols: Sequence[str], start: datetime, end: datetime) -> RunRefused:
    """The refusal for a run whose universe had no bars at all.

    Such a run returned a clean-looking result - every ratio 0.0 - and was journaled.
    Zero is a measurement; a run over no bars measured nothing.
    """
    named = ", ".join(symbols) if symbols else "an empty universe"
    return RunRefused(
        f"No bars for any of {named} between {start:%Y-%m-%d} and {end:%Y-%m-%d}, so there is "
        "nothing to measure and nothing was recorded. Check the symbols and the window."
    )
