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


class GeneratedCodeRefused(RunRefused):
    """A config or strategy name that points at code the research agent wrote.

    A saved config names its strategy, and a generated strategy has no registered name -
    only a ``draft:<Class>:<hash>`` identity. Resolving that identity must never mean
    "find the source and run it": loading a config is not a review.
    """

    def __init__(self, name: str):
        from quantred.settings import DOCS_URL

        super().__init__(
            f"{name!r} is a strategy the research agent wrote, and generated code is never run by "
            "loading a config. Review its source (recorded in the config's "
            "provenance.generated_code and on the journal's trial line), add it to a strategy "
            f"pack, and point the config at the registered name: {DOCS_URL}/usage/private-strategies"
        )
