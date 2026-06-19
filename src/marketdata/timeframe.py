"""Broker-agnostic bar timeframe.

Strategies declare their timeframe as a friendly string (``"5Min"``, ``"1Hour"``,
``"1Day"``). This module parses that once into a :class:`Timeframe` value object
so every layer agrees on what ``"5Min"`` means, and each broker adapter maps the
value object onto its own SDK type.
"""

import re
from dataclasses import dataclass
from typing import ClassVar, Dict

# Canonical unit names.
MINUTE = "min"
HOUR = "hour"
DAY = "day"
WEEK = "week"

# Map the suffixes people actually type onto canonical units.
_UNIT_ALIASES: Dict[str, str] = {
    "min": MINUTE,
    "m": MINUTE,
    "minute": MINUTE,
    "h": HOUR,
    "hr": HOUR,
    "hour": HOUR,
    "d": DAY,
    "day": DAY,
    "w": WEEK,
    "wk": WEEK,
    "week": WEEK,
}

_PATTERN = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")


@dataclass(frozen=True)
class Timeframe:
    """An amount + unit, e.g. ``Timeframe(5, "min")``."""

    amount: int
    unit: str

    #: pandas resample/offset aliases per canonical unit.
    _PANDAS_OFFSET: ClassVar[Dict[str, str]] = {MINUTE: "min", HOUR: "h", DAY: "D", WEEK: "W"}

    @classmethod
    def parse(cls, text: str) -> "Timeframe":
        """Parse a string like ``"5Min"`` / ``"1Hour"`` / ``"1Day"``."""
        match = _PATTERN.match(text)
        if not match:
            raise ValueError(f"Invalid timeframe: {text!r} (expected e.g. '5Min', '1Hour', '1Day')")

        amount = int(match.group(1))
        unit = _UNIT_ALIASES.get(match.group(2).lower())
        if unit is None:
            raise ValueError(f"Unsupported timeframe unit in {text!r}")
        if amount < 1:
            raise ValueError(f"Timeframe amount must be >= 1, got {amount}")
        return cls(amount=amount, unit=unit)

    def to_pandas_offset(self) -> str:
        """Return the pandas offset alias, e.g. ``"5min"``, ``"1D"``."""
        return f"{self.amount}{self._PANDAS_OFFSET[self.unit]}"

    def __str__(self) -> str:
        return f"{self.amount}{self.unit}"
