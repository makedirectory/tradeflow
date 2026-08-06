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

    #: US regular-session trading hours per day (09:30-16:00 ET).
    _TRADING_HOURS_PER_DAY: ClassVar[float] = 6.5
    #: US trading days per year.
    _TRADING_DAYS_PER_YEAR: ClassVar[int] = 252

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

    def periods_per_year(self) -> float:
        """How many bars of this timeframe occur in a trading year.

        Used to annualize ratios computed on a *per-bar* return series (a 5-minute
        bar is **not** 1/252 of a year). Note the backtest equity curve is daily-
        resampled, so its metrics use ``TRADING_DAYS_PER_YEAR`` directly; this is
        for callers that work on bar-frequency returns.
        """
        days = self._TRADING_DAYS_PER_YEAR
        if self.unit == DAY:
            return days / self.amount
        if self.unit == WEEK:
            return 52.0 / self.amount
        if self.unit == HOUR:
            return days * self._TRADING_HOURS_PER_DAY / self.amount
        # minutes
        return days * self._TRADING_HOURS_PER_DAY * 60.0 / self.amount

    def __str__(self) -> str:
        return f"{self.amount}{self.unit}"
