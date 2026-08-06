"""Small numeric helpers shared across layers.

Centralized so price/quantity rounding and step math are defined once and reused
by execution, the optimizer, and reporting - rather than re-derived inline.
"""

import math
from typing import Union


def round_price(price: float, places: int = 2) -> float:
    """Round a price to a tick size.

    Sub-dollar instruments are quoted to more decimals, so prices under $1 are
    rounded to 4 places by default.
    """
    if price < 1.0:
        places = max(places, 4)
    return round(float(price), places)


def round_quantity(qty: float, allow_fractional: bool = False) -> float:
    """Round an order quantity, flooring to whole shares unless fractional is allowed."""
    if allow_fractional:
        return float(qty)
    return float(math.floor(qty))


def safe_float(value, default: float = 0.0) -> float:
    """Best-effort float conversion that never raises."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def step_decimals(step: Union[int, float]) -> int:
    """Number of decimal places implied by a step size (capped at 6).

    e.g. ``0.05 -> 2``, ``1 -> 0``. Used by the optimizer to keep generated
    parameter values aligned to their declared grid.
    """
    text = str(step)
    if "." not in text:
        return 0
    return min(len(text.split(".")[-1]), 6)
