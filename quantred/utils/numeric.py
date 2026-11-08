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


#: Relative tolerance below which a series counts as having no dispersion. See
#: :func:`is_flat`.
FLAT_RTOL = 1e-12


def is_flat(values, *, ddof: int = 1, rtol: float = FLAT_RTOL) -> bool:
    """True when ``values`` has no dispersion beyond floating-point noise.

    This is the one definition of "zero variance" behind every ratio that divides by a
    standard deviation (Sharpe, information ratio, t-stats, correlations, z-scores).
    ``std == 0`` is not that test: the mean of twenty copies of ``0.01`` is not exactly
    ``0.01``, so the sample std of a constant series is of order ``1e-18``, passes an
    ``== 0`` or ``> 0`` guard, and becomes a Sharpe of order ``1e17``.

    The test is ``std <= rtol * max(1, mean|x|)``. Summing ``n`` values of magnitude
    ``m`` leaves rounding error of order ``n * eps * m`` (``eps`` about ``2.2e-16``), so
    the noise std of a constant series stays near ``1e-15 * m`` for any realistic ``n``.
    ``rtol = 1e-12`` sits three orders above that and still far below the dispersion any
    real return, IC or P&L series carries (a per-period return std of ``1e-12`` is a
    hundred-millionth of a basis point). Scaling by ``max(1, mean|x|)`` keeps
    large-magnitude series (dollar P&L) from being judged on an absolute scale, and the
    floor of ``1`` stops a series centred on zero - an exactly tracking book's active
    return, the difference of two numbers of order ``1e-2`` - from being judged against
    its own noise.

    NaN entries are ignored; fewer than ``ddof + 1`` finite entries has no dispersion;
    an infinite entry makes the series not flat. Pure numpy, so any layer may import it.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size <= ddof:
        return True
    if not np.all(np.isfinite(arr)):
        return False
    return bool(np.std(arr, ddof=ddof) <= rtol * max(1.0, float(np.mean(np.abs(arr)))))
