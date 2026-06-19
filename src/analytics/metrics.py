"""Pure performance-metric primitives.

One definition each for the ratios and risk measures used in the project, shared
by the backtest analytics (:mod:`src.analytics.performance`) and the scanner base
class - so "Sharpe" or "max drawdown" mean the same thing everywhere.

Every function is pure and defensive: empty/degenerate inputs yield 0.0 (or inf
for ratios with a zero denominator) rather than raising.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd

#: US trading days per year, used to annualise daily-return ratios.
TRADING_DAYS_PER_YEAR = 252

Numbers = Union[Sequence[float], pd.Series, np.ndarray]


def _as_series(values: Numbers) -> pd.Series:
    return values if isinstance(values, pd.Series) else pd.Series(list(values), dtype="float64")


def returns_from_equity(equity_curve: Numbers) -> pd.Series:
    """Period-over-period returns from an equity curve."""
    return _as_series(equity_curve).pct_change().dropna()


def sharpe_ratio(returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe ratio (risk-free rate assumed 0)."""
    r = _as_series(returns)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())


def sortino_ratio(returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sortino ratio (downside-deviation denominator)."""
    r = _as_series(returns)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / downside.std())


def max_drawdown(equity_curve: Numbers) -> float:
    """Maximum peak-to-trough decline of an equity curve, as a positive fraction."""
    equity = _as_series(equity_curve)
    if equity.empty:
        return 0.0
    running_max = equity.expanding().max()
    drawdowns = (equity - running_max) / running_max
    return float(abs(drawdowns.min())) if not drawdowns.empty else 0.0


def win_rate(pnl: Numbers) -> float:
    """Fraction of trades with positive P&L."""
    p = _as_series(pnl)
    return float((p > 0).sum() / len(p)) if len(p) else 0.0


def profit_factor(pnl: Numbers) -> float:
    """Gross profit / gross loss. ``inf`` when there are no losses."""
    p = _as_series(pnl)
    gross_loss = abs(p[p < 0].sum())
    if gross_loss == 0:
        return float("inf")
    return float(p[p > 0].sum() / gross_loss)


def calmar_ratio(total_return: float, max_drawdown_value: float) -> float:
    """Return / max drawdown. ``inf`` when drawdown is zero."""
    if max_drawdown_value == 0:
        return float("inf")
    return float(abs(total_return / max_drawdown_value))
