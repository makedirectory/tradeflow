"""Pure performance-metric primitives - the numbers that tell you, honestly,
whether you have an edge or just a flattering backtest.

One definition each for the ratios and risk measures used in the project, shared
by the backtest analytics (:mod:`tradeflow.analytics.performance`) and the scanner base
class - so "Sharpe" or "max drawdown" mean the same thing everywhere (a Sharpe
that changes depending on who's asking is how people end up trading their savings).

Every function is pure and defensive: empty/degenerate inputs yield 0.0 (or inf
for ratios with a zero denominator) rather than raising - a backtest with zero
trades shouldn't crash; it should just quietly tell you it found nothing.

A note on annualization: the headline ratios take a
``periods_per_year`` so they annualize to the *sampling frequency of the series
passed in*. The backtest equity curve is resampled to daily P&L
(:func:`tradeflow.analytics.performance.build_equity_curve`), so its returns are daily
and ``TRADING_DAYS_PER_YEAR`` is the correct factor; intraday or weekly series
pass their own factor via :meth:`tradeflow.marketdata.timeframe.Timeframe.periods_per_year`.
"""

import math
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: US trading days per year, used to annualize daily-return ratios.
TRADING_DAYS_PER_YEAR = 252

#: Euler-Mascheroni constant, used by the Deflated Sharpe expected-maximum.
_EULER_MASCHERONI = 0.5772156649015329

Numbers = Union[Sequence[float], pd.Series, np.ndarray]


def _as_series(values: Numbers) -> pd.Series:
    return values if isinstance(values, pd.Series) else pd.Series(list(values), dtype="float64")


# --------------------------------------------------------------------------- #
# Gaussian helpers (hand-rolled so the base install needs no scipy)
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (exact, stdlib only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (quantile) via Acklam's rational approximation.

    Accurate to ~1e-9 over the open interval (0, 1); clamps the endpoints.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    # Coefficients for Acklam's algorithm.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]

    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


# --------------------------------------------------------------------------- #
# Returns / growth
# --------------------------------------------------------------------------- #
def returns_from_equity(equity_curve: Numbers) -> pd.Series:
    """Period-over-period returns from an equity curve."""
    return _as_series(equity_curve).pct_change().dropna()


def cagr(equity_curve: Numbers, years: float) -> float:
    """Compound annual growth rate from first->last equity over ``years`` elapsed.

    Returns a fraction (0.20 == 20%/yr). ``0.0`` for non-positive elapsed time or
    a degenerate curve.
    """
    equity = _as_series(equity_curve)
    if len(equity) < 2 or years <= 0:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Standard deviation of returns scaled to a year."""
    r = _as_series(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std() * np.sqrt(periods_per_year))


def downside_deviation(returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized deviation of negative returns only (the Sortino denominator)."""
    r = _as_series(returns)
    downside = r[r < 0]
    if len(downside) < 2:
        return 0.0
    return float(downside.std() * np.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# Risk-adjusted ratios
# --------------------------------------------------------------------------- #
def sharpe_ratio(returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio (risk-free rate assumed 0)."""
    r = _as_series(returns)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())


def sortino_ratio(returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sortino ratio (downside-deviation denominator)."""
    r = _as_series(returns)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / downside.std())


def calmar_ratio(cagr_value: float, max_drawdown_value: float) -> float:
    """CAGR / max drawdown.

    Both arguments are fractions. ``inf`` when drawdown is zero.
    """
    if max_drawdown_value == 0:
        return float("inf")
    return float(abs(cagr_value / max_drawdown_value))


def treynor_ratio(returns: Numbers, beta: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized excess return per unit of market beta. ``0.0`` if beta is zero."""
    r = _as_series(returns)
    if len(r) < 2 or beta == 0:
        return 0.0
    return float(r.mean() * periods_per_year / beta)


# --------------------------------------------------------------------------- #
# Drawdown shape
# --------------------------------------------------------------------------- #
def drawdown_series(equity_curve: Numbers) -> pd.Series:
    """Per-point drawdown as a (negative) fraction of the running peak."""
    equity = _as_series(equity_curve)
    if equity.empty:
        return pd.Series(dtype="float64")
    running_max = equity.expanding().max()
    return (equity - running_max) / running_max


def max_drawdown(equity_curve: Numbers) -> float:
    """Maximum peak-to-trough decline of an equity curve, as a positive fraction."""
    dd = drawdown_series(equity_curve)
    return float(abs(dd.min())) if not dd.empty else 0.0


def max_drawdown_duration(equity_curve: Numbers) -> int:
    """Longest run of consecutive periods spent below a prior peak (underwater)."""
    equity = _as_series(equity_curve)
    if len(equity) < 2:
        return 0
    underwater = equity < equity.cummax()
    longest = current = 0
    for uw in underwater.to_numpy():
        current = current + 1 if uw else 0
        longest = max(longest, current)
    return int(longest)


def ulcer_index(equity_curve: Numbers) -> float:
    """Root-mean-square drawdown depth (%), penalizing deep *and* long drawdowns."""
    dd = drawdown_series(equity_curve)
    if dd.empty:
        return 0.0
    return float(np.sqrt(np.mean((dd.to_numpy() * 100.0) ** 2)))


def recovery_factor(net_profit: float, max_drawdown_value: float) -> float:
    """Net profit / max drawdown (absolute-dollar cousin of Calmar)."""
    if max_drawdown_value == 0:
        return float("inf")
    return float(net_profit / max_drawdown_value)


def martin_ratio(cagr_value: float, ulcer_index_value: float) -> float:
    """Ulcer Performance Index: CAGR / ulcer index. ``0.0`` if ulcer is zero."""
    if ulcer_index_value == 0:
        return 0.0
    return float(cagr_value * 100.0 / ulcer_index_value)


def sterling_ratio(cagr_value: float, average_drawdown: float) -> float:
    """CAGR / average drawdown. ``inf`` when average drawdown is zero."""
    if average_drawdown == 0:
        return float("inf")
    return float(abs(cagr_value / average_drawdown))


# --------------------------------------------------------------------------- #
# Tail / distribution risk
# --------------------------------------------------------------------------- #
def value_at_risk(returns: Numbers, level: float = 0.95) -> float:
    """Historical Value-at-Risk: the loss not exceeded with ``level`` confidence.

    Returned as a positive fraction (0.03 == a 3% loss).
    """
    r = _as_series(returns)
    if r.empty:
        return 0.0
    return float(abs(np.percentile(r.to_numpy(), (1.0 - level) * 100.0)))


def conditional_var(returns: Numbers, level: float = 0.95) -> float:
    """Expected shortfall: mean loss in the worst ``1 - level`` tail (positive)."""
    r = _as_series(returns)
    if r.empty:
        return 0.0
    cutoff = np.percentile(r.to_numpy(), (1.0 - level) * 100.0)
    tail = r[r <= cutoff]
    if tail.empty:
        return float(abs(cutoff))
    return float(abs(tail.mean()))


def skewness(returns: Numbers) -> float:
    """Sample skewness of returns (0.0 for < 3 points)."""
    r = _as_series(returns)
    if len(r) < 3:
        return 0.0
    return float(r.skew())


def kurtosis(returns: Numbers) -> float:
    """Excess kurtosis of returns (Fisher; 0.0 for a normal distribution)."""
    r = _as_series(returns)
    if len(r) < 4:
        return 0.0
    return float(r.kurtosis())


def tail_ratio(returns: Numbers) -> float:
    """Ratio of the right tail (95th pct) to the left tail (5th pct), in magnitude."""
    r = _as_series(returns)
    if r.empty:
        return 0.0
    left = abs(np.percentile(r.to_numpy(), 5))
    if left == 0:
        return float("inf")
    return float(abs(np.percentile(r.to_numpy(), 95)) / left)


def omega_ratio(returns: Numbers, threshold: float = 0.0) -> float:
    """Probability-weighted gains over losses relative to ``threshold``."""
    r = _as_series(returns)
    if r.empty:
        return 0.0
    excess = r - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


# --------------------------------------------------------------------------- #
# Statistical robustness (Bailey & Lopez de Prado) - the honest-evaluation core
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(returns: Numbers, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > ``benchmark_sr``) given the observed per-period Sharpe.

    Corrects the Sharpe estimate for sample length, skew and (excess) kurtosis.
    ``benchmark_sr`` is a *per-period* (non-annualized) Sharpe, matching the
    series passed in. Returns a probability in [0, 1].
    """
    r = _as_series(returns)
    n = len(r)
    if n < 3 or r.std() == 0:
        return 0.0
    sr = float(r.mean() / r.std())  # per-period Sharpe estimate
    skew = skewness(r)
    kurt = kurtosis(r) + 3.0  # Pearson kurtosis (normal == 3)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return 0.0
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(norm_cdf(z))


def expected_max_sharpe(var_of_trial_sr: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe across ``n_trials`` independent trials.

    The multiple-testing benchmark used by the Deflated Sharpe Ratio.
    """
    if n_trials <= 1 or var_of_trial_sr <= 0:
        return 0.0
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(var_of_trial_sr) * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(returns: Numbers, n_trials: int, var_of_trial_sr: float = None) -> float:
    """PSR against the expected best of ``n_trials`` configs (anti-overfitting).

    ``var_of_trial_sr`` is the variance of the *per-period* Sharpe estimates
    across the configs that were tried. When unknown (a standalone backtest),
    it falls back to the null-hypothesis estimator variance ``1/(n-1)`` so the
    deflation is still conservative. Returns a probability in [0, 1].
    """
    r = _as_series(returns)
    n = len(r)
    if n < 3 or r.std() == 0 or n_trials < 1:
        return 0.0
    if var_of_trial_sr is None:
        var_of_trial_sr = 1.0 / (n - 1)
    sr_star = expected_max_sharpe(var_of_trial_sr, n_trials)
    return probabilistic_sharpe_ratio(r, benchmark_sr=sr_star)


# --------------------------------------------------------------------------- #
# Trade-level statistics
# --------------------------------------------------------------------------- #
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


def expectancy(pnl: Numbers) -> float:
    """Mean P&L per trade (in account currency)."""
    p = _as_series(pnl)
    return float(p.mean()) if len(p) else 0.0


def payoff_ratio(pnl: Numbers) -> float:
    """Average win / average loss (magnitudes). ``inf`` with no losing trades."""
    p = _as_series(pnl)
    wins, losses = p[p > 0], p[p < 0]
    if losses.empty or losses.mean() == 0:
        return float("inf") if not wins.empty else 0.0
    if wins.empty:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def gain_to_pain_ratio(pnl: Numbers) -> float:
    """Sum of P&L over the sum of absolute losses. ``inf`` with no losses."""
    p = _as_series(pnl)
    pain = abs(p[p < 0].sum())
    if pain == 0:
        return float("inf")
    return float(p.sum() / pain)


def kelly_criterion(pnl: Numbers) -> float:
    """Kelly fraction from the win rate and payoff ratio: ``W - (1 - W)/R``.

    The theoretical optimal capital fraction; values are typically scaled down
    in practice. ``0.0`` for degenerate input.
    """
    p = _as_series(pnl)
    if p.empty:
        return 0.0
    w = win_rate(p)
    r = payoff_ratio(p)
    if r == 0 or math.isinf(r):
        return 0.0
    return float(w - (1.0 - w) / r)


def system_quality_number(pnl: Numbers) -> float:
    """Van Tharp's SQN: ``sqrt(n) * mean(pnl) / std(pnl)`` (expectancy quality)."""
    p = _as_series(pnl)
    if len(p) < 2 or p.std() == 0:
        return 0.0
    return float(np.sqrt(len(p)) * p.mean() / p.std())


def consecutive(pnl: Numbers, *, winning: bool) -> int:
    """Longest run of consecutive winning (or losing) trades, in trade order."""
    p = _as_series(pnl)
    longest = current = 0
    for value in p.to_numpy():
        hit = value > 0 if winning else value < 0
        current = current + 1 if hit else 0
        longest = max(longest, current)
    return int(longest)


# --------------------------------------------------------------------------- #
# Benchmark-relative
# --------------------------------------------------------------------------- #
def _align(returns: Numbers, benchmark_returns: Numbers) -> pd.DataFrame:
    paired = pd.concat([_as_series(returns), _as_series(benchmark_returns)], axis=1, keys=["r", "b"]).dropna()
    return paired


def alpha_beta(returns: Numbers, benchmark_returns: Numbers) -> Tuple[float, float, float]:
    """OLS of strategy returns on benchmark returns -> ``(alpha, beta, r_squared)``.

    ``alpha`` is the per-period intercept; ``beta`` the slope; ``r_squared`` the
    fraction of variance explained. ``(0.0, 0.0, 0.0)`` for degenerate input.
    """
    paired = _align(returns, benchmark_returns)
    if len(paired) < 3:
        return 0.0, 0.0, 0.0
    benchmark_var = paired["b"].var()
    if benchmark_var == 0:
        return 0.0, 0.0, 0.0
    beta = float(paired["r"].cov(paired["b"]) / benchmark_var)
    alpha = float(paired["r"].mean() - beta * paired["b"].mean())
    corr = paired["r"].corr(paired["b"])
    r_squared = float(corr**2) if pd.notna(corr) else 0.0
    return alpha, beta, r_squared


def information_ratio(
    returns: Numbers, benchmark_returns: Numbers, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualized mean active return over its tracking error vs a benchmark."""
    paired = _align(returns, benchmark_returns)
    if len(paired) < 2:
        return 0.0
    active = paired["r"] - paired["b"]
    if active.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * active.mean() / active.std())
