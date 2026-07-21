"""Conditional risk: a Σ that knows what month it is (spec 024).

006's Σ is **unconditional over its trailing window** — a flat, equal-weighted
average since the start of the estimation window. The fix here is old and cheap:
**condition the volatilities, keep the correlation structure slow.**

    Σ_t = D_t · R · D_t

``R`` is 006's own (shrunk/factor) correlation structure on the long window; ``D_t``
is a diagonal of *conditional* per-name variances from an EWMA (RiskMetrics
λ≈0.94 daily / 0.97 weekly) or HAR-lite forecast. This is **not** a multivariate
GARCH — correlations stay slow on purpose (conditioning them buys little at this
horizon and costs stability, spec §2 non-goals).

Two backends compose with 006's two estimators:

- **Shrinkage/sample** (:class:`~src.risk.base.RiskMatrix`): condition the diagonal,
  keep the LW-shrunk (or raw) correlation matrix fixed — :func:`condition_risk_matrix`.
- **Factor** (:class:`~src.risk.factor.FactorRiskMatrix`): condition *both*
  ``factor_cov`` (via an EWMA covariance on factor returns — small, ``K×K``, PSD by
  construction) and ``specific_var`` (via the same per-name EWMA/HAR family used by
  the shrinkage path) — never one without the other (hidden factor 5: a partial
  conditioning mis-splits the factor/specific attribution).

Everything here is an **as-of forward pass**: state at bar ``t`` is seeded from the
first ``min_obs`` rows of the panel handed in (the 006 window) and then updated row
by row through the rest of the panel — so the result depends only on rows at or
before the panel's last timestamp, never on what comes after (hidden factor 6;
verified by a property test in ``tests/test_conditional_risk.py``, not just a code
review). This mirrors the streaming accumulator shape of :mod:`src.risk.streaming`
(015) — a small O(N) / O(K²) state updated one row at a time — though, like
Ledoit–Wolf itself, the estimator here runs eagerly over a materialized panel; wiring
it to stream out of the Parquet store is future work, not done here.

**The alpha/risk vol seam (hidden factor 3).** :func:`src.data.features.add_risk_features`
computes a *different*, deliberately unconditional ``residual_vol`` that scales alphas
(``α = ω·IC·z``, Case 1) and 020's Case logic — that vol must stay slow. Only the
*risk*-model Σ used for portfolio construction / tracking-error conditions here. Do
not let the two merge into one "vol" concept.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.risk.base import RiskMatrix
from src.risk.factor import FactorRiskMatrix

#: RiskMetrics defaults, by cadence (spec §3.1). Global, not per-name (§7 lean —
#: cross-sectional λ fitting overfits small histories).
EWMA_LAMBDA_DAILY = 0.94
EWMA_LAMBDA_WEEKLY = 0.97

#: A cadence below this many periods/year is "weekly or slower" for the λ default.
_WEEKLY_PPY_CEILING = 100.0


def default_lambda(periods_per_year: float) -> float:
    """RiskMetrics' λ default for the bar cadence implied by ``periods_per_year``."""
    return EWMA_LAMBDA_DAILY if periods_per_year > _WEEKLY_PPY_CEILING else EWMA_LAMBDA_WEEKLY


# --------------------------------------------------------------------------- #
# The vol layer: EWMA and HAR-lite per-name variance forecasts
# --------------------------------------------------------------------------- #
def ewma_variance_series(x: np.ndarray, lambda_: float, min_obs: int) -> np.ndarray:
    """Per-column EWMA variance at every row ``t >= min_obs`` (rows ``< min_obs`` are
    NaN — no seed yet). ``x`` is raw returns, ``(T, N)`` (mean-zero assumed, the
    RiskMetrics convention). Row ``t``'s value depends only on ``x[:t+1]`` — the
    as-of property.

    Seeded from the raw (mean-zero-assumed) population variance of the first
    ``min_obs`` rows (006's own window), then folded forward one row at a time via
    the RiskMetrics recursion ``var_t = λ·var_{t-1} + (1-λ)·r_t²``. With ``λ = 1``
    the recursion never updates past the seed, so ``D_t`` stays exactly the seed
    window's variance no matter how much more data follows — the reduction test.
    """
    t, n = x.shape
    out = np.full((t, n), np.nan)
    if t < min_obs or min_obs < 1:
        return out
    var = np.mean(x[:min_obs] ** 2, axis=0)
    out[min_obs - 1] = var
    for i in range(min_obs, t):
        var = lambda_ * var + (1.0 - lambda_) * x[i] ** 2
        out[i] = var
    return out


def ewma_variance_path(returns: pd.DataFrame, lambda_: float, min_obs: int) -> Optional[pd.Series]:
    """Final per-name EWMA variance (per-bar, not annualized) as of ``returns``'
    last row. Returns are used **raw, not demeaned** — the RiskMetrics convention
    (daily/weekly return means are indistinguishable from zero at this horizon), and
    deliberately so: demeaning by a window's own mean would make the seed depend on
    every row in whatever window happened to be passed in, which breaks the as-of
    forward-pass property once more rows are appended (see the λ=1 reduction test).
    ``None`` if the panel is shorter than ``min_obs``.
    """
    x = returns.to_numpy(dtype=float)
    series = ewma_variance_series(x, lambda_, min_obs)
    if len(series) < min_obs:
        return None
    last = series[-1]
    if np.isnan(last).any():
        return None
    return pd.Series(last, index=returns.columns)


def ewma_covariance_path(returns: pd.DataFrame, lambda_: float, min_obs: int) -> Optional[np.ndarray]:
    """Final ``K×K`` EWMA covariance (per-bar) as of ``returns``' last row — the
    small-matrix analogue of :func:`ewma_variance_path`, used to condition a factor
    model's ``factor_cov`` (hidden factor 5: PSD by construction, a convex
    combination of PSD matrices, never eigenvalue surgery). Raw (not demeaned)
    returns, same convention as :func:`ewma_variance_path`.
    """
    x = returns.to_numpy(dtype=float)
    t, k = x.shape
    if t < min_obs or min_obs < 1:
        return None
    seed = x[:min_obs]
    cov = (seed.T @ seed) / min_obs
    for i in range(min_obs, t):
        r = x[i]
        cov = lambda_ * cov + (1.0 - lambda_) * np.outer(r, r)
    return cov


def har_variance_forecast(returns: pd.Series, min_obs: int = 60) -> Optional[float]:
    """HAR-lite one-step-ahead variance forecast (Corsi's Heterogeneous Autoregressive
    model, daily/weekly(5)/monthly(22) realized-variance regressors) — the
    term-structure-aware alternative to the EWMA's flat term structure (hidden
    factor 4). OLS-fit on the trailing window handed in, forecasting the *next* bar's
    ``r²`` from the current (day, week, month) realized-variance triple. ``None``
    below ``min_obs`` usable rows.
    """
    r2 = returns.to_numpy(dtype=float) ** 2
    t = len(r2)
    if t <= min_obs:
        return None
    day = r2
    week = pd.Series(r2).rolling(5, min_periods=5).mean().to_numpy()
    month = pd.Series(r2).rolling(22, min_periods=22).mean().to_numpy()

    x_day, x_week, x_month = day[:-1], week[:-1], month[:-1]
    y = day[1:]  # next bar's realized variance
    mask = ~(np.isnan(x_week) | np.isnan(x_month))
    if int(mask.sum()) < min_obs:
        return None
    design = np.column_stack([np.ones(int(mask.sum())), x_day[mask], x_week[mask], x_month[mask]])
    coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)

    x_last = np.array([1.0, day[-1], week[-1], month[-1]])
    if np.isnan(x_last).any():
        return None
    return float(max(coef @ x_last, 0.0))


def _forecast_variance(panel: pd.DataFrame, method: str, min_obs: int, lambda_: float) -> pd.Series:
    """Per-name variance forecast (per-bar), dispatched by ``method`` ('ewma'/'har')."""
    if method == "ewma":
        var = ewma_variance_path(panel, lambda_, min_obs)
        return var if var is not None else pd.Series(dtype=float)
    if method == "har":
        out: Dict[str, float] = {}
        for col in panel.columns:
            forecast = har_variance_forecast(panel[col], min_obs)
            if forecast is not None:
                out[col] = forecast
        return pd.Series(out, dtype=float)
    raise ValueError(f"conditional method must be 'ewma' or 'har', got {method!r}")


# --------------------------------------------------------------------------- #
# Σ_t = D_t R D_t assembly
# --------------------------------------------------------------------------- #
def _assemble_d_r_d(sigma: np.ndarray, unconditional_var: np.ndarray, conditional_var: np.ndarray) -> np.ndarray:
    """Rescale ``sigma`` to carry ``conditional_var`` on the diagonal, holding its
    implied correlation fixed. ``D R D`` with ``R`` PSD is PSD (hidden factor 5) —
    a diagonal congruence transform preserves the sign of every eigenvalue.
    Zero-variance (dead-flat) names get zero correlation with everything else
    rather than a NaN — the same "independent" treatment the ragged-panel fallback
    (:func:`src.risk.base.build_risk_matrix`) already gives thin-history names.
    """
    std_old = np.sqrt(np.maximum(unconditional_var, 0.0))
    std_new = np.sqrt(np.maximum(conditional_var, 0.0))
    denom = np.outer(std_old, std_old)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, sigma / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(std_new, std_new)


def _sigma_regime(
    symbols: List[str], unconditional_var: np.ndarray, conditional_var: np.ndarray, method: str, lambda_: float
) -> Dict[str, Any]:
    """The §5 diagnostic: current D_t vs the unconditional diagonal, as a ratio."""
    ratio = np.ones_like(conditional_var)
    positive = unconditional_var > 0
    ratio[positive] = conditional_var[positive] / unconditional_var[positive]
    return {
        "method": method,
        "lambda": lambda_,
        "unconditional_vol": {s: float(np.sqrt(max(v, 0.0))) for s, v in zip(symbols, unconditional_var)},
        "conditional_vol": {s: float(np.sqrt(max(v, 0.0))) for s, v in zip(symbols, conditional_var)},
        "sigma_regime": {s: float(r) for s, r in zip(symbols, ratio)},
        "mean_sigma_regime": float(np.mean(ratio)) if len(ratio) else 1.0,
    }


def condition_risk_matrix(
    matrix: RiskMatrix,
    panel: pd.DataFrame,
    method: str,
    periods_per_year: float,
    min_obs: int = 60,
    lambda_: Optional[float] = None,
    horizon: int = 1,
) -> RiskMatrix:
    """Return a Σ_t conditioned on ``method`` ('ewma'/'har'), or ``matrix`` unchanged
    if ``method`` is falsy or the panel is too short to seed the recursion.

    ``panel`` must be the SAME (as-of, ``<= t``) return panel used to estimate
    ``matrix`` — this function only rescales the diagonal (shrinkage/sample) or
    reassembles ``X F_t Xᵀ + Δ_t`` (factor); it never re-estimates the correlation
    structure or the exposures. Dispatches on ``matrix``'s type: a
    :class:`~src.risk.factor.FactorRiskMatrix` conditions ``factor_cov``/``specific_var``
    coherently (hidden factor 5); any other :class:`RiskMatrix` conditions the plain
    diagonal, keeping the correlation matrix fixed (``Σ_t = D_t R D_t``).
    """
    if not method:
        return matrix
    lam = lambda_ if lambda_ is not None else default_lambda(periods_per_year)
    if isinstance(matrix, FactorRiskMatrix):
        return _condition_factor(matrix, panel, method, periods_per_year, min_obs, lam, horizon)
    return _condition_shrinkage(matrix, panel, method, periods_per_year, min_obs, lam, horizon)


def _condition_shrinkage(
    matrix: RiskMatrix,
    panel: pd.DataFrame,
    method: str,
    periods_per_year: float,
    min_obs: int,
    lam: float,
    horizon: int,
) -> RiskMatrix:
    cols = [s for s in matrix.symbols if s in panel.columns]
    if len(cols) < 2 or len(panel) < min_obs:
        return matrix
    forecast = _forecast_variance(panel[cols], method, min_obs, lam)
    if forecast.empty:
        return matrix

    unconditional = np.diag(matrix.sigma).astype(float).copy()
    conditional = unconditional.copy()
    for i, sym in enumerate(matrix.symbols):
        if sym in forecast.index and np.isfinite(forecast[sym]):
            conditional[i] = max(float(forecast[sym]), 0.0) * horizon * periods_per_year

    sigma_t = _assemble_d_r_d(matrix.sigma, unconditional, conditional)
    diag = _sigma_regime(matrix.symbols, unconditional, conditional, method, lam)
    return RiskMatrix(
        symbols=list(matrix.symbols), sigma=sigma_t, shrinkage=matrix.shrinkage, conditional_diagnostics=diag
    )


def _condition_factor(
    matrix: FactorRiskMatrix,
    panel: pd.DataFrame,
    method: str,
    periods_per_year: float,
    min_obs: int,
    lam: float,
    horizon: int,
) -> FactorRiskMatrix:
    names = list(matrix.symbols)
    if not all(n in panel.columns for n in names) or len(panel) < min_obs:
        return matrix

    x = matrix.exposures.loc[names].to_numpy(dtype=float)  # N×K
    r = panel[names].to_numpy(dtype=float)  # T×N, the SAME panel the factor model was estimated on
    proj = x @ np.linalg.pinv(x.T @ x)  # N×K
    factor_returns = r @ proj  # T×K
    residuals = r - factor_returns @ x.T  # T×N

    factor_returns_df = pd.DataFrame(factor_returns, columns=matrix.factor_names)
    factor_cov_bar = ewma_covariance_path(factor_returns_df, lam, min_obs)
    residuals_df = pd.DataFrame(residuals, columns=names)
    specific_forecast = _forecast_variance(residuals_df, method, min_obs, lam)

    if factor_cov_bar is None or specific_forecast.empty:
        return matrix

    unconditional_specific = matrix.specific_var.astype(float).copy()
    conditional_specific = unconditional_specific.copy()
    for i, sym in enumerate(names):
        if sym in specific_forecast.index and np.isfinite(specific_forecast[sym]):
            conditional_specific[i] = max(float(specific_forecast[sym]), 0.0) * horizon * periods_per_year

    factor_cov_t = np.atleast_2d(factor_cov_bar) * horizon * periods_per_year
    sigma_t = x @ factor_cov_t @ x.T + np.diag(conditional_specific)

    diag = _sigma_regime(names, unconditional_specific, conditional_specific, method, lam)
    diag["factor_cov_conditioning"] = "ewma"  # F_t always conditions via EWMA (hidden factor 5's "same family")

    return FactorRiskMatrix(
        symbols=names,
        sigma=sigma_t,
        shrinkage=matrix.shrinkage,
        conditional_diagnostics=diag,
        exposures=matrix.exposures,
        factor_cov=factor_cov_t,
        specific_var=conditional_specific,
        factor_names=list(matrix.factor_names),
    )


# --------------------------------------------------------------------------- #
# Evidence gate: Mincer–Zarnowitz and QLIKE (spec §4 hidden factor 8, §6)
# --------------------------------------------------------------------------- #
def mincer_zarnowitz(realized_var: np.ndarray, forecast_var: np.ndarray) -> Dict[str, float]:
    """OLS of ``realized_var = a + b*forecast_var``. A well-calibrated forecast has
    ``a ≈ 0``, ``b ≈ 1``. Needs >= 3 finite pairs; returns NaNs otherwise.
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    mask = np.isfinite(realized_var) & np.isfinite(forecast_var)
    if int(mask.sum()) < 3:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"), "n": int(mask.sum())}
    y, f = realized_var[mask], forecast_var[mask]
    design = np.column_stack([np.ones(len(f)), f])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": float(coef[0]), "b": float(coef[1]), "r2": float(r2), "n": int(mask.sum())}


def qlike_loss(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss (Patton 2011): ``mean(realized/forecast - log(realized/forecast) - 1)``.
    A robust variance-forecast loss (lower is better; 0 for a perfect forecast).
    Non-positive forecasts/realized values are excluded (undefined ratio).
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    mask = np.isfinite(realized_var) & np.isfinite(forecast_var) & (forecast_var > 0) & (realized_var > 0)
    if not mask.any():
        return float("nan")
    ratio = realized_var[mask] / forecast_var[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


@dataclass
class VolForecastEvaluation:
    """Per-method forecast-quality evaluation (spec §6): MZ + QLIKE, pooled and by
    ex-post realized-vol tercile — the gate that decides whether conditioning is
    adopted (§4 hidden factor 8), never a preference. ``realized``/``forecasts``
    are the raw per-point arrays (one entry per sampled rebalance), kept so a
    caller can pool several names' points into one cross-sectional evaluation
    (:func:`src.services.analysis.evaluate_conditional_risk`)."""

    n_points: int
    by_method: Dict[str, Dict[str, Any]]
    realized: np.ndarray = None  # type: ignore[assignment]
    forecasts: Dict[str, np.ndarray] = None  # type: ignore[assignment]


def evaluate_vol_forecasts(
    returns: pd.Series,
    min_obs: int = 60,
    n_points: int = 60,
    lambda_: Optional[float] = None,
    periods_per_year: float = 252.0,
) -> VolForecastEvaluation:
    """Walk one name's return series forward, comparing EWMA / HAR / unconditional
    (expanding trailing) one-bar-ahead variance forecasts against the realized
    ``r_{t+1}²`` — Mincer–Zarnowitz and QLIKE, pooled and split by the realized-vol
    tercile the forecast landed in (ex-post labels only, §3.3 — never fed back into
    the model). All forecasts at ``t`` use only ``returns[:t+1]``.
    """
    x = returns.to_numpy(dtype=float)
    t_total = len(x)
    lam = lambda_ if lambda_ is not None else default_lambda(periods_per_year)
    last = t_total - 2  # need a realized r_{t+1}
    if last <= min_obs:
        return VolForecastEvaluation(n_points=0, by_method={})
    points = np.linspace(min_obs, last, num=min(n_points, last - min_obs + 1), dtype=int)
    points = sorted(set(int(p) for p in points))

    ewma_full = ewma_variance_series(x.reshape(-1, 1), lam, min_obs)[:, 0]
    cumsum2 = np.cumsum(x**2)

    realized, forecasts = [], {"ewma": [], "har": [], "unconditional": []}
    for t in points:
        realized.append(x[t + 1] ** 2)
        forecasts["ewma"].append(ewma_full[t] if t < len(ewma_full) else np.nan)
        forecasts["unconditional"].append(cumsum2[t] / (t + 1))
        har = har_variance_forecast(returns.iloc[: t + 1], min_obs)
        forecasts["har"].append(har if har is not None else np.nan)

    realized_arr = np.array(realized)
    by_method: Dict[str, Dict[str, Any]] = {}
    terciles = _tercile_labels(realized_arr)
    for name, series in forecasts.items():
        arr = np.array(series)
        entry: Dict[str, Any] = {
            "mincer_zarnowitz": mincer_zarnowitz(realized_arr, arr),
            "qlike": qlike_loss(realized_arr, arr),
            "by_regime": {},
        }
        for label in ("low", "mid", "high"):
            m = terciles == label
            entry["by_regime"][label] = {
                "mincer_zarnowitz": mincer_zarnowitz(realized_arr[m], arr[m]),
                "qlike": qlike_loss(realized_arr[m], arr[m]),
                "n": int(m.sum()),
            }
        by_method[name] = entry

    forecast_arrays = {name: np.array(series) for name, series in forecasts.items()}
    return VolForecastEvaluation(
        n_points=len(points), by_method=by_method, realized=realized_arr, forecasts=forecast_arrays
    )


def _tercile_labels(values: np.ndarray) -> np.ndarray:
    """Ex-post terciles ('low'/'mid'/'high') of ``values`` — report-time labels only."""
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return np.array(["mid"] * len(values))
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    labels = np.where(values <= q1, "low", np.where(values <= q2, "mid", "high"))
    return labels


# --------------------------------------------------------------------------- #
# Churn guard: risk-driven vs alpha-driven turnover (hidden factor 2)
# --------------------------------------------------------------------------- #
def turnover_risk_share(
    optimizer,
    alphas,
    current_weights: Dict[str, float],
    matrix_t: RiskMatrix,
    matrix_prev: RiskMatrix,
    **optimize_kwargs,
) -> Dict[str, float]:
    """Decompose realized turnover into alpha-driven vs Σ-driven (hidden factor 2):
    rerun the same solve with ``matrix_prev`` (Σ frozen at the prior rebalance) in
    place of ``matrix_t`` (today's conditioned Σ), holding alphas/cost/current_weights
    fixed. The gap between the two turnovers is turnover the frozen-Σ solve would NOT
    have traded — attributable to Σ_t having moved, not to the alpha view changing.
    If risk-driven turnover dominates in a *calm* regime, λ is too hot (over-reactive).
    """
    actual = optimizer.optimize(alphas, matrix_t, current_weights=current_weights, **optimize_kwargs)
    frozen = optimizer.optimize(alphas, matrix_prev, current_weights=current_weights, **optimize_kwargs)
    turnover_actual = float(actual.diagnostics.get("turnover", 0.0)) if actual.feasible else 0.0
    turnover_frozen = float(frozen.diagnostics.get("turnover", 0.0)) if frozen.feasible else 0.0
    risk_driven = max(turnover_actual - turnover_frozen, 0.0)
    share = risk_driven / turnover_actual if turnover_actual > 0 else 0.0
    return {
        "turnover_actual": turnover_actual,
        "turnover_frozen_sigma": turnover_frozen,
        "risk_driven_turnover": risk_driven,
        "risk_driven_share": share,
    }
