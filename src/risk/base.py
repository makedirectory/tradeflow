"""Risk model core - the covariance matrix Σ and the quantities built on it.

A portfolio's risk is **not** additive: two names that move together are one bet,
two that move oppositely are a hedge. The covariance matrix Σ is what lets the
optimizer (and the diagnostics) tell the difference - compute portfolio variance,
tracking error, and each name's marginal contribution to risk.

This module owns the shared types: :class:`RiskMatrix` (an annualized Σ plus the
derived quantities) and :class:`RiskModel` (the estimator interface). Concrete
estimators live in :mod:`src.risk.sample`. Everything is research-clock: Σ is a tool
for sizing conviction, never consulted to place an order.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

Weights = Union[Mapping[str, float], np.ndarray]


@dataclass
class RiskMatrix:
    """An annualized covariance matrix Σ over a universe, plus risk math on it."""

    symbols: List[str]
    sigma: np.ndarray  # N×N annualized covariance, positive-definite
    shrinkage: Optional[float] = None  # δ used (Ledoit–Wolf), for audit
    #: Set when Σ was conditioned (spec 024): method, λ, per-name D_t vs the
    #: unconditional diagonal (the "sigma_regime" diagnostic). ``None`` unconditional.
    conditional_diagnostics: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self._index = {sym: i for i, sym in enumerate(self.symbols)}

    # ------------------------------------------------------------------ #
    # Weight handling
    # ------------------------------------------------------------------ #
    def _vector(self, weights: Weights) -> np.ndarray:
        """Align a weight mapping to Σ's symbol order (missing names → 0)."""
        if isinstance(weights, np.ndarray):
            if weights.shape != (len(self.symbols),):
                raise ValueError(f"weight vector must have length {len(self.symbols)}")
            return weights.astype(float)
        vec = np.zeros(len(self.symbols))
        for sym, w in weights.items():
            idx = self._index.get(sym)
            if idx is not None:
                vec[idx] = w
        return vec

    # ------------------------------------------------------------------ #
    # Risk quantities (Σ is annualized, so these are annualized too)
    # ------------------------------------------------------------------ #
    def variance(self, weights: Weights) -> float:
        """Portfolio variance ``wᵀ Σ w``."""
        w = self._vector(weights)
        return float(w @ self.sigma @ w)

    def volatility(self, weights: Weights) -> float:
        """Portfolio volatility ``√(wᵀ Σ w)`` (annualized)."""
        return float(np.sqrt(max(self.variance(weights), 0.0)))

    def tracking_error(self, weights: Weights, benchmark: Weights) -> float:
        """Tracking error ``√(w_aᵀ Σ w_a)`` for active weights ``w_a = w − w_B``."""
        active = self._vector(weights) - self._vector(benchmark)
        return float(np.sqrt(max(active @ self.sigma @ active, 0.0)))

    def implied_beta(self, benchmark: Weights) -> pd.Series:
        """The Σ-implied benchmark beta per name: ``β = Σw_B / (w_Bᵀ Σ w_B)``.

        The one canonical beta for anything benchmark-portfolio-relative (reverse
        optimization, active-beta diagnostics, alpha neutralization) - spec 017 §4.3
        calls this "one β, everywhere": per-name regression betas (the alpha
        pipeline's ``beta`` feature) are a different, complementary quantity and
        must not be mixed with this one. Zero vector when the benchmark carries no
        risk (``w_B = 0`` or degenerate Σ), so callers reduce to "no benchmark"
        rather than dividing by zero.
        """
        wb = self._vector(benchmark)
        denom = float(wb @ self.sigma @ wb)
        if denom <= 0:
            return pd.Series(0.0, index=self.symbols)
        return pd.Series((self.sigma @ wb) / denom, index=self.symbols)

    def marginal_contribution_to_risk(
        self, weights: Weights, benchmark: Optional[Weights] = None
    ) -> pd.Series:
        """Per-name MCR ``(Σ w_a) / ψ`` - ``∂ψ/∂w_a``, the risk gradient."""
        active = self._vector(weights)
        if benchmark is not None:
            active = active - self._vector(benchmark)
        te = float(np.sqrt(max(active @ self.sigma @ active, 0.0)))
        if te == 0:
            return pd.Series(0.0, index=self.symbols)
        return pd.Series((self.sigma @ active) / te, index=self.symbols)

    def correlation(self) -> pd.DataFrame:
        """The implied correlation matrix (Σ scaled by its diagonal std)."""
        std = np.sqrt(np.diag(self.sigma))
        denom = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, self.sigma / denom, 0.0)
        return pd.DataFrame(corr, index=self.symbols, columns=self.symbols)

    def volatilities(self) -> pd.Series:
        """Annualized volatility per name (√diagonal of Σ)."""
        return pd.Series(np.sqrt(np.diag(self.sigma)), index=self.symbols)

    def condition_number(self) -> float:
        """Condition number of Σ - how close to singular (the invertibility guard)."""
        return float(np.linalg.cond(self.sigma))

    def is_positive_definite(self) -> bool:
        """Whether Σ is positive-definite (so ``Σ⁻¹`` exists for the optimizer)."""
        try:
            np.linalg.cholesky(self.sigma)
            return True
        except np.linalg.LinAlgError:
            return False


class RiskModel(ABC):
    """Estimates a per-bar covariance over a returns panel (columns = symbols)."""

    @abstractmethod
    def estimate(self, returns: pd.DataFrame) -> Tuple[np.ndarray, Optional[float]]:
        """Return ``(Σ_per_bar, shrinkage)`` over complete-case ``returns`` columns."""


def build_return_panel(bars: Dict[str, pd.DataFrame], min_obs: int = 60) -> Tuple[pd.DataFrame, List[str]]:
    """Build an aligned (dates × symbols) return panel and the under-sampled names.

    Names enter and leave the universe, so the raw panel is ragged. We keep names
    with at least ``min_obs`` returns and align them on their common (complete-case)
    dates; names below the floor are returned separately so the caller can apply the
    documented fallback (a cross-sectional median variance) rather than drop them.
    """
    returns = pd.DataFrame(
        {sym: frame["close"].pct_change() for sym, frame in bars.items() if not frame.empty}
    )
    if returns.empty:
        return returns, []
    counts = returns.notna().sum()
    kept = [sym for sym in returns.columns if counts[sym] >= min_obs]
    under_sampled = [sym for sym in returns.columns if sym not in kept]
    panel = returns[kept].dropna() if kept else returns.iloc[0:0]
    return panel, under_sampled


def build_risk_matrix(
    model: RiskModel,
    bars: Dict[str, pd.DataFrame],
    periods_per_year: float,
    min_obs: int = 60,
    conditional: Optional[str] = None,
    conditional_lambda: Optional[float] = None,
    conditional_horizon: int = 1,
) -> Optional[RiskMatrix]:
    """Estimate an annualized :class:`RiskMatrix` over a universe's scanned bars.

    Builds the aligned return panel, estimates Σ on the well-sampled names, annualizes
    it, and splices in under-sampled names with the documented fallback - a
    cross-sectional **median variance** and zero correlation - so the matrix spans the
    full universe and stays positive-definite rather than dropping names silently.
    Returns ``None`` if no name has enough history.

    ``conditional`` (spec 024, default ``None`` / off) conditions the well-sampled
    block's diagonal via an EWMA (``"ewma"``) or HAR-lite (``"har"``) per-name
    volatility forecast, keeping the correlation structure fixed
    (``Σ_t = D_t·R·D_t`` - see :mod:`src.risk.conditional`), *before* the
    under-sampled splice below - so thin-history names keep the same
    unconditional-fallback treatment regardless of conditioning.
    """
    panel, under_sampled = build_return_panel(bars, min_obs=min_obs)
    universe = [sym for sym in bars if not bars[sym].empty]
    if panel.empty or panel.shape[0] < 2:
        return None

    sigma_bar, shrinkage = model.estimate(panel)
    sigma = sigma_bar * periods_per_year
    kept = list(panel.columns)
    matrix = RiskMatrix(symbols=kept, sigma=sigma, shrinkage=shrinkage)
    if conditional:
        from src.risk.conditional import condition_risk_matrix

        matrix = condition_risk_matrix(
            matrix,
            panel,
            conditional,
            periods_per_year,
            min_obs=min_obs,
            lambda_=conditional_lambda,
            horizon=conditional_horizon,
        )

    if not under_sampled:
        return matrix

    # Fallback for thin-history names: independent, at the median estimated variance.
    median_var = float(np.median(np.diag(matrix.sigma))) if matrix.sigma.size else 0.0
    order = [sym for sym in universe if sym in kept or sym in under_sampled]
    idx = {sym: i for i, sym in enumerate(kept)}
    full = np.zeros((len(order), len(order)))
    for a, sa in enumerate(order):
        for b, sb in enumerate(order):
            if sa in idx and sb in idx:
                full[a, b] = matrix.sigma[idx[sa], idx[sb]]
        if sa in under_sampled:
            full[a, a] = median_var
    return RiskMatrix(
        symbols=order, sigma=full, shrinkage=matrix.shrinkage, conditional_diagnostics=matrix.conditional_diagnostics
    )
