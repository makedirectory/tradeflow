"""Structural factor risk model: Σ = X F Xᵀ + Δ.

A factor model is the preferred form for active risk: the parameter count drops from
``O(N²)`` to ``O(N·K)``, Σ is invertible and stable by construction (``K`` small), and
— crucially — risk becomes **attributable**, splitting into *factor* risk
(``w_aᵀ X F Xᵀ w_a``) and *specific* risk (``w_aᵀ Δ w_a``). That decomposition is what
the information-analysis attribution and the alpha factor-neutralization both consume.

Factor returns ``f_t`` are recovered each period by a cross-sectional regression of
that period's returns on the (slowly-varying) exposures ``X``; ``F`` is their
covariance and ``Δ`` the diagonal variance of the residuals. Everything is annualized.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.risk.base import RiskMatrix, build_return_panel
from src.risk.exposures import build_factor_exposures


@dataclass
class FactorRiskMatrix(RiskMatrix):
    """A :class:`RiskMatrix` whose Σ is ``X F Xᵀ + Δ``, with the pieces kept for attribution."""

    exposures: Optional[pd.DataFrame] = None  # N×K (standardized)
    factor_cov: Optional[np.ndarray] = None  # K×K, annualized
    specific_var: Optional[np.ndarray] = None  # N, annualized
    factor_names: List[str] = field(default_factory=list)

    def factor_variance(self, weights) -> float:
        """Active variance explained by the common factors: ``w_aᵀ X F Xᵀ w_a``."""
        w = self._vector(weights)
        x = self.exposures.loc[self.symbols].to_numpy()
        fx = x.T @ w
        return float(fx @ self.factor_cov @ fx)

    def specific_variance(self, weights) -> float:
        """Idiosyncratic active variance: ``w_aᵀ Δ w_a``."""
        w = self._vector(weights)
        return float(np.sum(self.specific_var * w * w))


def estimate_factor_model(
    returns: pd.DataFrame, exposures: pd.DataFrame, periods_per_year: float
) -> Optional[FactorRiskMatrix]:
    """Estimate ``Σ = X F Xᵀ + Δ`` from a returns panel and an exposure matrix.

    ``returns`` is (T × names), ``exposures`` (names × K). Factor returns are recovered
    per period by cross-sectional OLS ``f_t = (XᵀX)⁻¹ Xᵀ r_t``; ``F`` is their
    covariance and ``Δ`` the residual variance per name (both annualized). Returns
    ``None`` if too few names/periods overlap.
    """
    names = [c for c in returns.columns if c in exposures.index]
    if len(names) < 2 or len(returns) < 2:
        return None

    r = returns[names].to_numpy()  # T×N
    x = exposures.loc[names].to_numpy()  # N×K
    xtx_inv = np.linalg.pinv(x.T @ x)  # K×K (pinv guards a thin/collinear X)

    factor_returns = r @ x @ xtx_inv  # T×K: each row f_t = (XᵀX)⁻¹Xᵀ r_t
    residuals = r - factor_returns @ x.T  # T×N

    factor_cov = np.cov(factor_returns, rowvar=False) * periods_per_year
    factor_cov = np.atleast_2d(factor_cov)
    specific_var = residuals.var(axis=0, ddof=1) * periods_per_year
    sigma = x @ factor_cov @ x.T + np.diag(specific_var)

    return FactorRiskMatrix(
        symbols=names,
        sigma=sigma,
        shrinkage=None,
        exposures=exposures.loc[names],
        factor_cov=factor_cov,
        specific_var=specific_var,
        factor_names=list(exposures.columns),
    )


def build_factor_risk_matrix(
    bars,
    benchmark_bars,
    periods_per_year: float,
    min_obs: int = 60,
    conditional: Optional[str] = None,
    conditional_lambda: Optional[float] = None,
    conditional_horizon: int = 1,
) -> Optional[FactorRiskMatrix]:
    """Build factor exposures from ``bars`` then estimate the factor covariance Σ.

    ``conditional`` (default ``None`` / off) conditions ``factor_cov`` AND
    ``specific_var`` together via the same EWMA/HAR family (never one without the
    other — a partial conditioning mis-splits the factor/specific attribution); see
    :func:`src.risk.conditional.condition_risk_matrix`. Loadings ``X`` stay slow.
    """
    exposures = build_factor_exposures(bars, benchmark_bars)
    if exposures.empty:
        return None
    panel, _ = build_return_panel(bars, min_obs=min_obs)
    if panel.empty:
        return None
    matrix = estimate_factor_model(panel, exposures, periods_per_year)
    if matrix is None or not conditional:
        return matrix

    from src.risk.conditional import condition_risk_matrix

    return condition_risk_matrix(
        matrix,
        panel,
        conditional,
        periods_per_year,
        min_obs=min_obs,
        lambda_=conditional_lambda,
        horizon=conditional_horizon,
    )
