"""Sample-covariance estimators: raw, and Ledoit–Wolf shrinkage.

The raw sample covariance ``S`` over ``N`` names from ``T`` observations needs
``N(N+1)/2`` parameters; when ``T`` is not far greater than ``N`` it is noisy and
often **non-invertible** - fatal, because the optimiser needs ``Σ⁻¹``.
Ledoit–Wolf shrinks ``S`` toward a structured target ``F`` by the analytically
optimal intensity ``δ``:

    Σ̂ = δ·F + (1 − δ)·S

Here ``F`` is the **constant-correlation** target (every pairwise correlation = the
average sample correlation, variances kept), which is the right structure for
equity returns. The closed form is vendored in pure numpy (Ledoit & Wolf, 2004,
"Honey, I Shrunk the Sample Covariance Matrix") so the risk module needs no
compiled or sklearn dependency.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.risk.base import RiskModel


class SampleCovariance(RiskModel):
    """The raw sample covariance. Honest, but ill-conditioned when ``T ≲ N``."""

    def estimate(self, returns: pd.DataFrame) -> Tuple[np.ndarray, Optional[float]]:
        x = returns.to_numpy(dtype=float)
        x = x - x.mean(axis=0)
        t = x.shape[0]
        if t < 2:
            n = x.shape[1]
            return np.zeros((n, n)), None
        return (x.T @ x) / t, None


class LedoitWolfCovariance(RiskModel):
    """Ledoit–Wolf shrinkage toward a constant-correlation target (well-conditioned)."""

    def estimate(self, returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        x = returns.to_numpy(dtype=float)
        x = x - x.mean(axis=0)
        t, n = x.shape
        if t < 2 or n == 0:
            return np.zeros((n, n)), 0.0

        sample = (x.T @ x) / t  # MLE sample covariance
        var = np.diag(sample)
        std = np.sqrt(var)
        if n == 1 or np.any(std == 0):
            # No correlation structure to shrink (or a dead name); S is the answer.
            return sample, 0.0

        # Constant-correlation target F: average off-diagonal correlation, variances kept.
        outer_std = np.outer(std, std)
        corr = sample / outer_std
        r_bar = (corr.sum() - n) / (n * (n - 1))
        target = r_bar * outer_std
        np.fill_diagonal(target, var)

        # π̂ — sum of asymptotic variances of the sample-covariance entries.
        x2 = x**2
        pi_mat = (x2.T @ x2) / t - sample**2
        pi_hat = pi_mat.sum()

        # ρ̂ — asymptotic covariances; diagonal terms plus the constant-correlation
        # off-diagonal adjustment (the θ terms).
        rho_diag = np.trace(pi_mat)
        cubed = (x**3).T @ x / t  # A[i,j] = E[x_i^3 x_j]
        theta_ii = cubed - var[:, None] * sample  # θ_{ii,ij}
        theta_jj = cubed.T - var[None, :] * sample  # θ_{jj,ij}
        ratio = np.outer(std, 1.0 / std)  # ratio[i,j] = std_i / std_j
        off = (r_bar / 2.0) * (ratio.T * theta_ii + ratio * theta_jj)
        np.fill_diagonal(off, 0.0)
        rho_hat = rho_diag + off.sum()

        # γ̂ — misfit of the target to the sample covariance.
        gamma_hat = float(((target - sample) ** 2).sum())

        if gamma_hat == 0:
            delta = 0.0
        else:
            kappa = (pi_hat - rho_hat) / gamma_hat
            delta = max(0.0, min(1.0, kappa / t))

        sigma = delta * target + (1.0 - delta) * sample
        return sigma, float(delta)
