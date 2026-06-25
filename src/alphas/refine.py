"""Alpha refinement primitives - the pure-math steps that turn a raw signal into a
scaled, comparable residual-return forecast.

Every function here is a pure cross-sectional transform on a pandas Series indexed
by symbol: same names in, same names out, inputs untouched. They are deliberately
standalone (no AlphaModel, no I/O) so each step of the pipeline in
:mod:`src.alphas.base` is unit-testable in isolation and composable.

The pipeline order:

    raw scores  ->  winsorize  ->  zscore  ->  [neutralize]  ->  scale  ->  cap

All standardisation here is **cross-sectional** (across names at one rebalance),
never across time for one name - standardising over time would reintroduce the
look-ahead this module exists to avoid.
"""

import numpy as np
import pandas as pd


def winsorize(scores: pd.Series, lower: float = 0.025, upper: float = 0.975) -> pd.Series:
    """Clip scores to their ``[lower, upper]`` cross-sectional quantiles.

    Kills single-name outliers so one extreme score can't dominate the z-score's
    mean/std. With too few names the quantiles are meaningless; callers skip this
    step on thin universes (see :mod:`src.alphas.base`).
    """
    if scores.empty:
        return scores
    lo = scores.quantile(lower)
    hi = scores.quantile(upper)
    return scores.clip(lower=lo, upper=hi)


def demean(scores: pd.Series) -> pd.Series:
    """Subtract the cross-sectional mean (the thin-universe fallback for zscore).

    Centres the scores at 0 without dividing by the cross-sectional std, which is
    unstable on a handful of names. Preserves relative spacing; sets no scale.
    """
    if scores.empty:
        return scores
    return scores - scores.mean()


def zscore(scores: pd.Series) -> pd.Series:
    """Cross-sectional standardisation: ``z = (s - mean) / std``.

    Uses the population std (``ddof=0``) so the output has mean 0 and unit
    dispersion exactly. If every score is identical (zero std) the result is all
    zeros rather than NaN/inf - a degenerate universe expresses no view.
    """
    if scores.empty:
        return scores
    centered = scores - scores.mean()
    std = scores.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return centered * 0.0
    return centered / std


def neutralize(z: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    """Regress ``z`` on ``exposures`` and return the residual (orthogonal to them).

    Makes the alpha neutral to the supplied exposures: an intercept is always
    included (so the residual is mean-zero, i.e. benchmark/equal-weight neutral),
    and the residual is orthogonal to every exposure column by construction
    (the defining property of an OLS residual). Pass a single beta column for
    benchmark-beta neutralisation, or a factor-exposure matrix for factor
    neutralisation.

    ``exposures`` is a DataFrame indexed by symbol; it is aligned to ``z`` and any
    name missing an exposure is dropped from the regression and returned unchanged.
    """
    if z.empty or exposures is None or exposures.empty:
        return z

    frame = exposures.reindex(z.index)
    valid = frame.dropna().index
    if len(valid) < 2:
        return z

    y = z.loc[valid].to_numpy(dtype=float)
    # Design matrix: an intercept column plus the exposures.
    x = np.column_stack([np.ones(len(valid)), frame.loc[valid].to_numpy(dtype=float)])
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coeffs

    out = z.copy()
    out.loc[valid] = residual
    return out


def scale_to_alpha(z: pd.Series, residual_vol: pd.Series, ic: float) -> pd.Series:
    """Apply the refinement identity ``alpha_i = sigma_i * IC * z_i``.

    Turns a standardised score into a forecast of annualised residual return. The
    cross-sectional std of the result is ``~ IC * sigma`` (for unit-std ``z``), so
    the alphas carry the right information ratio for a mean-variance optimiser to
    size positions by genuine conviction rather than by an arbitrary signal scale.

    ``residual_vol`` is aligned to ``z`` by symbol.
    """
    vol = residual_vol.reindex(z.index)
    # Evaluated as (sigma * IC) * z to match the written identity exactly.
    return vol * ic * z


def cap(alpha: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Final sanity bound: clip alphas to ``+/- n_std * std(alpha)``.

    A guard against a single name carrying an implausible forecast after scaling.
    With a degenerate (zero-std) alpha vector this is a no-op.
    """
    if alpha.empty:
        return alpha
    std = alpha.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return alpha
    bound = n_std * std
    return alpha.clip(lower=-bound, upper=bound)
