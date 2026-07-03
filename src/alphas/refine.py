"""Alpha refinement primitives - the pure-math steps that turn a raw signal into a
scaled, comparable residual-return forecast.

Every function here is a pure cross-sectional transform on a pandas Series indexed
by symbol: same names in, same names out, inputs untouched. They are deliberately
standalone (no state, no I/O) so each step of the pipeline in
:mod:`src.alphas.base` is unit-testable in isolation and composable.

The pipeline order:

    raw scores  ->  winsorize  ->  zscore  ->  [neutralize]  ->  scale  ->  cap

All standardization here is **cross-sectional** (across names at one rebalance),
never across time for one name - standardizing over time would reintroduce the
look-ahead this module exists to avoid.
"""

import math

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

    Centers the scores at 0 without dividing by the cross-sectional std, which is
    unstable on a handful of names. Preserves relative spacing; sets no scale.
    """
    if scores.empty:
        return scores
    return scores - scores.mean()


def zscore(scores: pd.Series) -> pd.Series:
    """Cross-sectional standardization: ``z = (s - mean) / std``.

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
    benchmark-beta neutralization, or a factor-exposure matrix for factor
    neutralization.

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

    Turns a standardized score into a forecast of annualized residual return. The
    cross-sectional std of the result is ``~ IC * sigma`` (for unit-std ``z``), so
    the alphas carry the right information ratio for a mean-variance optimizer to
    size positions by genuine conviction rather than by an arbitrary signal scale.

    ``residual_vol`` is aligned to ``z`` by symbol. This is the **Case 1** scaling
    (see :func:`case_test`): correct when the signal's per-name time-series vol is
    roughly constant across names, so the cross-sectional z equals the time-series z
    the standard rule is stated in. For a **Case 2** signal pass a constant
    :func:`case_scale_factor` as ``residual_vol`` instead of the per-name vector.
    """
    vol = residual_vol.reindex(z.index)
    # Evaluated as (sigma * IC) * z to match the written identity exactly.
    return vol * ic * z


def case_scale_factor(
    raw_scores: pd.Series,
    residual_vol: pd.Series,
    winsorize_limits: tuple = (0.025, 0.975),
) -> float:
    """The **Case 2** constant scale ``c_g = Std_CS{g} / Std_CS{g/ω}``.

    A Case-2 signal (per-name signal vol ∝ the name's residual vol ``ω_n`` —
    empirically most price/estimate signals) already carries the volatility inside
    the raw score; multiplying by ``ω_n`` in :func:`scale_to_alpha` double-counts it
    and tilts the book into high-vol names. The fix replaces the per-name ``ω_n`` with
    one cross-sectional **constant** ``c_g`` (a vol-dimensioned, dispersion-matched
    representative vol), so ``alpha = IC · c_g · z`` sizes by conviction without the
    spurious vol tilt.

    Computed on the **raw** signal (winsorized the same way the pipeline winsorizes),
    because after standardization the raw dispersion that defines ``c_g`` is gone.
    Returns NaN when the ratio can't be formed (too few names, all-zero vol) so the
    caller can fall back.
    """
    g = raw_scores.dropna()
    if len(g) < 2:
        return float("nan")
    g = winsorize(g, *winsorize_limits)
    vol = residual_vol.reindex(g.index)
    ratio = (g / vol.where(vol > 0)).replace([np.inf, -np.inf], np.nan).dropna()
    num = g.std(ddof=0)
    den = ratio.std(ddof=0)
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def case_test(
    signal_history: pd.DataFrame,
    residual_vol: pd.Series,
    *,
    price_derived: bool = True,
    min_obs: int = 36,
    r2_case2: float = 0.25,
    r2_case1: float = 0.05,
    t_threshold: float = 2.0,
) -> dict:
    """Decide a signal's scaling: **Case 1** (``ω·IC·z``) vs **Case 2** (``IC·c_g·z``).

    The standard refinement rule ``α = ω·IC·z`` is stated in *time-series* z-scores,
    but the pipeline computes *cross-sectional* z. Whether the two coincide is an
    empirical property of each signal:

    - **Case 1** — per-name time-series signal vol ``Std_TS{g_n}`` is ~constant across
      names ⇒ ``z_TS ≈ z_CS`` and the ``ω_n`` multiply is right (the classic
      sector-momentum example).
    - **Case 2** — ``Std_TS{g_n}`` is ~proportional to the name's residual vol ``ω_n``
      (most price/estimate signals) ⇒ the vol is already inside the raw signal and the
      ``ω_n`` multiply double-counts it.

    The test regresses ``Std_TS{g_n} = a + b·ω_n`` across names: a strong, significant
    positive slope ⇒ Case 2. ``signal_history`` is a time×name frame of the **raw**
    signal; ``residual_vol`` the per-name ``ω_n``. Needs ``≥ min_obs`` time observations
    to engage; below that (or in the ambiguous band ``r2_case1 < R² < r2_case2``) it
    returns the empirical base rate — **Case 2 for price-derived signals, Case 1
    otherwise** — flagged ``ambiguous`` so a wrong call is visible, not silent.
    """
    default_case = 2 if price_derived else 1

    def _result(case, r2, t, slope, n_obs, n_names, engaged, ambiguous, reason):
        return {
            "case": case,
            "r_squared": float(r2),
            "t_stat": float(t),
            "slope": float(slope),
            "n_obs": int(n_obs),
            "n_names": int(n_names),
            "engaged": bool(engaged),
            "ambiguous": bool(ambiguous),
            "reason": reason,
        }

    if signal_history is None or signal_history.empty:
        return _result(default_case, 0.0, 0.0, 0.0, 0, 0, False, True, "no signal history")

    std_ts = signal_history.std(ddof=1)
    counts = signal_history.count()
    n_obs = int(counts.median()) if len(counts) else 0

    aligned = pd.concat([std_ts.rename("y"), residual_vol.rename("x")], axis=1).dropna()
    # A name needs enough time observations for its Std_TS to mean anything.
    aligned = aligned[counts.reindex(aligned.index).fillna(0) >= min(min_obs, max(n_obs, 2))]
    n_names = len(aligned)

    if n_obs < min_obs or n_names < 3:
        return _result(
            default_case,
            0.0,
            0.0,
            0.0,
            n_obs,
            n_names,
            False,
            True,
            f"insufficient history (n_obs={n_obs} < {min_obs} or n_names={n_names} < 3); "
            f"defaulting to Case {default_case} ({'price-derived' if price_derived else 'other'})",
        )

    x = aligned["x"].to_numpy()
    y = aligned["y"].to_numpy()
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx == 0:
        return _result(default_case, 0.0, 0.0, 0.0, n_obs, n_names, False, True, "no ω dispersion")
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    resid = y - (ym + slope * (x - xm))
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = n_names - 2
    se_b = math.sqrt((ss_res / dof) / sxx) if dof > 0 and sxx > 0 else float("inf")
    t_stat = slope / se_b if np.isfinite(se_b) and se_b > 0 else 0.0

    if r2 >= r2_case2 and t_stat >= t_threshold and slope > 0:
        return _result(
            2,
            r2,
            t_stat,
            slope,
            n_obs,
            n_names,
            True,
            False,
            f"Std_TS ∝ ω (R²={r2:.2f}, t={t_stat:.1f}): vol already in the signal",
        )
    if r2 <= r2_case1:
        return _result(
            1,
            r2,
            t_stat,
            slope,
            n_obs,
            n_names,
            True,
            False,
            f"Std_TS ⊥ ω (R²={r2:.2f}): per-name signal vol ~constant",
        )
    return _result(
        default_case,
        r2,
        t_stat,
        slope,
        n_obs,
        n_names,
        True,
        True,
        f"ambiguous (R²={r2:.2f}, t={t_stat:.1f}); defaulting to Case {default_case} "
        f"({'price-derived' if price_derived else 'other'})",
    )


def level_shrink_factor(ic: float, t_eff: float) -> float:
    """IC-uncertainty level shrink: ``1/(1 + 1/(T_eff·IC²)) = g/(g+1)``, ``g=T_eff·IC²``.

    The IC that scales alphas is itself estimated, and its sampling error dominates the
    mapping (``Var{Δβ/β} ≈ 1/(IC²·T)``). This Bayes-with-zero-prior factor
    shrinks the alpha **level** toward zero for that estimation error: ``→ 1`` as
    ``T_eff·IC² → ∞`` (skill well established), ``→ 0`` as ``IC → 0`` or ``T_eff → 0``.
    ``T_eff`` must be *effective independent* observations (overlapping horizons inflate
    a raw row count — see :func:`src.alphas.horizon.effective_sample_size`). Anchors:
    ``IC=0.05, T=60 → 0.13``; ``IC=0.10, T=120 → 0.55``.
    """
    g = t_eff * ic * ic
    return float(g / (g + 1.0)) if np.isfinite(g) and g > 0 else 0.0


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
