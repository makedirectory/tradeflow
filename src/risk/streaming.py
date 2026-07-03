"""Out-of-core risk estimation — stream the return panel instead of materializing it.

:func:`src.risk.base.build_risk_matrix` (and the factor model that shares its
:func:`build_return_panel`) is the only genuinely **panel-wide** (``T×N``) pandas
materialization in the stack: it builds an aligned returns DataFrame for the whole
history and hands it to an estimator. For a broad universe over many years that ``T×N``
panel is the thing that does not fit in RAM.

This module is the out-of-core counterpart (spec 015): it estimates the *same*
quantities by **streaming** the return panel out of a
:class:`~src.data.store.ParquetBarStore` in date-chunks and accumulating sufficient
statistics. It never holds the ``T×N`` panel: peak working memory is one chunk
(``chunk_obs × N``) plus the small accumulators (``N×N`` / ``K×K`` / ``N``) plus a 1-D
``O(T)`` index of complete-case timestamps — so on a long, wide history it peaks well
below the eager path (the dominant ``T×N`` matrix is never materialized), growing in
``T`` only as a thin timestamp vector.

Two estimators, both reproducing their eager oracle to machine epsilon for the
well-sampled names:

- :func:`streaming_sample_covariance` ↔ :func:`build_risk_matrix` with
  :class:`~src.risk.sample.SampleCovariance`.
- :func:`streaming_factor_risk_matrix` ↔ :func:`src.risk.factor.estimate_factor_model`
  (the structural ``Σ = X F Xᵀ + Δ``). Exposures stay a small ``N×K`` leaf computed by
  :func:`src.risk.exposures.build_factor_exposures`.

Ledoit–Wolf shrinkage needs higher moments and stays on the eager path for now — noted,
not faked. Requires the ``store`` extra (``polars``).
"""

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.marketdata.client import TimeframeLike
from src.risk.base import RiskMatrix
from src.risk.factor import FactorRiskMatrix

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.data.store import ParquetBarStore


def _kept_and_blocks(
    store: "ParquetBarStore",
    universe: List[str],
    timeframe: TimeframeLike,
    as_of: datetime,
    lookback_days: int,
    min_obs: int,
    chunk_obs: int,
) -> Optional[Tuple[List[str], Callable[[], Iterable[np.ndarray]]]]:
    """The shared streaming spine: the complete-case return panel, never materialized.

    Reproduces :func:`build_return_panel` exactly: keep names with ≥ ``min_obs`` finite
    returns (in ``universe`` order), take the timestamps where **every** kept name has a
    finite return (the lazy analogue of ``returns[kept].dropna()``), and return a factory
    that streams that complete-case panel in ``chunk_obs``-row blocks (columns in ``kept``
    order). Returns ``(kept, blocks)`` or ``None`` if fewer than two complete-case rows.
    """
    import polars as pl

    from src.data import compute

    lf = compute.with_returns(store.scan_lazy(universe, timeframe, as_of, lookback_days), src="close")
    # Present == finite return: mirror pandas' notna/dropna (null and NaN are missing).
    nn = lf.filter(pl.col("ret").is_not_null() & pl.col("ret").is_not_nan())

    counts = nn.group_by("symbol").agg(pl.len().alias("n")).collect(engine="streaming")
    count_map = dict(zip(counts["symbol"].to_list(), counts["n"].to_list()))
    kept: List[str] = []
    seen = set()
    for sym in universe:
        if sym in seen:
            continue
        seen.add(sym)
        if count_map.get(sym, 0) >= min_obs:
            kept.append(sym)
    if not kept:
        return None

    k = len(kept)
    complete = (
        nn.filter(pl.col("symbol").is_in(kept))
        .group_by("ts")
        .agg(pl.len().alias("c"))
        .filter(pl.col("c") == k)
        .select("ts")
        .collect(engine="streaming")
        .sort("ts")["ts"]
        .to_list()
    )
    if len(complete) < 2:
        return None

    def blocks() -> Iterable[np.ndarray]:
        for i in range(0, len(complete), chunk_obs):
            ts_chunk = complete[i : i + chunk_obs]
            block = (
                nn.filter(pl.col("symbol").is_in(kept) & pl.col("ts").is_in(ts_chunk))
                .select(["ts", "symbol", "ret"])
                .collect(engine="streaming")
                .pivot(values="ret", index="ts", on="symbol")
                .sort("ts")
            )
            yield block.select(kept).to_numpy()  # rows × N, columns in kept order

    return kept, blocks


def streaming_sample_covariance(
    store: "ParquetBarStore",
    universe: List[str],
    timeframe: TimeframeLike,
    as_of: datetime,
    *,
    lookback_days: int = 365,
    periods_per_year: float = 252.0,
    min_obs: int = 60,
    chunk_obs: int = 1024,
) -> Optional[RiskMatrix]:
    """Annualized sample :class:`RiskMatrix` by streaming the return panel.

    Mirrors :func:`build_risk_matrix` with :class:`~src.risk.sample.SampleCovariance`
    (population covariance), accumulating cross-products chunk-by-chunk via
    :func:`src.data.compute.streaming_covariance`. Returns ``None`` when no name clears
    ``min_obs`` or fewer than two complete-case rows exist; ``shrinkage`` is ``None``.
    Kept names are returned in ``universe`` order.
    """
    from src.data import compute

    prepared = _kept_and_blocks(store, universe, timeframe, as_of, lookback_days, min_obs, chunk_obs)
    if prepared is None:
        return None
    kept, blocks = prepared

    sigma_bar, n_obs = compute.streaming_covariance(blocks())
    if n_obs < 2:
        return None
    return RiskMatrix(symbols=kept, sigma=sigma_bar * periods_per_year, shrinkage=None)


def streaming_factor_risk_matrix(
    store: "ParquetBarStore",
    universe: List[str],
    timeframe: TimeframeLike,
    as_of: datetime,
    exposures: pd.DataFrame,
    *,
    lookback_days: int = 365,
    periods_per_year: float = 252.0,
    min_obs: int = 60,
    chunk_obs: int = 1024,
) -> Optional[FactorRiskMatrix]:
    """Structural factor :class:`FactorRiskMatrix` ``Σ = X F Xᵀ + Δ`` by streaming returns.

    Mirrors :func:`src.risk.factor.estimate_factor_model`: factor returns are recovered
    per period by the cross-sectional OLS ``f_t = (XᵀX)⁻¹Xᵀ r_t``, ``F`` is their
    covariance (``ddof=1``, as ``np.cov``) and ``Δ`` the per-name residual variance
    (``ddof=1``) — both annualized. The ``N×K`` ``exposures`` (a small cross-sectional
    leaf from :func:`src.risk.build_factor_exposures`) stay pandas; only the panel-wide
    ``T×N`` returns are streamed. Each chunk is projected to factor returns and residuals
    and folded into ``K×K`` / ``N`` accumulators, so the ``T×N`` panel is never held.

    Names are the well-sampled set ∩ ``exposures.index`` (the eager path's selection),
    in ``universe`` order. Returns ``None`` if fewer than two names/periods survive.
    """
    prepared = _kept_and_blocks(store, universe, timeframe, as_of, lookback_days, min_obs, chunk_obs)
    if prepared is None:
        return None
    kept, blocks = prepared

    names = [s for s in kept if s in exposures.index]
    if len(names) < 2:
        return None
    col = [kept.index(s) for s in names]  # pick the `names` columns out of each kept-ordered block
    x = exposures.loc[names].to_numpy(dtype=float)  # N×K
    proj = x @ np.linalg.pinv(x.T @ x)  # N×K; factor returns of a chunk R are R @ proj

    n_obs = 0
    k = x.shape[1]
    sf = np.zeros(k)  # Σ f_t
    sff = np.zeros((k, k))  # Σ f_t f_tᵀ
    sr = np.zeros(len(names))  # Σ residual_t (per name)
    srr = np.zeros(len(names))  # Σ residual_t² (per name)
    for block in blocks():
        r = block[:, col]  # rows × N (names columns)
        f = r @ proj  # rows × K factor returns
        resid = r - f @ x.T  # rows × N residuals
        sf += f.sum(axis=0)
        sff += f.T @ f
        sr += resid.sum(axis=0)
        srr += (resid * resid).sum(axis=0)
        n_obs += r.shape[0]

    if n_obs < 2:
        return None
    # ddof=1 sample stats from the streamed sums, to match np.cov / Series.var(ddof=1).
    mean_f = sf / n_obs
    factor_cov = (sff / n_obs - np.outer(mean_f, mean_f)) * (n_obs / (n_obs - 1)) * periods_per_year
    factor_cov = np.atleast_2d(factor_cov)
    specific_var = (srr - sr * sr / n_obs) / (n_obs - 1) * periods_per_year
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
