"""Tests for the out-of-core streaming sample covariance.

Proves the streaming estimator reproduces the eager ``build_risk_matrix`` oracle
(to machine epsilon) and that it runs in memory bounded by chunk size, not history.
Skipped unless the ``store`` extra (pyarrow + polars) is installed.
"""

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("polars")

import pandas as pd  # noqa: E402

from src.data import ParquetBarStore  # noqa: E402
from src.risk import (  # noqa: E402
    build_factor_exposures,
    build_factor_risk_matrix,
    build_risk_matrix,
    streaming_factor_risk_matrix,
    streaming_sample_covariance,
)
from src.risk.sample import SampleCovariance  # noqa: E402
from tests.fakes import make_ohlcv  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
PPY = 252.0


def _store(tmp_path, n=300):
    bars = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate(SYMBOLS)}
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    as_of = max(b.index[-1] for b in bars.values()).to_pydatetime()
    return store, bars, as_of


def _aligned(eager, streamed):
    """Reindex the eager Σ to the streamed symbol order for an apples-to-apples compare."""
    idx = [eager.symbols.index(s) for s in streamed.symbols]
    return eager.sigma[np.ix_(idx, idx)]


def test_matches_eager_build_risk_matrix(tmp_path):
    store, bars, as_of = _store(tmp_path)
    eager = build_risk_matrix(SampleCovariance(), bars, PPY, min_obs=60)
    streamed = streaming_sample_covariance(
        store, SYMBOLS, "1Day", as_of, lookback_days=10_000, periods_per_year=PPY, min_obs=60
    )
    assert set(streamed.symbols) == set(eager.symbols)
    np.testing.assert_allclose(streamed.sigma, _aligned(eager, streamed), atol=1e-12)
    assert streamed.shrinkage is None


def test_chunk_size_does_not_change_result(tmp_path):
    store, _, as_of = _store(tmp_path)
    ref = streaming_sample_covariance(
        store, SYMBOLS, "1Day", as_of, lookback_days=10_000, periods_per_year=PPY, chunk_obs=10_000
    )
    for chunk in (16, 37, 128):
        got = streaming_sample_covariance(
            store, SYMBOLS, "1Day", as_of, lookback_days=10_000, periods_per_year=PPY, chunk_obs=chunk
        )
        np.testing.assert_allclose(got.sigma, ref.sigma, atol=1e-12)


def test_is_positive_definite_and_annualised(tmp_path):
    store, _, as_of = _store(tmp_path)
    streamed = streaming_sample_covariance(store, SYMBOLS, "1Day", as_of, lookback_days=10_000)
    # Variances are annualized (×252) and the matrix is usable by the optimizer.
    assert streamed.is_positive_definite()
    daily = streaming_sample_covariance(
        store, SYMBOLS, "1Day", as_of, lookback_days=10_000, periods_per_year=1.0
    )
    np.testing.assert_allclose(streamed.sigma, daily.sigma * 252.0, atol=1e-12)


def test_under_sampled_names_dropped_like_oracle(tmp_path):
    # A name with < min_obs returns is excluded from the streamed kept-set, matching
    # build_return_panel's `kept` selection.
    bars = {s: make_ohlcv(n=300, seed=i, freq="1D") for i, s in enumerate(["AAA", "BBB"])}
    bars["SHORT"] = make_ohlcv(n=20, seed=99, freq="1D")  # only 19 returns < min_obs=60
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    as_of = max(b.index[-1] for b in bars.values()).to_pydatetime()
    streamed = streaming_sample_covariance(
        store, ["AAA", "BBB", "SHORT"], "1Day", as_of, lookback_days=10_000, min_obs=60
    )
    assert set(streamed.symbols) == {"AAA", "BBB"}


def test_kept_order_follows_universe(tmp_path):
    store, _, as_of = _store(tmp_path)
    universe = ["CCC", "AAA", "EEE", "BBB", "DDD"]
    streamed = streaming_sample_covariance(store, universe, "1Day", as_of, lookback_days=10_000)
    assert streamed.symbols == universe  # all well-sampled → preserved in universe order


def test_too_few_observations_returns_none(tmp_path):
    bars = {"AAA": make_ohlcv(n=20, seed=0, freq="1D")}  # < min_obs
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    as_of = bars["AAA"].index[-1].to_pydatetime()
    assert streaming_sample_covariance(store, ["AAA"], "1Day", as_of, min_obs=60) is None


def test_matches_oracle_across_year_boundary(tmp_path):
    # Multi-year history: the streamed estimate (which chunks across year partitions)
    # still matches the eager oracle exactly.
    idx = pd.date_range("2019-01-01", periods=365 * 4, freq="1D", tz="UTC")
    bars = {}
    for i, s in enumerate(["AAA", "BBB", "CCC"]):
        rng = np.random.default_rng(i)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        bars[s] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx
        )
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    as_of = idx[-1].to_pydatetime()
    eager = build_risk_matrix(SampleCovariance(), bars, PPY, min_obs=60)
    streamed = streaming_sample_covariance(
        store, ["AAA", "BBB", "CCC"], "1Day", as_of, lookback_days=10_000, chunk_obs=200
    )
    np.testing.assert_allclose(streamed.sigma, _aligned(eager, streamed), atol=1e-12)


def test_streaming_peak_below_eager_on_long_history(tmp_path):
    # The point of the migration: on a long, wide history the streaming path peaks well
    # below the eager build_risk_matrix, which materializes the full T×N return panel.
    import tracemalloc

    wide = [f"S{i}" for i in range(20)]

    def build(n, tag):
        bars = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate(wide)}
        store = ParquetBarStore(tmp_path / tag)
        store.write(bars, timeframe="1Day")
        as_of = max(b.index[-1] for b in bars.values()).to_pydatetime()
        return bars, store, as_of

    # Warm up Polars' one-time fixed allocations OUTSIDE the measured window.
    _, warm_store, warm_as_of = build(60, "warm")
    streaming_sample_covariance(warm_store, wide, "1Day", warm_as_of, lookback_days=10_000, chunk_obs=64)

    bars, store, as_of = build(4000, "big")
    tracemalloc.start()
    streaming_sample_covariance(store, wide, "1Day", as_of, lookback_days=10_000, chunk_obs=64)
    streaming_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    tracemalloc.start()
    build_risk_matrix(SampleCovariance(), bars, PPY, min_obs=60)
    eager_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert streaming_peak < eager_peak  # never holds the T×N panel


# --------------------------------------------------------------------------- #
# Streaming factor risk model (Σ = X F Xᵀ + Δ) vs the eager estimate_factor_model
# --------------------------------------------------------------------------- #
def _factor_setup(tmp_path, n=300, syms=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
    bars = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate(syms)}
    benchmark = make_ohlcv(n=n, seed=99, freq="1D")
    store = ParquetBarStore(tmp_path)
    store.write(bars, timeframe="1Day")
    as_of = max(b.index[-1] for b in bars.values()).to_pydatetime()
    return store, bars, benchmark, list(syms), as_of


def test_factor_matches_eager_estimate(tmp_path):
    store, bars, benchmark, syms, as_of = _factor_setup(tmp_path)
    eager = build_factor_risk_matrix(bars, benchmark, PPY, min_obs=60)
    exposures = build_factor_exposures(bars, benchmark)
    streamed = streaming_factor_risk_matrix(
        store, syms, "1Day", as_of, exposures, lookback_days=10_000, periods_per_year=PPY, min_obs=60
    )
    assert streamed.symbols == eager.symbols
    np.testing.assert_allclose(streamed.sigma, eager.sigma, atol=1e-12)
    np.testing.assert_allclose(streamed.factor_cov, eager.factor_cov, atol=1e-12)
    np.testing.assert_allclose(streamed.specific_var, eager.specific_var, atol=1e-12)
    assert streamed.factor_names == eager.factor_names


def test_factor_chunk_invariant(tmp_path):
    store, bars, benchmark, syms, as_of = _factor_setup(tmp_path)
    exposures = build_factor_exposures(bars, benchmark)
    ref = streaming_factor_risk_matrix(
        store, syms, "1Day", as_of, exposures, lookback_days=10_000, chunk_obs=10_000
    )
    for chunk in (16, 50, 128):
        got = streaming_factor_risk_matrix(
            store, syms, "1Day", as_of, exposures, lookback_days=10_000, chunk_obs=chunk
        )
        np.testing.assert_allclose(got.sigma, ref.sigma, atol=1e-12)
        np.testing.assert_allclose(got.factor_cov, ref.factor_cov, atol=1e-12)


def test_factor_attribution_quantities_usable(tmp_path):
    store, bars, benchmark, syms, as_of = _factor_setup(tmp_path)
    exposures = build_factor_exposures(bars, benchmark)
    streamed = streaming_factor_risk_matrix(store, syms, "1Day", as_of, exposures, lookback_days=10_000)
    w = {s: 1.0 / len(streamed.symbols) for s in streamed.symbols}
    # The factor/specific split is consistent and the streamed Σ is positive-definite.
    assert streamed.factor_variance(w) >= 0 and streamed.specific_variance(w) >= 0
    assert streamed.is_positive_definite()


def test_factor_drops_names_without_exposure(tmp_path):
    store, bars, benchmark, syms, as_of = _factor_setup(tmp_path)
    exposures = build_factor_exposures(bars, benchmark).drop(index="CCC")  # no exposure for CCC
    streamed = streaming_factor_risk_matrix(store, syms, "1Day", as_of, exposures, lookback_days=10_000)
    assert "CCC" not in streamed.symbols and set(streamed.symbols) <= set(syms)
