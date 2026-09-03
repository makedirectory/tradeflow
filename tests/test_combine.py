"""Tests for multi-signal alpha combination & shrinkage.

The combination math is the heart and is tested in closed form; the measurement and
the service flow are exercised for structure and the as-of/leakage guard.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from tests.fakes import DictMarketData, make_ohlcv
from tradeflow.alphas import strategy_scorer
from tradeflow.alphas.combine import (
    combination_weights,
    combine_scores,
    combined_ic,
    effective_ic,
    measure_signals,
    shrink_ic,
)
from tradeflow.data import ClientBarSource
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services import analysis
from tradeflow.services.analysis import _strategy


# --- two-signal closed form --------------------------------------------------
def test_two_signal_closed_form():
    ic1, ic2, rho = 0.05, 0.03, 0.4
    assert abs(effective_ic(ic1, ic2, rho) - (ic1 - rho * ic2) / (1 - rho**2)) < 1e-12
    corr = np.array([[1.0, rho], [rho, 1.0]])
    expected = np.sqrt((ic1**2 + ic2**2 - 2 * rho * ic1 * ic2) / (1 - rho**2))
    assert abs(combined_ic([ic1, ic2], corr, ridge=0.0) - expected) < 1e-12


# --- redundancy down-weighting -----------------------------------------------
def test_duplicate_signal_splits_weight_and_does_not_inflate_ic():
    corr = np.array([[1.0, 1.0], [1.0, 1.0]])  # identical signals
    w = combination_weights(np.array([0.05, 0.05]), corr)
    assert abs(w[0] - w[1]) < 1e-9  # symmetric
    # Combined IC ≈ a single signal's IC, not √2 larger (no double-counting).
    assert abs(combined_ic([0.05, 0.05], corr) - 0.05) < 1e-3


# --- independent uplift > redundant uplift -----------------------------------
def test_independent_signal_adds_more_than_redundant_one():
    base = 0.05
    uplift_independent = combined_ic([base, 0.02], np.array([[1.0, 0.0], [0.0, 1.0]])) - base
    uplift_redundant = combined_ic([base, base], np.array([[1.0, 0.98], [0.98, 1.0]])) - base
    assert uplift_independent > uplift_redundant
    assert uplift_independent > 0


# --- shrinkage limits --------------------------------------------------------
def test_shrinkage_limits():
    assert shrink_ic(0.05, 1) < 0.005  # tiny sample → ~0
    assert shrink_ic(0.001, 100) < 1e-4  # tiny IC → ~0
    assert abs(shrink_ic(0.05, 1_000_000) - 0.05) < 1e-4  # T·IC² → ∞ → unchanged
    assert shrink_ic(0.0, 100) == 0.0


def test_combine_scores_is_weighted_zscore():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    # Equal weights on opposite-ranked signals → ~flat combined score.
    combined = combine_scores(frame, np.array([1.0, 1.0]))
    assert abs(combined.sum()) < 1e-9
    assert len(combined) == 4


# --- measurement structure ---------------------------------------------------
def _setup(n=400, seed_offset=0):
    syms = [f"S{i}" for i in range(10)]
    data = {s: make_ohlcv(n=n, seed=i + seed_offset, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    return MarketDataClient(DictMarketData(data)), syms


def test_measure_signals_structure():
    client, syms = _setup()
    bars = ClientBarSource(client).scan([*syms, "SPY"], "1Day", datetime(2024, 6, 1), 365)
    bench = bars.pop("SPY")
    scorers = {n: strategy_scorer(_strategy(n, None)) for n in ("demo_trend", "demo_trend")}
    m = measure_signals(bars, scorers, bench, datetime(2024, 6, 1), horizon=5, n_points=10)

    assert m.n_periods > 0
    assert set(m.ics) == {"demo_trend", "demo_trend"}
    assert all(np.isfinite(v) for v in m.ics.values())
    # Shrinkage never increases the magnitude of an IC.
    for s in m.signals:
        assert abs(m.shrunk_ics[s]) <= abs(m.ics[s]) + 1e-12
    # Correlation matrix is symmetric with unit diagonal.
    c = m.correlation.to_numpy()
    assert np.allclose(c, c.T) and np.allclose(np.diag(c), 1.0)


# --- as-of / OOS discipline --------------------------------------------------
def test_combined_alphas_independent_of_post_as_of_bars():
    syms = [f"S{i}" for i in range(8)]
    full = {s: make_ohlcv(n=400, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    cutoff = full["S0"].index[300]
    as_of = cutoff.to_pydatetime()
    truncated = {s: f.loc[f.index <= cutoff] for s, f in full.items()}

    signals = ["demo_trend", "demo_trend"]
    a = analysis.compute_combined_alphas(MarketDataClient(DictMarketData(truncated)), signals, syms, as_of)
    b = analysis.compute_combined_alphas(MarketDataClient(DictMarketData(full)), signals, syms, as_of)
    assert a["alphas"] == b["alphas"]
    assert a["combined_ic"] == b["combined_ic"]
    assert a["signal_weights"] == b["signal_weights"]


def test_ics_differ_across_windows():
    """Weights/ICs are measured from data, so different windows give different ICs."""
    client, syms = _setup()
    bars_all = ClientBarSource(client).scan([*syms, "SPY"], "1Day", datetime(2025, 1, 1), 720)
    bench = bars_all["SPY"]
    universe = {s: bars_all[s] for s in syms}
    scorers = {n: strategy_scorer(_strategy(n, None)) for n in ("demo_trend", "demo_trend")}
    early = measure_signals(universe, scorers, bench, datetime(2024, 1, 1), horizon=5, n_points=10)
    late = measure_signals(universe, scorers, bench, datetime(2025, 1, 1), horizon=5, n_points=10)
    assert early.ics != late.ics  # measured, not assumed
