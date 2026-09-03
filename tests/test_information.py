"""Tests for information analysis: IC, breadth, IR reconciliation."""

from datetime import datetime

import numpy as np
import pandas as pd

from tests.fakes import DictMarketData, make_ohlcv
from tradeflow.analytics.information import (
    effective_breadth,
    ic_stats,
    ir_standard_error,
    multiple_testing_inflation,
    pearson_ic,
    predicted_ir,
    rank_ic,
)
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services import analysis


# --- breadth (the headline trap) ---------------------------------------------
def test_breadth_collapses_with_correlation():
    # ρ̄ = 1 → one effective bet per rebalance; ρ̄ = 0 → the full name count.
    assert effective_breadth(50, 12, 1.0)["br_eff"] == 12.0
    assert effective_breadth(50, 12, 0.0)["br_eff"] == 600.0


# --- IC stats ----------------------------------------------------------------
def test_zero_skill_is_not_significant():
    rng = np.random.default_rng(0)
    stats = ic_stats(list(rng.normal(0, 0.1, 200)))  # informationless
    assert abs(stats["mean_ic"]) < 0.02
    assert abs(stats["ic_tstat"]) < 2.0


def test_synthetic_skill_is_recovered():
    rng = np.random.default_rng(1)
    ic_true = 0.10
    ics = []
    for _ in range(60):
        a = pd.Series(rng.normal(0, 1, 50))
        noise = pd.Series(rng.normal(0, 1, 50))
        realized = ic_true * a + np.sqrt(1 - ic_true**2) * noise  # corr(a, realized) ≈ ic_true
        ics.append(pearson_ic(a, realized))
    stats = ic_stats(ics)
    assert abs(stats["mean_ic"] - ic_true) < 0.03
    assert stats["ic_tstat"] > 2.0  # 60 periods of real skill is distinguishable
    assert predicted_ir(stats["mean_ic"], effective_breadth(50, 12, 0.0)["br_eff"]) > 0


# --- alignment / look-ahead ----------------------------------------------------
def test_ic_detects_alignment():
    rng = np.random.default_rng(2)
    realized = pd.Series(rng.normal(0, 1, 50))
    assert abs(pearson_ic(realized.copy(), realized) - 1.0) < 1e-9  # forecast == realized → IC 1
    shuffled = pd.Series(realized.sample(frac=1, random_state=3).to_numpy())  # mis-aligned period
    assert abs(pearson_ic(shuffled, realized)) < 0.5  # wrong alignment → no IC


def test_rank_ic_is_robust_to_monotone_scaling():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    realized = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(rank_ic(a, realized) - 1.0) < 1e-9
    assert abs(rank_ic(a**3, realized) - 1.0) < 1e-9  # cubing preserves ranks


# --- guardrail formulas ------------------------------------------------------
def test_guardrail_formulas():
    assert abs(ir_standard_error(0.0, 1.0) - 1.0) < 1e-9  # SE ≈ 1/√T at IR 0
    assert ir_standard_error(0.5, 3.0) > 0.5  # a 3-yr IR of 0.5 is ~1 SE from 0
    assert abs(multiple_testing_inflation(1) - 0.05) < 1e-9
    assert multiple_testing_inflation(20) > 0.6


# --- service: zero-skill null + leakage --------------------------------------
def _client(n=600):
    syms = [f"S{i}" for i in range(12)]
    data = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    return MarketDataClient(DictMarketData(data)), syms


def test_compute_information_zero_skill_null():
    client, syms = _client()
    r = analysis.compute_information(
        client, "demo_trend", syms, datetime(2023, 1, 1), datetime(2024, 12, 31), n_points=20, n_trials=10
    )
    assert r["periods"] > 0
    assert abs(r["ic_tstat"]) < 2.0  # random walk → no distinguishable skill
    assert r["breadth_effective"] <= r["breadth_naive"]
    assert abs(r["multiple_testing_inflation"] - (1 - 0.95**10)) < 1e-9


# --- attribution -------------------------------------------------------------
def test_factor_split_closes():
    from tradeflow.services.analysis import _factor_split

    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, 8)
    x = rng.normal(0, 1, (8, 3))
    r = rng.normal(0, 0.02, 8)
    factor, specific = _factor_split(w, x, r)
    assert abs((factor + specific) - float(w @ r)) < 1e-12  # the split reconstructs w·r exactly


def test_compute_information_reports_attribution():
    client, syms = _client()
    r = analysis.compute_information(
        client, "demo_trend", syms, datetime(2023, 1, 1), datetime(2024, 12, 31), n_points=16
    )
    assert "factor_return" in r and "specific_return" in r
    assert np.isfinite(r["factor_return"]) and np.isfinite(r["specific_return"])


def test_compute_information_independent_of_post_end_bars():
    syms = [f"S{i}" for i in range(8)]
    full = {s: make_ohlcv(n=700, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    cutoff = full["S0"].index[600]
    end = cutoff.to_pydatetime()
    start = full["S0"].index[120].to_pydatetime()
    truncated = {s: f.loc[f.index <= cutoff] for s, f in full.items()}

    a = analysis.compute_information(
        MarketDataClient(DictMarketData(truncated)), "demo_trend", syms, start, end
    )
    b = analysis.compute_information(MarketDataClient(DictMarketData(full)), "demo_trend", syms, start, end)
    assert a["mean_ic"] == b["mean_ic"]
    assert a["predicted_ir"] == b["predicted_ir"]
    assert a["realized_ir"] == b["realized_ir"]
