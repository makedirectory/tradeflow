"""Tests for performance attribution: adding-up, null calibration, planted skill,
the cumulation trap, Bayesian-blend limits, and the years-to-significance formula,
in checklist order."""

from datetime import datetime

import numpy as np
import pandas as pd

from tests.fakes import DictMarketData, make_ohlcv
from tradeflow.analytics.attribution import (
    attribute_period,
    bayesian_blend_variance,
    cross_sectional_regression,
    cumulate_top_down,
    prior_weight_t0,
    prob_positive_over_years,
    series_stats,
    systematic_split,
    years_to_significance,
)
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services import analysis

RISK_COLS = ["market", "momentum", "volatility", "size"]


def _period(rng, n=15, k_signal=0, b_signal=0.0, r_bench=None, beta_mean=1.0, signal_names=None):
    """One synthetic rebalance: random weights/exposures/betas, a benchmark return,
    and (optionally) a planted signal contribution of known magnitude ``b_signal``."""
    names = [f"S{i}" for i in range(n)]
    w = pd.Series(rng.normal(0, 1, n), index=names)
    w = w - w.mean()  # mean-zero active book
    risk_x = pd.DataFrame(rng.normal(0, 1, (n, 4)), index=names, columns=RISK_COLS)
    beta = pd.Series(rng.normal(beta_mean, 0.2, n), index=names)
    if r_bench is None:
        r_bench = float(rng.normal(0, 0.01))
    signal_x = None
    noise = rng.normal(0, 0.01, n)
    r_raw = beta.to_numpy() * r_bench + noise
    if k_signal:
        cols = signal_names or [f"sig{i}" for i in range(k_signal)]
        signal_x = pd.DataFrame(rng.normal(0, 1, (n, k_signal)), index=names, columns=cols)
        r_raw = r_raw + b_signal * signal_x[cols[0]].to_numpy()
    return w, risk_x, pd.Series(r_raw, index=names), beta, r_bench, signal_x


# --- cross_sectional_regression (the shared projection) ----------------------
def test_cross_sectional_regression_reconstructs_r_exactly():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (10, 3))
    r = rng.normal(0, 0.02, 10)
    b, resid = cross_sectional_regression(x, r)
    assert np.allclose(x @ b + resid, r, atol=1e-10)


def test_cross_sectional_regression_degenerate_returns_zero_b():
    x = np.zeros((2, 3))  # fewer names than columns + 1
    r = np.array([0.01, -0.02])
    b, resid = cross_sectional_regression(x, r)
    assert np.allclose(b, 0.0)
    assert np.allclose(resid, r)


# --- adding-up ----------------------------------------------------------------
def test_attribute_period_adds_up_exactly():
    rng = np.random.default_rng(1)
    w, risk_x, r_raw, beta, r_bench, _ = _period(rng)
    result = attribute_period(w, risk_x, r_raw, beta, r_bench)
    assert result is not None
    assert abs(result.residual) < 1e-9
    reconstructed = (
        result.systematic
        + sum(result.factor_contributions.values())
        + sum(result.signal_contributions.values())
        + result.specific
    )
    assert abs(reconstructed - result.r_active) < 1e-9
    assert abs(result.systematic - result.beta_a * result.r_bench) < 1e-12


def test_attribute_period_adds_up_with_signal_column():
    rng = np.random.default_rng(2)
    w, risk_x, r_raw, beta, r_bench, signal_x = _period(rng, k_signal=2, b_signal=0.03)
    result = attribute_period(w, risk_x, r_raw, beta, r_bench, signal_x=signal_x)
    assert result is not None
    assert abs(result.residual) < 1e-9
    assert set(result.signal_contributions) == set(signal_x.columns)


def test_attribute_period_too_few_names_returns_none():
    idx = ["A", "B"]  # 2 names, 4 risk factors -> needs >= 6
    w = pd.Series([0.5, -0.5], index=idx)
    risk_x = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (2, 4)), index=idx, columns=RISK_COLS)
    r_raw = pd.Series([0.01, -0.01], index=idx)
    beta = pd.Series([1.0, 1.0], index=idx)
    assert attribute_period(w, risk_x, r_raw, beta, 0.01) is None


# --- null calibration ----------------------------------------------------------
def test_null_book_attribution_is_centered_on_zero():
    """A book with no true skill: random weights/exposures/returns every period.
    Averaged over many periods, every row's mean should be small relative to its
    own dispersion (t-stats mostly < 2) - the honest "this is luck" case."""
    rng = np.random.default_rng(3)
    factor_series = {c: [] for c in RISK_COLS}
    specific_series = []
    for _ in range(200):
        w, risk_x, r_raw, beta, r_bench, _ = _period(rng)
        result = attribute_period(w, risk_x, r_raw, beta, r_bench)
        assert result is not None
        for c in RISK_COLS:
            factor_series[c].append(result.factor_contributions[c])
        specific_series.append(result.specific)

    beyond_2 = 0
    for c in RISK_COLS:
        stats = series_stats(factor_series[c], 12.0, sigma2_prior=np.var(factor_series[c]), t0=0.0)
        beyond_2 += abs(stats["t_stat"]) > 2.0
    stats = series_stats(specific_series, 12.0, sigma2_prior=np.var(specific_series), t0=0.0)
    beyond_2 += abs(stats["t_stat"]) > 2.0
    # 5 rows measured; with no true skill, seeing all of them "significant" would be
    # the multiple-testing trap this module exists to catch - assert we don't see that.
    assert beyond_2 <= 2


def _planted_signal_period(rng, n=60, b_signal=0.04):
    """A paper book actually built FROM the 'planted' signal (active weights =
    its z-score, the same ``z/z.std()`` construction ``compute_information``
    uses) - so a real per-signal return, not just a recoverable regression
    coefficient with no bearing on the book, shows up in the contribution."""
    names = [f"S{i}" for i in range(n)]
    planted = rng.normal(0, 1, n)
    noise_signal = rng.normal(0, 1, n)
    signal_x = pd.DataFrame({"planted": planted, "noise": noise_signal}, index=names)
    z = planted - planted.mean()
    w = pd.Series(z / z.std(), index=names)
    risk_x = pd.DataFrame(rng.normal(0, 1, (n, 4)), index=names, columns=RISK_COLS)
    beta = pd.Series(rng.normal(1.0, 0.2, n), index=names)
    r_bench = float(rng.normal(0, 0.01))
    noise = rng.normal(0, 0.01, n)
    r_raw = pd.Series(beta.to_numpy() * r_bench + noise + b_signal * planted, index=names)
    return w, risk_x, r_raw, beta, r_bench, signal_x


# --- planted skill --------------------------------------------------------------
def test_planted_signal_skill_is_recovered_in_the_right_row():
    rng = np.random.default_rng(4)
    b_true = 0.04
    signal_contribs, other_signal_contribs, factor_contribs = [], [], []
    for _ in range(120):
        w, risk_x, r_raw, beta, r_bench, signal_x = _planted_signal_period(rng, b_signal=b_true)
        result = attribute_period(w, risk_x, r_raw, beta, r_bench, signal_x=signal_x)
        assert result is not None
        signal_contribs.append(result.signal_contributions["planted"])
        other_signal_contribs.append(result.signal_contributions["noise"])
        factor_contribs.append(sum(result.factor_contributions.values()))

    planted_stats = series_stats(signal_contribs, 12.0, sigma2_prior=np.var(signal_contribs), t0=0.0)
    noise_stats = series_stats(
        other_signal_contribs, 12.0, sigma2_prior=np.var(other_signal_contribs), t0=0.0
    )
    factor_stats = series_stats(factor_contribs, 12.0, sigma2_prior=np.var(factor_contribs), t0=0.0)

    assert planted_stats["t_stat"] > 4.0  # the planted signal lights up hard
    assert abs(noise_stats["t_stat"]) < 2.0  # the undoctored signal does not
    assert abs(factor_stats["t_stat"]) < 2.0  # neither do the risk factors


def test_planted_timing_skill_lights_up_only_the_timing_row():
    """beta_a(t) constructed to correlate with the future r_bench(t) - genuine
    timing skill - should light up the timing term; the systematic split's
    other two buckets are aggregate-only (no t-stat) by design."""
    rng = np.random.default_rng(5)
    t = 60
    r_bench_series = rng.normal(0, 0.02, t)
    # Planted timing: beta tilts up exactly when the benchmark is about to be up.
    beta_a_series = 1.0 + 2.0 * r_bench_series + rng.normal(0, 0.01, t)
    split = systematic_split(beta_a_series, r_bench_series, mu_b_period=0.0)

    timing_stats = series_stats(
        split["timing_series"], 12.0, sigma2_prior=np.var(split["timing_series"]), t0=0.0
    )
    assert timing_stats["t_stat"] > 4.0
    assert split["timing"] > 0

    # A null (no-timing) book: beta_a independent of r_bench.
    beta_null = 1.0 + rng.normal(0, 0.01, t)
    null_split = systematic_split(beta_null, r_bench_series, mu_b_period=0.0)
    null_stats = series_stats(
        null_split["timing_series"], 12.0, sigma2_prior=np.var(null_split["timing_series"]), t0=0.0
    )
    assert abs(null_stats["t_stat"]) < 2.0


# --- systematic split identity ------------------------------------------------
def test_systematic_split_identity():
    rng = np.random.default_rng(6)
    beta = rng.normal(1.0, 0.3, 40)
    r_bench = rng.normal(0, 0.01, 40)
    mu_b = 0.002
    split = systematic_split(beta, r_bench, mu_b)
    total = float(np.sum(beta * r_bench))
    assert abs((split["expected"] + split["surprise"] + split["timing"]) - total) < 1e-9


def test_systematic_split_empty():
    split = systematic_split([], [], 0.01)
    assert split["expected"] == 0.0 and split["timing"] == 0.0


# --- the cumulation trap ---------------------------------------------------------
def test_cumulate_top_down_identity():
    rng = np.random.default_rng(7)
    t = 24
    a = rng.normal(0.002, 0.01, t)
    b = rng.normal(0.003, 0.02, t)
    p = a + b  # r_active = r_portfolio - r_bench, by construction
    comp1 = rng.normal(0, 0.005, t)
    comp2 = a - comp1  # two components that sum exactly to r_active each period
    result = cumulate_top_down({"c1": comp1, "c2": comp2}, a, p, b)
    linked_sum = sum(result["linked_components"].values())
    assert abs(linked_sum - result["naive_cumulative"]) < 1e-9
    assert abs(linked_sum + result["delta_cp"] - result["honest_car"]) < 1e-9


def test_cumulation_trap_naive_diverges_materially_from_honest():
    """Large per-period returns: compounding the additive split is NOT the same
    as compounding the true portfolio/benchmark returns - the trap this module
    exists to catch. delta_cp must be non-trivial in this constructed case."""
    t = 20
    rng = np.random.default_rng(8)
    b = rng.normal(0.0, 0.08, t)  # large per-period moves
    a = rng.normal(0.01, 0.02, t)
    p = a + b
    result = cumulate_top_down({"whole": a}, a, p, b)
    assert abs(result["delta_cp"]) > 1e-3
    assert abs(result["naive_cumulative"] - result["honest_car"]) > 1e-3


def test_cumulate_top_down_empty():
    result = cumulate_top_down({"c": []}, [], [], [])
    assert result["naive_cumulative"] == 0.0
    assert result["honest_car"] == 0.0


# --- Bayesian blend limits -------------------------------------------------------
def test_bayesian_blend_limits():
    prior, realized = 0.02, 0.10
    # T -> 0: recovers the prior.
    assert abs(bayesian_blend_variance(prior, realized, t=0.0, t0=10.0) - prior) < 1e-12
    # T -> large (>> T0): recovers the realized variance.
    blended = bayesian_blend_variance(prior, realized, t=1e6, t0=10.0)
    assert abs(blended - realized) < 1e-3
    # Weights always sum to 1 (both terms nonnegative, interpolated).
    mid = bayesian_blend_variance(prior, realized, t=10.0, t0=10.0)
    assert min(prior, realized) <= mid <= max(prior, realized)


def test_prior_weight_t0_scales_with_bars_per_period():
    assert prior_weight_t0(60, 1) == 60.0
    assert prior_weight_t0(60, 5) == 12.0
    assert prior_weight_t0(60, 60) == 1.0


def test_series_stats_short_sample_leans_on_prior():
    # Two wild points with a modest prior: the blended vol should sit much
    # closer to the prior than the raw 2-point sample SD would suggest.
    values = [0.5, -0.5]
    raw_sample_std = float(np.std(values, ddof=1))  # ~0.707
    stats = series_stats(values, periods_per_year_series=12.0, sigma2_prior=0.0001, t0=50.0)
    assert stats["vol_blended"] < raw_sample_std * 0.3


# --- years-to-significance vs the SE(IR) formula ---------------------------------
def test_years_to_significance_solves_t_equals_2():
    for ir in (0.2, 0.5, 1.0):
        y_star = years_to_significance(ir)
        assert abs(ir * math_sqrt(y_star) - 2.0) < 1e-9


def math_sqrt(x):
    import math

    return math.sqrt(x)


def test_years_to_significance_zero_ir_is_infinite():
    assert years_to_significance(0.0) == float("inf")


def test_prob_positive_over_years_sanity():
    assert abs(prob_positive_over_years(0.0, 5.0) - 0.5) < 1e-9
    assert prob_positive_over_years(1.0, 100.0) > 0.99  # huge IR, long horizon -> ~certain
    assert prob_positive_over_years(0.5, 0.0) == 0.5  # no time elapsed -> coin flip


def test_years_to_significance_matches_direct_ir_simulation():
    """Direct Monte Carlo: at Y* periods (years, one obs/year for simplicity), the
    empirical t-stat of a true-IR return stream averages near 2, matching the
    closed-form Y* = (2/IR)^2."""
    rng = np.random.default_rng(9)
    ir = 0.5
    y_star = years_to_significance(ir)
    n = int(round(y_star))
    trials = 4000
    tstats = []
    for _ in range(trials):
        r = rng.normal(ir, 1.0, n)  # unit-vol annual returns with mean = ir
        tstats.append(r.mean() / r.std(ddof=1) * math_sqrt(n))
    assert abs(np.mean(tstats) - 2.0) < 0.25


# --- service: compute_attribution (wiring, leakage, insufficient-data) ------
def _client(n=600):
    syms = [f"S{i}" for i in range(12)]
    data = {s: make_ohlcv(n=n, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    return MarketDataClient(DictMarketData(data)), syms


def test_compute_attribution_reports_every_row():
    client, syms = _client()
    r = analysis.compute_attribution(
        client, "volume_spike", syms, datetime(2023, 1, 1), datetime(2024, 12, 31), n_points=16
    )
    assert r["periods"] > 0
    expected_rows = {
        "beta_expected",
        "beta_surprise",
        "timing",
        "specific",
        *r["risk_factor_names"],
        *r["signal_names"],
    }
    assert expected_rows <= set(r["rows"])
    assert r["signal_names"] == ["alpha:volume_spike"]
    assert np.isfinite(r["total_active_ir"])
    assert r["verdict"] in ("distinguishable from luck", "NOT distinguishable from luck")


def test_compute_attribution_cumulation_identity_holds():
    client, syms = _client()
    r = analysis.compute_attribution(
        client, "volume_spike", syms, datetime(2023, 1, 1), datetime(2024, 12, 31), n_points=16
    )
    linked_sum = sum(r["cumulation"]["linked_components"].values())
    assert abs(linked_sum - r["cumulation"]["naive_cumulative"]) < 1e-6
    assert abs(linked_sum + r["cumulation"]["delta_cp"] - r["cumulation"]["honest_car"]) < 1e-6


def test_compute_attribution_with_extra_signals():
    client, syms = _client()
    r = analysis.compute_attribution(
        client,
        "volume_spike",
        syms,
        datetime(2023, 1, 1),
        datetime(2024, 12, 31),
        n_points=12,
        signals=["ma_crossover"],
    )
    assert "ma_crossover" in r["signal_names"]
    assert "ma_crossover" in r["rows"]


def test_compute_attribution_insufficient_data_returns_note():
    client, syms = _client(n=20)
    r = analysis.compute_attribution(
        client, "volume_spike", syms, datetime(2023, 1, 1), datetime(2024, 12, 31), n_points=16
    )
    assert r["periods"] == 0 or "note" in r


def test_compute_attribution_independent_of_post_end_bars():
    syms = [f"S{i}" for i in range(8)]
    full = {s: make_ohlcv(n=700, seed=i, freq="1D") for i, s in enumerate([*syms, "SPY"])}
    cutoff = full["S0"].index[600]
    end = cutoff.to_pydatetime()
    start = full["S0"].index[120].to_pydatetime()
    truncated = {s: f.loc[f.index <= cutoff] for s, f in full.items()}

    a = analysis.compute_attribution(
        MarketDataClient(DictMarketData(truncated)), "ma_crossover", syms, start, end, n_points=12
    )
    b = analysis.compute_attribution(
        MarketDataClient(DictMarketData(full)), "ma_crossover", syms, start, end, n_points=12
    )
    assert a["periods"] == b["periods"]
    assert a["total_active_ir"] == b["total_active_ir"]
    assert a["cumulation"]["honest_car"] == b["cumulation"]["honest_car"]
