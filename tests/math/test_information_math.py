"""Math contract: forecast skill, breadth, signal combination, decay and attribution.

Independent references: ``scipy.stats`` for correlations, plain least squares for the
residual returns breadth is measured on, a brute-force grid for the IC-maximising blend,
and the attribution adding-up identities written out as sums.

Related existing tests: ``tests/test_information.py``, ``tests/test_breadth_correlation.py``,
``tests/test_horizon.py``, ``tests/test_combine.py``, ``tests/test_attribution.py``,
``tests/test_forecast_refinement_v2.py``.

Defects this module guards:

* breadth was deflated by the correlation of *total* returns, which include the market
  move the residual IC has already stripped out, instead of the residual returns;
* ``blend_weights`` was not the IC-maximising blend of the current and lagged signal.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantred.alphas import combine, horizon, refine
from quantred.analytics import attribution as attr
from quantred.analytics import information as info


# --------------------------------------------------------------------------- #
# Information coefficient and its statistics
# --------------------------------------------------------------------------- #
def _cross_section(seed=0, n=60, skill=0.3):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    r = skill * f + rng.normal(size=n)
    idx = [f"S{i}" for i in range(n)]
    return pd.Series(f, index=idx), pd.Series(r, index=idx)


def test_pearson_and_rank_ic_match_scipy():
    f, r = _cross_section(1)
    assert info.pearson_ic(f, r) == pytest.approx(stats.pearsonr(f, r)[0], rel=1e-12)
    assert info.rank_ic(f, r) == pytest.approx(stats.spearmanr(f, r)[0], rel=1e-12)


def test_rank_ic_is_invariant_to_monotone_transforms_of_the_forecast():
    f, r = _cross_section(2)
    assert info.rank_ic(np.exp(3 * f), r) == pytest.approx(info.rank_ic(f, r), abs=1e-12)
    assert info.pearson_ic(np.exp(3 * f), r) != pytest.approx(info.pearson_ic(f, r), abs=1e-3)


def test_ic_is_unavailable_not_zero_when_it_cannot_be_measured():
    f, r = _cross_section(3, 10)
    assert math.isnan(info.pearson_ic(f * 0 + 1, r)) and math.isnan(info.rank_ic(f * 0 + 1, r))
    assert math.isnan(info.pearson_ic(f.iloc[:1], r.iloc[:1]))
    with_nan = f.copy()
    with_nan.iloc[0] = np.nan
    assert info.pearson_ic(with_nan, r) == pytest.approx(stats.pearsonr(f.iloc[1:], r.iloc[1:])[0], rel=1e-12)


def test_ic_stats_known_answer():
    ics = [0.02, 0.06, 0.04, float("nan"), 0.08]
    xs = [0.02, 0.06, 0.04, 0.08]
    mean = sum(xs) / 4
    vol = math.sqrt(sum((x - mean) ** 2 for x in xs) / 3)
    out = info.ic_stats(ics)
    assert out["periods"] == 4 and out["mean_ic"] == pytest.approx(0.05)
    assert out["ic_vol"] == pytest.approx(vol, rel=1e-12)
    assert out["ic_tstat"] == pytest.approx(mean / (vol / 2), rel=1e-12)  # sqrt(4) = 2


@pytest.mark.xfail(
    strict=True,
    reason="a constant IC series has float-noise std (~1e-18), so the vol > 0 guard passes "
    "and the t-stat reads ~1e16 instead of the documented 0.0",
)
def test_ic_stats_with_a_constant_series_has_zero_tstat():
    assert info.ic_stats([0.05] * 6)["ic_tstat"] == 0.0


def test_ic_stats_with_no_data_is_zero_tstat_not_infinite():
    assert info.ic_stats([])["mean_ic"] == 0.0 and info.ic_stats([0.1])["ic_tstat"] == 0.0


# --------------------------------------------------------------------------- #
# Breadth and the fundamental law
# --------------------------------------------------------------------------- #
def test_effective_breadth_known_answers():
    out = info.effective_breadth(20, 12, 0.0)
    assert (out["br_naive"], out["br_eff"]) == (240.0, 240.0)  # independent bets keep the whole count
    assert info.effective_breadth(20, 12, 1.0)["br_eff"] == pytest.approx(12.0)  # one bet, rebalanced
    # 240 / (1 + 19 * 0.25)
    assert info.effective_breadth(20, 12, 0.25)["br_eff"] == pytest.approx(240 / 5.75, rel=1e-12)


def test_predicted_ir_is_mean_ic_times_root_breadth_and_floors_negative_breadth():
    assert info.predicted_ir(0.05, 400.0) == pytest.approx(1.0)
    assert info.predicted_ir(0.05, -4.0) == 0.0


def test_ir_standard_error_and_multiple_testing_inflation():
    assert info.ir_standard_error(0.0, 4.0) == pytest.approx(0.5)
    assert info.ir_standard_error(2.0, 1.0) == pytest.approx(math.sqrt(3.0))  # (1 + 4/2) / 1
    assert math.isinf(info.ir_standard_error(1.0, 0.0))
    assert info.multiple_testing_inflation(1) == pytest.approx(0.05)
    assert info.multiple_testing_inflation(20) == pytest.approx(1 - 0.95**20)
    assert info.multiple_testing_inflation(0) == 0.0


def _residual_world(seed=0, t=2000, n=6, loading=1.0):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, t)
    idio = rng.normal(0, 0.01, (t, n))
    beta = loading * rng.uniform(0.8, 1.2, n)
    rets = pd.DataFrame(market[:, None] * beta + idio, columns=[f"S{i}" for i in range(n)])
    return rets, pd.Series(market), idio


def test_mean_residual_correlation_uses_benchmark_residuals():
    """Reference: regress each name on the benchmark by least squares, correlate the
    residuals. Total-return correlation here is large (a strong shared market) while the
    residuals are independent noise, so the two answers are far apart."""
    rets, bench, _ = _residual_world(1)
    resid = []
    for c in rets:
        slope, intercept = np.polyfit(bench, rets[c], 1)
        resid.append(rets[c].to_numpy() - slope * bench.to_numpy() - intercept)
    corr = np.corrcoef(np.array(resid))
    n = corr.shape[0]
    expected = (corr.sum() - n) / (n * (n - 1))
    got = info.mean_residual_correlation(rets, bench)
    assert got == pytest.approx(expected, abs=1e-12)
    assert abs(got) < 0.05
    total = np.corrcoef(rets.to_numpy().T)
    assert (total.sum() - n) / (n * (n - 1)) > 0.3  # the number breadth used to be deflated by


def test_breadth_deflation_follows_the_residual_not_the_total_correlation():
    rets, bench, _ = _residual_world(2)
    rho = info.mean_residual_correlation(rets, bench)
    eff = info.effective_breadth(6, 12, rho)["br_eff"]
    assert eff == pytest.approx(72.0, rel=0.1)  # residuals are independent: almost no deflation


def test_mean_residual_correlation_degenerate_inputs_are_zero():
    rets, bench, _ = _residual_world(3, t=50)
    assert info.mean_residual_correlation(rets.iloc[:, :1], bench) == 0.0
    assert info.mean_residual_correlation(rets.iloc[:2], bench.iloc[:2]) == 0.0
    assert info.mean_residual_correlation(rets, bench * 0.0) == 0.0


# --------------------------------------------------------------------------- #
# Decay, blending and cadence
# --------------------------------------------------------------------------- #
def test_fit_decay_recovers_an_exact_exponential():
    ic0, delta = 0.1, 0.8
    out = horizon.fit_decay({n: ic0 * delta**n for n in range(6)})
    assert out["delta"] == pytest.approx(delta, rel=1e-10)
    assert out["ic0"] == pytest.approx(ic0, rel=1e-10)
    assert out["half_life"] == pytest.approx(-math.log(2) / math.log(delta), rel=1e-10)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert out["half_life_lower"] == pytest.approx(out["half_life"], rel=1e-6)  # zero residual: no band


def test_fit_decay_band_brackets_the_point_estimate_for_noisy_profiles():
    rng = np.random.default_rng(4)
    profile = {n: 0.1 * 0.85**n * math.exp(rng.normal(0, 0.1)) for n in range(8)}
    out = horizon.fit_decay(profile)
    assert out["half_life_lower"] <= out["half_life"] <= out["half_life_upper"]
    assert 0 < out["r_squared"] < 1


def test_fit_decay_with_too_little_data_is_nan_not_a_number():
    out = horizon.fit_decay({0: 0.1})
    assert math.isnan(out["delta"]) and math.isnan(out["half_life"])
    assert math.isnan(horizon.fit_decay({0: -0.1, 1: -0.2})["delta"])  # only positive ICs are usable


def test_peak_return_horizon_and_effective_sample_size():
    assert horizon.peak_return_horizon(10.0) == pytest.approx(12.566)
    assert math.isinf(horizon.peak_return_horizon(math.inf))
    assert horizon.effective_sample_size(210, 21, 1) == pytest.approx(10.0)
    assert (
        horizon.effective_sample_size(210, 21, 21) == 210.0
        and horizon.effective_sample_size(210, 5, 21) == 210.0
    )


def _blend_ic(w_now, w_lag, gamma, rho):
    """IC of the blend: w'c / sqrt(w' Omega w) with c = [1, gamma], Omega = [[1, rho], [rho, 1]]."""
    num = w_now + gamma * w_lag
    den = math.sqrt(w_now**2 + w_lag**2 + 2 * rho * w_now * w_lag)
    return num / den


@pytest.mark.parametrize(
    "gamma, rho", [(0.8, 0.5), (0.5, 0.8), (0.9, 0.1), (0.3, 0.3), (0.6, -0.2), (0.95, 0.9)]
)
def test_blend_weights_maximise_the_blend_ic_over_a_grid(gamma, rho):
    w_now, w_lag = horizon.blend_weights(gamma, rho)
    assert w_now + w_lag == pytest.approx(1.0, abs=1e-12)
    grid = np.linspace(-3, 4, 700_001)  # w_now on a grid, w_lag = 1 - w_now
    ics = [_blend_ic(w, 1 - w, gamma, rho) for w in grid[::100]]
    best = max(ics)
    assert _blend_ic(w_now, w_lag, gamma, rho) >= best - 1e-9
    # and the optimum is not at a grid edge: a finer search around it cannot beat it
    fine = [_blend_ic(w_now + e, w_lag - e, gamma, rho) for e in np.linspace(-1e-3, 1e-3, 201)]
    assert _blend_ic(w_now, w_lag, gamma, rho) >= max(fine) - 1e-12


def test_blend_weight_regimes_and_degenerate_cases():
    assert horizon.blend_weights(0.8, 0.5)[1] > 0  # gamma > rho: the lag carries independent information
    assert horizon.blend_weights(0.3, 0.7)[1] < 0  # gamma < rho: it mostly cancels current noise
    assert horizon.blend_weights(0.5, 0.5)[1] == pytest.approx(0.0, abs=1e-12)  # gamma = rho: latest alone
    assert horizon.blend_weights(0.9, 1.0) == (1.0, 0.0)  # identical signals: nothing to blend
    # closed form: (1 - g r) / ((1 + g)(1 - r))
    assert horizon.blend_weights(0.8, 0.5)[0] == pytest.approx((1 - 0.4) / (1.8 * 0.5), rel=1e-12)


def test_frequency_curve_and_recommended_cadence():
    ics = {1: 0.05, 4: 0.08, 16: 0.09, 64: 0.095}
    curve = horizon.frequency_ir_curve(ics)
    assert curve == pytest.approx({k: v / math.sqrt(k) for k, v in ics.items()})
    assert horizon.recommended_cadence(ics) == 1 and horizon.recommended_cadence({}) == 1
    assert horizon.recommended_cadence({1: 0.01, 4: 0.08}) == 4


# --------------------------------------------------------------------------- #
# Signal combination and the refinement identity
# --------------------------------------------------------------------------- #
def test_combined_ic_two_signal_closed_form_and_weights_are_gls():
    ic1, ic2, rho = 0.06, 0.04, 0.5
    omega = np.array([[1, rho], [rho, 1]])
    ics = np.array([ic1, ic2])
    expected = math.sqrt((ic1**2 + ic2**2 - 2 * rho * ic1 * ic2) / (1 - rho**2))
    assert combine.combined_ic(ics, omega, ridge=0.0) == pytest.approx(expected, rel=1e-12)
    w = combine.combination_weights(ics, omega, ridge=0.0)
    assert w == pytest.approx(np.linalg.inv(omega) @ ics, rel=1e-12)
    assert float(w @ ics) == pytest.approx(expected**2, rel=1e-12)  # IC' Omega^-1 IC


def test_redundant_signals_split_the_weight_and_a_weak_independent_one_adds():
    omega = np.array([[1, 0.99], [0.99, 1.0]])
    w = combine.combination_weights(np.array([0.05, 0.05]), omega, ridge=0.0)
    assert w == pytest.approx([0.05 / 1.99] * 2, rel=1e-10)  # each gets half of what one alone would
    alone = combine.combined_ic(np.array([0.05]), np.eye(1), ridge=0.0)
    with_weak = combine.combined_ic(np.array([0.05, 0.01]), np.eye(2), ridge=0.0)
    assert with_weak == pytest.approx(math.sqrt(0.05**2 + 0.01**2), rel=1e-12) and with_weak > alone


def test_effective_ic_and_shrink_ic_known_answers():
    assert combine.effective_ic(0.06, 0.04, 0.5) == pytest.approx((0.06 - 0.02) / 0.75, rel=1e-12)
    assert combine.effective_ic(0.06, 0.04, 1.0) == 0.0
    g = 100 * 0.05**2
    assert combine.shrink_ic(0.05, 100) == pytest.approx(0.05 * g / (g + 1), rel=1e-12)
    assert combine.shrink_ic(0.0, 100) == 0.0 and combine.shrink_ic(0.05, 0) == 0.0


def test_level_shrink_factor_anchors_and_limits():
    assert refine.level_shrink_factor(0.05, 60) == pytest.approx(0.15 / 1.15, rel=1e-12)  # ~0.13
    assert refine.level_shrink_factor(0.10, 120) == pytest.approx(1.2 / 2.2, rel=1e-12)  # ~0.55
    assert (
        refine.level_shrink_factor(0.0, 100) == 0.0 and refine.level_shrink_factor(0.1, float("nan")) == 0.0
    )
    assert refine.level_shrink_factor(0.5, 1e6) == pytest.approx(1.0, abs=1e-5)


def test_zscore_has_mean_zero_unit_population_std_and_degenerate_universe_is_flat():
    s = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0])
    z = refine.zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-15)
    assert math.sqrt(sum((v - z.mean()) ** 2 for v in z) / len(z)) == pytest.approx(1.0, abs=1e-12)
    assert (refine.zscore(pd.Series([2.0, 2.0, 2.0])) == 0).all()


def test_scale_to_alpha_is_vol_times_ic_times_z():
    z = pd.Series([1.0, -1.0, 0.5], index=list("abc"))
    vol = pd.Series([0.2, 0.3, 0.4], index=list("abc"))
    assert refine.scale_to_alpha(z, vol, 0.05).tolist() == pytest.approx([0.01, -0.015, 0.01], rel=1e-12)


def test_neutralize_leaves_a_residual_orthogonal_to_the_intercept_and_exposures():
    rng = np.random.default_rng(5)
    idx = [f"S{i}" for i in range(40)]
    exposures = pd.DataFrame(rng.normal(size=(40, 2)), index=idx, columns=["a", "b"])
    z = pd.Series(1.0 + exposures["a"] * 2 - exposures["b"] + rng.normal(size=40), index=idx)
    out = refine.neutralize(z, exposures)
    assert out.mean() == pytest.approx(0.0, abs=1e-12)
    assert exposures.T.to_numpy() @ out.to_numpy() == pytest.approx(0.0, abs=1e-10)


def test_cap_and_demean():
    a = pd.Series([0.0] * 20 + [10.0])
    capped = refine.cap(a, 3.0)
    assert capped.max() == pytest.approx(3.0 * a.std(ddof=0), rel=1e-12)
    assert refine.demean(pd.Series([1.0, 3.0])).tolist() == [-1.0, 1.0]
    assert refine.cap(pd.Series([1.0, 1.0])).tolist() == [1.0, 1.0]


# --------------------------------------------------------------------------- #
# Attribution identities
# --------------------------------------------------------------------------- #
def test_period_attribution_adds_up_exactly():
    rng = np.random.default_rng(6)
    n = 40
    idx = [f"S{i}" for i in range(n)]
    x = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=["m", "mom", "size"])
    sig = pd.DataFrame(rng.normal(size=(n, 1)), index=idx, columns=["sig"])
    w = pd.Series(rng.normal(0, 0.02, n), index=idx)
    beta = pd.Series(rng.uniform(0.7, 1.3, n), index=idx)
    r = pd.Series(rng.normal(0, 0.03, n), index=idx)
    r_b = 0.012
    out = attr.attribute_period(w, x, r, beta, r_b, sig)
    assert out.r_active == pytest.approx(float(w @ r), rel=1e-12)
    assert out.systematic == pytest.approx(float(w @ beta) * r_b, rel=1e-12)
    total = (
        out.systematic
        + sum(out.factor_contributions.values())
        + sum(out.signal_contributions.values())
        + out.specific
    )
    assert total == pytest.approx(out.r_active, abs=1e-14)
    assert abs(out.residual) < 1e-14


def test_period_attribution_recovers_a_planted_signal_return():
    rng = np.random.default_rng(7)
    n = 200
    idx = [f"S{i}" for i in range(n)]
    x = pd.DataFrame(rng.normal(size=(n, 1)), index=idx, columns=["m"])
    sig = pd.DataFrame(rng.normal(size=(n, 1)), index=idx, columns=["sig"])
    beta = pd.Series(1.0, index=idx)
    r = 0.01 * beta + 0.02 * sig["sig"] - 0.005 * x["m"]  # no noise
    out = attr.attribute_period(pd.Series(1.0 / n, index=idx), x, r, beta, 0.01, sig)
    assert out.signal_returns["sig"] == pytest.approx(0.02, abs=1e-12)
    assert out.factor_returns["m"] == pytest.approx(-0.005, abs=1e-12)
    assert abs(out.specific) < 1e-12


def test_attribution_with_too_few_names_is_none():
    idx = ["a", "b", "c"]
    x = pd.DataFrame(np.eye(3), index=idx, columns=["x", "y", "z"])
    assert (
        attr.attribute_period(
            pd.Series(1.0, index=idx), x, pd.Series(0.01, index=idx), pd.Series(1.0, index=idx), 0.0
        )
        is None
    )


def test_systematic_split_adds_back_to_beta_times_benchmark():
    rng = np.random.default_rng(8)
    beta, rb = rng.normal(0.1, 0.3, 24), rng.normal(0.01, 0.04, 24)
    out = attr.systematic_split(beta, rb, mu_b_period=0.008)
    assert out["expected"] + out["surprise"] + out["timing"] == pytest.approx(
        float((beta * rb).sum()), abs=1e-14
    )
    assert out["expected"] == pytest.approx(beta.mean() * 0.008 * 24, rel=1e-12)
    assert attr.systematic_split([], [], 0.01)["timing"] == 0.0


def test_cumulation_identity_with_compounding_residual():
    rng = np.random.default_rng(9)
    t = 18
    comps = {"a": rng.normal(0.002, 0.01, t), "b": rng.normal(0, 0.01, t)}
    r_a = comps["a"] + comps["b"]
    r_b = rng.normal(0.006, 0.03, t)
    r_p = r_b + r_a
    out = attr.cumulate_top_down(comps, r_a, r_p, r_b)
    growth = 1.0
    linked = {k: 0.0 for k in comps}
    for i in range(t):
        for k in comps:
            linked[k] += comps[k][i] * growth
        growth *= 1 + r_a[i]
    assert out["linked_components"] == pytest.approx(linked, rel=1e-12)
    assert out["naive_cumulative"] == pytest.approx(growth - 1, rel=1e-12)
    assert sum(out["linked_components"].values()) == pytest.approx(out["naive_cumulative"], abs=1e-14)
    assert sum(out["linked_components"].values()) + out["delta_cp"] == pytest.approx(
        float(np.prod(1 + r_p) - np.prod(1 + r_b)), abs=1e-14
    )


def test_bayesian_blend_variance_limits_and_weights():
    assert attr.bayesian_blend_variance(4.0, 1.0, 0.0, 6.0) == 4.0
    assert attr.bayesian_blend_variance(4.0, 1.0, 1e9, 6.0) == pytest.approx(1.0, abs=1e-6)
    assert attr.bayesian_blend_variance(4.0, 1.0, 6.0, 6.0) == pytest.approx(2.5)
    assert attr.prior_weight_t0(60, 20.0) == 3.0


def test_series_stats_known_answer_and_track_record_calculus():
    vals = [0.01, 0.03, 0.02, 0.0]
    mean = 0.015
    var = sum((v - mean) ** 2 for v in vals) / 3
    sigma2 = 1e-4 * (2.0 / 6.0) + var * (4.0 / 6.0)  # T0 = 2, T = 4
    out = attr.series_stats(vals, 12.0, 1e-4, 2.0)
    assert out["vol_blended"] == pytest.approx(math.sqrt(sigma2), rel=1e-12)
    assert out["t_stat"] == pytest.approx(mean / math.sqrt(sigma2) * 2, rel=1e-12)
    assert out["ir"] == pytest.approx(mean * 12 / (math.sqrt(sigma2) * math.sqrt(12)), rel=1e-12)
    assert attr.years_to_significance(0.5) == pytest.approx(16.0)
    assert math.isinf(attr.years_to_significance(0.0))
    assert attr.prob_positive_over_years(0.5, 4.0) == pytest.approx(stats.norm.cdf(1.0), rel=1e-12)
    assert attr.prob_positive_over_years(0.5, -1.0) == 0.5
