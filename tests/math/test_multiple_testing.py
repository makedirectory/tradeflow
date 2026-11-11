"""Math contract: the honest-evaluation core (probabilistic and deflated Sharpe, the
stationary bootstrap and the family-wise bootstrap test).

References use ``scipy.stats`` and plain loops. Related existing tests:
``tests/test_aggregate_annualisation.py`` (DSR units on the walk-forward clock),
``tests/test_bootstrap_skill.py``.

Defects this module guards:

* the walk-forward's Deflated Sharpe mixed a per-calendar-day trial variance with a
  per-trading-day Sharpe (numerator and benchmark in different units);
* the normal quantile feeding the expected-maximum benchmark drifting from the true
  inverse CDF.
"""

import math

import numpy as np
import pytest
from scipy import stats

from quantred.analytics import bootstrap as bs
from quantred.analytics import metrics as m

R = np.random.default_rng(31).normal(0.0006, 0.01, 500)
EULER = 0.5772156649015329


def _psr_reference(r, benchmark=0.0):
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=False, bias=False)  # Pearson: normal == 3
    z = (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf(z))


def test_norm_ppf_matches_the_exact_inverse_cdf():
    for p in [1e-9, 1e-6, 1e-3, 0.01, 0.02425, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97575, 0.99, 0.999999]:
        assert m.norm_ppf(p) == pytest.approx(stats.norm.ppf(p), abs=2e-8), p
    assert m.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
    assert m.norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-8)
    assert m.norm_ppf(0.0) == -math.inf and m.norm_ppf(1.0) == math.inf


def test_norm_cdf_matches_known_values():
    assert m.norm_cdf(0.0) == 0.5
    assert m.norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)
    assert m.norm_cdf(-8.0) + m.norm_cdf(8.0) == pytest.approx(1.0, abs=1e-15)


def test_probabilistic_sharpe_matches_the_formula_with_per_period_inputs():
    assert m.probabilistic_sharpe_ratio(R) == pytest.approx(_psr_reference(R), rel=1e-9)
    assert m.probabilistic_sharpe_ratio(R, benchmark_sr=0.03) == pytest.approx(
        _psr_reference(R, 0.03), rel=1e-9
    )


def test_probabilistic_sharpe_on_skewed_fat_tails_uses_pearson_kurtosis():
    t = np.random.default_rng(32).standard_t(4, 600) * 0.01 + 0.0008
    t[::37] -= 0.04  # negative skew
    assert m.probabilistic_sharpe_ratio(t) == pytest.approx(_psr_reference(t), rel=1e-9)
    # Using excess kurtosis in place of Pearson would shift the denominator by 3/4 sr^2.
    sr = t.mean() / t.std(ddof=1)
    wrong = 1 - stats.skew(t, bias=False) * sr + (stats.kurtosis(t, bias=False) - 1) / 4 * sr**2
    right = 1 - stats.skew(t, bias=False) * sr + (stats.kurtosis(t, bias=False) + 2) / 4 * sr**2
    assert right - wrong == pytest.approx(0.75 * sr**2)


def test_psr_is_half_at_its_own_sharpe_and_monotone_in_the_benchmark():
    sr = R.mean() / R.std(ddof=1)
    assert m.probabilistic_sharpe_ratio(R, benchmark_sr=sr) == pytest.approx(0.5, abs=1e-12)
    levels = [m.probabilistic_sharpe_ratio(R, b) for b in (0.0, 0.02, 0.05, 0.1)]
    assert levels == sorted(levels, reverse=True)


def test_psr_uses_sqrt_n_minus_one():
    """With zero skew and Pearson kurtosis 3 the denominator is 1 + sr^2/2: build that
    case exactly from a symmetric two-point-pair series and check z by hand."""
    r = np.array([0.03, -0.01, 0.03, -0.01, 0.03, -0.01])  # symmetric about its mean
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r, bias=False)
    assert skew == pytest.approx(0.0, abs=1e-12)
    kurt = stats.kurtosis(r, fisher=False, bias=False)
    z = sr * math.sqrt(5) / math.sqrt(1 + (kurt - 1) / 4 * sr**2)
    assert m.probabilistic_sharpe_ratio(r) == pytest.approx(stats.norm.cdf(z), rel=1e-12)


def test_expected_max_sharpe_matches_the_extreme_value_approximation():
    for var, n in [(0.01, 10), (0.0004, 100), (1e-4, 1000)]:
        z1 = stats.norm.ppf(1 - 1 / n)
        z2 = stats.norm.ppf(1 - 1 / (n * math.e))
        expected = math.sqrt(var) * ((1 - EULER) * z1 + EULER * z2)
        assert m.expected_max_sharpe(var, n) == pytest.approx(expected, rel=1e-7)


def test_expected_max_sharpe_known_answer_for_ten_trials():
    # sqrt(var)=0.1; z1 = Phi^-1(0.9) = 1.2815515655, z2 = Phi^-1(1-1/(10e)) = Phi^-1(0.963212...) = 1.7898..
    z1 = 1.2815515655446004
    z2 = float(stats.norm.ppf(1 - 1 / (10 * math.e)))
    assert m.expected_max_sharpe(0.01, 10) == pytest.approx(
        0.1 * (0.4227843350984671 * z1 + EULER * z2), rel=1e-7
    )


def test_expected_max_sharpe_grows_with_trials_and_with_variance_and_is_zero_without_either():
    assert m.expected_max_sharpe(0.01, 10) < m.expected_max_sharpe(0.01, 100)
    assert m.expected_max_sharpe(0.01, 10) < m.expected_max_sharpe(0.04, 10)
    assert m.expected_max_sharpe(0.01, 1) == 0.0 and m.expected_max_sharpe(0.0, 50) == 0.0


def test_deflated_sharpe_is_psr_against_the_expected_best():
    n = len(R)
    var = 1.0 / (n - 1)  # the documented default when the trial variance is unknown
    sr_star = math.sqrt(var) * (
        (1 - EULER) * stats.norm.ppf(1 - 1 / 25) + EULER * stats.norm.ppf(1 - 1 / (25 * math.e))
    )
    assert m.deflated_sharpe_ratio(R, 25) == pytest.approx(_psr_reference(R, sr_star), rel=1e-7)
    assert m.deflated_sharpe_ratio(R, 25, var_of_trial_sr=4e-4) == pytest.approx(
        _psr_reference(
            R,
            math.sqrt(4e-4)
            * ((1 - EULER) * stats.norm.ppf(1 - 1 / 25) + EULER * stats.norm.ppf(1 - 1 / (25 * math.e))),
        ),
        rel=1e-7,
    )


def test_deflation_never_helps_and_one_trial_is_the_plain_psr():
    assert m.deflated_sharpe_ratio(R, 1) == pytest.approx(m.probabilistic_sharpe_ratio(R), rel=1e-12)
    values = [m.deflated_sharpe_ratio(R, k, 1e-4) for k in (1, 5, 50, 500)]
    assert values == sorted(values, reverse=True)


def test_psr_and_dsr_degenerate_inputs_are_zero_not_nan():
    for series in ([], [0.01, 0.02], [0.0] * 20):
        assert m.probabilistic_sharpe_ratio(series) == 0.0
        assert m.deflated_sharpe_ratio(series, 10) == 0.0
    assert m.deflated_sharpe_ratio(R, 0) == 0.0


# --------------------------------------------------------------------------- #
# Stationary bootstrap and the family-wise bootstrap test
# --------------------------------------------------------------------------- #
def test_stationary_bootstrap_indices_follow_the_geometric_run_recurrence():
    t, b, length = 200, 400, 10.0
    idx = bs.stationary_bootstrap_indices(t, length, b, np.random.default_rng(0))
    assert idx.shape == (b, t) and idx.min() >= 0 and idx.max() < t
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % t).mean()
    # Continue with probability 1 - 1/L, plus the 1/(L*T) chance a restart lands on the successor.
    assert continues == pytest.approx(1 - 1 / length + 1 / (length * t), abs=0.01)


def test_block_length_one_is_an_iid_resample():
    idx = bs.stationary_bootstrap_indices(100, 1.0, 300, np.random.default_rng(1))
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 100).mean()
    assert continues == pytest.approx(1 / 100, abs=0.01)  # only by chance
    assert np.bincount(idx.ravel(), minlength=100).std() < 0.2 * 300  # roughly uniform


def test_bootstrap_p_value_uses_the_plus_one_correction():
    null = np.array([0.1, 0.2, 0.3, 0.4])
    p, se = bs._p_value(null, 0.25, 4)
    assert p == pytest.approx((1 + 2) / 5)  # two null draws at or above 0.25
    assert se == pytest.approx(math.sqrt(0.6 * 0.4 / 4))
    assert bs._p_value(null, 99.0, 4)[0] == pytest.approx(1 / 5)  # never exactly zero


def test_annualised_ir_helper_matches_a_loop_and_zero_variance_is_zero():
    x = np.random.default_rng(2).normal(0.001, 0.01, (3, 60))
    ref = [math.sqrt(252) * row.mean() / row.std(ddof=1) for row in x]
    assert bs._annualized_ir(x, 252, axis=-1) == pytest.approx(ref, rel=1e-12)
    assert bs._annualized_ir(np.zeros((2, 10)), 252, axis=-1).tolist() == [0.0, 0.0]


def test_bootstrap_null_separates_skill_from_noise_and_is_deterministic():
    rng = np.random.default_rng(3)
    skilled = rng.normal(0.002, 0.01, 500)
    noise = rng.normal(0.0, 0.01, 500)
    a = bs.bootstrap_null(skilled, B=400, seed=7)
    assert a == bs.bootstrap_null(skilled, B=400, seed=7)
    assert a["p_value"] < 0.05
    assert bs.bootstrap_null(noise, B=400, seed=7)["p_value"] > 0.05
    assert bs.bootstrap_null(-skilled, B=400, seed=7)["p_value"] > 0.9  # a loss needs no skill to explain
    ref = math.sqrt(252) * skilled.mean() / skilled.std(ddof=1)
    assert a["ir_observed"] == pytest.approx(ref, rel=1e-12)


def test_bootstrap_null_reports_insufficient_data_instead_of_a_p_value():
    out = bs.bootstrap_null([0.01] * 5)
    assert out["insufficient_data"] is True and out["B"] == 0


def test_reality_check_counts_the_whole_family():
    """The best of many noise columns is not evidence: the family p-value is higher than
    the same column judged alone."""
    rng = np.random.default_rng(4)
    panel = rng.normal(0.0, 0.01, (300, 30))
    alone = bs.reality_check(panel[:, :1], B=300, seed=5, block_length=5.0)
    family = bs.reality_check(panel, B=300, seed=5, block_length=5.0)
    best = int(np.argmax(panel.mean(0) / panel.std(0, ddof=1)))
    assert family["selected_trial_index"] == best
    assert family["observed_max_ir"] == pytest.approx(
        math.sqrt(252) * panel[:, best].mean() / panel[:, best].std(ddof=1), rel=1e-12
    )
    assert (
        family["family_p"] >= bs.reality_check(panel[:, [best]], B=300, seed=5, block_length=5.0)["family_p"]
    )
    assert family["k_trials"] == 30 and alone["k_trials"] == 1


def test_reality_check_detects_a_real_edge_among_noise():
    rng = np.random.default_rng(6)
    panel = rng.normal(0.0, 0.01, (400, 10))
    panel[:, 3] += 0.003
    out = bs.reality_check(panel, B=300, seed=1, block_length=5.0)
    assert out["selected_trial_index"] == 3 and out["family_p"] < 0.05


def test_reality_check_small_panels_are_flagged_not_scored():
    assert bs.reality_check(np.zeros((4, 3)))["insufficient_data"] is True
    assert bs.reality_check(np.zeros(5))["insufficient_data"] is True


def test_a_constant_nonzero_series_is_unavailable_not_a_perfect_sharpe():
    series = [0.01] * 20
    assert m.sharpe_ratio(series) == 0.0
    assert m.probabilistic_sharpe_ratio(series) == 0.0
    assert m.deflated_sharpe_ratio(series, 10) == 0.0
