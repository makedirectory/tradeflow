"""Math contract: tail risk and distribution shape.

Reference values are hand-derived literals, loop computations, or ``scipy.stats``
(an independent implementation, not part of the package).

Defects this module guards:

* historical VaR took the magnitude of the percentile, so a 0.2% *gain* at the 5th
  percentile was reported as a 0.2% loss (VaR/CVaR floor at zero);
* skewness/kurtosis conventions drifting from the sample (bias-corrected) forms the
  probabilistic Sharpe correction assumes.
"""

import math

import numpy as np
import pytest
from scipy import stats

from quantred.analytics import metrics as m

R = np.random.default_rng(21).standard_t(5, 800) * 0.01


def test_var_known_answer_with_linear_interpolation():
    # 20 returns -10%..+9% in 1% steps; the 5th percentile sits at rank 0.05*19 = 0.95
    # between -10% and -9%: -10% + 0.95*1% = -9.05%
    r = [(-10 + i) / 100 for i in range(20)]
    assert m.value_at_risk(r, 0.95) == pytest.approx(0.0905, rel=1e-12)


def test_var_matches_numpy_percentile_reference():
    for level in (0.9, 0.95, 0.99):
        ref = -float(np.percentile(R, (1 - level) * 100))
        assert m.value_at_risk(R, level) == pytest.approx(ref, rel=1e-12)


def test_cvar_is_the_mean_of_the_returns_at_or_below_the_var_cutoff():
    cutoff = float(np.percentile(R, 5))
    tail = [x for x in R if x <= cutoff]
    assert m.conditional_var(R, 0.95) == pytest.approx(-sum(tail) / len(tail), rel=1e-12)
    assert m.conditional_var(R, 0.95) >= m.value_at_risk(R, 0.95)  # expected shortfall dominates VaR


def test_var_and_cvar_floor_at_zero_when_the_tail_is_a_gain():
    """A series whose worst 5% is still positive has no loss to report."""
    gains = [0.002 + 0.001 * i for i in range(40)]
    assert m.value_at_risk(gains) == 0.0
    assert m.conditional_var(gains) == 0.0


def test_var_cvar_empty_series_are_zero():
    assert m.value_at_risk([]) == 0.0 and m.conditional_var([]) == 0.0


def test_skewness_and_excess_kurtosis_match_the_sample_estimators():
    assert m.skewness(R) == pytest.approx(stats.skew(R, bias=False), rel=1e-10)
    assert m.kurtosis(R) == pytest.approx(stats.kurtosis(R, fisher=True, bias=False), rel=1e-10)


def test_skewness_known_answer():
    # [0,0,0,4]: mean 1, deviations -1,-1,-1,3. m2 = 12/4 = 3, m3 = (−3+27)/4 = 6.
    # population skew = 6 / 3**1.5; sample (n=4) multiplies by sqrt(n(n-1))/(n-2) = sqrt(12)/2
    pop = 6 / 3**1.5
    assert m.skewness([0, 0, 0, 4]) == pytest.approx(pop * math.sqrt(12) / 2, rel=1e-12)


def test_symmetric_series_has_zero_skew():
    assert m.skewness([-2, -1, 0, 1, 2]) == pytest.approx(0.0, abs=1e-12)


def test_short_series_return_zero_not_nan():
    assert m.skewness([1, 2]) == 0.0 and m.kurtosis([1, 2, 3]) == 0.0


def test_zero_variance_series_do_not_leak_nan():
    for fn in (m.skewness, m.kurtosis):
        assert math.isfinite(fn([0.01] * 20))


def test_tail_ratio_known_answer_and_unbounded_left_tail():
    r = [(-10 + i) / 100 for i in range(20)]
    right = np.percentile(r, 95)  # rank 18.05 -> 0.08 + 0.05*0.01
    left = abs(np.percentile(r, 5))
    assert m.tail_ratio(r) == pytest.approx(abs(right) / left, rel=1e-12)
    assert m.tail_ratio(r) == pytest.approx(0.0805 / 0.0905, rel=1e-12)
    assert math.isinf(m.tail_ratio([0.0, 0.0, 0.0, 0.1]))  # no left tail: documented infinity


def test_omega_ratio_known_answer():
    # gains 0.03+0.01 = 0.04, losses 0.02 + 0.01 = 0.03
    assert m.omega_ratio([0.03, -0.02, 0.01, -0.01]) == pytest.approx(0.04 / 0.03, rel=1e-12)
    # against a 1% threshold: excess = .02,-.03,0,-.02 -> .02/.05
    assert m.omega_ratio([0.03, -0.02, 0.01, -0.01], threshold=0.01) == pytest.approx(0.02 / 0.05, rel=1e-12)
    assert math.isinf(m.omega_ratio([0.01, 0.02]))
    assert m.omega_ratio([]) == 0.0
