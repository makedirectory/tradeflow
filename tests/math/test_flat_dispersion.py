"""One definition of "no dispersion" (``utils.numeric.is_flat``) and the guards that use it.

Defect guarded: ``std == 0`` / ``std > 0`` tests let a constant series through, because the
sample std of twenty copies of ``0.01`` is ~1e-18, not 0, and the ratio that divides by it
reads ~1e17. Each case below hands a guard a constant series and requires the function's
documented "unavailable" value.
"""

import numpy as np
import pandas as pd
import pytest

from quantred.alphas import refine
from quantred.analytics import bootstrap as bs
from quantred.analytics import information as info
from quantred.analytics import metrics as m
from quantred.utils.numeric import is_flat

CONSTANT = [0.01] * 20


def test_constant_series_are_flat_whatever_their_magnitude():
    for level in (0.01, 1 / 3, 123456.789, -0.07):
        assert is_flat([level] * 37)


def test_real_dispersion_is_not_flat_even_when_tiny_or_far_from_zero():
    assert not is_flat([0.0, 1e-6, 0.0, 1e-6])
    assert not is_flat([1e6, 1e6 + 1.0, 1e6, 1e6 + 1.0])  # a dollar of spread on a million


def test_nan_is_ignored_and_too_short_is_flat():
    assert is_flat([float("nan"), 0.5]) and is_flat([])
    assert not is_flat([float("nan"), 0.1, 0.2])
    assert not is_flat([0.1, float("inf")])


def test_the_noise_floor_is_what_the_guard_exists_for():
    # not an exact zero: this is the value an == 0 guard misses
    assert pd.Series(CONSTANT).std() > 0
    assert is_flat(CONSTANT)


def test_sqn_of_constant_pnl_is_unavailable():
    assert m.system_quality_number(CONSTANT) == 0.0


def test_beta_against_a_constant_benchmark_is_unavailable():
    r = [0.01, -0.02, 0.015, 0.0, 0.003]
    assert m.alpha_beta(r, [0.004] * 5) == (0.0, 0.0, 0.0)


def test_information_ratio_of_a_constant_active_return_is_unavailable():
    assert m.information_ratio([0.02] * 10, [0.01] * 10) == 0.0


def test_ics_of_a_constant_forecast_are_nan_not_a_correlation():
    realized = pd.Series([0.1, -0.2, 0.3, 0.05, 0.0])
    constant = pd.Series([0.3] * 5)
    assert np.isnan(info.pearson_ic(constant, realized))
    assert np.isnan(info.rank_ic(constant, realized))


def test_zscore_and_winsorize_of_a_constant_alpha_are_no_view():
    scores = pd.Series([0.3] * 6)
    assert (refine.zscore(scores) == 0.0).all()
    assert refine.winsorize(scores).equals(scores)


def test_bootstrap_ir_of_a_constant_slice_is_zero_not_huge():
    panel = np.vstack([np.full(30, 0.01), np.linspace(-0.01, 0.02, 30)])
    out = bs._annualized_ir(panel, 252)
    assert out[0] == 0.0 and out[1] != 0.0


@pytest.mark.parametrize("n", [3, 11, 250])
def test_psr_and_dsr_of_a_constant_series_are_zero(n):
    assert m.probabilistic_sharpe_ratio([0.002] * n) == 0.0
    assert m.deflated_sharpe_ratio([0.002] * n, 5) == 0.0
