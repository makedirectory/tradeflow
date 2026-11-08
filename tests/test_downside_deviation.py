"""Downside deviation is a semideviation over every period, not a std of the losers.

The old form took the sample std of only the negative returns, about their own mean and
over their own count. A series whose losses were all the same size read as zero risk,
and rare losses were weighted as heavily as constant ones.
"""

import numpy as np

from quantred.analytics import metrics as m

R = np.random.default_rng(5).normal(0.0004, 0.01, 750)


def test_downside_deviation_matches_the_semideviation_closed_form():
    expected = np.sqrt(np.mean(np.minimum(R, 0.0) ** 2)) * np.sqrt(252)
    assert abs(m.downside_deviation(R) - expected) < 1e-12


def test_sortino_is_mean_over_semideviation():
    dd = np.sqrt(np.mean(np.minimum(R, 0.0) ** 2))
    assert abs(m.sortino_ratio(R) - np.sqrt(252) * R.mean() / dd) < 1e-9


def test_rare_losses_are_not_weighted_like_constant_ones():
    rare = np.array([0.01] * 98 + [-0.02, -0.02])
    often = np.array([0.01] * 50 + [-0.02] * 50)
    assert m.downside_deviation(rare) < m.downside_deviation(often)
    # Identical losses had zero "deviation" under the old std-of-losers definition.
    assert m.downside_deviation(rare) > 0
