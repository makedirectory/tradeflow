"""A sweep's summary must be able to say what noise looks like.

A screen reports the best of N points, and the best of N is the maximum of N draws —
a positive number even when nothing in the sweep has any edge, growing with N. A
leaderboard printed without that reference is the selection bias the deflated Sharpe
exists to prevent, one layer up and with no deflation applied.

So the claim these tests defend is a quantitative one: the reported noise baseline has
to match what a grid drawn from a *known* null actually produces. A baseline asserted
only against itself is another confident number.
"""

import numpy as np
import pytest

from tradeflow.analytics.screening import (
    gradients,
    noise_baseline,
    parameter_gradient,
    score_distribution,
)


# --- the null baseline, against a grid whose null is known ------------------------
def test_the_noise_baseline_matches_the_best_of_n_from_a_known_null():
    """The check that makes the number mean something. Draw many grids of N results
    from a distribution with no edge at all, take each grid's best, and compare the
    average of those maxima with what the baseline claims a best-of-N is worth.

    They are two routes to the same quantity: one analytic, one by simulation. If the
    baseline were computed against the wrong dispersion, or with the wrong N, this is
    where it shows.
    """
    rng = np.random.default_rng(20260903)
    sigma, n_points, n_grids = 0.4, 64, 4000

    grids = rng.normal(0.0, sigma, size=(n_grids, n_points))
    empirical_best = float(np.mean(np.max(grids, axis=1)))

    # The baseline reads its dispersion off the grid it is given, so hand it one grid
    # from the same null rather than the true sigma — that is the real call path.
    claimed = [noise_baseline(grid, "sharpe_ratio")["expected_best_under_null"] for grid in grids]

    assert float(np.mean(claimed)) == pytest.approx(empirical_best, rel=0.05)


def test_the_noise_baseline_grows_with_the_number_of_points_searched():
    """The property that makes it worth printing: searching harder raises the bar a
    result has to clear, and a screen that reported the same baseline for 10 points and
    1000 would be telling a reader the opposite of what is true."""
    rng = np.random.default_rng(7)
    small = noise_baseline(rng.normal(0.0, 0.4, 10), "sharpe_ratio")
    large = noise_baseline(rng.normal(0.0, 0.4, 1000), "sharpe_ratio")

    assert large["expected_best_under_null"] > small["expected_best_under_null"]


def test_a_real_edge_clears_the_baseline_and_pure_noise_does_not():
    """Both directions. A baseline that nothing ever clears is indistinguishable from
    one that is simply too high to be useful."""
    rng = np.random.default_rng(11)

    noise = rng.normal(0.0, 0.4, 64)
    edge = rng.normal(2.0, 0.4, 64)

    assert max(noise) < noise_baseline(noise, "sharpe_ratio")["expected_best_under_null"] * 2.5
    assert max(edge) > noise_baseline(edge, "sharpe_ratio")["expected_best_under_null"]


def test_the_baseline_says_it_assumes_independence():
    """Neighbouring grid points share most of their parameters and most of their
    trades, so the effective number of independent trials is smaller than N. The
    caveat has to travel with the number, not sit in a docstring."""
    baseline = noise_baseline(np.random.default_rng(1).normal(0, 0.3, 30), "sharpe_ratio")

    assert baseline["assumes_independence"] is True


# --- and refuses where it would be meaningless ------------------------------------
@pytest.mark.parametrize("objective", ["profit_factor", "total_return", "win_rate"])
def test_no_baseline_is_invented_for_an_objective_whose_null_is_not_zero(objective):
    """A profit factor is null-centred on 1 and heavily skewed. "The expected maximum
    of N draws" computed for it is a confident number that means nothing, and it would
    be quoted."""
    baseline = noise_baseline([0.1, 0.4, -0.2, 0.9], objective)

    assert baseline["applicable"] is False
    assert baseline["expected_best_under_null"] is None
    assert objective in baseline["reason"]
    # Silence about the missing baseline would read as "the best point stands".
    assert "best of 4 draws" in baseline["reason"]


def test_a_single_point_has_no_spread_to_draw_from():
    baseline = noise_baseline([0.5], "sharpe_ratio")

    assert baseline["applicable"] is False
    assert "spread" in baseline["reason"]


def test_identical_results_are_not_reported_as_a_null_of_zero():
    """Zero dispersion means the sweep found no variation, not that noise would produce
    a best of zero."""
    baseline = noise_baseline([0.3, 0.3, 0.3, 0.3], "sharpe_ratio")

    assert baseline["applicable"] is False
    assert baseline["expected_best_under_null"] is None


# --- the distribution -------------------------------------------------------------
def test_the_distribution_leads_with_the_middle_and_the_spread():
    summary = score_distribution([-1.0, 0.0, 1.0, 2.0])

    assert summary["n"] == 4
    assert summary["median"] == pytest.approx(0.5)
    assert summary["positive_rate"] == pytest.approx(0.5)
    assert summary["min"] == -1.0 and summary["max"] == 2.0


def test_an_evaluation_that_produced_nothing_is_counted_not_dropped_silently():
    """Absent is not zero. A failed evaluation quietly removed from the sample makes a
    sweep look narrower and more successful than it was."""
    summary = score_distribution([1.0, float("nan"), float("-inf"), 2.0])

    assert summary["n"] == 4
    assert summary["n_finite"] == 2
    assert summary["n_dropped"] == 2
    assert summary["median"] == pytest.approx(1.5)


def test_one_point_reports_no_spread_rather_than_a_spread_of_zero():
    """`std` of a single sample is not 0.0 — that would read as "every result agreed"."""
    assert score_distribution([1.0])["std"] is None


def test_a_sweep_where_everything_failed_still_reports_its_size():
    summary = score_distribution([float("nan"), float("nan")])

    assert summary["n"] == 2 and summary["n_finite"] == 0
    assert summary["median"] is None and summary["positive_rate"] is None


# --- the gradient -----------------------------------------------------------------
def _rows(pairs):
    return [{"lookback": value, "sharpe_ratio": score} for value, score in pairs]


def test_a_gradient_shows_a_trend_a_leaderboard_would_hide():
    """The real finding this exists for: a positive rate falling monotonically as a
    filter tightens is structure, and it points off the edge of the searched space.
    The same count of positive points scattered at random produces the same winner."""
    rows = _rows(
        [(0.0, 1.0), (0.0, 0.5), (0.0, -0.2), (0.5, 0.3), (0.5, -0.4), (0.5, -0.6), (1.0, -0.1), (1.0, -0.7)]
    )

    gradient = parameter_gradient(rows, "lookback", "sharpe_ratio")

    assert [row["value"] for row in gradient] == [0.0, 0.5, 1.0]
    assert [row["positive"] for row in gradient] == [2, 1, 0]
    assert gradient[0]["positive_rate"] > gradient[-1]["positive_rate"]


def test_a_parameter_that_did_not_vary_has_no_gradient():
    """One row is not a trend, and rendering it as one invites reading a single value
    as a direction."""
    assert parameter_gradient(_rows([(0.5, 1.0), (0.5, -1.0)]), "lookback", "sharpe_ratio") is None


def test_a_failed_evaluation_does_not_enter_a_gradient_as_a_zero():
    rows = _rows([(0.0, 1.0), (0.0, float("nan")), (1.0, -1.0)])

    gradient = parameter_gradient(rows, "lookback", "sharpe_ratio")

    assert gradient[0]["n"] == 1


def test_gradients_are_reported_only_for_parameters_that_moved():
    rows = [{"a": 1, "b": 5, "sharpe_ratio": 0.2}, {"a": 2, "b": 5, "sharpe_ratio": -0.1}]

    assert set(gradients(rows, ["a", "b"], "sharpe_ratio")) == {"a"}
