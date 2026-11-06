"""Labels that must say what a number is: sentinels, undeflated values, vacuous passes."""

import pytest

from tests.fakes import make_ohlcv


# --- capacity: a search bound is not a capacity ---------------------------------------
def _book():
    syms = [f"S{i}" for i in range(6)]
    weights = {s: 1.0 / len(syms) for s in syms}
    bars = {s: make_ohlcv(n=200, seed=i, freq="1D") for i, s in enumerate(syms)}
    return weights, bars


@pytest.mark.parametrize(
    "alpha, status",
    [(0.0, "no_edge"), (-0.01, "no_edge"), (1e-12, "below_search_floor"), (1e6, "above_search_ceiling")],
)
def test_capacity_names_the_bound_instead_of_returning_it(alpha, status):
    from tradeflow.services.analysis import _capacity

    weights, bars = _book()
    capital, got = _capacity(weights, bars, gross_alpha=alpha, holding_period_years=1 / 12)
    assert got == status
    assert capital is None


def test_capacity_text_says_the_bound_in_words_and_never_prints_a_sentinel():
    from tradeflow.analytics.reporting import capacity_text

    assert capacity_text({}) is None
    measured = capacity_text({"capacity_status": "measured", "capacity_capital": 2_500_000.0})
    assert "$2,500,000" in measured
    for status in ("no_edge", "below_search_floor", "above_search_ceiling"):
        text = capacity_text({"capacity_status": status, "capacity_capital": None})
        assert text.startswith("not ")
        assert "≈" not in text


def test_html_portfolio_does_not_render_a_capacity_number_for_a_bound():
    from tradeflow.analytics.htmlreport import _portfolio_card

    d = {"capacity_status": "no_edge", "capacity_capital": None, "expected_active_return": -0.01}
    card = _portfolio_card({"feasible": True, "weights": {"A": 1.0}, "diagnostics": d})
    assert "Capacity" in card and "not applicable" in card
