"""Math contract: position sizing and the two clocks' agreement about it.

Related existing tests: ``tests/test_sizing_parity.py`` (the same sequence through both
clocks) and ``tests/test_min_notional_parity.py`` (the execution floor); this module adds
the arithmetic they compare against, derived independently.

Defects this module guards:

* live sized every entry against the full capital while the backtest sized against what
  the book had left, so the same declared book deployed more per name live than it had
  been validated at;
* the execution floor existing on one clock only.
"""

import math

import pytest

from quantred.brokers.base import AccountSnapshot
from quantred.execution.sizing import (
    BetaSizer,
    PortfolioWeightSizer,
    RiskBasedSizer,
    below_min_notional,
    deployable_pool,
    free_buying_power,
)
from tests.test_backtest import ScriptedStrategy
from tests.test_sizing_parity import CAPITAL, PRICE, _backtest_notionals, _live_notionals

CONFIG = {
    "risk_per_trade": 0.02,
    "stop_loss": 0.05,
    "position_limits": {"max_positions": 5, "max_position_size": 30_000.0, "max_total_risk": 0.10},
}


def _strategy(**overrides):
    return ScriptedStrategy([], {**CONFIG, **overrides})


def test_risk_based_size_is_risk_budget_over_loss_per_share():
    # $100,000 * 2% = $2,000 at risk; a 5% stop on a $40 share loses $2 -> 1,000 shares
    # ($40,000 notional), above the $30,000 per-position cap -> 750 shares
    s = _strategy()
    assert s.calculate_position_size(100_000.0, 40.0) == pytest.approx(750.0)
    # lift the notional cap: risk budget binds -> 1,000
    uncapped = _strategy(position_limits={"max_position_size": 1e9, "max_total_risk": 1.0, "max_positions": 5})
    assert uncapped.calculate_position_size(100_000.0, 40.0) == pytest.approx(1_000.0)
    # total-risk cap: 0.5% * 100,000 / (40 * 5%) = 250 shares
    tight = _strategy(position_limits={"max_position_size": 1e9, "max_total_risk": 0.005, "max_positions": 5})
    assert tight.calculate_position_size(100_000.0, 40.0) == pytest.approx(250.0)
    assert uncapped.calculate_position_size(100_000.0, 40.0, risk_factor=0.5) == pytest.approx(500.0)


def test_sizer_reads_buying_power_not_equity():
    s = _strategy(position_limits={"max_position_size": 1e9, "max_total_risk": 1.0, "max_positions": 5})
    account = AccountSnapshot(cash=50_000.0, equity=100_000.0, buying_power=60_000.0, portfolio_value=100_000.0)
    # 60,000 * 2% / (40 * 5%) = 600
    assert RiskBasedSizer(s).size("A", 40.0, account) == pytest.approx(600.0)


def test_beta_sizer_divides_by_the_clamped_absolute_beta():
    s = _strategy(position_limits={"max_position_size": 1e9, "max_total_risk": 1.0, "max_positions": 5})
    account = AccountSnapshot(cash=0, equity=0, buying_power=100_000.0, portfolio_value=0)
    base = s.calculate_position_size(100_000.0, 40.0)
    sizer = BetaSizer(s, {"HI": 2.0, "NEG": -2.0, "ZERO": 0.01, "HUGE": 40.0})
    assert sizer.size("HI", 40.0, account) == pytest.approx(base / 2.0)
    assert sizer.size("NEG", 40.0, account) == pytest.approx(base / 2.0)  # the sign is ignored
    assert sizer.size("ZERO", 40.0, account) == pytest.approx(base / 0.25)  # clamped below at 0.25
    assert sizer.size("HUGE", 40.0, account) == pytest.approx(base / 4.0)  # clamped above at 4
    assert sizer.size("UNKNOWN", 40.0, account) == pytest.approx(base)  # neutral beta


def test_portfolio_weight_sizer_is_weight_times_equity_over_price():
    account = AccountSnapshot(cash=0, equity=80_000.0, buying_power=0, portfolio_value=80_000.0)
    sizer = PortfolioWeightSizer({"A": 0.25, "B": 0.0})
    assert sizer.size("A", 50.0, account) == pytest.approx(400.0)
    assert sizer.size("B", 50.0, account) == 0.0 and sizer.size("C", 50.0, account) == 0.0
    assert sizer.size("A", 0.0, account) == 0.0


def test_pool_and_free_buying_power_known_answers():
    assert deployable_pool(3_000.0, 8_000.0) == 3_000.0  # capital caps equity, never exceeds it
    assert deployable_pool(9_000.0, 8_000.0) == 8_000.0
    assert deployable_pool(9_000.0) == 9_000.0 and deployable_pool(9_000.0, 0) == 9_000.0
    assert free_buying_power(10_000.0, 2_500.0) == 7_500.0
    assert free_buying_power(10_000.0, 12_000.0) == 0.0  # over-deployed: nothing left, never negative
    assert free_buying_power(10_000.0, 2_500.0, venue_buying_power=5_000.0) == 5_000.0  # venue only lowers
    assert free_buying_power(10_000.0, 2_500.0, venue_buying_power=50_000.0) == 7_500.0
    assert free_buying_power(10_000.0, 2_500.0, venue_buying_power=-1.0) == 0.0


def test_execution_floor_boundaries_and_undeclared_floor():
    assert below_min_notional(10, 100.0, 1_001.0) is True
    assert below_min_notional(10, 100.0, 1_000.0) is False  # exactly at the floor is admitted
    assert below_min_notional(-10, 100.0, 1_001.0) is True  # a short is measured by magnitude
    assert below_min_notional(1, 1.0, None) is False and below_min_notional(1, 1.0, 0) is False  # absent, not zero


def _by_hand(free=CAPITAL, n=3):
    """Whole-share notionals of n simultaneous entries, each sizing against what is left.

    risk 1% of free cash, stop 4%: shares = floor(free * 0.01 / (50 * 0.04)) = floor(free / 200).
    """
    out = []
    for _ in range(n):
        shares = math.floor(free * 0.01 / (PRICE * 0.04))
        out.append(shares * PRICE)
        free -= shares * PRICE
    return out


def test_the_two_clocks_size_the_same_sequence_to_the_hand_derived_notionals():
    expected = _by_hand()
    assert expected == [2_500.0, 1_850.0, 1_400.0]
    assert _backtest_notionals() == expected
    assert _live_notionals(capital=CAPITAL) == expected
    # a margin account many times larger changes nothing: the book is held to its capital
    assert _live_notionals(capital=CAPITAL, equity=10 * CAPITAL, buying_power=40 * CAPITAL) == expected
