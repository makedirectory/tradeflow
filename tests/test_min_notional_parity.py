"""The execution floor, on both clocks.

`min_notional` refuses an order too small for a venue to be worth sending. The
backtest enforced it and the live path did not look at it at all, so a config
validated with a floor traded without one: the same declared book admitted different
orders depending on which clock was asking. Nothing errored, and the preflight
printed the floor on every run.

Two tests that each pass would not have caught it — the backtest's did pass. So the
load-bearing test here drives one book through *both* admission paths and asserts
they reach the same verdict, including at the boundary.
"""

import pytest

from tests.fakes import FakeBroker
from tradeflow.brokers.base import AccountSnapshot
from tradeflow.execution import decision as decisions
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.execution.sizing import below_min_notional
from tradeflow.services.registry import STRATEGIES
from tradeflow.strategies import signals

# One order, priced so the arithmetic is checkable by eye: 6 shares at $100 is $600.
QTY, PRICE = 6.0, 100.0
NOTIONAL = QTY * PRICE


def _trader(min_notional, capital=1_000.0):
    strategy = STRATEGIES["demo_trend"].create_with_defaults()
    strategy.config["position_limits"] = {
        **strategy.position_limits(),
        "max_positions": 10,
        "min_notional": min_notional,
    }
    return LiveTrader(
        FakeBroker(buying_power=100_000.0), strategy, capital=capital, respect_market_hours=False
    )


def _account(equity=100_000.0):
    return AccountSnapshot(cash=equity, equity=equity, buying_power=equity, portfolio_value=equity)


# --- the trade clock now applies it ------------------------------------------------
def test_an_order_below_the_floor_is_refused_and_never_reaches_the_broker():
    """The regression. The floor was declared, printed by the preflight, and a $600
    order went to the venue under a $5,000 minimum."""
    trader = _trader(min_notional=5_000.0)

    decision = trader.handle_signal("AAA", signals.BUY, price=PRICE)

    assert decision.allowed is False
    assert decision.reason_code == decisions.BELOW_MIN_NOTIONAL
    assert "$600.00" in decision.reason and "$5,000.00" in decision.reason
    # The thing that actually matters: nothing was sent.
    assert trader.broker.orders == []


def test_the_guard_is_recorded_as_consulted_even_when_it_passes():
    """A veto list naming only the guard that fired cannot distinguish a guard that
    passed from one that never ran, which is how a check silently stops applying."""
    passing = _trader(min_notional=10.0).handle_signal("AAA", signals.BUY, price=PRICE)

    assert decisions.MIN_NOTIONAL in passing.guards_consulted
    assert passing.allowed is True


def test_an_order_exactly_at_the_floor_is_admitted():
    """Both directions. A floor that rejects the order it was set to permit is
    indistinguishable from one that rejects everything."""
    trader = _trader(min_notional=NOTIONAL)
    at_the_floor = trader.handle_signal("AAA", signals.BUY, price=PRICE)

    assert at_the_floor.allowed is True
    assert len(trader.broker.orders) == 1


def test_an_undeclared_floor_bounds_nothing():
    """Absent is not zero: a book that never named a minimum has no minimum, not one
    of zero, and must trade exactly as it did before the floor existed."""
    for undeclared in (None, 0.0):
        decision = _trader(min_notional=undeclared).handle_signal("AAA", signals.BUY, price=PRICE)
        assert decision.allowed is True, undeclared


def test_a_size_that_rounds_to_zero_is_coded_rather_than_left_as_bare_text():
    """The other way a book too small to trade a name refuses it. Coded so it can be
    counted: the message carries the price, so grouping by message scatters one cause
    across as many rows as there were symbols."""
    # A price far above what the risk budget can buy a whole share of.
    decision = _trader(min_notional=None, capital=100.0).handle_signal("AAA", signals.BUY, price=50_000.0)

    assert decision.allowed is False
    assert decision.reason_code == decisions.ROUNDS_TO_ZERO


# --- and the two clocks agree about it ---------------------------------------------
@pytest.mark.parametrize(
    "floor",
    [None, 0.0, 1.0, NOTIONAL - 0.01, NOTIONAL, NOTIONAL + 0.01, 5_000.0],
)
def test_both_clocks_reach_the_same_verdict_about_the_same_order(floor):
    """The comparison, not two assertions. The same quantity, price and floor are put
    to the live path for real and to the rule the backtest admits entries with, and the
    two must agree — including on both sides of the boundary, where an inclusive and an
    exclusive comparison differ and everything else looks identical.
    """
    live = _trader(min_notional=floor).handle_signal("AAA", signals.BUY, price=PRICE)
    live_refused_for_the_floor = live.reason_code == decisions.BELOW_MIN_NOTIONAL

    # What the backtest asks before admitting a sized candidate.
    research_refuses = below_min_notional(QTY, PRICE, floor)

    assert live_refused_for_the_floor == research_refuses


def test_a_floor_refusal_reaches_the_binding_bucket_and_a_rounding_one_does_not():
    """Classified from a decision the trade clock really produced, not from a
    hand-written reason string — the bucket a code lands in is decided by code that a
    ready-made list never reaches.

    A declared floor refusing an order is a configured limit refusing it, which is what
    WOULD BIND means. A size rounding to zero is share granularity, which is not.
    """
    from tradeflow.services import dryrun

    floor = _trader(min_notional=5_000.0).handle_signal("AAA", signals.BUY, price=PRICE)
    rounding = _trader(min_notional=None, capital=100.0).handle_signal("AAA", signals.BUY, price=50_000.0)

    assert dryrun.classify(floor, "this is a dry run") == "would_bind"
    assert dryrun.classify(rounding, "this is a dry run") == "would_skip"


def test_the_backtest_admission_path_calls_the_shared_rule():
    """The convergence is what keeps the test above honest: were there two
    implementations, they could agree today and drift tomorrow with both suites green.
    Asserted against the source, because an engine run cannot show which function
    decided.
    """
    import pathlib

    from tradeflow.engine import backtest

    source = pathlib.Path(backtest.__file__).read_text()
    assert "below_min_notional(size, price, min_notional)" in source
    # And no second spelling of the same comparison survived.
    assert "size * price < min_notional" not in source
