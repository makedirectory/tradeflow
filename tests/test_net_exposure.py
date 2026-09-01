"""The directional cap, on both clocks.

`max_gross_exposure` bounds long + short and cannot see direction: a book sitting
inside a 90% gross cap can be entirely long, which is a bet on direction that nothing
was limiting. `max_net_exposure` bounds |long - short|.

The two clocks enforce this separately and must agree, so the same cases are asserted
against the backtest engine and the live trader.
"""

import pytest

from tests.fakes import FakeBroker
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.services.registry import STRATEGIES
from tradeflow.strategies import signals
from tradeflow.strategies.base import DEFAULT_POSITION_LIMITS


def _trader(limits, positions=None):
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    # max_positions is raised out of the way so each case exercises the cap it names;
    # the shipped default of 1 would otherwise reject every second entry first.
    strategy.config["position_limits"] = {**strategy.position_limits(), "max_positions": 10, **limits}
    strategy.positions = positions or {}
    return LiveTrader(FakeBroker(buying_power=100_000.0), strategy, respect_market_hours=False)


def _held(side, qty, price):
    return {"side": side, "qty": qty, "entry_price": price, "stop_loss": 0.0, "take_profit": 0.0}


def _account(equity=10_000.0):
    from tradeflow.brokers.base import AccountSnapshot

    return AccountSnapshot(cash=equity, equity=equity, buying_power=equity, portfolio_value=equity)


# --- trade clock ------------------------------------------------------------------
def test_a_book_inside_its_gross_cap_can_still_be_refused_for_direction():
    """The gap this closes: gross sees $8,000 of exposure either way, so a wholly
    long book and a balanced one are identical to it."""
    trader = _trader(
        {"max_gross_exposure": 0.9, "max_net_exposure": 0.3, "max_total_risk": None},
        {"AAA": _held(signals.BUY, 40, 100.0)},  # $4,000 long
    )

    code, detail = trader._limit_breach("BBB", 40, 100.0, _account(), signals.BUY)

    assert code == "net_exposure_capped"  # a stable family, not the dollar text
    assert "net exposure" in detail and "long" in detail  # the direction, not just a number


def test_a_hedge_that_moves_the_book_toward_flat_is_admitted():
    """Both directions, and the case a naive addition check gets wrong: judged on the
    resulting |net|, so correcting a tilt is never refused for being a trade."""
    trader = _trader(
        {"max_gross_exposure": None, "max_net_exposure": 0.3, "max_total_risk": None},
        {"AAA": _held(signals.BUY, 40, 100.0)},  # $4,000 long, already over a $3,000 cap
    )

    assert trader._limit_breach("BBB", 40, 100.0, _account(), signals.SELL) is None


def test_a_short_that_deepens_a_short_tilt_is_refused():
    """The cap is on magnitude: 90% net short is as directional as 90% net long."""
    trader = _trader(
        {"max_gross_exposure": None, "max_net_exposure": 0.3, "max_total_risk": None},
        {"AAA": _held(signals.SELL, 40, 100.0)},
    )

    code, detail = trader._limit_breach("BBB", 40, 100.0, _account(), signals.SELL)

    assert code == "net_exposure_capped" and "short" in detail


def test_the_sign_comes_from_the_recorded_side_not_the_quantity():
    """A broker reports a short's quantity negative while an entry this trader records
    itself is positive. Reading the sign off qty would count adopted shorts and freshly
    opened ones with opposite signs, and net would silently be wrong on a resumed book.
    """
    trader = _trader(
        {"max_gross_exposure": None, "max_net_exposure": 0.3, "max_total_risk": None},
        {"AAA": _held(signals.SELL, -40, 100.0)},  # adopted short: negative qty
    )

    # Deepening the short must be refused; it is only visible as a short via "side".
    assert trader._limit_breach("BBB", 40, 100.0, _account(), signals.SELL) is not None
    # And netting against it must be allowed.
    assert trader._limit_breach("BBB", 40, 100.0, _account(), signals.BUY) is None


def test_an_undeclared_net_cap_binds_nothing():
    """Absent is not zero: no declared limit must not read as "no exposure allowed"."""
    trader = _trader(
        {"max_gross_exposure": None, "max_net_exposure": None, "max_total_risk": None},
        {"AAA": _held(signals.BUY, 90, 100.0)},
    )

    assert trader._limit_breach("BBB", 90, 100.0, _account(), signals.BUY) is None


def test_the_default_limits_leave_it_off():
    """Off unless configured, like the gross cap — turning it on by default would
    change every existing config's book."""
    assert DEFAULT_POSITION_LIMITS["max_net_exposure"] is None


# --- research clock ---------------------------------------------------------------
def _book(positions):
    from tradeflow.engine.backtest import _Book

    book = _Book(cash=10_000.0)
    for symbol, (side, size, price) in positions.items():
        book.positions[symbol] = {
            "side": side,
            "size": size,
            "last_price": price,
            "risk": 0.0,
            "entry_price": price,
        }
    return book


def test_the_engine_nets_longs_against_shorts():
    book = _book({"AAA": (signals.BUY, 40, 100.0), "BBB": (signals.SELL, 40, 100.0)})

    assert book.gross_exposure() == pytest.approx(8_000.0)
    assert book.net_exposure() == pytest.approx(0.0)


def test_the_engine_reports_a_one_sided_book_as_fully_net():
    book = _book({"AAA": (signals.BUY, 40, 100.0), "BBB": (signals.BUY, 40, 100.0)})

    assert book.gross_exposure() == pytest.approx(8_000.0)
    assert book.net_exposure() == pytest.approx(8_000.0)


def test_the_engine_reports_a_short_book_as_negative_net():
    book = _book({"AAA": (signals.SELL, 40, 100.0)})

    assert book.net_exposure() == pytest.approx(-4_000.0)
