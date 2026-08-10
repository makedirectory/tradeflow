"""Broker failure, typed.

Every broker call used to fail identically — ``None`` or ``False`` — so a rate limit,
an expired token, and an order the venue deliberately refused all produced the same
non-answer and the same response. These cover the distinctions that change what a
caller should do, and the two places where the old behavior was actively wrong: a
duplicate order read as a failed submission, and revoked credentials read as "the
market is open".
"""

from datetime import datetime

import pytest

from tests.fakes import FailingBroker
from tradeflow.brokers.alpaca.broker import classify_error
from tradeflow.brokers.errors import (
    AuthenticationError,
    BrokerError,
    BrokerUnavailableError,
    DuplicateOrderError,
    InsufficientFundsError,
    OrderRejectedError,
    RateLimitedError,
)
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.strategies import signals
from tradeflow.strategies.volume_spike import VolumeSpikeStrategy


class _APIErrorish(Exception):
    """Shaped like the vendor's APIError: status and message as properties."""

    def __init__(self, message="", status=None, raising=False):
        super().__init__(message)
        self._message = message
        self._status = status
        self._raising = raising

    @property
    def status_code(self):
        if self._raising:
            raise ValueError("malformed error body")
        return self._status

    @property
    def message(self):
        if self._raising:
            raise ValueError("malformed error body")
        return self._message


@pytest.fixture(autouse=True)
def _api_error(monkeypatch):
    """Classify against our stand-in rather than importing the vendor's class."""
    monkeypatch.setattr("tradeflow.brokers.alpaca.broker.APIError", _APIErrorish)


# --- classification ---------------------------------------------------------
@pytest.mark.parametrize(
    "message,status,expected",
    [
        ("nope", 401, AuthenticationError),
        ("forbidden", 403, AuthenticationError),
        ("slow down", 429, RateLimitedError),
        ("bad gateway", 502, BrokerUnavailableError),
        ("insufficient buying power", 403, AuthenticationError),  # status wins over text
        ("insufficient buying power", 422, InsufficientFundsError),
        ("client_order_id must be unique", 422, DuplicateOrderError),
        ("some validation problem", 422, OrderRejectedError),
    ],
)
def test_a_vendor_error_is_classified_by_what_the_caller_should_do(message, status, expected):
    assert isinstance(classify_error(_APIErrorish(message, status)), expected)


def test_a_connection_failure_is_unavailability_not_rejection():
    assert isinstance(classify_error(ConnectionError("refused")), BrokerUnavailableError)


def test_an_unclassifiable_error_still_becomes_a_broker_error():
    """The fallback must never be "no error"."""
    assert isinstance(classify_error(Exception("who knows")), BrokerError)


def test_classifying_never_raises_on_a_malformed_error_body():
    """The vendor parses its payload lazily in properties, so both can throw. A
    classifier that raises while classifying replaces a diagnosable broker failure
    with an undiagnosable one."""
    result = classify_error(_APIErrorish(raising=True))
    assert isinstance(result, BrokerError)


def test_an_already_typed_error_passes_through_unchanged():
    original = RateLimitedError("already classified")
    assert classify_error(original) is original


# --- what the trader does with them -----------------------------------------
def _trader(broker, **kwargs):
    return LiveTrader(broker, VolumeSpikeStrategy.create_with_defaults(), **kwargs)


def test_a_duplicate_order_is_not_treated_as_a_failed_submission():
    """The venue already holds this order. Resubmitting is the one thing that must
    not happen, and reporting it as a failure invites exactly that."""
    broker = FailingBroker({"submit_bracket_order": DuplicateOrderError("already exists")})
    trader = _trader(broker, respect_market_hours=False)

    result = trader.handle_signal("AAA", signals.BUY, 100.0, bar_timestamp=datetime(2024, 1, 2, 10, 0))

    assert result is None
    assert broker.orders == []  # nothing resubmitted


def test_a_refused_entry_does_not_break_the_loop():
    broker = FailingBroker({"submit_bracket_order": InsufficientFundsError("no")})
    assert _trader(broker, respect_market_hours=False).handle_signal("AAA", signals.BUY, 100.0) is None


def test_revoked_credentials_mean_closed_not_open():
    """The old fallback said "assume open" for every clock failure, so an expired
    token produced "the market is open" and the system kept placing orders on the
    strength of an answer nobody had given it."""
    broker = FailingBroker({"get_market_status": AuthenticationError("token expired")})
    assert _trader(broker)._market_open() is False


def test_an_unreachable_clock_still_assumes_open():
    """The other direction: freezing trading over one failed request would be worse,
    and the bar stream only delivers during sessions anyway."""
    broker = FailingBroker({"get_market_status": BrokerUnavailableError("timeout")})
    assert _trader(broker)._market_open() is True


def test_an_unreadable_account_stops_the_entry_rather_than_guessing_a_size():
    broker = FailingBroker({"get_account": BrokerUnavailableError("timeout")})
    assert _trader(broker, respect_market_hours=False).handle_signal("AAA", signals.BUY, 100.0) is None
    assert broker.orders == []


def test_a_position_that_could_not_be_closed_stays_in_the_book():
    """Dropping it would leave the strategy holding a position it is no longer
    entitled to exit — the exact failure the book was hydrated to prevent."""
    from tradeflow.brokers.base import Position

    broker = FailingBroker({"close_position": BrokerUnavailableError("timeout")})
    broker.positions["AAA"] = Position(
        symbol="AAA",
        qty=10.0,
        side="long",
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=1000.0,
        unrealized_pl=0.0,
    )
    trader = _trader(broker, respect_market_hours=False)
    trader.sync_strategy_book()

    trader.handle_signal("AAA", signals.CLOSE_BUY, 100.0)

    assert "AAA" in trader._strategy.positions


def test_an_unreadable_broker_never_empties_the_book():
    """`list_positions` returning [] on failure would mean "you are flat", and the
    book would be wiped — silently un-exiting every real position."""
    from tradeflow.brokers.base import Position

    broker = FailingBroker()
    broker.positions["AAA"] = Position(
        symbol="AAA",
        qty=10.0,
        side="long",
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=1000.0,
        unrealized_pl=0.0,
    )
    trader = _trader(broker)
    trader.sync_strategy_book()
    assert "AAA" in trader._strategy.positions

    broker.failures["list_positions"] = BrokerUnavailableError("timeout")
    with pytest.raises(BrokerError):
        trader.sync_strategy_book()
    assert "AAA" in trader._strategy.positions  # preserved, not cleared
