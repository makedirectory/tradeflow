"""Which Alpaca market-data feed a run reads, and why it must stay opt-in.

The SDK's two halves default differently: a historical request resolves to the full
consolidated tape, while the live stream defaults to a single venue. An account
entitled to one and not the other warms up on nothing and streams normally — which
reads as an empty market rather than as a wrong feed.
"""

from datetime import datetime
from unittest import mock

import pytest

from tradeflow.brokers.alpaca.market_data import AlpacaMarketData
from tradeflow.marketdata.timeframe import Timeframe
from tradeflow.settings import SettingsError, data_feed


def _provider(feed=None):
    historical = mock.Mock()
    historical.get_stock_bars.return_value = mock.Mock(df=None)
    return AlpacaMarketData(historical, "key", "secret", feed=feed), historical


def _request_for(feed):
    provider, historical = _provider(feed)
    provider.get_bars(["AAA"], Timeframe.parse("1Day"), datetime(2024, 1, 1), datetime(2024, 2, 1))
    return historical.get_stock_bars.call_args[0][0]


def test_a_pinned_feed_reaches_the_historical_request():
    """The bug: warm-up sent no feed, so it resolved to the full tape and unentitled
    keys got `subscription does not permit querying recent SIP data` — 0 of 61 symbols,
    while the stream connected happily to a venue the same keys could read."""
    assert _request_for("iex").feed.value == "iex"


def test_an_unpinned_feed_leaves_the_request_exactly_as_before():
    """The constraint that makes this opt-in: pinning a default would put an entitled
    account on a partial venue or a delayed tape with nothing saying so."""
    assert _request_for(None).feed is None


def test_a_pinned_feed_reaches_the_stream_too():
    """Both halves, or the mismatch this exists to close simply moves."""
    provider, _ = _provider("iex")

    with mock.patch("tradeflow.brokers.alpaca.market_data.StockDataStream") as stream:
        provider._new_stream()

    assert stream.call_args.kwargs["feed"].value == "iex"


def test_an_unpinned_stream_is_constructed_exactly_as_before():
    provider, _ = _provider(None)

    with mock.patch("tradeflow.brokers.alpaca.market_data.StockDataStream") as stream:
        provider._new_stream()

    assert "feed" not in stream.call_args.kwargs


def test_the_environment_can_pin_a_feed(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    assert data_feed() == "iex"


def test_an_unset_environment_pins_nothing(monkeypatch):
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    assert data_feed() is None


def test_a_feed_nobody_supports_is_refused_rather_than_passed_through(monkeypatch):
    """Absent is not zero, and a typo is not a default: a misspelled feed must not
    quietly fall back to the behaviour the setting exists to change."""
    monkeypatch.setenv("ALPACA_DATA_FEED", "iexx")

    with pytest.raises(SettingsError) as exit_info:
        data_feed()

    assert "iexx" in str(exit_info.value)
