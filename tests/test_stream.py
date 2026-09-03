"""Live-stream reconnection tests for AlpacaMarketData.

The reconnect loop is exercised without any network by monkeypatching the stream
factory with a fake whose `_run_forever` is scripted to fail, then to cancel.
"""

import asyncio

import pytest
from alpaca.data.historical import StockHistoricalDataClient

from tradeflow.brokers.alpaca.market_data import AlpacaMarketData


class _FakeStream:
    def __init__(self, behavior):
        self._behavior = behavior
        self.subscribed = []
        self.stopped = False

    def subscribe_bars(self, callback, symbol):
        self.subscribed.append(symbol)

    async def _run_forever(self):
        await self._behavior()

    def stop(self):
        self.stopped = True


def _market_data():
    # Construction does not touch the network; keys can be placeholders.
    client = StockHistoricalDataClient("key", "secret")
    return AlpacaMarketData(client, "key", "secret", base_reconnect_delay=0, max_reconnect_delay=0)


def test_stream_reconnects_after_error_then_exits_on_cancel():
    md = _market_data()
    attempts = {"n": 0}

    async def behavior():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("socket dropped")  # first attempt fails -> reconnect
        raise asyncio.CancelledError()  # second attempt: shut down cleanly

    md._new_stream = lambda: _FakeStream(behavior)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(md.stream_bars(["AAA", "BBB"], lambda event: None))

    assert attempts["n"] == 2  # it reconnected exactly once before cancellation


def test_stream_returns_on_normal_completion():
    md = _market_data()

    async def behavior():
        return  # stream ends without error

    md._new_stream = lambda: _FakeStream(behavior)
    # Should complete without raising or looping forever.
    asyncio.run(md.stream_bars(["AAA"], lambda event: None))


def test_supports_trade_updates_requires_credentials():
    from alpaca.trading.client import TradingClient

    from tradeflow.brokers.alpaca.broker import AlpacaBroker

    with_keys = AlpacaBroker(TradingClient("k", "s", paper=True), "k", "s", paper=True)
    without_keys = AlpacaBroker(TradingClient("k", "s", paper=True))
    assert with_keys.supports_trade_updates() is True
    assert without_keys.supports_trade_updates() is False


def test_live_engine_runs_market_and_trade_update_streams():
    from tests.fakes import StreamingFakeMarketData, TradeUpdateFakeBroker
    from tradeflow.brokers.base import TradeUpdate
    from tradeflow.demo.strategies import DemoTrendStrategy
    from tradeflow.engine.live import LiveEngine
    from tradeflow.execution.live_trader import LiveTrader
    from tradeflow.marketdata.client import MarketDataClient

    symbols = ["AAA"]
    market_data = StreamingFakeMarketData(symbols, bars_to_emit=1)
    broker = TradeUpdateFakeBroker(
        updates=[TradeUpdate("fill", "AAA", "o1", "filled", 5, 100.0)], market_open=True
    )
    strategy = DemoTrendStrategy.create_with_defaults()

    engine = LiveEngine(strategy, MarketDataClient(market_data), LiveTrader(broker, strategy))
    received = []
    engine._on_trade_update = received.append

    asyncio.run(engine.start(symbols))
    assert received and received[0].symbol == "AAA"
