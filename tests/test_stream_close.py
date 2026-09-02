"""Closing a websocket without deadlocking the loop that has to close it.

The shutdown hang that survived three fixes came from here. alpaca-py's synchronous
``stop()`` is::

    asyncio.run_coroutine_threadsafe(self.stop_ws(), self._loop).result(timeout=5)

Called from inside that loop, the submitted work can never run, so the loop stays
blocked for the whole timeout — and every bound in the shutdown path is scheduled on
that loop, including the signal handlers. One blocking call took all of them down.
"""

import asyncio

from tradeflow.utils.streaming import close_stream


class _Stream:
    """A stream offering both shapes, recording which one was used."""

    def __init__(self, *, has_stop_ws=True, has_close=True, slow=False):
        self.used = None
        self.slow = slow
        if not has_stop_ws:
            del self.stop_ws
        if not has_close:
            del self.close

    def stop(self):
        self.used = "stop"
        raise AssertionError("the blocking stop() must never be called from the loop")

    async def stop_ws(self):
        self.used = "stop_ws"
        if self.slow:
            await asyncio.sleep(30)

    async def close(self):
        self.used = "close"


class _OnlyClose:
    def __init__(self):
        self.used = None

    def stop(self):
        raise AssertionError("the blocking stop() must never be called from the loop")

    async def close(self):
        self.used = "close"


class _Nothing:
    def stop(self):
        raise AssertionError("the blocking stop() must never be called from the loop")


def test_the_coroutine_is_preferred_over_the_blocking_stop():
    """`stop_ws` is the coroutine that `stop()` wraps, so awaiting it is the same
    shutdown without the self-deadlock."""
    stream = _Stream()

    asyncio.run(close_stream(stream))

    assert stream.used == "stop_ws"


def test_close_is_used_when_there_is_no_stop_ws():
    """Not every SDK names it the same; the blocking path is still not an option."""
    stream = _OnlyClose()

    asyncio.run(close_stream(stream))

    assert stream.used == "close"


def test_a_stream_with_no_async_closer_is_left_alone():
    """Rather than falling back to the blocking call, which is the whole bug."""
    asyncio.run(close_stream(_Nothing()))  # must not raise, must not call stop()


def test_a_close_that_will_not_finish_is_bounded():
    """A teardown that hangs here hangs the loop, and the loop is what every other
    bound in the shutdown path depends on."""
    import time

    stream = _Stream(slow=True)
    started = time.monotonic()

    asyncio.run(close_stream(stream))

    assert time.monotonic() - started < 5.0


def test_a_raising_close_never_propagates():
    """Cleanup must not mask the real reason a session is ending."""

    class Raises:
        async def stop_ws(self):
            raise RuntimeError("socket already gone")

    asyncio.run(close_stream(Raises()))  # must not raise


def test_a_synchronous_closer_is_not_awaited():
    """A stop_ws that is not a coroutine returns nothing to await, and treating its
    return value as awaitable would raise inside cleanup."""

    class Sync:
        def __init__(self):
            self.used = False

        def stop_ws(self):
            self.used = True

    stream = Sync()
    asyncio.run(close_stream(stream))

    assert stream.used is True


def test_the_alpaca_provider_delegates_rather_than_calling_stop():
    """The provider's own teardown must go through the shared helper — two copies of
    this reasoning existed once, and one of them was wrong."""
    from unittest import mock

    from tradeflow.brokers.alpaca.market_data import AlpacaMarketData

    provider = AlpacaMarketData(mock.Mock(), "k", "s")
    stream = _Stream()

    asyncio.run(provider._safe_stop(stream))

    assert stream.used == "stop_ws"
