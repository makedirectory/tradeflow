"""The bar path must not block the loop, and must not overlap with itself.

The broker SDK is synchronous, and an entry makes several HTTP round trips. Running
those on the event loop stalls everything the loop is also carrying — other symbols
that signalled on the same bar, the trade-update stream that delivers fills, and the
reconciliation sweep.

The fix is a thread per blocking call and one semaphore around the order path. The
semaphore is not an optimisation: concurrency here is a correctness bug, because two
entries reading the book at once can both pass a limit only one of them fits inside.
"""

import asyncio
import time

from tests.fakes import RecordingBroker, ScriptedFeed
from tradeflow.engine.live import LiveEngine
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.marketdata.base import BarEvent
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services.registry import STRATEGIES
from tradeflow.strategies import signals
from tradeflow.utils.timeutils import NEW_YORK


def _engine(handle_signal):
    strategy = STRATEGIES["demo_trend"].create_with_defaults()
    strategy.process_bar = lambda symbol, bar, ts: signals.BUY
    trader = LiveTrader(RecordingBroker(), strategy, respect_market_hours=False)
    trader.handle_signal = handle_signal
    engine = LiveEngine(
        strategy,
        MarketDataClient(ScriptedFeed(["AAA"], events=[], n=10, freq="1D")),
        trader,
        reconcile_every=0,  # isolate the order path
    )
    return engine


def _bar(symbol):
    from datetime import datetime

    return BarEvent(
        symbol=symbol,
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=NEW_YORK),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1_000,
    )


def test_a_blocking_broker_call_does_not_stall_the_loop():
    """The bug: `handle_signal` ran inline on the loop, so a slow submission froze bar
    delivery, fill handling and reconciliation for its whole duration."""
    from tradeflow.execution import decision as decisions

    def slow(symbol, signal, price, bar_timestamp=None):
        time.sleep(0.3)  # a blocking HTTP round trip
        return decisions.decline(symbol, signal, "test", ())

    ticks = []
    ticks_when_entry_finished = []

    def slow_and_observe(symbol, signal, price, bar_timestamp=None):
        result = slow(symbol, signal, price, bar_timestamp)
        # How much the loop got done *while this was blocking* — the only thing that
        # distinguishes the two arrangements. Total ticks at the end is 20 either way,
        # which is how a first version of this test passed without the fix.
        ticks_when_entry_finished.append(len(ticks))
        return result

    engine = _engine(slow_and_observe)

    async def heartbeat():
        for _ in range(20):
            ticks.append(1)
            await asyncio.sleep(0.01)

    async def main():
        await asyncio.gather(engine._on_bar(_bar("AAA")), heartbeat())

    asyncio.run(main())

    # Inline, the loop is frozen for the whole call and this is 1.
    assert ticks_when_entry_finished[0] > 10, ticks_when_entry_finished


def test_entries_never_overlap():
    """One slot, deliberately. Two entries checking the book at once can both pass a
    gross-exposure limit that only one of them fits inside."""
    from tradeflow.execution import decision as decisions

    overlapping = []
    inside = []

    def tracked(symbol, signal, price, bar_timestamp=None):
        inside.append(symbol)
        overlapping.append(len(inside))
        time.sleep(0.05)
        inside.remove(symbol)
        return decisions.decline(symbol, signal, "test", ())

    engine = _engine(tracked)

    async def main():
        await asyncio.gather(*(engine._on_bar(_bar(s)) for s in ("AAA", "BBB", "CCC")))

    asyncio.run(main())

    assert overlapping == [1, 1, 1], f"entries overlapped: {overlapping}"


def test_submission_order_follows_arrival_order():
    """Serialized, not merely mutually excluded: a book assembled in a different order
    from the signals that produced it is not the book that was validated.

    Bars are dispatched concurrently here, which is how they actually arrive when
    several symbols print on the same bar — sequential awaits would pass whatever the
    lock did.
    """
    from tradeflow.execution import decision as decisions

    submitted = []

    def record(symbol, signal, price, bar_timestamp=None):
        # Descending sleeps: without a lock the shortest finishes first and the
        # recorded order inverts.
        time.sleep({"AAA": 0.08, "BBB": 0.06, "CCC": 0.04, "DDD": 0.02}[symbol])
        submitted.append(symbol)
        return decisions.decline(symbol, signal, "test", ())

    engine = _engine(record)
    order = ["AAA", "BBB", "CCC", "DDD"]

    async def main():
        await asyncio.gather(*(engine._on_bar(_bar(s)) for s in order))

    asyncio.run(main())

    assert submitted == order


def test_the_book_is_not_replaced_underneath_an_entry():
    """The reconciliation sweep replaces the position book wholesale. Doing that under
    an entry that has already checked the book would let it act on one that no longer
    exists, so both take the same lock."""
    from tradeflow.execution import decision as decisions

    events = []

    def slow_entry(symbol, signal, price, bar_timestamp=None):
        events.append("entry-start")
        time.sleep(0.1)
        events.append("entry-end")
        return decisions.decline(symbol, signal, "test", ())

    engine = _engine(slow_entry)
    engine.reconcile_every = 0.0001
    engine._last_reconcile = None

    def slow_sync():
        events.append("sync-start")
        time.sleep(0.05)
        events.append("sync-end")
        return 0

    engine.live_trader.sync_strategy_book = slow_sync

    async def main():
        await asyncio.gather(engine._on_bar(_bar("AAA")), engine._maybe_reconcile())

    asyncio.run(main())

    # Whichever ran first, it finished before the other started.
    for i in range(0, len(events), 2):
        assert events[i].endswith("-start") and events[i + 1].endswith("-end")
        assert events[i].split("-")[0] == events[i + 1].split("-")[0], events


def test_one_slot_only():
    """A semaphore sized above one would silently reintroduce the race."""
    engine = _engine(lambda *a, **k: None)

    assert engine._order_lock._value == 1
