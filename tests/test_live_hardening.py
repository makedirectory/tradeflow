"""The live path under adversarial conditions.

The trade clock is the smallest and least-tested part of this project, which is
backwards: it is the only part that can lose money. This drives the **real**
`LiveEngine` and the **real** `LiveTrader`, faking only the two edges — data in,
broker out — so the fence cannot pass by encoding the same assumptions the code
makes.

Two properties matter more than the rest. A guard must **reject, never repair**: the
moment the live path fixes its inputs it stops being the thing the backtest
validated. And the ledger must **report, never remediate**: an automated system that
notices a missed fill and corrects it is one that can double a position while nobody
is watching.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from tests.fakes import (
    FakeTradeUpdate,
    RecordingBroker,
    ScriptedFeed,
    bar_event,
)
from tradeflow.brokers.base import Position
from tradeflow.engine.barcheck import BarChecks, BarQualityFilter
from tradeflow.engine.live import LiveEngine
from tradeflow.execution.ledger import MISSING, QUANTITY_DRIFT, UNEXPECTED, PositionLedger
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services.registry import STRATEGIES

SYMBOL = "AAA"


def _filter(**kwargs):
    return BarQualityFilter(checks=BarChecks(**kwargs), interval=timedelta(minutes=5))


def _bar(close=100.0, **kwargs):
    return {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0, **kwargs}


def _ts(minute=0):
    return datetime(2024, 1, 2, 10, minute)


# --- bar guards: internal consistency ---------------------------------------
def test_a_well_formed_bar_is_accepted():
    assert _filter().check(SYMBOL, _bar(), _ts(), now=_ts()).accepted


@pytest.mark.parametrize(
    "bar,reason",
    [
        ({"open": 100, "high": 98, "low": 99, "close": 99, "volume": 1}, "inverted_range"),
        ({"open": 100, "high": 101, "low": 99, "close": 105, "volume": 1}, "outside_range"),
        ({"open": 0, "high": 101, "low": 99, "close": 100, "volume": 1}, "non_positive"),
        ({"open": -5, "high": -1, "low": -9, "close": -5, "volume": 1}, "non_positive"),
        ({"open": 100, "high": 101, "low": 99, "close": 100, "volume": -3}, "negative_volume"),
        ({"open": 100, "high": 101, "close": 100, "volume": 1}, "malformed"),
    ],
)
def test_a_broken_bar_is_rejected_with_its_reason(bar, reason):
    """A bar failing these is not a market event; it is a broken message."""
    verdict = _filter().check(SYMBOL, bar, _ts(), now=_ts())
    assert not verdict.accepted
    assert verdict.reason == reason
    assert verdict.detail  # never a bare rejection


# --- bar guards: sequencing and staleness -----------------------------------
def test_an_out_of_order_bar_is_rejected():
    f = _filter()
    assert f.check(SYMBOL, _bar(), _ts(10), now=_ts(10)).accepted
    assert f.check(SYMBOL, _bar(), _ts(5), now=_ts(10)).reason == "out_of_order"
    # A repeat of the same timestamp is equally not-after.
    assert f.check(SYMBOL, _bar(), _ts(10), now=_ts(10)).reason == "out_of_order"


def test_a_stale_bar_is_rejected_but_a_merely_late_one_is_not():
    now = _ts(30)
    assert _filter().check(SYMBOL, _bar(), now - timedelta(minutes=10), now=now).accepted  # 2 intervals
    # A fresh filter: otherwise the older bar trips the ordering check first, which
    # is correct behavior but not what this test is about.
    assert _filter().check(SYMBOL, _bar(), now - timedelta(minutes=60), now=now).reason == "stale"


def test_a_feed_that_switches_timestamp_awareness_does_not_crash_the_loop():
    """A vendor serializing naive on one bar and aware on the next must produce an
    answer, not a TypeError from inside the order path."""
    f = _filter()
    assert f.check(SYMBOL, _bar(), datetime.now(), now=None).accepted
    verdict = f.check(SYMBOL, _bar(), datetime.now(timezone.utc) + timedelta(seconds=1), now=None)
    assert verdict is not None  # a verdict either way, never an exception


# --- bar guards: spikes -----------------------------------------------------
def test_an_implausible_single_bar_move_is_rejected():
    f = _filter(max_return=0.35)
    assert f.check(SYMBOL, _bar(100.0), _ts(1), now=_ts(1)).accepted
    assert f.check(SYMBOL, _bar(1000.0), _ts(2), now=_ts(2)).reason == "spike"


def test_a_large_but_real_move_is_not_rejected():
    """A guard that vetoes real moves silently removes the strategy's best
    opportunities — the threshold catches a bad tick, not a violent day."""
    f = _filter(max_return=0.35)
    f.check(SYMBOL, _bar(100.0), _ts(1), now=_ts(1))
    assert f.check(SYMBOL, _bar(125.0), _ts(2), now=_ts(2)).accepted  # +25%


def test_a_rejected_bar_does_not_become_the_baseline():
    """Otherwise one bad tick poisons the comparison for every bar after it."""
    f = _filter(max_return=0.35)
    f.check(SYMBOL, _bar(100.0), _ts(1), now=_ts(1))
    assert f.check(SYMBOL, _bar(1000.0), _ts(2), now=_ts(2)).reason == "spike"
    # Back to normal: judged against 100, not against the rejected 1000.
    assert f.check(SYMBOL, _bar(101.0), _ts(3), now=_ts(3)).accepted


def test_zero_volume_is_only_suspicious_once_a_symbol_has_traded():
    f = _filter()
    assert f.check(SYMBOL, _bar(volume=0.0), _ts(1), now=_ts(1)).accepted  # never seen trading
    assert f.check(SYMBOL, _bar(volume=500.0), _ts(2), now=_ts(2)).accepted
    assert f.check(SYMBOL, _bar(volume=0.0), _ts(3), now=_ts(3)).reason == "zero_volume"


def test_symbols_are_judged_independently():
    f = _filter()
    assert f.check("AAA", _bar(), _ts(10), now=_ts(10)).accepted
    assert f.check("BBB", _bar(), _ts(5), now=_ts(10)).accepted  # not out of order for BBB


def test_every_check_can_be_disabled():
    f = _filter(check_spike=False, check_ohlc=False, check_staleness=False, check_monotonic=False)
    f.check(SYMBOL, _bar(100.0), _ts(5), now=_ts(5))
    assert f.check(SYMBOL, _bar(9999.0), _ts(1), now=_ts(59)).accepted


def test_no_guard_ever_modifies_a_bar():
    """Reject, never repair. A live path that fixes its inputs stops being the thing
    the backtest validated."""
    f = _filter()
    for bar in (_bar(100.0), {"open": 1, "high": 0, "low": 5, "close": 3, "volume": -1}):
        before = dict(bar)
        f.check(SYMBOL, bar, _ts(1), now=_ts(1))
        assert bar == before


# --- bar guards: visibility -------------------------------------------------
def test_the_filter_reports_what_it_discarded():
    """A guard quietly eating a third of the feed looks, from the strategy's side,
    exactly like a quiet market."""
    f = _filter()
    f.check(SYMBOL, _bar(100.0), _ts(1), now=_ts(1))
    for i in range(9):
        f.check(SYMBOL, {"open": 1, "high": 0, "low": 5, "close": 3, "volume": 1}, _ts(i + 2), now=_ts(20))

    report = f.report()
    assert report["seen"] == 10
    assert report["rejected"] == 9
    assert report["rate"] == pytest.approx(0.9)
    # high=0 is caught by the non-positive check before the inverted-range one.
    assert report["by_reason"]["non_positive"] == 9
    assert report["elevated"] is True


def test_a_clean_feed_is_not_flagged_as_elevated():
    f = _filter()
    for i in range(20):
        f.check(SYMBOL, _bar(100.0 + i * 0.1), _ts(i), now=_ts(i))
    assert f.report()["elevated"] is False


# --- the ledger -------------------------------------------------------------
def _ledger(tmp_path):
    return PositionLedger(tmp_path / "ledger.jsonl")


def test_the_ledger_replays_its_own_file_rather_than_holding_state(tmp_path):
    """A restarted process must recover its expectation exactly."""
    led = _ledger(tmp_path)
    led.record_fill("AAA", "buy", 10)
    led.record_fill("AAA", "buy", 5)
    led.record_fill("BBB", "sell", 3)

    assert PositionLedger(tmp_path / "ledger.jsonl").expected_positions() == {"AAA": 15.0, "BBB": -3.0}


def test_a_flat_symbol_looks_the_same_as_one_never_traded(tmp_path):
    led = _ledger(tmp_path)
    led.record_fill("AAA", "buy", 10)
    led.record_fill("AAA", "sell", 10)
    assert led.expected_positions() == {}


def test_a_close_zeroes_the_expectation(tmp_path):
    led = _ledger(tmp_path)
    led.record_fill("AAA", "buy", 10)
    led.record_close("AAA")
    assert led.expected_positions() == {}


def test_each_divergence_class_is_detected(tmp_path):
    led = _ledger(tmp_path)
    led.record_fill("MISS", "buy", 10)  # broker will not have it
    led.record_fill("DRIFT", "buy", 10)  # broker will have 4

    broker = RecordingBroker(
        positions=[
            Position("DRIFT", 4, "long", 100, 100, 400, 0),
            Position("SURPRISE", 7, "long", 50, 50, 350, 0),
        ]
    )
    report = led.reconcile(broker)

    assert not report.clean
    kinds = {d.symbol: d.kind for d in report.divergences}
    assert kinds == {"MISS": MISSING, "DRIFT": QUANTITY_DRIFT, "SURPRISE": UNEXPECTED}
    assert "authoritative" in report.summary()


def test_a_matching_book_reconciles_clean(tmp_path):
    led = _ledger(tmp_path)
    led.record_fill("AAA", "buy", 10)
    report = led.reconcile(RecordingBroker(positions=[Position("AAA", 10, "long", 100, 100, 1000, 0)]))
    assert report.clean and report.summary().endswith("no divergence.")


def test_a_short_position_reconciles_by_sign(tmp_path):
    led = _ledger(tmp_path)
    led.record_fill("AAA", "sell", 10)
    report = led.reconcile(RecordingBroker(positions=[Position("AAA", 10, "short", 100, 100, 1000, 0)]))
    assert report.clean


def test_an_unreachable_broker_is_not_reported_as_divergence(tmp_path):
    """ "I could not check" and "your book is wrong" must not look the same."""

    class Unreachable(RecordingBroker):
        def list_positions(self):
            raise ConnectionError("broker unreachable")

    led = _ledger(tmp_path)
    led.record_fill("AAA", "buy", 10)
    report = led.reconcile(Unreachable())
    assert report.clean and report.n_actual == 0


def test_reconciliation_reads_the_broker_once_not_once_per_symbol(tmp_path):
    """It runs on the trade clock; its API usage must not scale with the universe."""
    led = _ledger(tmp_path)
    for i in range(25):
        led.record_fill(f"S{i}", "buy", 1)
    broker = RecordingBroker(positions=[])
    led.reconcile(broker)
    assert broker.calls.count("list_positions") == 1


def test_a_torn_final_line_does_not_cause_total_amnesia(tmp_path):
    """A process killed mid-write must lose one entry, not the whole history."""
    path = tmp_path / "ledger.jsonl"
    led = PositionLedger(path)
    led.record_fill("AAA", "buy", 10)
    with path.open("a") as fh:
        fh.write('{"event": "fill", "symbol": "BBB", "qty"')  # truncated

    assert PositionLedger(path).expected_positions() == {"AAA": 10.0}


def test_the_ledger_never_places_an_order(tmp_path):
    """Report, never remediate. Detection is the feature; action is a decision."""
    import inspect

    source = inspect.getsource(PositionLedger)
    for forbidden in ("submit_", "close_position", "cancel_order"):
        assert forbidden not in source


def test_an_unwritable_ledger_never_breaks_the_caller(tmp_path):
    led = PositionLedger(tmp_path / "nested" / "ledger.jsonl")
    led.path = tmp_path / "nope" / "deeper" / "ledger.jsonl"  # parent does not exist
    led.record_fill("AAA", "buy", 1)  # must not raise
    assert led.expected_positions() == {}


# --- the loop fence ---------------------------------------------------------
def _engine(events, *, broker=None, bar_filter=None, ledger=None, raise_after=None):
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    feed = ScriptedFeed(["AAA", "BBB"], events=events, n=200, freq="1D", raise_after=raise_after)
    broker = broker or RecordingBroker()
    engine = LiveEngine(
        strategy,
        MarketDataClient(feed),
        LiveTrader(broker, strategy, respect_market_hours=False),
        bar_filter=bar_filter,
        ledger=ledger,
        reconcile_every=0,  # off unless a test asks for it
    )
    return engine, feed, broker


def test_the_loop_survives_a_feed_of_malformed_bars():
    """Every bar is broken; the loop must complete and place no orders."""
    events = [
        bar_event(minute=1, high=0, low=99),  # inverted
        bar_event(minute=2, close=-5),  # non-positive
        bar_event(minute=3, close=1e9),  # spike
    ]
    engine, feed, broker = _engine(events, bar_filter=_filter())
    asyncio.run(engine.start(["AAA"]))

    assert feed.delivered == 3
    assert not [c for c in broker.calls if c.startswith(("bracket", "market"))]
    assert engine.bar_filter.report()["rejected"] == 3


def test_out_of_order_and_duplicate_bars_never_reach_the_strategy():
    events = [
        bar_event(minute=10),
        bar_event(minute=5),  # out of order
        bar_event(minute=10),  # duplicate
    ]
    engine, _, _ = _engine(events, bar_filter=_filter(check_staleness=False))
    asyncio.run(engine.start(["AAA"]))
    assert engine.bar_filter.report()["by_reason"]["out_of_order"] == 2


def test_a_dropped_stream_propagates_rather_than_being_swallowed():
    """A silently-dead stream is worse than a crash: it looks like a quiet market."""
    engine, _, _ = _engine([bar_event(minute=i) for i in range(5)], raise_after=2)
    with pytest.raises(ConnectionError):
        asyncio.run(engine.start(["AAA"]))


def test_a_broker_that_rejects_orders_does_not_stop_the_loop():
    events = [bar_event(minute=i, close=100 + i) for i in range(1, 6)]
    engine, feed, broker = _engine(events, broker=RecordingBroker(reject_orders=True))
    asyncio.run(engine.start(["AAA"]))
    assert feed.delivered == 5  # consumed the whole stream regardless


def test_warm_up_with_no_history_still_processes_the_first_live_bar():
    """An empty warm-up must not leave the loop unable to start."""

    class Empty(ScriptedFeed):
        def get_bars(self, symbols, timeframe, start, end):
            return {}

    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    feed = Empty(["AAA"], events=[bar_event(minute=1)], n=10, freq="1D")
    broker = RecordingBroker()
    engine = LiveEngine(
        strategy, MarketDataClient(feed), LiveTrader(broker, strategy, respect_market_hours=False)
    )
    asyncio.run(engine.start(["AAA"]))
    assert feed.delivered == 1


def test_without_a_filter_the_loop_behaves_exactly_as_before():
    """The guards are opt-in; passing none must change nothing."""
    events = [bar_event(minute=i, close=100 + i) for i in range(1, 4)]
    engine, feed, _ = _engine(events, bar_filter=None)
    asyncio.run(engine.start(["AAA"]))
    assert feed.delivered == 3 and engine.bar_filter is None


def test_the_loop_records_intent_and_fills_in_the_ledger(tmp_path):
    led = _ledger(tmp_path)
    engine, _, _ = _engine([bar_event(minute=1)], ledger=led)
    engine._on_trade_update(FakeTradeUpdate(event="fill", symbol="AAA", filled_qty=7, side="buy"))

    assert led.expected_positions() == {"AAA": 7.0}
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert any(r["event"] == "fill" for r in records)


def test_a_ledger_failure_never_breaks_the_order_path(tmp_path):
    class Exploding(PositionLedger):
        def record_fill(self, *a, **k):
            raise RuntimeError("disk on fire")

        def record_intent(self, *a, **k):
            raise RuntimeError("disk on fire")

    led = Exploding(tmp_path / "ledger.jsonl")
    engine, feed, _ = _engine([bar_event(minute=i, close=100 + i) for i in range(1, 4)], ledger=led)
    asyncio.run(engine.start(["AAA"]))  # must not raise
    engine._on_trade_update(FakeTradeUpdate())
    assert feed.delivered == 3


def test_scheduled_reconciliation_is_rate_limited(tmp_path):
    """It runs inside the loop; it cannot fire per bar."""
    led = _ledger(tmp_path)
    events = [bar_event(minute=i, close=100 + i) for i in range(1, 11)]
    engine, _, broker = _engine(events, ledger=led)
    engine.reconcile_every = 3600.0
    asyncio.run(engine.start(["AAA"]))
    assert broker.calls.count("list_positions") == 1  # once, not ten times


def test_the_first_sweep_happens_immediately_whatever_the_machines_uptime(tmp_path):
    """`time.monotonic()` counts from an arbitrary origin, so seeding "last swept"
    with 0.0 made the first sweep depend on how long the host had been up — skipped
    entirely on a freshly booted one, which is exactly when a process is most likely
    to have missed fills while it was down."""
    led = _ledger(tmp_path)
    engine, _, broker = _engine([bar_event(minute=1)], ledger=led)
    engine.reconcile_every = 3600.0
    assert engine._last_reconcile is None  # never swept, not "swept at zero"

    asyncio.run(engine.start(["AAA"]))
    assert broker.calls.count("list_positions") == 1


def test_reconciliation_can_be_disabled_outright(tmp_path):
    led = _ledger(tmp_path)
    engine, _, broker = _engine([bar_event(minute=i) for i in range(1, 4)], ledger=led)
    engine.reconcile_every = 0
    asyncio.run(engine.start(["AAA"]))
    assert broker.calls.count("list_positions") == 0


# --- two clocks -------------------------------------------------------------
def test_the_live_path_imports_no_research_machinery():
    """The invariant the whole project is shaped around. Hardening must not smuggle
    a research import into the order path."""
    from pathlib import Path

    for module in (
        "tradeflow/engine/live.py",
        "tradeflow/engine/barcheck.py",
        "tradeflow/execution/ledger.py",
    ):
        source = Path(module).read_text()
        for forbidden in (
            "tradeflow.services",
            "tradeflow.analytics",
            "tradeflow.optimization",
            "tradeflow.research",
        ):
            assert forbidden not in source, f"{module} reaches into {forbidden}"
