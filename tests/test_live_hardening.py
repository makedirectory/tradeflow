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
    ScriptedStrategy,
    bar_event,
)
from tradeflow.brokers.base import Position
from tradeflow.engine.barcheck import BarChecks, BarQualityFilter
from tradeflow.engine.live import BlindStartError, LiveEngine
from tradeflow.execution.ledger import MISSING, QUANTITY_DRIFT, UNEXPECTED, PositionLedger
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services.registry import STRATEGIES
from tradeflow.strategies import signals
from tradeflow.utils.timeutils import NEW_YORK

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


class _EmptyWarmUp(ScriptedFeed):
    """A feed whose historical half returns nothing — an unentitled key's symptom."""

    def get_bars(self, symbols, timeframe, start, end):
        return {}


def _blind_engine(**kwargs):
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    feed = _EmptyWarmUp(["AAA"], events=[bar_event(minute=1)], n=10, freq="1D")
    engine = LiveEngine(
        strategy,
        MarketDataClient(feed),
        LiveTrader(RecordingBroker(), strategy, respect_market_hours=False),
        **kwargs,
    )
    return engine, feed


def test_a_run_that_warmed_up_on_nothing_refuses_to_start():
    """The bug: it started, streamed normally, and logged one line per symbol.

    Every indicator then computed from history it never had, and from inside the loop
    that is indistinguishable from a strategy that simply is not triggering — so the
    run looks healthy for as long as you let it go.
    """
    engine, feed = _blind_engine()

    with pytest.raises(BlindStartError) as exit_info:
        asyncio.run(engine.start(["AAA"]))

    assert feed.delivered == 0  # refused before the stream, not after
    # The message must name the likeliest cause and both remedies.
    assert "--feed iex" in str(exit_info.value)
    assert "--allow-blind-start" in str(exit_info.value)


def test_a_blind_start_is_allowed_when_it_is_asked_for_explicitly():
    """Both directions. Previously the only behaviour, now opt-in."""
    engine, feed = _blind_engine(allow_blind_start=True)

    asyncio.run(engine.start(["AAA"]))

    assert feed.delivered == 1


def test_partial_warm_up_is_not_treated_as_blind():
    """One symbol with history is not the failure this guards, and refusing it would
    make a single delisted name stop an otherwise valid book."""
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    feed = ScriptedFeed(["AAA", "BBB"], events=[bar_event(minute=1)], n=10, freq="1D")
    original = feed.get_bars

    def only_one(symbols, timeframe, start, end):
        return {"AAA": original(symbols, timeframe, start, end)["AAA"]}

    feed.get_bars = only_one
    engine = LiveEngine(
        strategy,
        MarketDataClient(feed),
        LiveTrader(RecordingBroker(), strategy, respect_market_hours=False),
    )

    asyncio.run(engine.start(["AAA", "BBB"]))  # must not raise


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
    # The one remaining read is the cold-start book hydration, which is not governed
    # by `reconcile_every` — an unhydrated book breaks exits outright, so it is a
    # correctness step rather than a cadence choice. No per-bar sweep fires.
    assert broker.calls.count("list_positions") == 1


# --- the strategy's position book -------------------------------------------
def _long_position(symbol=SYMBOL, qty=10.0, entry=205.0):
    return Position(
        symbol=symbol,
        qty=qty,
        side="long",
        avg_entry_price=entry,
        current_price=entry,
        market_value=qty * entry,
        unrealized_pl=0.0,
    )


#: Warm-up history is a random walk around 100, so a pivot well above it keeps the
#: strategy unambiguously flat until the scripted bars arrive.
PIVOT = 200.0


def _scripted_engine(closes, *, positions=None):
    """The real engine and trader, driven by a strategy that goes long above PIVOT."""
    config = {name: spec["default"] for name, spec in ScriptedStrategy.PARAM_RANGES.items()}
    config["pivot"] = PIVOT
    strategy = ScriptedStrategy(config)
    # Day 15: the warm-up frame is ten daily bars from the 2nd, and a scripted bar
    # timestamped before them would not be the latest signal in the buffer.
    events = [bar_event(day=15, minute=i, close=close) for i, close in enumerate(closes, start=1)]
    feed = ScriptedFeed([SYMBOL], events=events, n=10, freq="1D")
    broker = RecordingBroker(positions=None)
    for position in positions or []:
        broker.positions[position.symbol] = position
    engine = LiveEngine(
        strategy,
        MarketDataClient(feed),
        LiveTrader(broker, strategy, respect_market_hours=False),
        reconcile_every=0,
    )
    return engine, broker, strategy


def test_a_strategy_that_holds_a_position_can_actually_exit_it():
    """The regression. An exit is only legitimate if the strategy believes it holds
    the position, and nothing in live mode used to give it that belief — so every
    CLOSE_BUY was rewritten to HOLD and the position could be opened but never
    closed. This drives the whole path: bar -> process_bar -> handle_signal ->
    broker.close_position."""
    engine, broker, _ = _scripted_engine([205.0, 206.0, 195.0], positions=[_long_position()])

    asyncio.run(engine.start([SYMBOL]))

    assert broker.closed == [SYMBOL]


def test_the_book_is_hydrated_from_the_broker_before_the_first_bar():
    """Cold start: a restarted process must know what it holds."""
    engine, _, strategy = _scripted_engine([205.0], positions=[_long_position()])
    assert strategy.positions == {}  # nothing known before start

    asyncio.run(engine.start([SYMBOL]))

    assert strategy.positions[SYMBOL]["side"] == signals.BUY
    assert strategy.positions[SYMBOL]["qty"] == 10.0


def test_broker_truth_replaces_a_stale_belief_rather_than_merging_with_it():
    """A belief that disagrees with the account is just a stale belief."""
    engine, broker, strategy = _scripted_engine([205.0], positions=[_long_position()])
    strategy.positions = {"GONE": {"side": signals.BUY}, SYMBOL: {"side": signals.SELL}}

    engine.live_trader.sync_strategy_book()

    assert set(strategy.positions) == {SYMBOL}  # the phantom is dropped, not kept
    assert strategy.positions[SYMBOL]["side"] == signals.BUY  # and the side corrected


def test_an_entry_registers_in_the_book_immediately():
    """Intent, recorded at submission. Without it the strategy forgets its own entry
    until the next sweep — so it would neither recognize the position to exit it nor
    know to stop re-entering it."""
    engine, broker, strategy = _scripted_engine([205.0, 206.0])

    asyncio.run(engine.start([SYMBOL]))

    assert [o["type"] for o in broker.orders] == ["bracket"]  # entered once, not twice
    assert strategy.positions[SYMBOL]["side"] == signals.BUY


def test_a_naive_streamed_bar_does_not_silence_a_strategy_warmed_from_history():
    """Warm-up history is localized to New York; a feed is free to stream a naive
    timestamp. One naive value in an otherwise aware index made every later comparison
    raise inside `process_bar` — whose blanket except turned that into a strategy that
    emitted nothing at all, indefinitely, with no error anywhere."""
    from tests.fakes import make_ohlcv

    config = {name: spec["default"] for name, spec in ScriptedStrategy.PARAM_RANGES.items()}
    config["pivot"] = PIVOT
    strategy = ScriptedStrategy(config)
    strategy.warm_up(SYMBOL, strategy.process_data(make_ohlcv(n=10, freq="1D")))
    assert strategy.get_real_time_buffer(SYMBOL).index.tz is not None  # aware history

    naive = datetime(2024, 1, 15, 10, 0)  # what the feed hands us
    signal = strategy.process_bar(SYMBOL, _bar(close=PIVOT + 5.0), naive)

    assert signal == signals.BUY  # an actual opinion, not silence


def test_reading_the_book_never_places_an_order():
    """Report, never remediate: hydration is a read of broker truth."""
    engine, broker, _ = _scripted_engine([205.0], positions=[_long_position()])

    engine.live_trader.sync_strategy_book()

    assert broker.orders == []
    assert broker.closed == []


def test_a_broker_that_cannot_be_read_at_start_up_does_not_stop_the_engine():
    """Starting flat is wrong but recoverable; refusing to start is not better."""

    class Unreadable(RecordingBroker):
        def list_positions(self):
            raise ConnectionError("broker unreachable")

    strategy = ScriptedStrategy.create_with_defaults()
    feed = ScriptedFeed([SYMBOL], events=[bar_event(day=15, minute=1, close=205.0)], n=10, freq="1D")
    engine = LiveEngine(
        strategy,
        MarketDataClient(feed),
        LiveTrader(Unreadable(), strategy, respect_market_hours=False),
        reconcile_every=0,
    )

    asyncio.run(engine.start([SYMBOL]))  # must not raise

    assert feed.delivered == 1


# --- two clocks -------------------------------------------------------------
def test_the_live_path_imports_no_research_machinery():
    """The invariant the whole project is shaped around. Hardening must not smuggle
    a research import into the order path."""
    from pathlib import Path

    for module in (
        "tradeflow/engine/live.py",
        "tradeflow/engine/barcheck.py",
        "tradeflow/execution/ledger.py",
        "tradeflow/execution/live_trader.py",
        "tradeflow/execution/sizing.py",
    ):
        source = Path(module).read_text()
        for forbidden in (
            "tradeflow.services",
            "tradeflow.analytics",
            "tradeflow.optimization",
            "tradeflow.research",
        ):
            assert forbidden not in source, f"{module} reaches into {forbidden}"


# --- warm-up window ---------------------------------------------------------
@pytest.mark.parametrize("spec,periods", [("1Min", 50), ("5Min", 50), ("1Hour", 50), ("1Day", 50)])
def test_the_warm_up_window_actually_spans_enough_sessions(spec, periods):
    """It used to convert bars to wall-clock time directly, which treats the
    overnight gap, the weekend and every holiday as tradeable: 50 one-minute bars
    became "100 minutes ago", so a 09:35 start fetched five bars for a fifty-bar
    indicator. Daily under-fetched too — 100 calendar days is about 70 sessions."""
    from tradeflow.marketdata.timeframe import Timeframe

    timeframe = Timeframe.parse(spec)
    start = LiveEngine._lookback_start(timeframe, periods)

    calendar_days = (datetime.now(NEW_YORK) - start).total_seconds() / 86400
    sessions = calendar_days * 5 / 7
    assert sessions * timeframe.bars_per_trading_day() >= periods


def test_an_intraday_warm_up_reaches_past_the_previous_session():
    """The specific failure: a window that never leaves the current morning."""
    from tradeflow.marketdata.timeframe import Timeframe

    start = LiveEngine._lookback_start(Timeframe.parse("1Min"), 50)
    assert (datetime.now(NEW_YORK) - start) > timedelta(days=1)


def test_a_short_warm_up_is_reported_rather_than_absorbed(caplog):
    """Too little history produces confident-looking signals the backtest never
    validated, and nothing else in the loop can tell that from a quiet market."""
    strategy = ScriptedStrategy.create_with_defaults()
    strategy.config["required_lookback_periods"] = 500  # more than the feed holds
    feed = ScriptedFeed([SYMBOL], events=[], n=10, freq="1D")
    engine = LiveEngine(
        strategy,
        MarketDataClient(feed),
        LiveTrader(RecordingBroker(), strategy, respect_market_hours=False),
        reconcile_every=0,
    )

    with caplog.at_level("WARNING"):
        engine._warm_up([SYMBOL])

    assert any("only 10 of the 500" in record.getMessage() for record in caplog.records)


# --- missed edges -----------------------------------------------------------
def test_an_entry_edge_missed_during_warm_up_is_still_acted_on():
    """Entries are edge-triggered, so a crossing that happened inside the warm-up
    history left the score saying "should be long" while every bar emitted HOLD —
    and the position was never opened at all."""
    engine, broker, strategy = _scripted_engine([206.0, 207.0])
    # The crossing happens in history: warm the buffer already above the pivot, so no
    # live bar carries the edge.
    import pandas as pd

    history = pd.DataFrame(
        {"open": 205.0, "high": 206.0, "low": 204.0, "close": 205.0, "volume": 1000.0},
        index=pd.date_range("2024-01-02 09:30", periods=5, freq="1D", tz=NEW_YORK),
    )
    strategy.warm_up(SYMBOL, history)

    signal = strategy.process_bar(SYMBOL, _bar(close=206.0), datetime(2024, 1, 15, 10, 0))

    assert signal == signals.BUY  # re-affirmed, not silently held


def test_an_exit_edge_missed_while_holding_is_still_acted_on():
    """The mirror case, and the worse one: a real position nothing will close."""
    engine, broker, strategy = _scripted_engine([195.0], positions=[_long_position()])
    engine.live_trader.sync_strategy_book()
    import pandas as pd

    history = pd.DataFrame(
        {"open": 195.0, "high": 196.0, "low": 194.0, "close": 195.0, "volume": 1000.0},
        index=pd.date_range("2024-01-02 09:30", periods=5, freq="1D", tz=NEW_YORK),
    )
    strategy.warm_up(SYMBOL, history)  # already below the pivot: the exit edge is gone

    signal = strategy.process_bar(SYMBOL, _bar(close=194.0), datetime(2024, 1, 15, 10, 0))

    assert signal == signals.CLOSE_BUY


def test_entry_reaffirmation_can_be_turned_off():
    """A legitimate preference: wait for a fresh crossing rather than open a position
    on a signal this process never saw fire."""
    engine, broker, strategy = _scripted_engine([206.0, 207.0])
    strategy.config["reaffirm_entries"] = False
    import pandas as pd

    history = pd.DataFrame(
        {"open": 205.0, "high": 206.0, "low": 204.0, "close": 205.0, "volume": 1000.0},
        index=pd.date_range("2024-01-02 09:30", periods=5, freq="1D", tz=NEW_YORK),
    )
    strategy.warm_up(SYMBOL, history)

    signal = strategy.process_bar(SYMBOL, _bar(close=206.0), datetime(2024, 1, 15, 10, 0))

    assert signal == signals.HOLD


def test_turning_it_off_still_does_not_strand_an_open_position():
    """Declining to *enter* is a preference. Declining to *close* something the
    strategy no longer wants is a stuck position, so the exit side is never gated."""
    engine, broker, strategy = _scripted_engine([195.0], positions=[_long_position()])
    strategy.config["reaffirm_entries"] = False
    engine.live_trader.sync_strategy_book()
    import pandas as pd

    history = pd.DataFrame(
        {"open": 195.0, "high": 196.0, "low": 194.0, "close": 195.0, "volume": 1000.0},
        index=pd.date_range("2024-01-02 09:30", periods=5, freq="1D", tz=NEW_YORK),
    )
    strategy.warm_up(SYMBOL, history)

    signal = strategy.process_bar(SYMBOL, _bar(close=194.0), datetime(2024, 1, 15, 10, 0))

    assert signal == signals.CLOSE_BUY


def test_a_fresh_crossing_still_enters_with_reaffirmation_off():
    """The flag suppresses re-affirmation, not trading."""
    engine, broker, strategy = _scripted_engine([205.0, 206.0])
    strategy.config["reaffirm_entries"] = False

    asyncio.run(engine.start([SYMBOL]))

    assert [o["type"] for o in broker.orders] == ["bracket"]


def test_reaffirmation_defaults_to_on():
    """The default is the assumption: a trend-follower started mid-trend takes the
    position rather than sitting flat until the next crossing."""
    from tradeflow.strategies.base import REAFFIRM_ENTRIES_DEFAULT

    assert REAFFIRM_ENTRIES_DEFAULT is True
    strategy = ScriptedStrategy.create_with_defaults()
    assert strategy.config.get("reaffirm_entries", REAFFIRM_ENTRIES_DEFAULT) is True


def test_a_book_that_already_matches_the_score_stays_quiet():
    """The other direction: re-affirmation must not re-fire every bar on a position
    that is exactly as intended."""
    engine, broker, strategy = _scripted_engine([205.0, 206.0, 207.0])

    asyncio.run(engine.start([SYMBOL]))

    assert [o["type"] for o in broker.orders] == ["bracket"]  # once, not three times


def test_the_preflight_runs_the_same_warm_up_the_run_would():
    """The gap: --preflight printed the whole contract but never touched market data,
    so the one number that decides whether a run is viable — how many symbols actually
    have history — could not be confirmed from it. A lighter probe would not do: a
    preflight that fetches differently from the run it precedes confirms nothing.
    """
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    feed = ScriptedFeed(["AAA", "BBB"], events=[bar_event(minute=1)], n=10, freq="1D")
    engine = LiveEngine(
        strategy,
        MarketDataClient(feed),
        LiveTrader(RecordingBroker(), strategy, respect_market_hours=False),
    )

    assert engine.warm_up_coverage(["AAA", "BBB"]) == (2, 2)


def test_the_preflight_reports_a_blind_warm_up_without_raising():
    """It reports; the refusal belongs to the start path. A preflight that raised would
    lose the rest of the contract it exists to print."""
    engine, _ = _blind_engine()

    assert engine.warm_up_coverage(["AAA"]) == (0, 1)
