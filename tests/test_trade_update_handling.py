"""From a broker event to a ledger record — the path that lost the side.

Two links, both previously untestable. The Alpaca mapping lived in a stream callback
reachable only through a socket, and the engine handler read a field the TradeUpdate
type did not have, so `getattr(update, "side", "buy")` resolved to buy forever.
"""

import asyncio
from enum import Enum

import pytest

from tests.fakes import RecordingBroker
from tradeflow.brokers.alpaca.broker import to_trade_update
from tradeflow.brokers.base import TradeUpdate
from tradeflow.engine.live import LiveEngine
from tradeflow.execution.ledger import PositionLedger
from tradeflow.execution.live_trader import LiveTrader
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.services.registry import STRATEGIES


class _Side(Enum):
    BUY = "buy"
    SELL = "sell"


class _Status(Enum):
    FILLED = "filled"


class _Order:
    def __init__(self, side=_Side.SELL, filled_qty="31", symbol="NKE"):
        self.symbol, self.id, self.status = symbol, "order-1", _Status.FILLED
        self.filled_qty, self.side = filled_qty, side


class _Payload:
    def __init__(self, order=None, event="fill"):
        self.order, self.event, self.price = order or _Order(), event, None


# --- the Alpaca mapping -----------------------------------------------------------
def test_the_side_survives_the_mapping():
    """The bug: TradeUpdate had no side field, so this was dropped on the floor."""
    assert to_trade_update(_Payload()).side == "sell"


def test_a_buy_side_survives_too():
    assert to_trade_update(_Payload(_Order(side=_Side.BUY))).side == "buy"


def test_an_enum_side_is_unwrapped_to_its_value():
    """The SDK hands back enums; a str() of one would record 'Side.SELL'."""
    assert to_trade_update(_Payload()).side == "sell"


def test_a_plain_string_side_is_accepted():
    """Not every payload shape wraps it, and neither should be assumed."""

    class Bare(_Order):
        def __init__(self):
            super().__init__()
            self.side = "sell"

    assert to_trade_update(_Payload(Bare())).side == "sell"


def test_a_missing_side_maps_to_none_not_to_buy():
    """None is answerable downstream; "buy" is a wrong answer that looks right."""

    class NoSide(_Order):
        def __init__(self):
            super().__init__()
            del self.side

    assert to_trade_update(_Payload(NoSide())).side is None


def test_the_cumulative_quantity_is_carried_as_a_number():
    assert to_trade_update(_Payload()).filled_qty == 31.0


# --- the engine handler -----------------------------------------------------------
def _engine(tmp_path):
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    ledger = PositionLedger(tmp_path / "ledger.jsonl")
    engine = LiveEngine(
        strategy,
        MarketDataClient(None),
        LiveTrader(RecordingBroker(), strategy, respect_market_hours=False),
        ledger=ledger,
    )
    return engine, ledger


def _update(**kwargs):
    base = dict(
        event="fill",
        symbol="NKE",
        order_id="o1",
        status="filled",
        filled_qty=31.0,
        side="sell",
        filled_avg_price=None,
        price=None,
        filled_at=None,
        fee=None,
    )
    return TradeUpdate(**{**base, **kwargs})


def test_a_short_fill_reaches_the_ledger_negative(tmp_path):
    """End to end for the reported defect: broker holds -31, ledger must agree."""
    engine, ledger = _engine(tmp_path)

    asyncio.run(engine._on_trade_update(_update()))

    assert ledger.expected_positions() == {"NKE": -31.0}


def test_a_long_fill_reaches_the_ledger_positive(tmp_path):
    engine, ledger = _engine(tmp_path)

    asyncio.run(engine._on_trade_update(_update(symbol="COP", side="buy", filled_qty=8.0)))

    assert ledger.expected_positions() == {"COP": 8.0}


def test_partial_fills_of_one_order_do_not_accumulate(tmp_path):
    """The quantity half, through the real handler: three reports of a running total
    that ends at 8 must leave 8, not 21."""
    engine, ledger = _engine(tmp_path)

    for total in (5, 8, 8):
        asyncio.run(engine._on_trade_update(_update(symbol="COP", side="buy", filled_qty=float(total))))

    assert ledger.expected_positions() == {"COP": 8.0}


def test_a_fill_with_no_side_is_refused_rather_than_guessed(tmp_path, caplog):
    """Recording it as a buy is what produced a book off by twice the position. A
    dropped record shows up as a visible divergence; a wrong one does not."""
    engine, ledger = _engine(tmp_path)

    with caplog.at_level("ERROR"):
        asyncio.run(engine._on_trade_update(_update(side=None)))

    assert ledger.expected_positions() == {}
    assert "no side" in caplog.text


@pytest.mark.parametrize("event", ["new", "canceled", "rejected", "expired", "replaced"])
def test_non_fill_events_record_nothing(tmp_path, event):
    """Only fills move a position. Counting an acknowledgement would invent one."""
    engine, ledger = _engine(tmp_path)

    asyncio.run(engine._on_trade_update(_update(event=event)))

    assert ledger.expected_positions() == {}


def test_a_partial_fill_event_is_recorded(tmp_path):
    """Both directions on the event filter — one that drops partials would under-count."""
    engine, ledger = _engine(tmp_path)

    asyncio.run(engine._on_trade_update(_update(event="partial_fill", filled_qty=11.0)))

    assert ledger.expected_positions() == {"NKE": -11.0}


def test_a_zero_quantity_fill_records_nothing(tmp_path):
    engine, ledger = _engine(tmp_path)

    asyncio.run(engine._on_trade_update(_update(filled_qty=0.0)))

    assert ledger.expected_positions() == {}


def test_bookkeeping_never_breaks_the_order_path(tmp_path):
    """A ledger that cannot be written must not raise into the trade clock."""
    engine, ledger = _engine(tmp_path)

    def explode(*args, **kwargs):
        raise OSError("disk full")

    ledger.record_fill = explode

    asyncio.run(engine._on_trade_update(_update()))  # must not raise


def test_no_ledger_is_not_an_error(tmp_path):
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    engine = LiveEngine(
        strategy,
        MarketDataClient(None),
        LiveTrader(RecordingBroker(), strategy, respect_market_hours=False),
    )

    asyncio.run(engine._on_trade_update(_update()))  # must not raise


# --- execution telemetry ----------------------------------------------------------
def test_the_fill_price_reaches_the_ledger(tmp_path):
    """TradeUpdate carried a price all along and the ledger discarded it, so the run
    could prove what filled but not what it cost."""
    engine, ledger = _engine(tmp_path)

    asyncio.run(
        engine._on_trade_update(_update(symbol="MSFT", side="buy", filled_qty=1.0, filled_avg_price=500.91))
    )

    (record,) = [r for r in ledger._read() if r["event"] == "fill"]
    assert record["fill_price"] == 500.91


def test_the_average_price_is_preferred_over_this_event_s_print(tmp_path):
    """filled_qty is cumulative, so the price beside it must be the average across the
    order — pairing a running total with one print's price misstates the cost."""
    engine, ledger = _engine(tmp_path)

    asyncio.run(
        engine._on_trade_update(_update(symbol="MSFT", filled_qty=2.0, price=499.00, filled_avg_price=500.50))
    )

    (record,) = [r for r in ledger._read() if r["event"] == "fill"]
    assert record["fill_price"] == 500.50


def test_a_venue_fee_that_is_not_reported_stays_none(tmp_path):
    """None means "not reported", which every paper fill is, and which is not zero."""
    engine, ledger = _engine(tmp_path)

    asyncio.run(engine._on_trade_update(_update(filled_qty=1.0)))

    (record,) = [r for r in ledger._read() if r["event"] == "fill"]
    assert record["broker_fee"] is None


def test_the_mapping_carries_price_time_and_fee():
    """All three were dropped by the old closure-bound mapping."""

    class Order(_Order):
        def __init__(self):
            super().__init__(side=_Side.BUY, filled_qty="2", symbol="MSFT")
            self.filled_avg_price = "500.50"

    class Payload(_Payload):
        def __init__(self):
            super().__init__(Order())
            self.timestamp = "2026-09-01T17:13:00.850000+00:00"
            self.fee = "0.35"

    update = to_trade_update(Payload())

    assert update.filled_avg_price == 500.50
    assert update.filled_at == "2026-09-01T17:13:00.850000+00:00"
    assert update.fee == 0.35


def test_a_paper_venue_reporting_no_fee_maps_to_none_not_zero(tmp_path):
    """The mapping used `safe_float`, whose default is 0.0. A paper account reports
    no fee at all, so every paper fill arrived claiming an *observed* zero — and the
    ledger it lands in is append-only, with nothing behind it to correct from."""

    class Payload(_Payload):
        def __init__(self):
            super().__init__(_Order(side=_Side.BUY, filled_qty="2", symbol="MSFT"))
            # No `fee` attribute at all: exactly what a paper stream sends.

    assert to_trade_update(Payload()).fee is None


def test_an_unreported_average_price_maps_to_none_not_zero():
    """Same shape, same field contract: None means the venue did not say."""
    assert to_trade_update(_Payload()).filled_avg_price is None


def test_an_unreported_fee_reaches_the_ledger_as_not_reported(tmp_path):
    """Both links at once, which is where the zero actually did its damage: the
    mapping turned silence into 0.0 and the ledger recorded it as an observed fee,
    which `cost_summary` then reports with `fees_reported: True`."""
    engine, ledger = _engine(tmp_path)

    asyncio.run(engine._on_trade_update(to_trade_update(_Payload())))

    (record,) = [r for r in ledger._read() if r["event"] == "fill"]
    assert record["broker_fee"] is None


# --- the book gets refreshed whether or not the ledger records -------------------
def _refresh_recorder(engine):
    refreshed = []
    engine.live_trader.refresh_position = lambda symbol: refreshed.append(symbol)
    return refreshed


def test_a_fill_with_no_side_still_refreshes_the_strategy_s_book(tmp_path):
    """The `return` that refuses to guess a side sat in the handler body, so it also
    skipped the refresh. A fill the ledger declines to record is still a fill the
    strategy's book has to learn about — and a strategy that believes it is flat in a
    symbol it holds cannot emit an exit for it."""
    engine, ledger = _engine(tmp_path)
    refreshed = _refresh_recorder(engine)

    asyncio.run(engine._on_trade_update(_update(side=None)))

    assert [r for r in ledger._read() if r["event"] == "fill"] == []
    assert refreshed == ["NKE"]


def test_a_fill_refreshes_the_book_even_with_no_ledger(tmp_path):
    """Same defect, one line earlier: the ledger is optional, the book is not."""
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    engine = LiveEngine(
        strategy,
        MarketDataClient(None),
        LiveTrader(RecordingBroker(), strategy, respect_market_hours=False),
        ledger=None,
    )
    refreshed = _refresh_recorder(engine)

    asyncio.run(engine._on_trade_update(_update()))

    assert refreshed == ["NKE"]


def test_a_failed_ledger_write_still_refreshes_the_book(tmp_path):
    """Bookkeeping never breaks the order path — and never costs the book either."""
    engine, ledger = _engine(tmp_path)
    refreshed = _refresh_recorder(engine)
    engine.ledger.record_fill = _raise

    asyncio.run(engine._on_trade_update(_update()))

    assert refreshed == ["NKE"]


def _raise(*args, **kwargs):
    raise RuntimeError("disk full")
