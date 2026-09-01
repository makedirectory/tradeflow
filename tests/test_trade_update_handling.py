"""From a broker event to a ledger record — the path that lost the side.

Two links, both previously untestable. The Alpaca mapping lived in a stream callback
reachable only through a socket, and the engine handler read a field the TradeUpdate
type did not have, so `getattr(update, "side", "buy")` resolved to buy forever.
"""

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
    base = dict(event="fill", symbol="NKE", order_id="o1", status="filled", filled_qty=31.0, side="sell")
    return TradeUpdate(**{**base, **kwargs})


def test_a_short_fill_reaches_the_ledger_negative(tmp_path):
    """End to end for the reported defect: broker holds -31, ledger must agree."""
    engine, ledger = _engine(tmp_path)

    engine._on_trade_update(_update())

    assert ledger.expected_positions() == {"NKE": -31.0}


def test_a_long_fill_reaches_the_ledger_positive(tmp_path):
    engine, ledger = _engine(tmp_path)

    engine._on_trade_update(_update(symbol="COP", side="buy", filled_qty=8.0))

    assert ledger.expected_positions() == {"COP": 8.0}


def test_partial_fills_of_one_order_do_not_accumulate(tmp_path):
    """The quantity half, through the real handler: three reports of a running total
    that ends at 8 must leave 8, not 21."""
    engine, ledger = _engine(tmp_path)

    for total in (5, 8, 8):
        engine._on_trade_update(_update(symbol="COP", side="buy", filled_qty=float(total)))

    assert ledger.expected_positions() == {"COP": 8.0}


def test_a_fill_with_no_side_is_refused_rather_than_guessed(tmp_path, caplog):
    """Recording it as a buy is what produced a book off by twice the position. A
    dropped record shows up as a visible divergence; a wrong one does not."""
    engine, ledger = _engine(tmp_path)

    with caplog.at_level("ERROR"):
        engine._on_trade_update(_update(side=None))

    assert ledger.expected_positions() == {}
    assert "no side" in caplog.text


@pytest.mark.parametrize("event", ["new", "canceled", "rejected", "expired", "replaced"])
def test_non_fill_events_record_nothing(tmp_path, event):
    """Only fills move a position. Counting an acknowledgement would invent one."""
    engine, ledger = _engine(tmp_path)

    engine._on_trade_update(_update(event=event))

    assert ledger.expected_positions() == {}


def test_a_partial_fill_event_is_recorded(tmp_path):
    """Both directions on the event filter — one that drops partials would under-count."""
    engine, ledger = _engine(tmp_path)

    engine._on_trade_update(_update(event="partial_fill", filled_qty=11.0))

    assert ledger.expected_positions() == {"NKE": -11.0}


def test_a_zero_quantity_fill_records_nothing(tmp_path):
    engine, ledger = _engine(tmp_path)

    engine._on_trade_update(_update(filled_qty=0.0))

    assert ledger.expected_positions() == {}


def test_bookkeeping_never_breaks_the_order_path(tmp_path):
    """A ledger that cannot be written must not raise into the trade clock."""
    engine, ledger = _engine(tmp_path)

    def explode(*args, **kwargs):
        raise OSError("disk full")

    ledger.record_fill = explode

    engine._on_trade_update(_update())  # must not raise


def test_no_ledger_is_not_an_error(tmp_path):
    strategy = STRATEGIES["ma_crossover"].create_with_defaults()
    engine = LiveEngine(
        strategy,
        MarketDataClient(None),
        LiveTrader(RecordingBroker(), strategy, respect_market_hours=False),
    )

    engine._on_trade_update(_update())  # must not raise
