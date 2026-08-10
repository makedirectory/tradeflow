"""Test doubles for offline verification.

A :class:`FakeMarketData` implements the :class:`MarketDataProvider` interface
with deterministic synthetic OHLCV, so the engine/scanner/optimizer can be
exercised end-to-end without network access, API keys, or the ``alpaca`` SDK.
This is exactly what the broker abstraction buys us.
"""

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import pandas as pd

from tradeflow.brokers.base import AccountSnapshot, Broker, MarketStatus, OrderResult, OrderSide, Position
from tradeflow.marketdata.base import BarHandler, MarketDataProvider
from tradeflow.marketdata.timeframe import Timeframe
from tradeflow.strategies.base import Strategy
from tradeflow.utils.timeutils import NEW_YORK


def make_ohlcv(n: int = 600, seed: int = 0, freq: str = "5min") -> pd.DataFrame:
    """Deterministic random-walk OHLCV with periodic volume spikes."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-02 09:30", periods=n, freq=freq, tz=NEW_YORK)

    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))

    volume = rng.integers(100_000, 500_000, n).astype(float)
    volume[rng.choice(n, size=max(n // 20, 1), replace=False)] *= 5  # spikes

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


class FakeMarketData(MarketDataProvider):
    """Serves canned per-symbol frames; ignores the requested window."""

    def __init__(self, symbols: List[str], n: int = 600, freq: str = "5min"):
        self._data: Dict[str, pd.DataFrame] = {
            symbol: make_ohlcv(n=n, seed=i, freq=freq) for i, symbol in enumerate(symbols)
        }

    def get_bars(self, symbols, timeframe: Timeframe, start, end) -> Dict[str, pd.DataFrame]:
        return {s: self._data[s].copy() for s in symbols if s in self._data}

    async def stream_bars(self, symbols, handler: BarHandler) -> None:  # pragma: no cover
        raise NotImplementedError("FakeMarketData does not stream")

    def supports_streaming(self) -> bool:
        return False


class DictMarketData(MarketDataProvider):
    """Serves caller-supplied fixture frames verbatim (for precise engine tests)."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self._data = data

    def get_bars(self, symbols, timeframe: Timeframe, start, end) -> Dict[str, pd.DataFrame]:
        return {s: self._data[s].copy() for s in symbols if s in self._data}

    async def stream_bars(self, symbols, handler: BarHandler) -> None:  # pragma: no cover
        raise NotImplementedError

    def supports_streaming(self) -> bool:
        return False


class FakeBroker(Broker):
    """In-memory broker that records orders for assertions."""

    def __init__(
        self,
        buying_power: float = 100_000.0,
        positions: Optional[List[Position]] = None,
        tradable: bool = True,
        market_open: bool = True,
    ):
        self.account = AccountSnapshot(
            cash=buying_power,
            equity=buying_power,
            buying_power=buying_power,
            portfolio_value=buying_power,
        )
        self.positions: Dict[str, Position] = {p.symbol: p for p in (positions or [])}
        self.tradable = tradable
        self.market_open = market_open
        self.orders: List[dict] = []  # every submitted order, as a dict
        self.open_orders_list: List[OrderResult] = []  # still-open orders
        self.closed: List[str] = []  # symbols passed to close_position
        self.cancelled: List[str] = []  # order ids passed to cancel_order

    def get_account(self) -> Optional[AccountSnapshot]:
        return self.account

    def list_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def is_tradable(self, symbol: str) -> bool:
        return self.tradable

    def _record(self, record: dict, symbol, qty, side) -> OrderResult:
        self.orders.append(record)
        result = OrderResult(id=f"o{len(self.orders)}", symbol=symbol, side=side, qty=qty, status="accepted")
        self.open_orders_list.append(result)
        return result

    def submit_market_order(self, symbol, qty, side: OrderSide) -> Optional[OrderResult]:
        return self._record({"type": "market", "symbol": symbol, "qty": qty, "side": side}, symbol, qty, side)

    def submit_bracket_order(self, symbol, qty, side, stop_loss, take_profit) -> Optional[OrderResult]:
        return self._record(
            {
                "type": "bracket",
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
            symbol,
            qty,
            side,
        )

    def list_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        return [o for o in self.open_orders_list if symbol is None or o.symbol == symbol]

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        self.open_orders_list = [o for o in self.open_orders_list if o.id != order_id]
        return True

    def cancel_all_orders(self) -> bool:
        return True

    def close_position(self, symbol: str) -> bool:
        self.closed.append(symbol)
        self.positions.pop(symbol, None)
        return True

    def close_all_positions(self, cancel_orders: bool = True) -> bool:
        self.positions.clear()
        return True

    def get_market_status(self) -> Optional[MarketStatus]:
        t = datetime(2024, 1, 2, 12, 0)
        return MarketStatus(timestamp=t, is_open=self.market_open, next_open=t, next_close=t)


class StreamingFakeMarketData(FakeMarketData):
    """FakeMarketData that emits a few live bars then ends (for engine tests)."""

    def __init__(self, symbols, bars_to_emit: int = 1, **kwargs):
        super().__init__(symbols, **kwargs)
        self._symbols = symbols
        self._bars_to_emit = bars_to_emit

    def supports_streaming(self) -> bool:
        return True

    async def stream_bars(self, symbols, handler):
        import asyncio

        from tradeflow.marketdata.base import BarEvent

        for i in range(self._bars_to_emit):
            for symbol in symbols:
                event = BarEvent(
                    symbol=symbol,
                    timestamp=datetime(2024, 1, 2, 10, i % 60),
                    open=100,
                    high=101,
                    low=99,
                    close=100,
                    volume=1000,
                )
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result


class TradeUpdateFakeBroker(FakeBroker):
    """FakeBroker that supports trade-update streaming and replays canned updates."""

    def __init__(self, updates=(), **kwargs):
        super().__init__(**kwargs)
        self._updates = list(updates)

    def supports_trade_updates(self) -> bool:
        return True

    async def stream_trade_updates(self, handler):
        import asyncio

        for update in self._updates:
            result = handler(update)
            if asyncio.iscoroutine(result):
                await result


class ScriptedFeed(FakeMarketData):
    """Emits an exact, caller-supplied sequence of bar events.

    The adversarial-feed harness. :class:`StreamingFakeMarketData` emits well-formed
    bars, which is the case that was never in doubt; this one emits whatever the test
    scripts — duplicates, inverted ranges, stale timestamps, out-of-order arrivals —
    so the live loop's behavior under a misbehaving vendor is asserted rather than
    assumed.

    It fakes only the edge (data in). The engine, the strategy, and the trader under
    test are the real ones, so the fence cannot pass by encoding the same assumptions
    the code makes.
    """

    def __init__(self, symbols, events=(), *, raise_after: Optional[int] = None, **kwargs):
        super().__init__(symbols, **kwargs)
        self._events = list(events)
        #: Simulates a dropped connection mid-stream, after this many events.
        self._raise_after = raise_after
        self.delivered = 0

    def supports_streaming(self) -> bool:
        return True

    async def stream_bars(self, symbols, handler):
        import asyncio

        for i, event in enumerate(self._events):
            if self._raise_after is not None and i == self._raise_after:
                raise ConnectionError("stream dropped")
            result = handler(event)
            self.delivered += 1
            if asyncio.iscoroutine(result):
                await result


def bar_event(
    symbol="AAA",
    *,
    minute=0,
    close=100.0,
    open=None,
    high=None,
    low=None,
    volume=1000.0,
    day=2,
):
    """One well-formed BarEvent, with any field overridable to make it malformed."""
    from tradeflow.marketdata.base import BarEvent

    return BarEvent(
        symbol=symbol,
        timestamp=datetime(2024, 1, day, 10, minute),
        open=close if open is None else open,
        high=(close + 1) if high is None else high,
        low=(close - 1) if low is None else low,
        close=close,
        volume=volume,
    )


class RecordingBroker(FakeBroker):
    """A broker that records every call, and can be told to misbehave.

    ``reject_orders`` makes submission return ``None`` (the broker refusing), which
    is the case the loop has to survive without losing its place in the stream.
    """

    def __init__(self, *, reject_orders: bool = False, positions=None, **kwargs):
        super().__init__(**kwargs)
        self.reject_orders = reject_orders
        self.calls: List[str] = []
        self._forced_positions = positions

    def submit_bracket_order(self, symbol, qty, side, stop_loss, take_profit):
        self.calls.append(f"bracket:{symbol}")
        if self.reject_orders:
            return None
        return super().submit_bracket_order(symbol, qty, side, stop_loss, take_profit)

    def submit_market_order(self, symbol, qty, side):
        self.calls.append(f"market:{symbol}")
        if self.reject_orders:
            return None
        return super().submit_market_order(symbol, qty, side)

    def list_positions(self):
        self.calls.append("list_positions")
        if self._forced_positions is not None:
            return self._forced_positions
        return super().list_positions()


class FakeTradeUpdate:
    """A broker trade update, shaped like the live path expects one."""

    def __init__(
        self, event="fill", symbol="AAA", order_id="o-1", status="filled", filled_qty=10.0, side="buy"
    ):
        self.event = event
        self.symbol = symbol
        self.order_id = order_id
        self.status = status
        self.filled_qty = filled_qty
        self.side = side


class ScriptedStrategy(Strategy):
    """A strategy whose conviction is a plain function of price.

    Score is ``close - pivot``, so a test drives it long or flat by choosing closes
    either side of the pivot — no warm-up period to wait out and no indicator math
    to reason about.

    Only the indicator layer is faked. Hysteresis, signal derivation, position
    validation and the real-time buffer all come from the real :class:`Strategy`
    base class, which is where the behavior under test actually lives.
    """

    TIMEFRAME = "1Day"

    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "pivot": {"type": "float", "min": 1.0, "max": 1e6, "step": 1.0, "default": 100.0},
        "risk_per_trade": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01, "default": 0.02},
        "stop_loss": {"type": "float", "min": 0.01, "max": 0.08, "step": 0.01, "default": 0.03},
        "take_profit": {"type": "float", "min": 0.02, "max": 0.15, "step": 0.01, "default": 0.06},
    }

    def __init__(self, config: Dict[str, Any]):
        config["timeframe"] = self.TIMEFRAME
        config.setdefault(
            "position_limits",
            {"max_positions": 1, "max_position_size": 100_000.0, "max_total_risk": 0.05},
        )
        super().__init__(config)

    def calculate_required_lookback(self) -> int:
        # One bar, so every scripted bar is actually evaluated. A larger lookback
        # would swallow the first bars before the strategy ever sees them, which is
        # realistic but makes a test about exits depend on warm-up arithmetic.
        return 1

    def initialize(self) -> None:
        return None

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].astype(float) - self.config["pivot"]
