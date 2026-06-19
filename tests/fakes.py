"""Test doubles for offline verification.

A :class:`FakeMarketData` implements the :class:`MarketDataProvider` interface
with deterministic synthetic OHLCV, so the engine/scanner/optimizer can be
exercised end-to-end without network access, API keys, or the ``alpaca`` SDK.
This is exactly what the broker abstraction buys us.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.brokers.base import AccountSnapshot, Broker, MarketStatus, OrderResult, OrderSide, Position
from src.marketdata.base import BarHandler, MarketDataProvider
from src.marketdata.timeframe import Timeframe
from src.utils.timeutils import NEW_YORK


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

    def __init__(self, buying_power: float = 100_000.0, positions: Optional[List[Position]] = None,
                 tradable: bool = True):
        self.account = AccountSnapshot(
            cash=buying_power, equity=buying_power, buying_power=buying_power,
            portfolio_value=buying_power,
        )
        self.positions: Dict[str, Position] = {p.symbol: p for p in (positions or [])}
        self.tradable = tradable
        self.orders: List[dict] = []     # every submitted order, as a dict
        self.closed: List[str] = []      # symbols passed to close_position
        self.cancelled: List[str] = []   # order ids passed to cancel_order

    def get_account(self) -> Optional[AccountSnapshot]:
        return self.account

    def list_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def is_tradable(self, symbol: str) -> bool:
        return self.tradable

    def submit_market_order(self, symbol, qty, side: OrderSide) -> Optional[OrderResult]:
        self.orders.append({"type": "market", "symbol": symbol, "qty": qty, "side": side})
        return OrderResult(id=f"o{len(self.orders)}", symbol=symbol, side=side, qty=qty, status="accepted")

    def submit_bracket_order(self, symbol, qty, side, stop_loss, take_profit) -> Optional[OrderResult]:
        self.orders.append({"type": "bracket", "symbol": symbol, "qty": qty, "side": side,
                            "stop_loss": stop_loss, "take_profit": take_profit})
        return OrderResult(id=f"o{len(self.orders)}", symbol=symbol, side=side, qty=qty, status="accepted")

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
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
        return None
