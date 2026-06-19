"""Broker abstraction.

Everything above this layer (execution, engine, scanners, strategies, main)
depends only on the :class:`Broker` interface and the broker-agnostic domain
types defined here - never on a specific vendor SDK. To support a new venue,
implement :class:`Broker` (see :mod:`src.brokers.alpaca.broker`) and inject it;
nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


class OrderSide(str, Enum):
    """Side of an order."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class AccountSnapshot:
    """A point-in-time view of the trading account."""

    cash: float
    equity: float
    buying_power: float
    portfolio_value: float
    trading_blocked: bool = False


@dataclass
class Position:
    """An open position, normalised across brokers.

    ``side`` is ``"long"`` or ``"short"``; ``qty`` is always non-negative.
    """

    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float

    @property
    def is_long(self) -> bool:
        return self.side == "long"


@dataclass
class OrderResult:
    """The outcome of submitting an order."""

    id: str
    symbol: str
    side: OrderSide
    qty: float
    status: str
    raw: object = None  # underlying SDK object, if a caller needs broker specifics


@dataclass
class MarketStatus:
    """Current market clock."""

    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


class Broker(ABC):
    """Interface every brokerage adapter must implement.

    Kept intentionally small: account/position reads, the order types the engine
    actually uses, and order/position lifecycle management.
    """

    # --- account & positions -------------------------------------------------
    @abstractmethod
    def get_account(self) -> Optional[AccountSnapshot]:
        """Return the current account snapshot, or None on failure."""

    @abstractmethod
    def list_positions(self) -> List[Position]:
        """Return all open positions (empty list if none)."""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for ``symbol``, or None if flat."""

    @abstractmethod
    def is_tradable(self, symbol: str) -> bool:
        """Whether ``symbol`` can currently be traded on this venue."""

    # --- orders --------------------------------------------------------------
    @abstractmethod
    def submit_market_order(self, symbol: str, qty: float, side: OrderSide) -> Optional[OrderResult]:
        """Submit a market order."""

    @abstractmethod
    def submit_bracket_order(
        self, symbol: str, qty: float, side: OrderSide, stop_loss: float, take_profit: float
    ) -> Optional[OrderResult]:
        """Submit a bracket order (entry + stop-loss + take-profit)."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order. Returns True on success."""

    @abstractmethod
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders. Returns True on success."""

    # --- position lifecycle --------------------------------------------------
    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """Liquidate the position in ``symbol``. Returns True on success."""

    @abstractmethod
    def close_all_positions(self, cancel_orders: bool = True) -> bool:
        """Liquidate all positions, optionally cancelling orders first."""

    # --- market clock --------------------------------------------------------
    @abstractmethod
    def get_market_status(self) -> Optional[MarketStatus]:
        """Return the current market clock, or None on failure."""
