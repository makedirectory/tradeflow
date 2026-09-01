"""Broker abstraction.

Everything above this layer (execution, engine, scanners, strategies, main)
depends only on the :class:`Broker` interface and the broker-agnostic domain
types defined here - never on a specific vendor SDK. To support a new venue,
implement :class:`Broker` (see :mod:`tradeflow.brokers.alpaca.broker`) and inject it;
nothing else changes.

**Failure is typed, not erased.** Anything that moves money or reports the account's
actual state raises a :class:`~tradeflow.brokers.errors.BrokerError` subclass, because
the caller's correct response differs by cause: back off, stop and get a human, size
down, or carry on. The methods that answer a genuine yes/no question still return one
- :meth:`Broker.get_position` returning ``None`` means *flat*, which is an answer
rather than a failure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, List, Optional, Union


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
    """An open position, normalized across brokers.

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


@dataclass
class TradeUpdate:
    """A streamed account/order event (fill, new, canceled, rejected, ...)."""

    event: str
    symbol: str
    order_id: str
    status: str
    #: Cumulative filled quantity for the order, not this event's increment. Alpaca
    #: re-reports the running total on every partial fill and again on the final fill.
    filled_qty: float = 0.0
    price: Optional[float] = None
    #: "buy" / "sell". Without it a short fill is indistinguishable from a long one,
    #: and anything defaulting the side records the whole book long.
    side: Optional[str] = None
    #: Average price across everything filled on this order so far. Pairs with
    #: ``filled_qty``, which is also cumulative; ``price`` is this event's own print.
    filled_avg_price: Optional[float] = None
    #: When the venue reported the fill, as an ISO timestamp. Distinct from when we
    #: recorded it — the gap between the two is latency we would otherwise attribute
    #: to the venue.
    filled_at: Optional[str] = None
    #: What the venue charged, when it says. ``None`` means "not reported", which a
    #: paper account always does, and which is not the same as zero.
    fee: Optional[float] = None


#: A trade-update handler may be sync or async.
TradeUpdateHandler = Callable[[TradeUpdate], Union[None, Awaitable[None]]]


class Broker(ABC):
    """Interface every brokerage adapter must implement.

    Kept intentionally small: account/position reads, the order types the engine
    actually uses, and order/position lifecycle management.
    """

    # --- account & positions -------------------------------------------------
    @abstractmethod
    def get_account(self) -> Optional[AccountSnapshot]:
        """Return the current account snapshot.

        Raises a :class:`~tradeflow.brokers.errors.BrokerError` if the account cannot
        be read.
        """

    @abstractmethod
    def list_positions(self) -> List[Position]:
        """Return all open positions; an empty list means the account is flat.

        Raises a :class:`~tradeflow.brokers.errors.BrokerError` rather than returning
        an empty list when positions cannot be read - "flat" and "unknown" are
        different answers, and callers rebuild state from this one.
        """

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for ``symbol``, or None if flat."""

    @abstractmethod
    def is_tradable(self, symbol: str) -> bool:
        """Whether ``symbol`` can currently be traded on this venue."""

    # --- orders --------------------------------------------------------------
    @abstractmethod
    def submit_market_order(
        self, symbol: str, qty: float, side: OrderSide, client_order_id: Optional[str] = None
    ) -> Optional[OrderResult]:
        """Submit a market order.

        ``client_order_id``, when given, is the caller's own identity for this order.
        A venue that has already accepted it must reject the duplicate rather than
        place a second order - that rejection is the idempotency guarantee, and it is
        the only one that survives the submitting process restarting.
        """

    @abstractmethod
    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        stop_loss: float,
        take_profit: float,
        client_order_id: Optional[str] = None,
    ) -> Optional[OrderResult]:
        """Submit a bracket order (entry + stop-loss + take-profit).

        ``client_order_id`` carries the same meaning as in :meth:`submit_market_order`.
        """

    @abstractmethod
    def list_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Return open (unfilled) orders, optionally filtered to ``symbol``."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order. Raises a ``BrokerError`` if the venue refuses."""

    @abstractmethod
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders. Raises a ``BrokerError`` if the venue refuses."""

    # --- position lifecycle --------------------------------------------------
    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """Liquidate the position in ``symbol``. Raises a ``BrokerError`` on failure."""

    @abstractmethod
    def close_all_positions(self, cancel_orders: bool = True) -> bool:
        """Liquidate all positions, optionally canceling orders first.

        Raises a ``BrokerError`` on failure - a flatten that quietly did nothing is
        the worst possible outcome for this call.
        """

    # --- market clock --------------------------------------------------------
    @abstractmethod
    def get_market_status(self) -> Optional[MarketStatus]:
        """Return the current market clock.

        Raises a :class:`~tradeflow.brokers.errors.BrokerError` if the clock cannot be
        read, so a caller can tell "closed" apart from "could not tell".
        """

    # --- trade-update streaming (optional capability) ------------------------
    def supports_trade_updates(self) -> bool:
        """Whether this broker can stream account/order updates. Override if so."""
        return False

    async def stream_trade_updates(self, handler: TradeUpdateHandler) -> None:
        """Stream account/order updates to ``handler`` until canceled.

        Optional: only meaningful when :meth:`supports_trade_updates` is True.
        """
        raise NotImplementedError("This broker does not support trade-update streaming")
