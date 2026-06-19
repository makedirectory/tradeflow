"""Alpaca implementation of the :class:`Broker` interface.

This is the single place where ``alpaca-py`` trading types are imported. It maps
Alpaca SDK objects to the broker-agnostic domain types in
:mod:`src.brokers.base` so the rest of the system stays vendor-neutral.
"""

import logging
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderType, TimeInForce
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.requests import (
    MarketOrderRequest,
    OrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from src.brokers.base import (
    AccountSnapshot,
    Broker,
    MarketStatus,
    OrderResult,
    OrderSide,
    Position,
)
from src.utils.numeric import safe_float

logger = logging.getLogger(__name__)

# Map our side enum onto Alpaca's.
_SIDE_TO_ALPACA = {OrderSide.BUY: AlpacaOrderSide.BUY, OrderSide.SELL: AlpacaOrderSide.SELL}


class AlpacaBroker(Broker):
    """Trade and account operations backed by an Alpaca ``TradingClient``."""

    def __init__(self, trading_client: TradingClient):
        self._client = trading_client
        self._tradable_cache: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Account & positions
    # ------------------------------------------------------------------ #
    def get_account(self) -> Optional[AccountSnapshot]:
        try:
            account = self._client.get_account()
            return AccountSnapshot(
                cash=safe_float(account.cash),
                equity=safe_float(account.equity),
                buying_power=safe_float(account.buying_power),
                portfolio_value=safe_float(account.portfolio_value),
                trading_blocked=bool(account.trading_blocked),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch account: %s", exc)
            return None

    def list_positions(self) -> List[Position]:
        try:
            return [self._to_position(p) for p in self._client.get_all_positions()]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to list positions: %s", exc)
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            return self._to_position(self._client.get_open_position(symbol))
        except Exception:  # noqa: BLE001 - "no position" is an expected 404
            return None

    def is_tradable(self, symbol: str) -> bool:
        if symbol not in self._tradable_cache:
            try:
                self._tradable_cache[symbol] = bool(self._client.get_asset(symbol).tradable)
            except Exception as exc:  # noqa: BLE001
                logger.info("Could not determine tradability for %s: %s", symbol, exc)
                self._tradable_cache[symbol] = False
        return self._tradable_cache[symbol]

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #
    def submit_market_order(self, symbol: str, qty: float, side: OrderSide) -> Optional[OrderResult]:
        try:
            order = self._client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=_SIDE_TO_ALPACA[side],
                    time_in_force=TimeInForce.GTC,
                )
            )
            return self._to_order_result(order, side)
        except Exception as exc:  # noqa: BLE001
            logger.error("Market order failed for %s: %s", symbol, exc)
            return None

    def submit_bracket_order(
        self, symbol: str, qty: float, side: OrderSide, stop_loss: float, take_profit: float
    ) -> Optional[OrderResult]:
        try:
            order = self._client.submit_order(
                OrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=_SIDE_TO_ALPACA[side],
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=take_profit),
                    stop_loss=StopLossRequest(stop_price=stop_loss),
                )
            )
            return self._to_order_result(order, side)
        except Exception as exc:  # noqa: BLE001
            logger.error("Bracket order failed for %s: %s", symbol, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    def cancel_all_orders(self) -> bool:
        try:
            self._client.cancel_orders()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to cancel all orders: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Position lifecycle
    # ------------------------------------------------------------------ #
    def close_position(self, symbol: str) -> bool:
        try:
            self._client.close_position(symbol)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to close position %s: %s", symbol, exc)
            return False

    def close_all_positions(self, cancel_orders: bool = True) -> bool:
        try:
            self._client.close_all_positions(cancel_orders=cancel_orders)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to close all positions: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Market clock
    # ------------------------------------------------------------------ #
    def get_market_status(self) -> Optional[MarketStatus]:
        try:
            clock = self._client.get_clock()
            return MarketStatus(
                timestamp=clock.timestamp,
                is_open=clock.is_open,
                next_open=clock.next_open,
                next_close=clock.next_close,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch market clock: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Mapping helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_position(pos) -> Position:
        return Position(
            symbol=pos.symbol,
            qty=abs(safe_float(pos.qty)),
            side=pos.side.value,  # "long" / "short"
            avg_entry_price=safe_float(pos.avg_entry_price),
            current_price=safe_float(pos.current_price),
            market_value=safe_float(pos.market_value),
            unrealized_pl=safe_float(pos.unrealized_pl),
        )

    @staticmethod
    def _to_order_result(order, side: OrderSide) -> OrderResult:
        return OrderResult(
            id=str(order.id),
            symbol=order.symbol,
            side=side,
            qty=safe_float(order.qty),
            status=str(getattr(order.status, "value", order.status)),
            raw=order,
        )
