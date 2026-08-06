"""Alpaca implementation of the :class:`Broker` interface.

Along with :mod:`src.brokers.alpaca.factory`, this is the only place where
``alpaca-py`` trading types are imported. It maps
Alpaca SDK objects to the broker-agnostic domain types in
:mod:`src.brokers.base` so the rest of the system stays vendor-neutral.
"""

import inspect
import logging
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.requests import (
    GetOrdersRequest,
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
    TradeUpdate,
    TradeUpdateHandler,
)
from src.utils.numeric import safe_float
from src.utils.streaming import run_with_reconnect

logger = logging.getLogger(__name__)

# Map our side enum onto Alpaca's.
_SIDE_TO_ALPACA = {OrderSide.BUY: AlpacaOrderSide.BUY, OrderSide.SELL: AlpacaOrderSide.SELL}


class AlpacaBroker(Broker):
    """Trade and account operations backed by an Alpaca ``TradingClient``."""

    def __init__(
        self,
        trading_client: TradingClient,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ):
        self._client = trading_client
        self._tradable_cache: dict[str, bool] = {}
        # Credentials are only needed for the trade-update WebSocket.
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = paper

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

    def list_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[symbol] if symbol else None,
            )
            orders = self._client.get_orders(filter=request)
            return [
                OrderResult(
                    id=str(o.id),
                    symbol=o.symbol,
                    side=OrderSide(o.side.value),
                    qty=safe_float(o.qty),
                    status=str(getattr(o.status, "value", o.status)),
                    raw=o,
                )
                for o in orders
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to list open orders: %s", exc)
            return []

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
    # Trade-update streaming
    # ------------------------------------------------------------------ #
    def supports_trade_updates(self) -> bool:
        return bool(self._api_key and self._api_secret)

    async def stream_trade_updates(self, handler: TradeUpdateHandler) -> None:
        from alpaca.trading.stream import TradingStream

        async def on_alpaca_update(data) -> None:
            order = data.order
            update = TradeUpdate(
                event=str(getattr(data, "event", "")),
                symbol=getattr(order, "symbol", ""),
                order_id=str(getattr(order, "id", "")),
                status=str(getattr(getattr(order, "status", ""), "value", getattr(order, "status", ""))),
                filled_qty=safe_float(getattr(order, "filled_qty", 0)),
                price=safe_float(getattr(data, "price", None)) if getattr(data, "price", None) else None,
            )
            result = handler(update)
            if inspect.isawaitable(result):
                await result

        async def connect() -> None:
            stream = TradingStream(self._api_key, self._api_secret, paper=self._paper)
            try:
                stream.subscribe_trade_updates(on_alpaca_update)
                logger.info("Subscribed to trade updates")
                await stream._run_forever()
            finally:
                try:
                    stop = stream.stop()
                    if inspect.isawaitable(stop):
                        await stop
                except Exception:  # noqa: BLE001
                    pass

        await run_with_reconnect("trade-updates", connect)

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
