---
sidebar_position: 3
title: Broker abstraction
---

# Broker abstraction

The single most important design decision: **the system depends on interfaces,
not on Alpaca.** Trading and market data are two small abstract contracts, and
Alpaca is one implementation of each.

## The two interfaces

### `Broker` (`src/brokers/base.py`)

Account reads, the order types the engine uses, and position/order lifecycle:

```python
class Broker(ABC):
    def get_account(self) -> Optional[AccountSnapshot]: ...
    def list_positions(self) -> List[Position]: ...
    def get_position(self, symbol) -> Optional[Position]: ...
    def is_tradable(self, symbol) -> bool: ...
    def submit_market_order(self, symbol, qty, side: OrderSide) -> Optional[OrderResult]: ...
    def submit_bracket_order(self, symbol, qty, side, stop_loss, take_profit) -> Optional[OrderResult]: ...
    def list_open_orders(self, symbol=None) -> List[OrderResult]: ...
    def cancel_order(self, order_id) -> bool: ...
    def cancel_all_orders(self) -> bool: ...
    def close_position(self, symbol) -> bool: ...
    def close_all_positions(self, cancel_orders=True) -> bool: ...
    def get_market_status(self) -> Optional[MarketStatus]: ...

    # Optional capability (default: unsupported)
    def supports_trade_updates(self) -> bool: ...
    async def stream_trade_updates(self, handler) -> None: ...
```

`list_open_orders` backs two live-trading safeguards: skipping an entry when an
order is already pending (no double-submits between placement and fill) and
canceling resting bracket legs before a discretionary close (no orphaned
stop/take orders). `stream_trade_updates` is an *optional* capability — brokers
that can't stream account events simply return `False` from
`supports_trade_updates`.

### `MarketDataProvider` (`src/marketdata/base.py`)

```python
class MarketDataProvider(ABC):
    def get_bars(self, symbols, timeframe: Timeframe, start, end) -> Dict[str, pd.DataFrame]: ...
    async def stream_bars(self, symbols, handler: BarHandler) -> None: ...
    def supports_streaming(self) -> bool: ...
```

## Vendor-neutral domain types

Callers never see Alpaca objects. The broker layer defines plain dataclasses /
enums — `OrderSide`, `AccountSnapshot`, `Position` (`side` is `"long"`/`"short"`,
`qty` non-negative), `OrderResult`, `MarketStatus`, and `BarEvent`. The Alpaca
adapter maps SDK objects to these.

## The Alpaca adapter

`src/brokers/alpaca/` is the **only** place `import alpaca` appears
(`AlpacaBroker`, `AlpacaMarketData`, and `factory.py`, whose `build_broker` /
`build_market_data` construct them from credentials — entry points call the
factories and never touch SDK clients). `AlpacaMarketData` also converts the
project's `Timeframe` into Alpaca's `TimeFrame` and normalizes bars into
per-symbol, New-York-localized OHLCV frames.

## Dropping in another broker

1. Implement `Broker` (and optionally `MarketDataProvider`) for the new venue.
2. Construct it in `main.build_data_and_broker()` instead of the Alpaca classes.

That's it — `engine/`, `execution/`, `strategies/`, `scanners/`, and the optimizer
are untouched because they only ever knew the interface. The test suite proves
this: it runs the entire stack against an in-memory `FakeBroker` /
`FakeMarketData` with no Alpaca involved at all.
