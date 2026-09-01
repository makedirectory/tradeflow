---
sidebar_position: 3
title: Broker abstraction
---

# Broker abstraction

The single most important design decision: **the system depends on interfaces,
not on Alpaca.** Trading and market data are two small abstract contracts, and
Alpaca is one implementation of each.

## The two interfaces

### `Broker` (`tradeflow/brokers/base.py`)

Account reads, the order types the engine uses, and position/order lifecycle:

```python
class Broker(ABC):
    def get_account(self) -> Optional[AccountSnapshot]: ...
    def list_positions(self) -> List[Position]: ...
    def get_position(self, symbol) -> Optional[Position]: ...
    def is_tradable(self, symbol) -> bool: ...
    def submit_market_order(self, symbol, qty, side, client_order_id=None) -> Optional[OrderResult]: ...
    def submit_bracket_order(self, symbol, qty, side, stop_loss, take_profit,
                             client_order_id=None) -> Optional[OrderResult]: ...
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
order is already pending, and canceling resting bracket legs before a
discretionary close (no orphaned stop/take orders). `stream_trade_updates` is an
*optional* capability — brokers that can't stream account events simply return
`False` from `supports_trade_updates`.

### `client_order_id` — idempotency that survives a restart

Asking the broker "are there open orders for this symbol?" immediately before
placing one is a check-then-act race, and it has no memory: a process that
restarts between submitting an order and seeing its fill asks again, gets an
answer that no longer reflects what it did, and submits the same order twice.

So every order carries an id derived from the decision behind it — strategy,
its parameters, symbol, signal, and bar timestamp — and a venue that has already
accepted that id must reject the duplicate. Idempotency becomes a property of
the request rather than of how carefully the caller looked first. The pending-
order check remains, but only as a cheap way to avoid a pointless round trip.

The same bar redelivered after a reconnect hashes to the same id and is refused;
the same symbol on the next bar hashes differently and trades normally. See
`tradeflow/execution/order_id.py`.

### `MarketDataProvider` (`tradeflow/marketdata/base.py`)

```python
class MarketDataProvider(ABC):
    def get_bars(self, symbols, timeframe: Timeframe, start, end) -> Dict[str, pd.DataFrame]: ...
    async def stream_bars(self, symbols, handler: BarHandler) -> None: ...
    def supports_streaming(self) -> bool: ...
```

## Typed failure (`tradeflow/brokers/errors.py`)

Every broker call used to fail the same way — `None`, or `False`. A rate limit, an
expired token, insufficient buying power, a closed market, and an order the venue
deliberately refused all arrived as the same absence of information, so the only
possible response was the same one. That is the wrong response to most of them, and
for a duplicate order it is actively misleading: the order *was* placed.

Anything that moves money or reports the account's actual state now raises a
`BrokerError` subclass, chosen by what a caller would do differently:

| Error | What it means | Correct response |
|---|---|---|
| `RateLimitedError` | Throttled; the request was fine | Back off |
| `AuthenticationError` | Credentials rejected | Stop; never transient |
| `InsufficientFundsError` | Account can't support this size | Size down |
| `MarketClosedError` | Right request, wrong time | Wait |
| `DuplicateOrderError` | The venue already has this order | Nothing — it succeeded |
| `OrderRejectedError` | Refused on the venue's own terms | Inspect |
| `BrokerUnavailableError` | Couldn't reach the venue at all | Retry |
| `NotTradableError` | Symbol can't be traded now | Skip the symbol |

Methods that answer a genuine yes/no question still return one: `get_position`
returning `None` means *flat*, which is an answer rather than a failure.
`list_positions` is the opposite case and raises — an empty list is the factual
claim that the account holds nothing, and the strategy's position book is rebuilt
from it, so turning an unreachable broker into "you are flat" would silently make
every real position un-exitable.

Two consequences worth stating outright. A `DuplicateOrderError` is not a failed
submission, so the trader logs it and leaves the existing order alone rather than
resubmitting. And an unreadable market clock no longer uniformly means "assume
open": that is right for a transient blip, but revoked credentials are never
transient, so authentication failures fail closed.

## Vendor-neutral domain types

Callers never see Alpaca objects. The broker layer defines plain dataclasses /
enums — `OrderSide`, `AccountSnapshot`, `Position` (`side` is `"long"`/`"short"`,
`qty` non-negative), `OrderResult`, `MarketStatus`, `BarEvent`, and `TradeUpdate`. The
Alpaca adapter maps SDK objects to these.

`TradeUpdate` carries `side`, `filled_qty`, `filled_avg_price`, `filled_at` and `fee`,
and two of those fields need care from any adapter:

- **`filled_qty` is cumulative**, not this event's increment — a venue re-reports the
  order's running total on every partial fill and again on the final one. Summing those
  events counts the same shares repeatedly. `filled_avg_price` is the price that pairs
  with it; `price` is this event's own print.
- **`side` is never defaulted.** An update that arrives without one is dropped and
  logged rather than guessed, because assuming `buy` records a short as a long and puts
  the ledger out by twice the position. A dropped record shows up as a visible
  reconciliation divergence; a wrongly-signed one does not.
- **`fee` is `None` when the venue does not report one**, which paper accounts never do.
  That is not the same as zero and must not be averaged as though it were.

## The Alpaca adapter

`tradeflow/brokers/alpaca/` is the **only** place `import alpaca` appears
(`AlpacaBroker`, `AlpacaMarketData`, and `factory.py`, whose `build_broker` /
`build_market_data` construct them from credentials — entry points call the
factories and never touch SDK clients). `AlpacaMarketData` also converts the
project's `Timeframe` into Alpaca's `TimeFrame` and normalizes bars into
per-symbol, New-York-localized OHLCV frames.

Two adapter details worth knowing before writing another one:

**The SDK's two halves can default to different data feeds.** Alpaca's historical
requests resolve to the full consolidated tape while its stream defaults to a single
venue, so an account entitled to one and not the other warms up on nothing and streams
normally — which reads as an empty market rather than a wrong feed. `build_market_data`
takes a `feed` that pins both halves, unset by default so an entitled account is never
silently moved to a partial or delayed source.

**A synchronous `stop()` must not be called from inside the event loop.** Alpaca's
submits a coroutine back to the running loop and blocks the caller waiting for it, which
from inside that loop can never complete. The shared `close_stream` helper awaits the
coroutine that wrapper wraps instead.

## Dropping in another broker

1. Implement `Broker` (and optionally `MarketDataProvider`) for the new venue.
2. Construct it in `main.build_data_and_broker()` instead of the Alpaca classes.

That's it — `engine/`, `execution/`, `strategies/`, `scanners/`, and the optimizer
are untouched because they only ever knew the interface. The test suite proves
this: it runs the entire stack against an in-memory `FakeBroker` /
`FakeMarketData` with no Alpaca involved at all.
