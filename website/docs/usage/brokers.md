---
sidebar_position: 8
title: Brokers
---

# Brokers

The system is **broker-agnostic**: the engine, strategies, scanners, execution,
and optimizer all depend on two interfaces — `Broker` (trading) and
`MarketDataProvider` (data) — never on a specific vendor. A broker is just an
adapter that implements those. Today **Alpaca** is the one bundled adapter; others
can be added without touching anything above the broker layer.

## Supported

### Alpaca (US equities)

| Capability | Status |
|---|---|
| Historical bars (`get_bars`) | ✅ |
| Live bar stream (`stream_bars`) | ✅ minute bars |
| Trade-update / fill stream (`stream_trade_updates`) | ✅ |
| Account & positions | ✅ |
| Market & bracket orders | ✅ |
| Open-order listing / cancel | ✅ |
| Close position(s) | ✅ |
| Market clock (hours gating) | ✅ |
| Paper **and** live | ✅ (paper by default) |

Implemented in `src/brokers/alpaca/` — `AlpacaBroker` (trading) and
`AlpacaMarketData` (data). This is the **only** place the `alpaca` SDK is imported.

## Not yet supported

These aren't built, but the design has a clear seam for each:

- **Crypto / other Alpaca asset classes** — `AlpacaBroker`/`AlpacaMarketData` are
  equity-focused (`StockHistoricalDataClient` / `StockDataStream`). A crypto
  adapter would use Alpaca's crypto clients behind the same interfaces.
- **Options, trailing-stop, and other order types** — the `Broker` interface
  exposes the order types the engine uses (market, bracket); more can be added.
- **Other brokerages** (Interactive Brokers, Tradier, ...) — implement `Broker`.
- **Other data vendors** (Polygon, Databento, ...) — implement
  `MarketDataProvider`; you can mix a data vendor from one provider with order
  routing through another.

## What it takes to add a broker

1. Create `src/brokers/<vendor>/` implementing `Broker` (and optionally
   `MarketDataProvider`), mapping the SDK's objects to the project's vendor-neutral
   domain types (`Position`, `OrderResult`, `AccountSnapshot`, `BarEvent`, ...).
2. Construct it in `main.build_data_and_broker()`.

That's the whole change — `engine/`, `execution/`, `strategies/`, `scanners/`, and
the optimizer are untouched. See [Broker abstraction](../engineering/broker-abstraction)
for the interface contract and [Extending](../engineering/extending) for a worked
example. Trade-update streaming is optional: a broker that can't stream account
events just returns `False` from `supports_trade_updates`.
