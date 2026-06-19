---
sidebar_position: 1
title: Architecture
---

# Architecture

The system is a stack of single-responsibility layers. Dependencies point
**downward only**, and nothing above the broker layer imports a vendor SDK.

```
                 main.py  (CLI: wires everything per command)
                    │
        ┌───────────┼─────────────┬───────────────┐
     engine/     optimization/   portfolio/        │
   (orchestrate)  (tune params)  (weight positions)│
        │                                          │
   ┌────┴─────┬───────────┬──────────┐             │
strategies/  scanners/  execution/  analytics/     │
        │        │          │                      │
        └────────┴────┬─────┴──────────────────────┘
                 marketdata/        brokers/  (interfaces)
                      │                 │
                      └── brokers/alpaca/ (the only SDK adapter)
                 indicators/   utils/   (leaf helpers)
```

## The layers

| Package | Responsibility |
|---------|----------------|
| `brokers/` | `Broker` interface + domain types (`OrderSide`, `Position`, `AccountSnapshot`, `OrderResult`, `MarketStatus`) |
| `brokers/alpaca/` | `AlpacaBroker` + `AlpacaMarketData` — the only modules that import `alpaca` |
| `marketdata/` | `MarketDataProvider` interface, `Timeframe`, `MarketDataClient`, `BarEvent` |
| `indicators/` | Pure pandas/numpy technical indicators |
| `strategies/` | `Strategy` base, the signal vocabulary, and concrete strategies |
| `scanners/` | `ScannerStrategy` base, concrete scanners, and the `SymbolScanner` |
| `execution/` | `LiveTrader` — turns signals into broker orders |
| `analytics/` | Performance metric primitives, backtest metrics, reporting |
| `engine/` | `BacktestEngine` + `LiveEngine` — orchestration only |
| `optimization/` | `ParameterSpace` + `ParameterOptimizer` |
| `portfolio/` | `PortfolioAllocator` — OR-Tools position weighting |
| `utils/` | logging, numeric, and timezone helpers |

## Guiding principles

- **Interfaces at the boundary.** The engine talks to a `Broker` and a
  `MarketDataClient`, never to Alpaca directly. See
  [Broker Abstraction](broker-abstraction).
- **One job per module.** Signals, sizing, fills, metrics, and execution each
  live in different layers. See [Separation of Concerns](separation-of-concerns).
- **No hidden heavy dependencies.** scikit-learn and OR-Tools are optional extras,
  imported lazily.
- **Everything is testable offline.** Because the boundary is an interface, tests
  inject in-memory fakes. See [Testing](testing).
