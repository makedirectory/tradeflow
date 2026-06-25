---
sidebar_position: 1
title: Architecture
---

# Architecture

## The mental model: two clocks

Before the layers, the one idea that explains the most: TradeFlow runs on **two
clocks**, and they never touch.

```
  RESEARCH CLOCK  (offline, slow, exploratory)        TRADE CLOCK  (live, fast, deterministic)
  ───────────────────────────────────────────        ──────────────────────────────────────
  hypothesis → backtest → optimize → walk-forward      live bar → signal → order
  may be non-deterministic; LLMs allowed here          deterministic, auditable, LLM-free
  emits configs + rationale to disk                    reads a human-approved config
                              │                                    ▲
                              └──────── a human promotes ──────────┘
                                       (nothing auto-flips to live)
```

- **The research clock** is where intelligence and non-determinism live —
  parameter search, walk-forward validation, and the optional AI research agent.
  It only ever *proposes*: it writes provenance-stamped candidate configs to disk.
- **The trade clock** (`src/engine/live.py` → `src/execution/`) is deliberately
  dumb: a live bar produces a signal produces an order. No model sits in the order
  path, so there's nothing to prompt-inject and nothing non-deterministic to debug
  when real money is at stake.
- **Promotion is a manual human step.** Automation never flips `PAPER_TRADE` or
  routes an order. The [MCP server](mcp-server) enforces this *structurally* — it
  builds only a data client, so it physically cannot trade.

Everything below is how the code is arranged to keep those two clocks separate.

## The layers

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

### The packages

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
