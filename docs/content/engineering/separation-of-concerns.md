---
sidebar_position: 2
title: Separation of concerns
---

# Separation of concerns

The original code conflated responsibilities — a strategy that also computed
portfolio metrics, an engine that simulated *and* executed *and* reported. This
refactor pulls those apart so each concern can change independently and be tested
in isolation.

## Who owns what

| Concern | Owner | Explicitly *not* its job |
|---------|-------|--------------------------|
| Indicators → signals | `strategies/` | fetching data, placing orders |
| Position sizing & risk validation | `strategies/` (config-driven) | knowing about a broker |
| Which symbols to trade | `scanners/` | running the strategy |
| Fetching/streaming bars | `marketdata/` | indicators, signals |
| Simulating fills (backtest) | `engine/` | metric formulas |
| Placing real orders (live) | `execution/` | deciding *whether* to trade |
| Performance math | `analytics/` | trading, data access |
| Wiring it together | `engine/` + `main.py` | business logic |

## Why it matters here

- **A `Strategy` emits signals and sizes positions — nothing else.** It has no
  reference to a broker or data vendor, so the *same* strategy object runs
  unchanged in both backtest and live mode.
- **Metrics moved out of the strategy and engine into `analytics/`.** The numbers
  can be consumed programmatically (by the optimizer) or rendered (by reporting)
  without dragging trading logic along.
- **Execution is isolated.** `LiveTrader` is the only thing that mutates the
  account; everything else just produces signals.

## A concrete example

When a live bar arrives:

1. `MarketDataClient` delivers a `BarEvent` (data concern).
2. `Strategy.process_real_time_data` updates indicators and returns a signal
   (signal concern).
3. `LiveEngine` forwards an actionable signal (orchestration concern).
4. `LiveTrader` sizes it via the strategy config and places a bracket order
   (execution concern).
5. `AlpacaBroker` maps that to the SDK call (vendor concern).

Each step can be replaced or tested without touching the others.
