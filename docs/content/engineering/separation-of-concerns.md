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

## The trade clock is single-threaded on purpose

The broker SDK is synchronous, so its calls run in a worker thread — an entry makes
several blocking round trips, and running those on the event loop stalls every other
thing the loop carries. But there is still exactly **one loop and one order at a time**:
everything that reads the position book and then acts on it is serialized behind a
single semaphore.

That is a correctness requirement rather than a performance choice. Two entries checking
the book at once can both pass an exposure limit only one of them fits inside, and a
book assembled in a different order from the signals that produced it is not the book
that was validated. The threads exist only to keep blocking I/O off the loop; the trade
clock's determinism comes from doing one thing at a time, and that has not changed.

The same reasoning bounds shutdown. Every timeout scheduled on the loop fails at once
the moment third-party code blocks it — including the signal handlers, which asyncio
delivers as loop callbacks — so the one bound that must always hold lives on a daemon
thread instead.

`tradeflow/costs/` is importable from the trade clock, and deliberately so: a live run
records what the same cost model expected a fill to cost, and a separate live-only
formula would make that number incomparable with the backtest a candidate was judged on.
It imports nothing from `services/`, `analytics/`, `optimization/` or `research/`, which
is what keeps it on the right side of the line.

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
