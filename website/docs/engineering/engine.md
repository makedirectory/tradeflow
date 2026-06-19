---
sidebar_position: 8
title: The engine
---

# The engine

`src/engine/` contains two orchestrators. They wire the other layers together and
own the per-bar loop — but contain no indicator math, no metric formulas, and no
vendor specifics.

## `BacktestEngine`

`run(symbols, start, end, initial_capital) -> BacktestResult`:

1. Fetch bars for all symbols via the `MarketDataClient`.
2. For each symbol: `process_data` → `generate_signals` → `_simulate_symbol`.
3. Aggregate trades, build the equity curve, compute metrics
   (`analytics.performance`), and return a `BacktestResult`.

### `_simulate_symbol`

A bar-by-bar replay holding **one position at a time**:

- On an entry signal with no open position, size via
  `Strategy.calculate_position_size` and open (if affordable) with stop/take
  levels from the strategy config.
- On each subsequent bar, check **stop-loss**, **take-profit**, then a
  **signal-based exit**, in that order. Stop/take fill at their level; a signal
  exit fills at the next open.
- Any position still open at the final bar is force-closed (`END_OF_PERIOD`).

Realised cash (`_available`) carries across symbols so the run can't spend the
same dollar twice. P&L is `(exit − entry) × size × direction`.

`BacktestResult` carries `metrics`, the `trades` DataFrame, the `equity_curve`,
capital, dates, and the strategy config.

## `LiveEngine`

`start(symbols)`:

1. `_warm_up` — fetch a lookback window, run `process_data`, and seed each
   symbol's rolling buffer via `Strategy.warm_up`, so indicators are valid on the
   very first live bar.
2. Subscribe to the live stream through the `MarketDataClient`.
3. `_on_bar` — feed each `BarEvent` to `process_real_time_data`; forward any
   actionable signal to the `LiveTrader`.

The engine never calls the broker directly — it delegates to
[execution](broker-abstraction). That boundary is exactly why the same strategy
object backtests and trades live unchanged.
