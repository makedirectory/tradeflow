---
sidebar_position: 8
title: The engine
---

# The engine

`src/engine/` contains two orchestrators. They wire the other layers together and
own the per-bar loop — but contain no indicator math, no metric formulas, and no
vendor specifics.

## `BacktestEngine`

`run(symbols, start, end, initial_capital, trade_from=None) -> BacktestResult`:

1. Fetch bars for all symbols via the `MarketDataClient`.
2. `_prepare`: per symbol, `process_data` → `generate_signals` → `calculate_scores`,
   then align every symbol to one merged timeline (`_Panel`).
3. `_replay`: walk that timeline once against a shared capital pool.
4. Compute metrics (`analytics.performance`) and return a `BacktestResult`.

`trade_from` separates warmup from trading: earlier bars feed the indicators but
open no positions, and the equity curve starts there. Walk-forward uses it so an
out-of-sample window is measured on its own portfolio curve.

### Portfolio accounting

The simulation runs on **one clock against one capital pool** — every symbol on a
single merged timeline, positions competing for the same dollars as they would
live. Each step:

1. **Mark** open positions to market (and track excursion extremes).
2. **Exit** — stop-loss, take-profit, then signal exit, in that order. Stop/take
   fill at their level; a signal exit fills at the next open. Exits run *before*
   entries, so capital freed this bar is reusable this bar.
3. **Rank** entry candidates across the whole universe by the strategy's own
   conviction score, descending, ties broken by symbol so a run is reproducible.
4. **Admit** in that order while `max_positions`, `max_total_risk` and free cash
   allow. Sizing goes through `Strategy.calculate_position_size` as before, but
   against *free cash*, which is what makes positions actually compete.
5. **Record** portfolio equity — cash plus marked-to-market positions.

Anything still open at the end is force-closed (`END_OF_PERIOD`). P&L is
`(exit − entry) × size × direction`, less costs on both legs.

This matters more than it sounds. The engine originally simulated each symbol
independently across its whole history and summed the P&L onto one capital base —
so symbol A's positions returned their capital before symbol B started, and two
positions that would have competed never met. Absolute metrics therefore scaled
with universe size: the same strategy on the same window returned 23% on one
symbol and 411% on fourteen. Position limits were per-symbol too, so
`max_positions: 1` meant one position *per name*, not one in the book.

Because the promotion gates and the research agent's out-of-sample selection both
read those numbers, widening the universe was an undocumented way to make any
strategy look better — the exact overfit surface the engine exists to close.

The equity curve is emitted from portfolio state per bar, so **open positions are
marked to market**. It previously accumulated realized P&L at exit time and
resampled to calendar days, which made a long-held position invisible until it
closed and then land as a single spike — overstating volatility and distorting
Sharpe, drawdown and VaR. See the
[gate calibration](walk-forward.md#why-the-thresholds-are-what-they-are) for how
that measurement change was carried into the thresholds.

`BacktestResult` carries `metrics`, the `trades` DataFrame, the `equity_curve`,
capital, dates, and the strategy config.

## `LiveEngine`

`start(symbols)`:

1. `_warm_up` — fetch a lookback window, run `process_data`, and seed each
   symbol's rolling buffer via `Strategy.warm_up`, so indicators are valid on the
   very first live bar.
2. Subscribe to the live stream through the `MarketDataClient`.
3. `_on_bar` — feed each full streamed bar to `process_bar`; forward any
   actionable signal to the `LiveTrader`.
4. If the broker supports it, run the **trade-update stream concurrently**
   (`asyncio.gather`) so fills/cancels/rejects are logged alongside trading.

Both live streams (market data and trade updates) auto-reconnect with capped
backoff via a shared `run_with_reconnect` helper, and shut down cleanly on
cancellation.

The engine never calls the broker directly — it delegates to
[execution](broker-abstraction). That boundary is exactly why the same strategy
object backtests and trades live unchanged.
