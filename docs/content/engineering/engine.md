---
sidebar_position: 8
title: The engine
---

# The engine

`quantred/engine/` contains two orchestrators. They wire the other layers together and
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

0. **Decide on the previous bar.** A signal at bar `i` comes from `calculate_scores`
   at bar `i`, and scores are computed from that bar's *close*. So the signal a bar may
   act on is always the one before it, executed at this bar's open. Through accounting
   v3 the engine executed `signal[i]` at `open[i]` — a one-bar look-ahead on every entry
   and every signal exit, applied to the whole history. A feed shift cannot detect it:
   shifting moves signal and price together, so the relationship survives intact, which
   is why the leakage probe passed over it for as long as it did.
1. **Mark** open positions at *this bar's open*, and track excursion extremes over the
   whole bar. The open, because every decision below transacts at it: the exposure caps
   in step 4 are tested against `equity × cap`, so marking at the close first meant
   whether an entry was admitted could depend on where the bar finished, hours after the
   price it filled at. That was the intra-bar look-ahead left after accounting v4,
   removed at v5 — the same class as the signal one above, in the admission gate rather than the signal, and just
   as invisible to a feed shift. The excursion extremes stay full-bar: they are reported,
   never consulted by a decision, and the worst and best a position saw is the point.
2. **Exit at the open** — only the exits that were decided before the bar began: a
   signal exit, which fills at this open (the signal came from the previous close, so
   the live trader already has the close ordered when it decides the next entries).
   Capital and the slot it frees are reusable by entries at this same open.
3. **Rank** entry candidates across the whole universe by the strategy's own
   conviction score, descending, ties broken by symbol so a run is reproducible.
4. **Admit** in that order while `max_positions`, `max_total_risk`,
   `max_gross_exposure`, `max_net_exposure` and free cash allow. Sizing goes through
   `Strategy.calculate_position_size`, but against *free buying power* — the pool less
   the notional already held at cost (`execution.sizing.free_buying_power`) — which is
   what makes positions actually compete. Live calls the same definition, so three
   entries in a row shrink on both clocks instead of only in the backtest. The two portfolio fractions are not the
   same measurement — see [what `max_total_risk` caps](../usage/configuration.md#what-max_total_risk-caps).
   A request larger than free cash is opened at what the cash can pay, entry cost
   included, and counted (`positions_capped_to_pool`; `cap_to_buying_power` is the one
   rule both clocks apply). An order that rounds to zero, or falls under the declared
   `min_notional` floor (`below_min_notional`), is refused and counted separately
   (`positions_below_min_notional`); with no floor declared there is none, not a floor
   of zero. The size and the bracket's stop
   and target are computed from the close the signal came from, as the live order's
   are, and the levels are rounded to the tick; the fill itself is this bar's open.
5. **Stops and targets**, after admission, for every position still held (entries made
   on this bar included - the bracket rests from the fill, and the signal that opened a
   position cannot also close it). A level the bar *opens* beyond fills at the open: a
   long with a stop at 97 on a bar opening at 90 exits at 90, and a target gapped
   through fills at the better open. The open is the first print, so when it is beyond
   one level that level fills even if the same bar's range later spans the other (a
   long with stop 90 and target 110 on a bar opening 115 books the target at 115, not
   the stop). A level the open was inside of and the range then touched fills at the
   level; if one bar touches both, the stop is taken first - bars cannot say which came
   first, and the pessimistic order is the convention. **Capital and slots freed here
   are available from the next bar, not this open.** Freeing them for the same open
   made admission depend on a price later than the open (with one slot, a new name was
   admitted at the open only if the held name's low went on to break its stop). That
   includes a stop that gaps through at the open: it fills at the open, but the entry
   decision is made before that print exists, so the live trader still sees the slot
   occupied; the backtest takes the conservative reading. Before accounting v7 the
   entry bar was never checked, and before this change intra-bar exits freed the slot.
6. **Record** portfolio equity — cash plus positions re-marked to *this bar's close*.
   Everything that transacts on the bar has happened by now, so the curve is an
   end-of-bar mark-to-market as it has always been. Only the decisions are held to the
   open, and only because they priced there.

Anything still open at the end is force-closed (`END_OF_PERIOD`). P&L is
`(exit − entry) × size × direction`, less costs on both legs.

**Shorts are fully cash-collateralized.** Opening debits the whole notional whichever
way the position faces, rather than crediting short proceeds against margin the way a
real margin account would. This is deliberate: a short costs the same buying power as
the equivalent long, so the book can never quietly take on leverage the engine isn't
modelling. The trade-off to keep in mind when reading results is that short capacity
is understated, so a long-short configuration and a long-only one are not compared on
exactly equal footing. Entry and exit are symmetric, so realized P&L is unaffected.

### What the pool is, and what the two clocks share

The pool entries are sized from is `deployable_pool(equity, capital, realized_pnl)`: the
configured capital, capped at the account's equity, or the equity itself when no capital
was set (never margin buying power, which is a lending limit and not a budget). It moves
with realized P&L on both clocks, so after a loss the next entry sizes from less. Live
keeps the realized figure in memory from closing fills, so a restart resumes at the
configured capital. These four functions in `quantred/execution/sizing.py` are the shared
definitions; the backtest imports them from the trade clock's package, never the reverse.

Entry edges are acted on once. Signals are edge-triggered, the backtest takes or declines
each entry exactly once, and live marks a side spent after the book has held it or the
trader has submitted or book-refused its entry, until the score stops implying that side —
see [edges, and what live mode adds](strategies#edges-and-what-live-mode-adds). The shared
rules are guarded by tests that feed one input to both clocks and compare the results.

### Annualizing per-step quantities

The merged timeline is the **union** of every symbol's timestamps, so it is at least as
dense as any single symbol's bars and strictly denser whenever symbols don't share one
grid — halts, differing listing calendars, a mixed-venue universe. Two things are
measured in steps on that timeline: the equity curve and short carry accrual.

Annualizing them at the strategy's timeframe rate would therefore assume a coarser
sampling frequency than the series actually has, inflating Sharpe and volatility by
`√density` and understating carry by `density`. The engine instead scales the timeframe
rate by the observed density (merged steps ÷ the densest symbol's bars). For a universe
whose symbols share a grid that ratio is exactly 1, so ordinary backtests are unchanged;
it corrects only the ragged case, which was previously wrong in the flattering
direction.

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
   very first live bar. The window is measured in **sessions, not wall-clock
   time**: converting bars to a `timedelta` directly counts the overnight gap and
   the weekend as tradeable, which at intraday frequencies fetches a fraction of
   the history the indicators asked for. A short or empty warm-up is logged,
   because from inside the loop it is indistinguishable from a quiet market.
2. `_cold_start` — hydrate the strategy's position book from broker truth *before*
   the first bar. Warm-up seeds indicators; it says nothing about what is held, and
   a strategy that wrongly believes it is flat cannot emit an exit.
3. Subscribe to the live stream through the `MarketDataClient`.
4. `_on_bar` — feed each full streamed bar to `process_bar`; forward any
   actionable signal to the `LiveTrader`, which returns a `Decision` recording what
   it did and why.
5. If the broker supports it, run the **trade-update stream concurrently** so
   fills/cancels/rejects are logged alongside trading, and each fill re-reads its own
   symbol from the broker — a reconciliation sweep landing between an entry's
   submission and its fill would otherwise leave the book believing it is flat in a
   symbol it holds, and a strategy that believes it is flat cannot emit an exit.

Inside the loop, on a timer rather than per bar, the engine re-reads the position
book and reconciles the ledger against the account. Both are bounded by
construction — one `list_positions` call per sweep, never one per symbol — because
this runs on the trade clock and must not scale with universe size.

**The broker SDK is synchronous, so its calls run in a worker thread.** An entry makes
several blocking round trips, and running those on the event loop stalls everything else
it carries: other symbols that signalled on the same bar, the trade-update stream, and
the reconciliation sweep. There is still exactly one loop and one order at a time —
everything that reads the position book and then acts on it is serialized behind a
single semaphore. That is a correctness requirement, not a speed-up: two entries
checking the book at once can both pass an exposure limit only one of them fits inside.

Both live streams (market data and trade updates) auto-reconnect with capped backoff via
a shared `run_with_reconnect` helper. Shutdown is signal-driven — `SIGINT` and `SIGTERM`
take the same path — and every bound in it is deliberately duplicated off the loop,
because third-party code that blocks the loop takes every loop-scheduled timeout down
with it, signal handlers included. See [stopping a
session](../usage/live-trading.md#stopping-a-session).

This matches the live path rather than being more conservative than it. Live has always
been causal — a closed bar arrives, `process_bar` emits a signal, and a market order
fills strictly afterwards — so through v3 the backtest was transacting at a price live
could never get, and every validated number overstated what a deployment could achieve.

That is the general hazard of the two-clock design: the same rule is implemented twice,
in code that cannot reference itself, so a defect on one side is a defect on the other
until someone checks. See [separation of concerns](separation-of-concerns.md).

The engine never calls the broker directly — it delegates to
[execution](broker-abstraction). That boundary is exactly why the same strategy
object backtests and trades live unchanged.
