---
sidebar_position: 15
title: Walk-forward validation
---

# Walk-forward validation

`src/optimization/walk_forward.py` is the honest fitness function for the whole
automation effort. The optimizer tunes parameters in-sample; the
`WalkForwardValidator` makes the chosen config prove itself on out-of-sample data
the optimizer never saw, across folds, with a holdout scored exactly once.

> Without this, more automation just means faster overfitting. With it, the
> optimizer's power becomes an asset because the fitness function is honest.

## Fold geometry

Given `start`, `end`, and either `n_folds` or `(train_days, test_days)`:

```
anchored (expanding IS):
  fold k:  IS = [start ............ t_k]  | embargo |  OOS = (t_k+gap, t_k+gap+test]
rolling (sliding IS, width = train_days):
  fold k:  IS = [t_k-train, t_k]          | embargo |  OOS = (t_k+gap, t_k+gap+test]
holdout: [end - holdout_days, end]  — carved off FIRST, excluded from every fold
```

The holdout is computed before fold generation and subtracted from the fold
region, so it is provably disjoint from every IS/OOS window and never reaches an
optimizer call.

## Correctness properties

- **No fold-boundary leakage.** Each OOS backtest fetches `embargo` (≥ lookback)
  bars *before* `oos_start` so indicators are valid, but only trades entered
  at/after `oos_start` are counted (`_filter_trades_from`). The embargo separates
  IS from OOS.
- **Variable-length folds ⇒ CAGR / annualized metrics**, never raw total return.
- **Honest aggregate.** The OOS aggregate recomputes metrics over the
  concatenation of every fold's OOS trades and a stitched curve — cross-fold
  Sharpe/drawdown are real, not an average of per-fold numbers.
- **Determinism.** The optimizer `seed` is threaded, so a run is reproducible.
- **Prefetch once, slice per fold.** The full window (plus warmup) is fetched a
  single time and sliced in memory per fold via `_PrefetchedProvider` — fetching
  per fold would dominate the cost.

## Diagnostics

- **Walk-forward efficiency** = mean(OOS objective) / mean(IS objective).
- **Degradation** = IS − OOS per headline metric, surfaced (not hidden in an
  average) so a big Sharpe drop is visible.
- **Deflated Sharpe** with `n_trials_total` (and a session-wide `n_trials_offset`
  for the research agent) so the multiple-testing correction reflects how many
  configs were tried — **within a run**. See the scope limit below.
- Optional, behind flags: **PBO** (CSCV-style probability of backtest
  overfitting), **Monte-Carlo** block-bootstrap (5th-percentile Sharpe),
  **parameter sensitivity** (±10% perturbation), and a **leakage probe** (shift
  the feed forward; identical results ⇒ the strategy reads future data ⇒ fail).

## Promotion gates

`WalkForwardResult.gate_report()` turns the scorecard into a keep/reject decision
so an agent (or human) can't cherry-pick. Thresholds are config-driven (a dict,
not hardcoded) and use **median** (not mean) for efficiency and OOS Sharpe so one
lucky fold can't inflate the verdict. Default gates: median OOS Sharpe, OOS profit
factor, walk-forward efficiency, OOS-vs-IS drawdown ratio, a minimum-OOS-trades
floor, the deflated Sharpe, and — when computed — parameter sensitivity and the
leakage probe. A config is `promotable` only if it clears **every** gate.

### Known limit: `n_trials` counts a run, not a campaign

The Deflated Sharpe raises the bar as you try more configurations — the more
lottery tickets you buy, the better your best one must be before it counts as
skill. Today `n_trials` **resets when the process exits**: walk-forward accumulates
across folds, and the research agent accumulates across a session, and then the
count starts over.

So a researcher on their tenth session is deflating against that session's few
dozen trials rather than the campaign's few thousand. The error runs in the
dangerous direction — the correction gets *weaker* the harder you search:

| Trials counted | DSR (same Sharpe-2.0 series) | vs the 0.50 gate |
|---:|---:|---|
| 37 (one session) | 0.90 | PASS |
| 370 | 0.69 | PASS |
| 3700 (a real campaign) | 0.43 | **FAIL** |

Treat a reported deflated Sharpe as a **lower bound on how much deflation is
warranted**, and remember that configs tried in earlier sessions are invisible to it.

Two further gaps worth knowing while reading any DSR number here:

- Only the **research agent** journals its trials. Ad-hoc `backtest` and `optimize`
  runs from the CLI are not recorded anywhere, so even a future campaign-level
  count would be a lower bound until those paths journal too.
- `var_of_trial_sr`, the DSR's other input, is estimated per run rather than from
  the real distribution of tried configs.

Closing this needs a queryable index over the journal — the journal already records
every trial, nothing reads it back. That is planned as a trial store: record first,
then a separate evidence-backed decision about wiring campaign counts into the
gates, since doing so makes every gate strictly harder and would reclassify configs
already saved as promotable.

### Why the thresholds are what they are

A threshold is only meaningful relative to how the quantity is measured, so when
the engine moved to [portfolio-level accounting](engine.md#portfolio-accounting)
the gates had to be re-checked — not to make strategies pass, but to keep the
numbers meaning what they meant.

Holding the trades fixed and varying *only* the equity-curve construction across
12 runs (2 strategies × 3 universes × 2 windows):

| Quantity | Mark-to-market ÷ realized-P&L | Threshold |
|---|---|---|
| Sharpe | 1.04–1.27, median **1.19** | `min_oos_sharpe` 1.0 → **1.2** |
| Max drawdown | 1.00–1.28, median **1.04** | `max_dd_ratio` unchanged |

The old curve booked a position's P&L as a single spike when it closed, which
overstated volatility and so understated Sharpe. The new curve marks open
positions to market. Rescaling `min_oos_sharpe` keeps the original bar; leaving it
at 1.0 would have quietly made the gate ~16% easier. Ratio gates
(`max_dd_ratio`, walk-forward efficiency) compare two same-construction numbers,
so the factor cancels and they are untouched.

`min_oos_trades` deliberately **did not move**, even though portfolio accounting
made it much harder to clear: one book with `max_positions` slots simply takes
fewer positions than every symbol trading its own full capital. That is a real
loss of evidence rather than a change of units — the sample genuinely is smaller —
and relaxing a statistical-power floor because results got worse is the precise
form of self-deception the gates exist to prevent. If a strategy cannot reach 100
out-of-sample trades, the honest reading is that it has not earned a verdict yet.

## Config persistence

`src/optimization/config_store.py` saves a chosen config as JSON with a
`provenance` block (method, windows, objective, OOS metrics, `n_trials`, seed, git
SHA, timestamp, `accounting`). Configs land in a gitignored `configs/` directory.
Saving a config never alters live behavior — it's a file a human chooses to promote.

### The `accounting` stamp

Metrics only mean something relative to how capital was accounted for, so every
provenance block records the engine's `ACCOUNTING_VERSION`:

- **1** — pre-[spec 025](engine.md#portfolio-accounting): each symbol simulated
  independently against full capital, equity accumulated from realized P&L at exit.
- **2** — current: one merged timeline, one capital pool, per-bar mark-to-market.

Records written before the field existed carry no version, so absence reads as 1 —
which is exactly what they are. `load_config` warns when a stored version differs
from the running engine, and `is_current_accounting(payload)` is the check to use
before ranking or comparing stored results. The *params* in an old config remain
perfectly usable; it is the `oos_metrics` beside them that were measured a
different way and must not be compared with a fresh run without re-running it.

The same stamp goes on every `audit_log` record, so a research journal spanning an
engine change stays interpretable on replay.
