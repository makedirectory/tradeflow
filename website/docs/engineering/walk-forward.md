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
- **Variable-length folds ⇒ CAGR / annualised metrics**, never raw total return.
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
  configs were tried.
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

## Config persistence

`src/optimization/config_store.py` saves a chosen config as JSON with a
`provenance` block (method, windows, objective, OOS metrics, `n_trials`, seed, git
SHA, timestamp). Configs land in a gitignored `configs/` directory. Saving a
config never alters live behaviour — it's a file a human chooses to promote.
