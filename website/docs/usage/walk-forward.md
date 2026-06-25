---
sidebar_position: 9
title: Walk-forward validation
---

# Walk-forward validation

A backtest optimized and reported on the **same** date range tells you how well a
config fit that window's noise — not whether it will work tomorrow. Walk-forward
validation is the honest alternative: **optimize on one slice of time, measure on
a later slice the optimizer never saw, and reserve a final slice nobody touches
until the very end.**

```bash
uv run python main.py walkforward \
  --strategy volume_spike --scanner volume --symbols NVDA,AAPL,MSFT \
  --start 2024-01-01 --end 2025-12-31 --capital 100000 \
  --mode anchored --folds 6 \
  --embargo-days 5 --holdout-days 60 \
  --method grid --objective sharpe_ratio --max-evals 50
```

## Why it matters

The failure mode this guards against is **overfitting** — mistaking the random
noise in a particular stretch of history for a real, repeatable edge.

It's easy to do by accident. Search enough parameter combinations against one
date range and *something* will look brilliant, purely by luck — the more configs
you try, the better the best one looks even if none of them has any edge. (This
is the multiple-testing problem, and it's why TradeFlow also reports a **deflated
Sharpe** that discounts your result by how many configs you tried.) Optimize and
report on the *same* window and you've measured how well you fit that window's
noise, not whether the strategy will work next month.

`make demo` shows the trap in one screen: a moving-average crossover posts
+16.8% in-sample, then collapses to a **−0.42 median Sharpe out-of-sample** and
fails every promotion gate. Same strategy, same data — the only thing that
changed is that it had to perform on bars it wasn't tuned on.

Walk-forward closes the gap by splitting time into three roles:

- **In-sample (IS)** — the only data the optimizer is allowed to fit.
- **Out-of-sample (OOS)** — later data the optimizer never saw, used to measure
  whether the chosen config *generalizes*. Repeated across folds so the verdict
  doesn't hinge on one lucky window.
- **Holdout** — a final slice carved off first and scored exactly once, at the
  very end. It's the closest thing to "live" you get before risking real money,
  so it must never influence any decision along the way.

The payoff isn't a higher return — it's an *honest* one. A config that clears the
promotion gates here has at least been asked the right question: does this work
on data it has never seen?

## What it does

For each fold it optimizes parameters on an **in-sample (IS)** window, then scores
the chosen config on the following **out-of-sample (OOS)** window. An
**embargo** gap separates IS from OOS so indicator warm-up can't leak across the
boundary. A **holdout** window is carved off the end first and scored exactly
once, at the very end — it never reaches any optimizer call.

The honest performance number is the **OOS aggregate**: metrics recomputed over
the concatenation of every fold's OOS trades, not an average of per-fold numbers.

## Key options

| Flag | Meaning |
|------|---------|
| `--mode anchored\|rolling` | Expanding IS window (anchored) or fixed-width sliding IS (rolling). |
| `--folds N` | Number of folds. Alternatively set `--train-days` / `--test-days`. |
| `--embargo-days N` | IS→OOS gap. Defaults to the strategy's required lookback in calendar days. |
| `--holdout-days N` | Final sacred window, scored once. |
| `--method grid\|random\|bayesian` | Search method used per fold (`bayesian` needs the `optimize` extra). |
| `--objective` | Metric to optimize in-sample (default `sharpe_ratio`). |
| `--pbo` | Also estimate the Probability of Backtest Overfitting (slower). |
| `--monte-carlo` | Block-bootstrap the OOS trades for a 5th-percentile Sharpe. |
| `--param-sensitivity` | Perturb the chosen params ±10% and re-test robustness. |
| `--leakage-probe` | Shift the data feed forward to detect future-data leakage. |
| `--save-config PATH` | Save the chosen config (with provenance) for a human to review. |
| `--results-csv PATH` | Write the per-fold table to CSV. |

## Reading the output

The report prints a per-fold table (IS vs OOS headline metrics, OOS trade count),
the OOS aggregate block, walk-forward efficiency, IS→OOS degradation, the holdout
block, and the **promotion-gate verdict** — a pass/fail per gate plus an overall
`promotable`. A config is only `promotable` if it clears *every* gate (median OOS
Sharpe, profit factor, efficiency, drawdown ratio, minimum OOS trades, deflated
Sharpe, and — when requested — parameter sensitivity and the leakage probe).

> Saving a config never changes live behaviour. It writes a JSON file to a
> gitignored `configs/` directory; promoting it to live trading is a manual human
> step.

See the engineering wiki's **Walk-forward validation** page for the design,
fold geometry, and the leakage-safety guarantees.
