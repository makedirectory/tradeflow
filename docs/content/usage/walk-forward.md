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
| `--bootstrap-skill` | Nonparametric own p-value (stationary block bootstrap) next to the FAMILY p from White's Reality Check over every OOS return series the trial store has recorded for this strategy/universe/accounting — advisory only, not a gate. See [below](#nonparametric-skill-check). |
| `--save-config PATH` | Save the chosen config — params *and* run inputs — for a human to review. See [Reusing a saved config](#reusing-a-saved-config). |
| `--results-csv PATH` | Write the per-fold table to CSV. |

## Reading the output

The report prints a per-fold table (IS vs OOS headline metrics, OOS trade count),
the OOS aggregate block, walk-forward efficiency, IS→OOS degradation, the holdout
block, and the **promotion-gate verdict** — a pass/fail per gate plus an overall
`promotable`. A config is only `promotable` if it clears *every* gate (median OOS
Sharpe, profit factor, efficiency, drawdown ratio, minimum OOS trades, deflated
Sharpe, and — when requested — parameter sensitivity and the leakage probe).

> Saving a config never changes live behavior. It writes a JSON file to a
> gitignored `configs/` directory; promoting it to live trading is a manual human
> step.

## Promotion prerequisites

`promotable` stays **statistical**: median OOS Sharpe, profit factor, walk-forward
efficiency, drawdown ratio, trade count, parameter sensitivity, deflated Sharpe. It
means the same thing it meant for every trial already recorded, and nothing added since
has changed it.

Two further questions come *after* a candidate clears those, and are reported beside it:

```
=== Promotion prerequisites (separate from `promotable`) ===
  [PASS] cost_stress        5 vs 3
  [ -- ] family_bootstrap   not evaluated - needs 10 usable return-series trials to
                            mean anything; 2 available
  Prerequisites: 1 of 3 evaluated - clear so far; 2 unknown
  An unevaluated check is not a passed one - what is unknown stays unknown.
```

- **`cost_stress`** — the edge survives at least 3x its own assumed cost. Run it with
  `walkforward --cost-stress`, which stresses the config the folds actually chose.
- **`family_bootstrap`** — still notable once every trial the campaign tried is priced
  in. It does **not** run below 10 usable return-series trials: a family test over two
  series is arithmetic rather than evidence, and a striking p-value on K=2 is exactly
  the kind of number that should not gate anything.
- **`benchmark_relative`** — the median per-fold information ratio against
  `--benchmark` is positive. **Per fold, then median**, because every other fold
  statistic here is a median and a second aggregation convention in one report would
  differ from its neighbours most exactly when the folds disagree. They do: a real run
  produced per-fold IRs of `[0.13, -1.25, 2.10]` for a median of `+0.13`, and a single
  figure over the stitched curve would have hidden that spread entirely.

**An unevaluated check is not a passed one.** A cost curve nobody ran and a family too
thin to test are both *unknown*, and `ready` always travels with the count of what was
actually evaluated — a clear verdict over one of two checks is not a clearance.

## Reusing a saved config

The file holds the whole run configuration, not just tuned params: `strategy`,
`params`, `scanner`, `symbols` (the universe the scanner *resolved*, not the candidate
list), `capital` and the cost model. So one file drives any run type, and can be
versioned in a private repository beside the strategies it belongs to:

```bash
tradeflow backtest --config configs/alpha.json --start 2024-01-02 --end 2024-06-01
tradeflow verdict  --config configs/alpha.json --start 2024-01-02 --end 2024-06-01
tradeflow alphas   --config configs/alpha.json
tradeflow risk     --config configs/alpha.json
```

`--config` is accepted by `backtest`, `live`, `verdict`, `info`, `alphas`, `horizon`,
`allocate` and `risk`. Three rules make it predictable:

- **Anything you type wins.** The file fills in only what the command line left
  unsaid, so `--symbols ZZZ` beats the file's universe. Each run prints where every
  value came from.
- **A command takes only the fields it has.** `risk` summarizes a universe's
  covariance and has no strategy, so it uses the file's `symbols` and nothing else —
  and says so rather than naming a strategy it never ran.
- **A contradictory `--strategy` is refused.** The params in the file belong to the
  strategy in the file; handing one strategy's tuned params to another is not
  something to guess at.

**The window is never stored.** A config carrying its own tuning dates would make
every later run silently re-evaluate that period, so `--start`/`--end` always come
from the run and the output says so. What the config *was* tuned on is recorded under
`provenance.windows`, for reading rather than replaying.

## Nonparametric skill check

`--bootstrap-skill` adds a second, assumption-free verdict next to the deflated
Sharpe: an **own** p-value from a stationary block bootstrap of this run's OOS
returns (no assumption about the return distribution's shape), always reported
next to the **family** p-value from White's Reality Check over every other
trial recorded for the same strategy/universe/accounting in the
[trial store](#the-trial-store) — a great own p and a terrible family p is
exactly the selection-luck signature this test exists to catch. Family scoring
is advisory, not a hard gate. See
[Evaluation metrics — bootstrap skill inference](../engineering/evaluation-metrics#bootstrap-skill-inference).

## The trial store

Every walk-forward run (and every `backtest`/`optimize`/`alphas` run, and the
research agent) is dual-written into a queryable SQLite index over the research
journal — `logs/trials.db`. It's what makes campaign-wide counts answerable
without reading the whole journal:

```bash
python main.py trials query --strategy volume_spike --symbols NVDA,AAPL,META
python main.py trials status      # row/journal-line counts + a drift check
python main.py trials rebuild     # rebuild from the journal — safe, it's derived
```

`trials query --strategy ... --symbols ...` prints the campaign's real
`n_trials` — useful to check by hand, since the promotion gate above still only
counts the current run (deliberately: wiring the campaign count into the gate
automatically would make every gate strictly harder and reclassify configs
already saved as promotable — an open, evidence-backed decision, not an
oversight). See
[Walk-forward validation — the trial store (engineering)](../engineering/walk-forward#the-trial-store).

See the engineering wiki's **Walk-forward validation** page for the design,
fold geometry, and the leakage-safety guarantees.

## Running folds faster (`--workers N`)

`--workers` parallelizes each fold's in-sample candidate search across worker
processes. Folds stay sequential — the candidates are where the work is, and
per-fold progress stays readable.

It changes wall-clock only: the same seed produces the same folds, the same chosen
config, and the same campaign trial count as a sequential run, because workers only
execute and this process still does every journal write. See
[parallel candidates](optimization#running-candidates-in-parallel---workers-n) for
the full contract and its costs.
