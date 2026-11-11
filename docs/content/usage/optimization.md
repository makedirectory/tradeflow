---
sidebar_position: 6
title: Parameter optimization
---

# Parameter optimization

Optimization searches a strategy's parameter space for the configuration that
maximizes a chosen backtest objective (Sharpe ratio, total return, ...). Each
candidate configuration is scored by running a full backtest.

A warning that bears repeating: optimization is *very* good at finding the
settings that would have printed money on this exact slice of history. That's not
the same as edge — it's often just a flattering fit to noise. Treat the result as
a hypothesis and put it through [walk-forward validation](walk-forward) before
believing a word of it.

```bash
quantred optimize \
    --strategy demo_trend --scanner none --symbols NVDA,META,TSLA \
    --start 2024-01-02 --end 2024-04-01 --method grid --max-evals 50
```

Like a backtest it needs bars: Alpaca keys from [`quantred init`](first-run), or
`--offline` against a warmed cache. It follows [`backtest`](backtesting) and
[`verdict`](verdict) in the workflow, and is followed by [`walkforward`](walk-forward).

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `grid` | `grid`, `random` or `bayesian` ([below](#methods)) |
| `--objective` | `sharpe_ratio` | Metric the search maximises (for example `sortino_ratio`, `total_return`) |
| `--max-evals` | `50` | Evaluation budget; a grid larger than this is sampled, not truncated |
| `--seed` | `42` | Seed for sampling |
| `--output PATH` | `optimization_results.csv` | CSV of every candidate, relative to the current directory |
| `--workers N` | `1` | Evaluate candidates in parallel ([below](#running-candidates-in-parallel---workers-n)) |
| `--strategy`, `--scanner`, `--symbols`, `--start`, `--end`, `--capital`, `--scan-as-of` | as for [`backtest`](backtesting#options) | The universe and window |
| `--gross`, `--commission-bps`, `--impact-eta`, `--borrow-bps` | net, `1.0`, `0.3`, `50.0` | The cost model every candidate is scored under |
| `--cache` / `--offline` / `--cache-dir` | off | Serve bars from the local cache |
| `--no-journal` / `--force` | off | Do not record the search; or re-run instead of serving identical candidates |

`optimize` has no `--config` and no book-limit flags: every candidate runs at the book
the strategy declares, which for the demo strategy is one position. To explore a
parameter space against a stated book first, use [`screen`](screening), which takes
`--max-positions` and `--range`.

## Methods

| `--method` | What it does | Needs |
|-----------|--------------|-------|
| `grid` | Sweep the step grid; randomly samples it when larger than `--max-evals` | — |
| `random` | Random step-aligned sampling (`--max-evals` samples) | — |
| `bayesian` | Trains a Gaussian-Process **surrogate model** of the objective and proposes promising configs | `scikit-learn` |

All three consume `--seed` (default 42): `random` and `bayesian` always, and `grid`
once `--max-evals` is smaller than the grid, because the cap is *sampled* rather than
truncated. It matches `walkforward --seed`, so the same value reproduces the search a
walk-forward ran inside each fold. Vary it to draw a genuinely second sample — without
it every random search on a given space returned the same candidates.

Bayesian needs the `optimize` extra (see [optional extras](extras)); with it installed:

```bash
quantred optimize --method bayesian --scanner none --symbols NVDA,META
```

## Output

The best parameters and score are printed, and every evaluated configuration is
written to `optimization_results.csv` (or `--output`) in the current directory:

```
Best sharpe_ratio: 1.83
Best parameters: {'fast_ema_period': 10, 'slow_ema_period': 40, ...}
Full results written to optimization_results.csv

IN-SAMPLE: the best of 50 configs scored on the data that chose it. Validate out of sample
with `quantred walkforward`, which deflates across every trial. ...
```

(Illustrative values.) The `IN-SAMPLE` line is part of the output on purpose. The best
score is the maximum of the candidates, so it is flattering by construction; read it as
"this is the most this slice of history could be made to say", never as a forecast. The CSV's
`deflated_sharpe_ratio` column is not deflated here, since each row is one backtest.
`optimize` does not write a config file: it is a search, not a validation, and nothing
here is promotable.

Every evaluated configuration is also recorded as one **trial** in
`logs/research_journal.jsonl` and the queryable
[trial store](../engineering/walk-forward#the-trial-store) — a 50-point search is
50 trials, which is exactly what a campaign-level deflated Sharpe needs to count
(the store makes that count queryable by hand; wiring it into the gate itself is
a [separate, open item](../engineering/walk-forward#n_trials-still-counts-a-run-at-gate-time-not-a-campaign)).
Pass `--no-journal` to keep an exploratory sweep out of that total.

:::tip Avoid overfitting
A configuration that looks great in-sample often disappoints out-of-sample.
Validate the winner on a *different* date range before trusting it.

This is also the multiple-testing trap: try enough configs and one looks good by
luck. The journaled trial count is what lets the deflated Sharpe raise the bar
accordingly — so `--no-journal` on a real search understates how many tickets you
bought.
:::

### Validating a winner

There are two different questions, and `walkforward` answers only one of them. It runs
its **own** search inside each fold and scores whatever that search picks on the next
slice, so it validates the **tuning procedure** (method, objective, budget), not the one
set of parameters `optimize` just printed. To judge those exact parameters on dates the
search never saw, write them to a config (a `walkforward --save-config` run, or
[`trials promote`](trials#promoting-a-validated-trial) on a validated trial) and run
[`backtest --config`](run-configs#consuming-one) over a later window. Count that later
window as spent once you have looked at it: reuse it to choose again and it stops being
out of sample.

How the search avoids materializing astronomically large grids, and how the
surrogate model works, is covered in
**[Optimization (engineering)](../engineering/optimization)**.

## Running candidates in parallel (`--workers N`)

Candidate evaluation is embarrassingly parallel — each candidate is an independent
backtest over the same read-only bars — so a search can use the cores the machine
already has:

```bash
quantred optimize --strategy demo_trend --symbols NVDA,AAPL,META \
    --start 2024-01-01 --end 2024-12-31 --method grid --max-evals 200 --workers 4
```

`--workers` also applies to [`walkforward`](walk-forward), where it parallelizes
each fold's in-sample candidate search. Folds themselves stay sequential: the
candidates are where the work is, and per-fold progress stays readable.

**It changes wall-clock and nothing else.** The same seed produces the same trials,
the same chosen config, and the same campaign trial count as a sequential run.
Measured on a 64-candidate grid over 8 symbols and 3 years of synthetic daily bars:
**17.6s sequential → 9.5s with `--workers 4`** (1.85×), identical winner, identical
trial count. The speedup is sublinear because process startup and per-worker bar
loading are real costs — on a search small enough, sequential wins.

Three things make the "nothing else" part true:

- **Workers execute; the parent records.** No worker touches the research journal,
  the trial-store index, or stdout. Memoization is resolved before dispatch, so a
  candidate this campaign already scored costs no compute and no worker slot. The
  trial store's single-writer contract is untouched — there is no schema change and
  no locking, because there is no concurrent writing.
- **Seeds come from identity, not from order.** Each candidate's seed derives from
  its own dedup hash, so it simulates identically whether it ran first, last, or in
  the sequential path. Results are collected in submission order regardless of
  completion order.
- **Ranking is a total order.** Tied candidates break on their parameter values, so
  a parallel run completing in a different order cannot pick a different winner from
  an identical set of results.

Duplicate candidates are dispatched **once**. Two identical candidates are one
trial; running both would inflate the campaign's multiple-testing total with work
that produced no new information.

### What it costs

**Memory scales with `--workers`.** Each worker holds its own copy of the bar
frames, so a wide universe times many workers can dwarf the sequential footprint.
The flag defaults to 1 and is capped at the machine's core count; raise it
deliberately.

**It implies the bar cache.** A live data client cannot be handed to a spawned
worker, and N workers independently fetching the same bars from the vendor is
strictly worse than one warmed local cache — so the parent warms the union of the
requested ranges once, before dispatch, and workers read local Parquet. The command
says so when it turns the cache on for you.

### When something goes wrong

A worker crash fails **that candidate**, not the campaign: the failure is reported
with its error, and the search completes. Ctrl-C cancels what has not started, keeps
what already finished (that compute was really spent), and prints a partial-run
notice rather than pretending the search completed.

With Bayesian search, the surrogate proposes a **batch** per round instead of one
point — `--workers` points, spread apart so they explore rather than cluster — then
refits. Same evaluation budget, fewer rounds.

## From a source checkout (contributors)

The commands above are the installed `quantred` command. In a checkout, `uv run quantred <command> ...` runs the same thing (`uv run python main.py <command> ...` also works). The Makefile shortcut for this page is `make optimize`. Extras are installed with `uv sync --inexact --extra <name>`
(see [optional extras](extras)).
