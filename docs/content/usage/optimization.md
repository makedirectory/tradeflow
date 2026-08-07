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
make optimize
# or
uv run python main.py optimize \
    --strategy volume_spike --scanner none --symbols NVDA,META,TSLA \
    --start 2024-01-02 --end 2024-04-01 --method grid --max-evals 50
```

## Methods

| `--method` | What it does | Needs |
|-----------|--------------|-------|
| `grid` | Sweep the step grid; randomly samples it when larger than `--max-evals` | — |
| `random` | Random step-aligned sampling (`--max-evals` samples) | — |
| `bayesian` | Trains a Gaussian-Process **surrogate model** of the objective and proposes promising configs | `scikit-learn` |

Bayesian needs the optional extra:

```bash
make install-optimize     # or: uv sync --extra optimize
uv run python main.py optimize --method bayesian --scanner none --symbols NVDA,META
```

## Output

The best parameters and score are printed, and every evaluated configuration is
written to `optimization_results.csv`:

```
Best sharpe_ratio: 1.83
Best parameters: {'rsi_period': 10, 'volume_threshold': 1.4, ...}
Full results written to optimization_results.csv
```

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

How the search avoids materializing astronomically large grids, and how the
surrogate model works, is covered in
**[Optimization (engineering)](../engineering/optimization)**.

## Running candidates in parallel (`--workers N`)

Candidate evaluation is embarrassingly parallel — each candidate is an independent
backtest over the same read-only bars — so a search can use the cores the machine
already has:

```bash
python main.py optimize --strategy volume_spike --symbols NVDA,AAPL,META \
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
