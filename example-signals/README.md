# example-signals — a private pack you can copy

This is what your own work looks like from TradeFlow's side: a separate package with its
own `pyproject.toml`, registering strategies and a scanner through two entry-point
groups. The engine imports nothing from here and knows nothing about it beyond the base
interface.

**Every name in it is a placeholder.** The package inside is `my_signals`, because that
is what a pack of your own would look like; the entry points are `example_breakout` and
friends, because from the engine's side these are demonstrations. Renaming both is the
first thing you do, and the next section says how.

```bash
tradeflow init --example-pack ./my-signals   # the destination is yours to name
cd my-signals && uv pip install -e .
tradeflow init --check                       # lists it under "private packs installed"
```

## Make it yours

Nothing here needs to keep its name. Rename the `my_signals/` directory and the matching
`name` and `packages` entries in `pyproject.toml`, then reinstall. The entry-point names
on the left-hand side (`example_breakout` and friends) are what you type at `--strategy`;
rename those too, and the only rule is that they must not collide with a name the engine
reserves.

## Layout

```
pyproject.toml                    the entry-point groups — this is the whole registration
my_signals/
  strategies/
    breakout.py                   long-only Donchian breakout with a trend filter
    pairs_reversion.py            long/short mean reversion
  scanners/
    liquidity.py                  dollar-volume screen with point-in-time ranking
configs/
  breakout.json                   a long-only run config
  reversion_longshort.json        a long/short one — note what it carries that the other does not
.gitignore                        what a private pack keeps out of git
```

One class per module, re-exported from each package's `__init__.py`, so the entry points
name a stable import path while files stay free to move. Everything is written to be
**read**: the comments explain decisions that are not obvious from the code — why the
breakout level is shifted by a bar, why the scanner ranks rather than only flags, why
dispersion is guarded before dividing by it.

**Two strategies on purpose.** A long-only book cannot exercise half of what TradeFlow
measures. Leg diagnostics, `max_net_exposure`, the directional-tilt derivation and the
short-borrow side of the cost model only mean anything for a book that trades both
sides. `pairs_reversion` is there so those surfaces have something to act on.

Neither is an edge. They are ordinary ideas, chosen because the point is the shape.

## The runbook

Each step answers a different question, and clearing one says nothing about the others.
Substitute your own dates and symbols.

### 1. Does it run, and where did the money come from?

```bash
tradeflow backtest --config configs/breakout.json --start 2023-01-02 --end 2025-01-02
```

The weakest evidence this system produces, and worth exactly one thing: confirming the
strategy executes and trades. Read the **where the P&L came from** block before the
return — a book whose entire gain arrives through one exit path is a bet on that path's
fill assumption.

### 2. What is the result actually resting on?

```bash
tradeflow backtest --config configs/breakout.json --start ... --end ... --fill-stress
tradeflow backtest --config configs/breakout.json --start ... --end ... --cost-stress
```

`--fill-stress` requires the price to trade progressively further *through* each
take-profit before it counts as filled. `--cost-stress` re-runs under worse cost
assumptions. Neither is journaled — they are one candidate under stated assumptions, not
new candidates, so you can run them freely.

For the long/short config, the same command also prints the **directional tilt** the
book carried and what each candidate `max_net_exposure` would have done to it. That is
how you choose a cap from the strategy's own history instead of picking a number.

### 3. Which parameters, and how many did you try?

```bash
tradeflow optimize --strategy example_breakout --scanner example_liquidity \
  --symbols AAPL,MSFT,NVDA,AMD --start 2023-01-02 --end 2025-01-02 \
  --method grid --max-evals 40
```

Every candidate is counted. That count is what the deflated Sharpe deflates against, so
a search is not free — it raises the bar for everything in that family afterwards.

### 4. Does it survive out of sample?

```bash
tradeflow walkforward --strategy example_breakout --scanner example_liquidity \
  --symbols AAPL,MSFT,NVDA,AMD --start 2020-01-02 --end 2025-01-02 \
  --folds 4 --save-config configs/breakout_validated.json
```

Searches inside each fold and scores on data the search never saw. `--save-config`
writes the result as a run config carrying the params, the resolved universe, the book
limits and a provenance block recording how they were arrived at.

### 5. How would it sit in a portfolio?

```bash
tradeflow allocate --strategy example_breakout --symbols AAPL,MSFT,NVDA,AMD \
  --capital 25000 --objective utility
```

The strategy becomes the alpha source and the optimizer builds a book from it, cost-aware
by default. `--objective weights` for a plain mean-variance allocation.

### 6. One verdict across the pipeline

```bash
tradeflow verdict --strategy example_breakout --symbols AAPL,MSFT,NVDA,AMD \
  --start 2023-01-02 --end 2025-01-02
```

Scan → alphas → portfolio → information, reported together. Statistical validation,
execution viability and evidence completeness are three separate facts; clearing one
says nothing about the others, and the report keeps them apart.

### 7. Paper, and only then

```bash
tradeflow live --config configs/breakout_validated.json \
  --capital 8000 --feed iex --preflight
```

`--preflight` prints the whole contract — broker mode, capital, universe, data feed,
every book limit in the units it is enforced in, warm-up coverage, telemetry
destinations — and exits without starting the order path. Drop the flag only when that
output is boring.

Afterwards:

```bash
tradeflow execution-report --orders     # slippage, latency, refusals by kind
tradeflow reconcile                     # ledger against the broker
```

## Where your evidence goes

Nowhere near here. TradeFlow writes its journal, trial store, saved configs and position
ledger under one state root — `~/.tradeflow` by default, `TRADEFLOW_HOME` to move it —
deliberately outside any working tree, so a private strategy's results are not sitting in
a git repository waiting for an ignore-file mistake.

The `configs/` in this pack is different: it is *input*, hand-written and versioned with
the code. Configs a run produces land under the state root.

## The names

`example_breakout`, `example_reversion` and `example_liquidity` are what you type at
`--strategy` and `--scanner`. They come from the left-hand side of the entry-point
declarations, not from the class names. Rename them to whatever you like; they only have
to avoid colliding with a built-in, and the engine refuses a pack rather than shadowing
one if they do.

## Then read

- [Your own strategies](https://tradeflow.mk-dir.com/docs/usage/private-strategies)
- [Run configs](https://tradeflow.mk-dir.com/docs/usage/run-configs)
- [Validation diagnostics](https://tradeflow.mk-dir.com/docs/usage/validation-diagnostics)
- [Extending](https://tradeflow.mk-dir.com/docs/engineering/extending) — the interface reference
