---
sidebar_position: 3
title: Your own strategies
---

# Your own strategies

**The strategies and scanners that ship with TradeFlow are examples.** Three strategies
and one scanner, there to demonstrate what the interface expects and to give the demo
something to run. They are not the product, and none of them is an edge.

The product is everything around them — the walk-forward validation, the
multiple-testing correction, the cost model, the trade clock, the execution telemetry.
That machinery is worth something only when it is pointed at *your* idea, kept
somewhere that is yours.

This page is the route from an idea to a strategy running privately against your own
data. For the interface reference — every method, every contract — see
[extending the engine](../engineering/extending.md).

## Start from a working pack

There is a complete one in the repository, and `init` will copy it somewhere you own:

```bash
tradeflow init --example-pack ./my-signals
cd my-signals && uv pip install -e .
tradeflow init --check          # lists it under "private packs installed"
```

That gives you a strategy, a scanner, a saved config, a `.gitignore`, and a
`pyproject.toml` with the entry points already declared — a starting point to edit
rather than a sample to translate. Rename the entry points and the classes and it is
your pack.

The rest of this page explains what you just copied.

## Your code stays out of this repository

A strategy lives in a separate installed package and registers itself through an
entry-point group. Nothing about it is committed here, imported from here, or visible
to anyone reading this project.

```toml
# pyproject.toml, in your own package
[project.entry-points."tradeflow.strategies"]
my_breakout = "yourfirm_signals.strategies:BreakoutStrategy"

[project.entry-points."tradeflow.scanners"]
my_liquidity = "yourfirm_signals.scanners:LiquidityScanner"
```

The names on the left are what you type at `--strategy` and `--scanner`. They come from
the entry points, not from the class names, and they only have to avoid colliding with a
built-in — the engine refuses a pack rather than shadowing one if they do.

Install it into the same environment and it appears everywhere a built-in does — in
`--strategy`, in the walk-forward search, over MCP, in the research agent. Sizing,
fills, execution, limits and metrics come for free, because they only ever depended on
the base interface.

```bash
uv pip install -e ../yourfirm-signals     # or from your private index
tradeflow init --check                    # confirms which packs are installed
```

`init --check` lists them under **private packs installed**. If your strategy is not
there, the entry point is not being found, and nothing further will work.

## What a strategy has to provide

Four methods. The engine calls them and nothing else.

| Method | What it answers |
| --- | --- |
| `calculate_required_lookback()` | How many bars before the indicators are valid |
| `process_data(data)` | The frame with your indicator columns added |
| `calculate_scores(data)` | Conviction per bar — the ranking signal |
| `initialize()` | Any one-time setup |

Signals are derived from your scores by the base class, so most strategies never
implement `generate_signals` at all.

**One thing about timing is worth knowing before you write anything.** A signal at bar
`i` is derived from bar `i`'s *close*, and the engine executes it at the open of bar
`i + 1`. You do not need to shift anything yourself, and you should not: a strategy
that lags its own scores would be lagged twice. This is also what live does — a closed
bar produces a signal, an order fills after it — so a backtest and a deployment agree
about what was knowable when.

## The path from idea to evidence

Each step answers a different question, and passing one says nothing about the others.

**1. Does it run at all?**

```bash
tradeflow backtest --strategy my_breakout --symbols AAPL,MSFT,NVDA \
  --start 2024-01-02 --end 2025-01-02
```

A single backtest over one window is the weakest evidence this system produces. It is
worth exactly one thing: confirming the strategy executes and trades. Read the
[where the P&L came from](backtesting.md) block before reading the return.

**2. Does it survive out of sample?**

```bash
tradeflow walkforward --strategy my_breakout --symbols AAPL,MSFT,NVDA \
  --start 2020-01-02 --end 2025-01-02 --folds 4 --save-config configs/breakout.json
```

Walk-forward searches parameters inside each fold and scores them on data the search
never saw. It also counts every candidate it tried, so the deflated Sharpe knows how
many chances the idea had. `--save-config` writes the result as a
[run config](run-configs.md) — the artefact everything downstream uses.

**3. Would it have been tradeable?**

Statistical validation says nothing about execution. The
[validation diagnostics](validation-diagnostics.md) are where you find out whether the
edge survives its own fill assumptions, whether one exit path is carrying the entire
result, and what the book was actually exposed to.

**4. Paper, then a long look.**

```bash
tradeflow live --config configs/breakout.json --capital 8000 --feed iex --preflight
```

`--preflight` prints the contract and exits without starting the order path. Read
[live trading](live-trading.md) before dropping that flag.

## Where your evidence is kept

Everything a run records — the research journal, the trial store, saved configs, the
position ledger — lives under one state root, and **it is never inside this
repository**. The default is `~/.tradeflow`; `TRADEFLOW_HOME` moves it.

That matters most when the strategy is private. Evidence in a working tree is one
ignore-file edit from disclosure, and `git clean -xd` deletes it outright — live
position ledger included. `init --check` tells you where the root is and warns if it
has ended up inside a repository.

One journal per root, deliberately. The multiple-testing correction counts every trial
your campaign has run; split it across two roots and the deflated Sharpe is computed
against half the evidence, with nothing erroring.

## What ships, and what it is good for

`demo_trend` is the smallest complete strategy — long-only, one indicator, a score that
is just a normalized EMA gap. Read it for the shape of the interface, not for the idea.

For anything past that shape, read `example-signals` instead — it is a real
installable pack, and `tradeflow init --example-pack ./my-signals` writes you a copy.
Its `example_reversion` is long/short, which is what the leg diagnostics and the
directional cap exist for, and none of that is exercisable by a long-only book.

Then point the machinery at something of your own.
