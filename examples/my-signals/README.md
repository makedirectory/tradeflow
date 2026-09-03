# my-signals — an example private pack

This is what your own work looks like from TradeFlow's side: a separate package, with
its own `pyproject.toml`, that registers a strategy and a scanner through two
entry-point groups. The engine imports nothing from here and knows nothing about it
beyond the base interface.

Copy it and start editing:

```bash
tradeflow init --example-pack ./my-signals
cd my-signals
uv pip install -e .
tradeflow init --check          # lists it under "private packs installed"
```

## What is in it

```
pyproject.toml            the two entry-point groups — this is the whole registration
my_signals/strategies.py  a Donchian breakout with a trend filter
my_signals/scanners.py    a dollar-volume liquidity screen
configs/                  a saved run config, the artefact backtest/walk-forward/live all read
.gitignore                what a private pack should keep out of git
```

The strategy and scanner are written to be **read**, not traded. Both are deliberately
ordinary ideas; the point is the shape, and the comments explain the decisions that are
not obvious from the code — why the breakout level is shifted by a bar, why the scanner
ranks rather than just flags, why book limits are declared rather than defaulted.

## Try it

```bash
tradeflow backtest --strategy example_breakout --scanner none \
  --symbols AAPL,MSFT,NVDA --start 2024-01-02 --end 2025-01-02

tradeflow backtest --config configs/example_breakout.json \
  --start 2024-01-02 --end 2025-01-02
```

The second form reads everything — universe, capital, book limits, cost assumptions and
params — from the config, which is how a validated run is reproduced rather than
retyped.

## The name matters

`example_breakout` and `example_liquidity` are what you type at `--strategy` and
`--scanner`. They come from the left-hand side of the entry-point declarations, not from
the class names. Rename them to whatever you like; they only have to avoid colliding
with a built-in, and the engine refuses a pack rather than shadowing one if they do.

## Where your evidence goes

Nowhere near here. TradeFlow writes its journal, trial store, saved configs and position
ledger under one state root — `~/.tradeflow` by default, `TRADEFLOW_HOME` to move it —
deliberately outside any working tree, so a private strategy's results are not sitting
in a git repository waiting for an ignore-file mistake.

`tradeflow init --check` tells you where that root is and warns if it has ended up
inside a repository.

## Then read

- [Your own strategies](https://tradeflow.mk-dir.com/docs/usage/private-strategies) —
  the path from an idea to a validated result.
- [Run configs](https://tradeflow.mk-dir.com/docs/usage/run-configs) — the artefact in
  `configs/`.
- [Extending](https://tradeflow.mk-dir.com/docs/engineering/extending) — the full
  interface reference.
