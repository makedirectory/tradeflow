---
sidebar_position: 10
title: Run configs
---

# Run configs

A run config is the artefact the whole workflow turns on. Walk-forward produces one;
backtest, walk-forward and live all consume one. It is what makes a result reproducible
across run types, and what a private repository versions alongside the strategies it
describes.

Not to be confused with the strategy `config` dict, which holds a single strategy's
parameters and limits — see [configuration](configuration.md). This page is about the
saved **file**.

## Producing one

```bash
tradeflow walkforward --strategy my_breakout --symbols AAPL,MSFT,NVDA \
  --start 2020-01-02 --end 2025-01-02 --folds 4 \
  --save-config configs/breakout.json
```

Saving a config never trades it. Promotion to live is a human step, deliberately.

## What is in it

```json
{
  "strategy": "my_breakout",
  "scanner": "my_liquidity",
  "symbols": ["AAPL", "MSFT", "NVDA"],
  "candidate_symbols": ["AAPL", "MSFT", "NVDA", "AMD", "..."],
  "capital": 8000.0,
  "position_limits": {"max_positions": 8, "max_position_size": 1200.0, "max_gross_exposure": 0.9},
  "cost": {"gross": false, "commission_bps": 1.0, "impact_eta": 0.3, "borrow_bps": 50.0},
  "params": {"entry_period": 40, "exit_period": 10},
  "provenance": {"method": "grid", "seed": 27, "n_trials": 32, "git_sha": "...", "oos_metrics": {}}
}
```

Everything except `provenance` is an **input** — what to run. `provenance` is the
opposite: a record of how those params were arrived at, and it is never read back as
input. That separation is what stops a config quietly re-applying a stale search.

Two fields deserve attention.

**`symbols` and `candidate_symbols` are different things.** `symbols` is the universe
the scanner actually resolved — the book that was validated. `candidate_symbols` is the
list it was resolved *from*. Both are kept so a later run can honestly choose between
replaying the validated universe and re-resolving it.

**`position_limits` is written in full**, not left to the strategy's declaration. The
limits are part of what was validated; a config that omitted them would describe a
different book from the one the evidence came from.

## Consuming one

```bash
tradeflow backtest    --config configs/breakout.json --start 2024-01-02 --end 2025-01-02
tradeflow walkforward --config configs/breakout.json --start 2020-01-02 --end 2025-01-02
tradeflow live        --config configs/breakout.json --capital 8000 --preflight
```

Each command prints what it took from the file and what it ignored, because a config
silently supplying half a run is how the validated setup and the executed one diverge.

**The window is never stored.** A config describes *what* to run, not *when*; the dates
always come from the command line. A stored window would make every re-run of a config
a re-run of one moment in history.

### The universe is replayed, not re-scanned

By default `--config` pins `scanner=none` and trades the saved `symbols` as-is. That is
the validated book. Re-running the scanner would give you today's universe attached to
yesterday's evidence — a different book with the same provenance stamped on it.

```bash
tradeflow backtest --config configs/breakout.json --re-resolve-universe
```

`--re-resolve-universe` opts into re-scanning from `candidate_symbols`. It is worth
doing deliberately, to see how much the universe has drifted; it is not worth doing by
accident.

### Flags win over the file

Anything you type beats what the file says, and the run reports which is which. One
contradiction is refused rather than resolved: a `--strategy` that disagrees with the
config's strategy exits, because the params in the file belong to the strategy in the
file and handing them to another one is not an outcome worth guessing at.

## Configs and the trial store

A config's identity — strategy, params, universe, window, cost assumptions and book
limits — is what the trial store deduplicates on. Two runs differing in any of them are
different trials; two runs matching in all of them serve the earlier result rather than
re-simulating.

That identity is also scoped to the engine's accounting version, so a result computed
under an older model is never served to a newer one. See [trials](trials.md).

## Keeping them

Configs live under the state root, outside this repository — `~/.tradeflow/configs` by
default. For a private strategy, version them in the same private repository as the
strategy itself: the config is a description of your work, and its `provenance` block
records the search that produced it.
