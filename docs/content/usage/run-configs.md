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
quantred walkforward --strategy my_breakout --symbols AAPL,MSFT,NVDA \
  --start 2020-01-02 --end 2025-01-02 --folds 4 \
  --save-config breakout.json
```

Saving a config never trades it. Promotion to live is a human step, deliberately.

**Where the file lands.** `--save-config` with a bare filename (`breakout.json`) writes
to `configs/` under the state root (`~/.quantred/configs/` by default). A path that
carries a directory (`./runs/breakout.json`, `/data/breakout.json`) is used as given,
relative to the current directory. `--config` does no such lookup: it reads the path you
pass, so the examples below spell out the state-root location.

## What is in it

```json
{
  "strategy": "my_breakout",
  "scanner": "my_liquidity",
  "symbols": ["AAPL", "MSFT", "NVDA"],
  "candidate_symbols": ["AAPL", "MSFT", "NVDA", "AMD", "..."],
  "capital": 25000.0,
  "position_limits": {"max_positions": 6, "max_position_size": 5000.0, "max_total_risk": 0.05,
                      "max_gross_exposure": 0.8, "max_net_exposure": null, "min_notional": null},
  "cost": {"gross": false, "commission_bps": 1.0, "impact_eta": 0.3, "borrow_bps": 50.0},
  "params": {"entry_period": 40, "exit_period": 10},
  "provenance": {"method": "grid", "objective": "sharpe_ratio", "windows": {"...": "..."},
                 "oos_metrics": {}, "n_trials": 32, "seed": 27, "git_sha": "...", "accounting": 8}
}
```

(Illustrative values; `oos_metrics` is the full metric set in a real file.)

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
quantred backtest    --config ~/.quantred/configs/breakout.json --start 2024-01-02 --end 2025-01-02
quantred walkforward --config ~/.quantred/configs/breakout.json --start 2020-01-02 --end 2025-01-02
quantred live        --config ~/.quantred/configs/breakout.json --capital 25000 --preflight
```

`--config` is read by `backtest`, `walkforward`, `verdict`, `screen`, `live` and the
other commands listed under [reusing a saved config](walk-forward#reusing-a-saved-config).
Each command prints what it took from the file and what it ignored, because a config
silently supplying half a run is how the validated setup and the executed one diverge:

```
Config ~/.quantred/configs/breakout.json: strategy='my_breakout', scanner=<config>, symbols=<config>, capital=<config>
  window 2024-01-02..2025-01-02 (from this run, not the config)
  candidates      3 names from the saved config
  universe        replayed from config
```

**Common errors.**

| Message | What to do |
|---|---|
| `Refusing to run: <path> cannot be used by this build.` | The path is mistyped, the file is not valid config JSON, or its strategy is not installed here (a private strategy needs its package installed). The message names the cause |
| `... holds strategy 'a' and its tuned params, but --strategy 'b' was given` | Drop `--strategy`, or drop `--config` and tune `b` on its own |
| A bare `--save-config name.json` is not where you expected | It lands in `configs/` under the state root; a path with a directory is used as given |
| A config trades one position though the search used more | Check `position_limits`; a config written before limits were recorded falls back to the strategy's own |

**The window is never stored.** A config describes *what* to run, not *when*; the dates
always come from the command line. A stored window would make every re-run of a config
a re-run of one moment in history.

### The universe is replayed, not re-scanned

By default `--config` pins `scanner=none` and trades the saved `symbols` as-is. That is
the validated book. Re-running the scanner would give you today's universe attached to
yesterday's evidence — a different book with the same provenance stamped on it.

```bash
quantred backtest --config ~/.quantred/configs/breakout.json --re-resolve-universe
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

A saved config is also written by [`trials promote`](trials#promoting-a-validated-trial),
from a recorded trial and without re-running it. Where each fits in the workflow:
[walk-forward](walk-forward) validates and saves, [backtest](backtesting) replays on a later
window, and [live](live-trading) runs it, with promotion always a manual step.

## Keeping them

Configs live under the state root, outside this repository — `~/.quantred/configs` by
default. For a private strategy, version them in the same private repository as the
strategy itself: the config is a description of your work, and its `provenance` block
records the search that produced it.

## From a source checkout (contributors)

The commands above are the installed `quantred` command. In a checkout, `uv run quantred <command> ...` runs the same thing (`uv run python main.py <command> ...` also works). Extras are installed with `uv sync --inexact --extra <name>`
(see [optional extras](extras)).
