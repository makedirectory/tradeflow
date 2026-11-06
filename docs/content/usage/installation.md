---
sidebar_position: 1
title: Installation
---

# Installation

## As a command (no clone)

### Fresh install

```bash
uv tool install quantred
quantred demo
```

In an activated virtual environment: `python -m pip install quantred`.

The distribution and the command are both **`quantred`**; the importable package is
still `tradeflow`, and a `tradeflow` command is installed alongside as an alias so
existing scripts and client configs keep working. The project was previously
distributed as `tradeflow-engine`: that distribution owns the same import package, so
uninstall it *before* installing `quantred` (`uv tool uninstall tradeflow-engine`, or
`python -m pip uninstall tradeflow-engine`) rather than installing both into one
environment. The full steps, including what to do if the order was reversed, are in
[migrating to QuantRed](migrating-to-quantred).

This gives you a `quantred` command with every verb — `quantred demo`,
`quantred init`, `quantred verdict`, `quantred mcp` — and needs no repository, no
keys, and no network for the demo. Next: [first run](first-run).

### Adding a capability later

Optional capabilities are opt-in extras. To choose them at install time:

```bash
uv tool install 'quantred[viz,store,mcp]'
```

Adding one to an installed copy later is *not* additive with `uv tool`: it replaces
the environment, so list every extra you want. The [optional extras](extras) page has
the table of what each unlocks and the exact commands; `quantred init --check` prints
them with your current extras filled in.

### Where state lives

State is the same wherever the software came from: the research journal, trial
store, bar cache, and promoted configs go to `~/.tradeflow`. A source checkout does
**not** keep its own state; it uses the same directory, so that a private strategy's
trial evidence never sits inside a git working tree.

Resolution: `TRADEFLOW_HOME` if set, otherwise `~/.tradeflow`. Contributors who want
state inside their checkout again set `TRADEFLOW_HOME` to it explicitly;
`quantred init --dev-local-state` prints the line to use.

This matters more than it looks. The multiple-testing correction rests on **one**
journal accumulating every trial, so a campaign split across two roots would deflate
its Sharpe against half the evidence — and nothing would error. `quantred --version`
and `quantred init --check` both print the resolved root, so it is never a mystery.

## From a checkout (contributors)

The rest of this page is for working on QuantRed itself. The Makefile targets and
`uv run python main.py` below are the checkout forms of the `quantred` commands used
everywhere else in these docs.

## Prerequisites

You need **either** of these (not both):

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — runs the app
  locally, **or**
- **[Docker](https://docs.docker.com/get-docker/)** — runs the app in a container,
  no local Python/uv required.

Plus free Alpaca **paper-trading** API keys from the
[Alpaca dashboard](https://app.alpaca.markets/) (*Paper Account → API Keys*).
(Node.js 18+ is only needed to build these docs.)

The `Makefile` targets run through **uv**; the **Docker** path uses
`make docker-build` / `make docker-run` (uv runs inside the image).

## Option A — local with uv

```bash
make install        # == uv sync --inexact
```

`uv` reads `pyproject.toml`, creates a virtual environment, and installs the
pinned dependencies from `uv.lock`. The base install is intentionally small:
`alpaca-py`, `pandas`, `numpy`, `pytz`. There is **no compiler step** — indicators
are pure pandas/numpy, not TA-Lib.

## Option B — Docker

```bash
make docker-build                     # build the image
make docker-run                       # paper live-trading; mounts your .env
```

No local Python or uv needed. Override the command to backtest/scan, e.g.:

```bash
docker run --rm -v $(pwd)/.env:/app/.env quantred \
    uv run python main.py backtest --symbols NVDA,META --start 2024-01-02 --end 2024-04-01
```

## Optional extras (checkout)

The full list and what each unlocks is on the [optional extras](extras) page. In a
checkout:

```bash
make install-optimize     # scikit-learn  -> Bayesian parameter optimization
make install-portfolio    # Google OR-Tools -> portfolio allocation
```

Or install everything used by the test suite:

```bash
uv sync --inexact --extra dev
```

## Verify

```bash
make test           # offline test suite — no API keys or network needed
make demo           # run the whole pipeline on synthetic data (also keyless)
```

A green `make test` confirms the engine, scanner, optimizer, and portfolio
allocator are wired correctly; `make demo` shows them working end-to-end and ends
in an honest promotion verdict. Next: **[Configuration](configuration)**.
