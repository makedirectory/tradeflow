---
sidebar_position: 1
title: Installation
---

# Installation

## As a command (no clone)

```bash
uv tool install git+https://github.com/makedirectory/tradeflow
tradeflow demo
```

`pipx install` works the same way. This gives you a `tradeflow` command with every
verb — `tradeflow demo`, `tradeflow init`, `tradeflow verdict`, `tradeflow mcp` —
and needs no repository, no keys, and no network for the demo.

Optional capabilities stay opt-in extras:

```bash
uv tool install "tradeflow[viz,store,mcp] @ git+https://github.com/makedirectory/tradeflow"
```

### Where state lives

An installed copy has no repository to write into, so the research journal, trial
store, bar cache, and promoted configs go to `~/.tradeflow`. Run from a checkout,
they stay in the checkout, exactly as they always have.

Resolution order: `TRADEFLOW_HOME` if set → the current directory if it is a
TradeFlow checkout → `~/.tradeflow`.

This matters more than it looks. The multiple-testing correction rests on **one**
journal accumulating every trial, so a campaign split across two roots would deflate
its Sharpe against half the evidence — and nothing would error. `tradeflow --version`
and `tradeflow init --check` both print the resolved root, so it is never a mystery.

## From a checkout

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
make install        # == uv sync
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
docker run --rm -v $(pwd)/.env:/app/.env tradeflow \
    uv run python main.py backtest --symbols NVDA,META --start 2024-01-02 --end 2024-04-01
```

## Optional extras

Two features are opt-in so the base install stays lean:

```bash
make install-optimize     # scikit-learn  -> Bayesian parameter optimization
make install-portfolio    # Google OR-Tools -> portfolio allocation
```

Or install everything used by the test suite:

```bash
uv sync --extra dev
```

## Verify

```bash
make test           # offline test suite — no API keys or network needed
make demo           # run the whole pipeline on synthetic data (also keyless)
```

A green `make test` confirms the engine, scanner, optimizer, and portfolio
allocator are wired correctly; `make demo` shows them working end-to-end and ends
in an honest promotion verdict. Next: **[Configuration](configuration)**.
