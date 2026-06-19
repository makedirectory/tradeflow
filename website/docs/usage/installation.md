---
sidebar_position: 1
title: Installation
---

# Installation

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
make docker-run                       # paper live-trading; mounts your config.py
```

No local Python or uv needed. Override the command to backtest/scan, e.g.:

```bash
docker run --rm -v $(pwd)/config.py:/app/config.py trading-engine \
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
```

A green run confirms the engine, scanner, optimizer, and portfolio allocator are
wired correctly. Next: **[Configuration](configuration)**.
