---
sidebar_position: 1
title: Installation
---

# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (the project's package manager)
- An Alpaca account with **paper-trading** API keys
- Optionally Docker, and Node.js 18+ to build these docs

## Install dependencies

```bash
make install        # == uv sync
```

`uv` reads `pyproject.toml`, creates a virtual environment, and installs the
pinned dependencies from `uv.lock`. The base install is intentionally small:
`alpaca-py`, `pandas`, `numpy`, `pytz`. There is **no compiler step** — indicators
are pure pandas/numpy, not TA-Lib.

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
