# Contributing

Thanks for contributing! This project values being **simple, well-separated, and
testable**. Please skim the conventions before opening a PR.

## Setup

Use **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (or
[Docker](https://docs.docker.com/get-docker/)):

```bash
uv sync --extra dev      # installs deps + ruff + pytest (and optimize/portfolio extras)
```

Working on the documentation site as well? It pins its Node version in `.nvmrc`:

```bash
nvm use                  # or: fnm use — reads .nvmrc
cd docs && npm ci
```

CI reads the same file, so a local build and a CI build are on the same major.

## Before you push

These are exactly what CI runs:

```bash
uv run ruff check .            # lint
uv run ruff format .           # format (use --check in CI)
make test                      # offline test suite (no API keys/network)
```

## Coding standards

The full conventions live in the engineering wiki:
**[Coding standards](docs/content/engineering/coding-standards.md)**. The rules
that matter most:

- **Dependencies point downward**, and **no vendor SDK (`alpaca`, ...) is imported
  above the broker layer** — use the `Broker` / `MarketDataProvider` interfaces.
- **One concern per module** (signals vs sizing vs fills vs execution vs metrics).
- **Indicators stay pure pandas/numpy** — no TA-Lib or other compiled deps.
- **Type hints + docstrings** on public APIs; reuse the shared `utils`/`analytics`
  helpers instead of re-deriving logic.
- **Tests are offline and deterministic** via the fakes in `tests/fakes.py`.

## Pull requests

- Keep them small and focused on one layer/concern.
- Fill out the PR template (it has the checklist above).
- CI must be green (lint, format, tests).

## Adding things

Extending a strategy, scanner, or broker? See
[Extending](docs/content/engineering/extending.md) — each is a one-file change at
the right layer.
