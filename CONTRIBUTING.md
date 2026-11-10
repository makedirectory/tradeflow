# Contributing

Thanks for contributing! This project values being **simple, well-separated, and
testable**. Please skim the conventions before opening a PR.

## Setup

Use **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (or
[Docker](https://docs.docker.com/get-docker/)):

```bash
uv sync --inexact --extra dev       # installs deps + ruff + pytest (and the extras the suite uses)
uv pip install --no-deps -e example # the example pack, as CI installs it (entry-point discovery)
```

Working on the documentation site as well? It pins its Node version in `.nvmrc`:

```bash
nvm use                  # or: fnm use — reads .nvmrc
cd docs && npm ci
```

CI reads the same file, so a local build and a CI build are on the same major.

## Before you push

These are the checks CI runs (`.github/workflows/ci.yml`), in its order:

```bash
uv run ruff check .                 # lint
uv run ruff format --check .        # format check (`uv run ruff format .` fixes it)
make check-links                    # Markdown links and anchors; no tracked link into specs/
docker compose config --quiet       # the compose file parses (needs Docker, not a daemon)
make test                           # offline test suite (no API keys/network)
make secret-scan                    # committed secrets over the whole history (needs gitleaks)
```

`--inexact` matters locally: plain `uv sync` is exact, so it uninstalls the example pack
and any other package installed with `uv pip install -e`. CI starts from a clean runner and
uses plain `uv sync --extra dev`, then installs the example pack as above. Before a release,
`make release-check` also builds the package and installs it into a clean venv.

## Keys and secrets

This repository is public, so a committed key is disclosed the moment it lands —
rotating it is the only fix. Keep credentials in the state root's `.env`
(`~/.quantred/.env`, outside the working tree; `make init` writes it), never in a file
inside the checkout: a `.env` in the repository root is not read, and is one ignore-file
edit away from being committed. `.env.example` holds placeholder values only, and is the
right place to document a new setting.

Two checks back this up: GitHub's push protection rejects a push containing a
recognized provider key, and CI runs [gitleaks](https://github.com/gitleaks/gitleaks)
over **everything reachable from the pushed ref** — not just the diff, since deleting
a secret in a later commit does not remove it from the one that added it. That scan
runs on every branch push, not only on PRs: a secret pushed to a topic branch is
already public whether or not a PR exists.

If the scan fails, treat the key as compromised and **rotate it first**. Rewriting
history is cleanup, not a fix — anyone may already have fetched the commit.

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
