---
sidebar_position: 13
title: Coding standards
---

# Coding standards

These are the conventions the codebase follows. CI enforces the mechanical ones
(lint, format, tests); the rest are reviewed in PRs.

## Tooling

- **[uv](https://docs.astral.sh/uv/)** for dependencies and running everything.
- **[ruff](https://docs.astral.sh/ruff/)** for both linting and formatting.
  - Lint rules: `E`, `W` (pycodestyle), `F` (pyflakes), `I` (import sorting).
  - Line length **110**; the formatter owns wrapping (so `E501` is not linted).
  - Run locally: `uv run ruff check .` and `uv run ruff format .`.
- **[pytest](https://docs.pytest.org/)** for tests: `make test` (offline).

CI (`.github/workflows/ci.yml`) runs lint, format-check, and tests on every push
and PR.

## Architecture rules

These keep the layering honest — they matter more than any style nit:

1. **Dependencies point downward.** A layer may import from layers below it, never
   above. See [Architecture](architecture).
2. **No vendor SDK above the broker layer.** Only `src/brokers/<vendor>/` imports
   `alpaca` (or any other broker SDK). Everything else uses the `Broker` /
   `MarketDataProvider` interfaces. See [Broker abstraction](broker-abstraction).
3. **One concern per module.** Signals, sizing, fills, execution, metrics, and
   data access live in different layers. See
   [Separation of concerns](separation-of-concerns).
4. **Heavy/optional dependencies are lazy-imported** behind extras (scikit-learn
   for Bayesian optimization, OR-Tools for portfolio allocation), so the base
   install stays small.

## Conventions

- **Type hints** on public functions, methods, and dataclasses.
- **Docstrings** on modules and public classes/methods — explain *why*, not just
  what. Note explicitly what a class is *not* responsible for when it clarifies a
  boundary.
- **Domain types are vendor-neutral dataclasses/enums** (`Position`, `OrderSide`,
  `BarEvent`, ...); adapters map SDK objects to them.
- **Shared vocabulary lives in one place** — e.g. trade signals in
  `strategies/signals.py`, scan signals in `scanners/base.py`. Never hard-code a
  signal string in two layers.
- **Indicators are pure pandas/numpy** (`src/indicators`). No TA-Lib or other
  compiled dependency in the base install.
- **Reuse the shared helpers** (`utils/numeric`, `utils/timeutils`,
  `utils/streaming`, `analytics/metrics`) rather than re-deriving the same logic.

## Error handling

- **Real-time handlers never raise** — `process_bar` and stream callbacks log and
  continue so a single bad bar can't kill a live session.
- **Streams reconnect**, they don't crash — via `utils/streaming.run_with_reconnect`.
- **Fail fast on configuration** — invalid parameters raise at construction
  (`Strategy._validate_parameters`), not mid-run.
- Catch-all `except Exception` is acceptable only at stream/loop boundaries, and
  is annotated `# noqa: BLE001` with a reason.

## Tests

- **Offline and deterministic.** Use the in-memory fakes (`tests/fakes.py`); seed
  any randomness. No network or API keys.
- **One behaviour per test**, named for what it asserts.
- Cover the boundary you changed: a new sizer, scanner, or broker gets a test
  through its interface.
