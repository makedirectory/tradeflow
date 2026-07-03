<!-- Keep PRs small and focused on one layer/concern where possible. -->

## What & why

<!-- What does this change do, and why? Link any issue. -->

## How

<!-- Notable implementation choices, especially anything affecting a layer boundary. -->

## Checklist

- [ ] `make test` passes (offline; no keys needed)
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
- [ ] New/changed behavior is covered by tests
- [ ] No vendor SDK (`alpaca`, ...) imported above the broker layer
- [ ] Touches one concern per layer (see [Separation of concerns](../website/docs/engineering/separation-of-concerns.md))
- [ ] Docs updated if usage or architecture changed (README / `website/docs`)

## Screenshots / output

<!-- e.g. a backtest report, scan output, or relevant logs. Optional. -->
