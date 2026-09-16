# How a change gets built here

The durable *how we work* rule: the order a change goes through and the gate it has to
pass. The topic rules carry the detail and the procedures live in skills — this file is
the sequence and the standard, and points at the rest rather than restating it.

The one invariant is [the trade clock](trade-clock.md) and its
[cross-clock parity](cross-clock-parity.md) cost. The architecture it enforces is
documented in the engineering wiki
([architecture](../../docs/content/engineering/architecture.md),
[coding standards](../../docs/content/engineering/coding-standards.md)); setup and PR
mechanics are in [CONTRIBUTING.md](../../CONTRIBUTING.md).

## The cycle

A step that does not apply is skipped with a one-line reason **in the PR description**,
never silently.

1. **Plan** — for anything non-trivial, name the failure modes and sketch the test plan
   before writing code. How much you write down (a spec, a note, an issue) is your call;
   [specs](specs.md) governs the ones that become files.
2. **Implement** in the correct layer. Domain meaning lives in a strategy, scanner,
   engine or service; reusable mechanics live in `utils/` or `analytics/`. No vendor SDK
   above `tradeflow/brokers/`. Where behavior turns on a library, use the
   **dependency-truth** skill rather than recalling what it does.
3. **Test** offline and deterministically through `tests/fakes.py` — see
   [testing](testing.md). Unit coverage for the mechanic, integration coverage for the
   flow, a regression test for every bug review finds.
4. **Wire every applicable surface**: the CLI, an MCP tool (research clock only), a
   Makefile target. A knob one surface lacks is a gate that does not run there and
   nothing says so — [surfaces](surfaces.md), and the CLI↔MCP entry in
   [parity points](parity-points.md).
5. **Document** — the engineering wiki for architecture and behavior, the usage guide
   for how to run it, the README if the headline workflow changed.
6. **Consolidate** with the **consolidation-pass** skill, then review with the
   **review-gate** skill, then [check for divergence](check-for-divergence.md).
7. **Land** a focused PR on green CI. `/ship-check` walks the definition of done below
   and reports what is not met.

Build in dependency order: stabilize an interface (`Broker`, `MarketDataProvider`,
`Strategy`, `Scanner`) before building consumers on it, and prove research-clock
machinery offline before it acquires any trade-clock surface.

## Definition of done

None of it optional:

- Logic sits in the right layer and dependencies point downward; public APIs carry type
  hints and docstrings; configuration is explicit and validated in `settings.py`.
- Offline deterministic tests cover the mechanic and the flow, and each new test has
  been watched failing without the fix.
- Every applicable surface is wired, and anything on [parity points](parity-points.md)
  was changed on both sides in the same commit.
- Docs updated.
- `uv run ruff check .`, `uv run ruff format --check .`, `make test`, `make check-links`
  and `make release-check` are clean.
- Where automation cannot reach — a live paper-trading path — the manual smoke test is
  recorded in the PR.

A green gate is not evidence that an intended artifact exists. Verify the text, the
file, or the behavior itself, and **say what you actually verified and what you did
not.**

## Compatibility: clean-slate

There are no production users and no stored production data; live trading is paper
trading. Prefer simple, direct changes over compatibility shims, and add one only to
protect a local developer or a paper account.

Records already on disk are the exception and are not covered by this — see
[durable records](durable-records.md), where the contract is every shape ever written,
forever.

If this project ever manages real capital, that decision is made deliberately and this
section changes with it.

## Recording what changed

The PR description is where the history stays legible. Say what was built and in which
layers, what was verified (test results, lint, any manual smoke test), which docs moved,
how the two-clocks invariant stayed intact, what review found and fixed, and what was
deferred and to where.
