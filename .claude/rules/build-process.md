# How a change gets built here

This is the durable *how we work* rule: the cycle a change goes through, what "done"
means, and the review it has to survive. The topic rules carry the detail — this one
carries the order and the gate.

The one invariant lives in [the trade clock](trade-clock.md) and its
[cross-clock parity](cross-clock-parity.md) cost; the architecture it enforces is
documented in the engineering wiki
([architecture](../../docs/content/engineering/architecture.md),
[coding standards](../../docs/content/engineering/coding-standards.md)).

## The cycle

Steps that don't apply are skipped with a one-line reason **in the PR description**,
never silently.

1. **Plan** — for anything non-trivial, name the failure modes and sketch the test plan
   before writing code. How much you write down (a spec, a note, an issue) is your call;
   [specs](specs.md) governs the ones that become files.
2. **Implement** in the correct layer. Domain meaning lives in a strategy, scanner,
   engine or service; reusable mechanics live in `utils/` or `analytics/`. No vendor SDK
   above `tradeflow/brokers/`.
3. **Test** offline and deterministically through `tests/fakes.py` — see
   [testing](testing.md). Unit coverage for the mechanic, integration coverage for the
   flow, a regression test for every bug review finds.
4. **Wire every applicable surface**: the CLI, an MCP tool (research clock only), a
   Makefile target. A knob one surface lacks is a gate that does not run there and
   nothing says so — [surfaces](surfaces.md), and the CLI↔MCP entry in
   [parity points](parity-points.md).
5. **Document** — the engineering wiki for architecture and behavior, the usage guide
   for how to run it, the README if the headline workflow changed.
6. **Review** against the angles below, then [check for divergence](check-for-divergence.md).
7. **Land** a focused PR on green CI, with the record below in its description.

Build in dependency order: stabilize an interface (`Broker`, `MarketDataProvider`,
`Strategy`, `Scanner`) before building consumers on it, and prove research-clock
machinery offline before it acquires any trade-clock surface.

## Definition of done

None of it optional:

- Logic sits in the right layer and dependencies point downward; public APIs carry type
  hints and docstrings; configuration is explicit and validated in `settings.py`.
- Offline deterministic tests cover the mechanic and the flow, and each new test has
  been watched to fail without the fix.
- Every applicable surface is wired, and anything on [parity points](parity-points.md)
  was changed on both sides in the same commit.
- Docs updated.
- `uv run ruff check .`, `uv run ruff format --check .`, `make test`, `make check-links`
  and `make release-check` are clean.
- Where automation cannot reach — a live paper-trading path — the manual smoke test is
  recorded in the PR.

## Consolidate before merging

After the change works and before it lands:

1. Scan the new code for duplication against `services/`, `utils/`, `analytics/`.
2. Route through the existing helper where one exists. Extract a new one only when a
   second caller actually needs it — premature abstraction is also debt, and a
   single-caller domain rule belongs where it is.
3. Confirm the import graph still points downward with no new cycle.
4. Record in the PR what was extracted, which callers moved, and which test proves the
   behavior held.

`services/` is the layer the CLI and MCP route through; adding to it means the surfaces
stop being able to disagree.

## Read the dependency, don't recall it

When a correctness or security decision hinges on how a third-party library behaves,
read its installed source and cite file, symbol and version in the PR. This has mattered
most for `alpaca-py`'s order and data clients, pandas/numpy NaN, alignment and dtype
edges, and the `scikit-learn`, `ortools`, `mcp` and `anthropic` contracts.

```bash
uv pip show <pkg>
python -c "import x, inspect; print(inspect.getsource(x.fn))"
rg "<symbol>" .venv/lib/python*/site-packages/<pkg>
```

The habit generalises: check the resolved environment rather than the declaration that
describes it.

## Review angles

Review is a quality gate, not a courtesy. Every time:

- Line-by-line correctness and regression risk.
- **The two clocks** — can research-clock code reach the order path, or a
  model/optimizer/LLM/DB reach `engine/live.py` or `execution/`? Does automation still
  only *propose*, with a human promoting? Can the MCP server still not construct a
  trading client?
- **Layering** — downward dependencies, one concern per module, no vendor SDK above the
  broker layer, no business logic in an entry point or a tool handler.
- Numeric correctness — NaN, alignment, dtype — in indicators, metrics and walk-forward.
- Test quality: offline, deterministic, and failing without the fix.
- Surface and contract completeness, and whether anything changed has a twin.
- What the change makes the tool *say* — [honest output](honest-output.md).
- Consolidation opportunities.

An independent read is available through the `/peer-review` command and the
cross-model-review skill; it is advisory, and an answer that violates an invariant here
is wrong regardless of how sound it looks in general.

### Severity

| Severity | Meaning |
|----------|---------|
| Critical | Crosses the two-clocks line, loses data, leaks a credential, or breaks a core flow outright |
| High | Serious bug, broken edge case, bad contract, layering violation |
| Medium | Narrower incorrect behavior, test gap, maintainability |
| Low | Cleanup, clarity, minor doc or UX issue |

Findings are fixed before merge or deliberately deferred against a tracked issue.
Nothing is deferred by not mentioning it.

## Compatibility: clean-slate

There are no production users and no stored production data; live trading is paper
trading. Prefer simple, direct changes over compatibility shims, and add one only to
protect a local developer or a paper account. Records already on disk are the exception
and are not covered by this — see [durable records](durable-records.md), where the
contract is every shape ever written, forever.

If this project ever manages real capital, that decision is made deliberately and this
section changes with it.

## Recording what changed

The PR description is where the history stays legible. Say what was built and in which
layers, what was verified (test results, lint, any manual smoke test), which docs moved,
how the two-clocks invariant stayed intact, what review found and fixed, and what was
deferred and to where. Say what you actually verified, and what you did not.
