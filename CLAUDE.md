# Working on TradeFlow

Conventions for AI assistants working in this repository. The authoritative process
document is [PROCESS.md](PROCESS.md) — this file is the short version plus the things
that are easy to get wrong.

## The one invariant

**Two clocks.** The research clock (backtest, optimize, walk-forward, analytics, the
AI agent) is slow, exploratory, and only ever *proposes*. The trade clock
(`tradeflow/engine/live.py`, `tradeflow/execution/`) is deterministic and imports
**nothing** from `services/`, `analytics/`, `optimization/`, or `research/`.
Promotion between them is a manual human step.

A change that blurs this is wrong regardless of how well it works.

## Verify, don't assert

This is the rule that has caught the most real defects here, and every one of them
looked fine in the source:

- **Run the thing.** A published package installed into a clean venv found four
  onboarding dead ends that passed every test. An MCP dependency that every test and
  lockfile agreed was fine shipped broken for a week.
- **Check the environment, not the config.** Verify an action tag resolves before
  pushing it; verify a URL serves before linking it; verify a dependency's *resolved*
  version, not what the lockfile pinned.
- **A fixture that agrees with the test proves nothing.** A checkout-detection test
  wrote its own `pyproject.toml` and kept passing after the real one was renamed.
  Build fixtures from the same constant the code reads.
- **Say what you actually verified**, and what you did not.

## Definition of done

Per PROCESS.md §2, and none of it is optional:

1. Logic in the right layer; dependencies point downward; no vendor SDK above
   `tradeflow/brokers/`.
2. Offline, deterministic tests via `tests/fakes.py` — unit for the mechanic,
   integration for the flow, regression for anything found in review.
3. Every applicable surface wired: CLI, MCP tool, Makefile target.
4. Docs updated — usage guide, engineering wiki, README if the headline changed.
5. `uv run ruff check .`, `uv run ruff format --check .`, `make test`,
   `make check-links` all clean.

## Writing rules

- **Code and docs are self-contained.** Never reference a planning spec by number or
  section in code, docstrings, tests, or docs — `specs/` is gitignored, so a reader
  cannot follow it. Link to the engineering wiki instead.
- **No external-source citations.** Explain the math directly; do not name the book,
  paper, author, or equation number a technique came from. Cite the concept, not its
  provenance.
- **Comments explain *why*, not *what*.** Prefer the reason a threshold is loose over
  a restatement of the comparison.
- **Absent is not zero.** A metric that was never recorded is not a metric of zero.
  This governs sorting and filtering as much as rendering.

## Honesty in output

The project's whole value is refusing to flatter a result. Anything it prints
inherits that:

- A summary that collapses disagreeing gates into one reassuring number is a bug.
  Show the checks and say "mixed".
- Reused, memoized, degraded, and partial results say so prominently.
- A guard rejects; it never repairs. Repairing an input makes the live path stop
  being the thing the backtest validated.
- Report; never remediate. Detection is a feature; automatic correction of a
  financial position is not.

## Messages differ by how the software was reached

An installed copy has no repository. Any instruction assuming one — `.env.example`,
a `make` target, `uv sync` — is a dead end for that reader, and worse than no
instruction because it sends them looking for a file that was never there. This has
been fixed in four separate places; check for a fifth before adding a message.

## Git

- Branch off `main`; never commit to it directly.
- Commit messages explain the *why* and any deviation from plan. End with
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- One concern per PR where possible; state what was verified and what was deferred.
- Commit as you go. Reconstructing per-change commits afterwards is not possible
  once edits to shared files interleave.

## Cross-model consultation

Codex CLI may be consulted as an independent engineering reviewer:

```
codex exec "<focused question>"
```

**Check it is installed first** (`command -v codex`) and say so plainly if it is not,
rather than reporting a `command not found` as a consultation.

Good reasons to consult it:

- architecture decisions
- difficult debugging
- security-sensitive changes
- unfamiliar code
- reviewing a proposed implementation
- checking assumptions
- comparing multiple approaches

Codex is **advisory**. Evaluate its response independently before acting — it has no
context on this project's invariants, and an answer that violates the two-clocks rule
or the honesty rules above is wrong here regardless of how sound it looks in general.

Do not ask Codex to invoke Claude Code. This prevents recursive agent-to-agent loops.

## Specs

`specs/` is gitignored and local-only. A spec moves to `specs/complete/` only when
its §Testing items exist and pass — and moving one means updating its status header,
all three indexes, and re-running `make check-links`, which catches the sibling
links the move breaks in both directions.

A spec that ships with deviations records them under *Implementation notes*, so the
next reader learns them from the spec rather than from the code.
