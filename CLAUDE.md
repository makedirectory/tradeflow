# Working on TradeFlow

Topic rules live in `.claude/rules/`; the authoritative process document is
[PROCESS.md](PROCESS.md). This file is only what is true in every session.

## The one invariant

**Two clocks.** The research clock (backtest, optimize, walk-forward, analytics, the
AI agent) is slow, exploratory, and only ever *proposes*. The trade clock
(`tradeflow/engine/live.py`, `tradeflow/execution/`) is deterministic and imports
**nothing** from `services/`, `analytics/`, `optimization/`, or `research/`.
Promotion between them is a manual human step.

A change that blurs this is wrong regardless of how well it works.

## Verify, don't assert

The rule that has caught the most real defects here — every one of them looked
correct in the source:

- **Run the thing.** Installing the published package into a clean venv found four
  onboarding dead ends that passed every test. An MCP dependency that every test and
  lockfile agreed was fine shipped broken.
- **Check the environment, not the config.** Confirm an action tag resolves before
  pushing it, a URL serves before linking it, a dependency's *resolved* version
  rather than what the lockfile pinned.
- **Say what you actually verified**, and what you did not.

## Commands

```bash
uv run ruff check . && uv run ruff format --check .   # exactly what CI runs
make test                                             # offline, no keys
make check-links                                      # includes gitignored specs/
make release-check                                    # build + install into a clean venv
```

## Definition of done

Per PROCESS.md §2, none of it optional: right layer · offline deterministic tests via
`tests/fakes.py` · every applicable surface wired (CLI, MCP tool, Makefile) · docs
updated · all four commands above clean.

## Git

- Branch off `main`; never commit to it directly.
- **Commit as you go.** Reconstructing per-change commits afterwards is impossible
  once edits to shared files interleave.
- Messages explain the *why* and any deviation from plan; end with
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
