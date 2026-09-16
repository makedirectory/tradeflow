---
description: Walk the definition of done for the current change and report what is not met
argument-hint: [what the change is, if the diff does not make it obvious]
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(uv run ruff:*), Bash(make test), Bash(make check-links), Bash(make release-check), Read, Grep, Glob
---

Check whether the change in the working tree and on this branch is actually done, per
[build-process](../rules/build-process.md). Report; do not fix. `$ARGUMENTS` is what the
change is meant to be, if given.

Run the four gates and report each result verbatim — a failure keeps its output:

```bash
uv run ruff check . && uv run ruff format --check .
make test
make check-links
make release-check          # only if packaging, the CLI entry points, or docs shipped in the distribution changed; say if you skipped it and why
```

Then work the checklist, and for each item say **met**, **not met**, or **not checked** —
never infer one from a passing gate:

- **Layer.** Does the logic sit in the right layer, with dependencies pointing downward
  and no vendor SDK above `tradeflow/brokers/`?
- **Two clocks.** Does `engine/live.py` or `execution/` import anything from `services/`,
  `analytics/`, `optimization/` or `research/`? Does automation still only propose?
- **Tests.** Offline and deterministic through `tests/fakes.py`, state at `tmp_path`, and
  a regression test for every bug this change fixed. Name any new test you did not watch
  fail without the fix.
- **Parity.** List what the diff touches that appears on
  [parity points](../rules/parity-points.md), and whether both sides changed in the same
  commit. If something belongs on that list and is not there, say so.
- **Surfaces.** Every applicable surface wired — CLI, MCP tool, Makefile target — or an
  absence with a stated reason.
- **Docs.** Engineering wiki for behavior, usage guide for how to run it, README if the
  headline workflow changed. Quote the lines that changed rather than reporting that a
  docs commit exists.
- **Commits.** On a branch off `main`, not on `main` itself, with the work committed as
  it went.

Finish with the PR description block, filled in from what you found — scope, verification
(the actual results above, including any manual paper smoke test), docs, how the
two-clocks invariant stayed intact, what review fixed, and what is deferred and where.

Anything you could not check gets named as unchecked. Do not report the change as done
on the strength of a green suite.
