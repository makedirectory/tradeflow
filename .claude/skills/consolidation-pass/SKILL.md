---
name: consolidation-pass
description: Sweep a finished change for duplication against services/, utils/ and analytics/ before it merges — route through the existing helper, extract a new one only when a second caller needs it, and confirm the import graph still points downward. Use after a change works and before opening the PR, and whenever you are about to write a second implementation of something that sounds familiar.
---

# The consolidation pass

Run this when the change works, before it lands. It is the step that decides whether
this codebase ends up with one definition of an idea or two — and two is how the
defects here are shaped: both copies read correctly, every test passes, and they
disagree.

## The pass

1. **Scan the new code for duplication** against `services/`, `utils/` and
   `analytics/`. The mechanical check finds names defined in two modules:

   ```bash
   rg -N '^\s*def ([a-zA-Z_]\w*)' -or '$1' tradeflow --sort path | sort | uniq -d
   ```

   A duplicate name is either a delegate (fine) or a parity point (not fine yet).

2. **Route through the existing helper** where one exists. The preferred shape is
   `cli.resolve_universe`: four lines that call `services.data.resolve_universe`. One
   definition, reached two ways.

3. **Extract a new helper only when a second caller actually needs it.** Premature
   abstraction is also debt, and single-caller domain logic belongs where it is. The
   trigger for extraction is a real second caller, the same operational code appearing
   twice, or a bug fix that should propagate everywhere doing the same thing.

4. **Confirm the import graph still points downward** with no new cycle. Keep the
   neutral modules (`utils/`, `analytics/`) free of imports from heavy domain graphs.

5. **Run the gates** — `uv run ruff check .`, `uv run ruff format --check .`,
   `make test`.

6. **Record in the PR** what was extracted, which callers moved, and which test proves
   the behavior held. Log intentional deferrals against a tracked issue.

## When delegation is impossible

Sometimes it genuinely is: the trade clock cannot import the research clock, the CLI and
the MCP server are separate transports, an argparse namespace is not a function
signature. Then the duplication is a decision rather than an accident, and it gets
written down — add the pair to [parity points](../../rules/parity-points.md) with what
it costs when the two drift, and guard it with a test that **builds the thing both ways
and compares**. Two tests that each pass are exactly what a parity bug looks like.

`limits_key` exists because this check was skipped once: a trial recorded over the CLI
stopped being found over MCP.
