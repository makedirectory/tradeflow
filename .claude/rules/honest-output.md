---
paths:
  - "tradeflow/analytics/**/*.py"
  - "tradeflow/services/**/*.py"
  - "tradeflow/store/**/*.py"
  - "tradeflow/cli.py"
  - "tradeflow/mcp/**/*.py"
---

# Honesty rules for anything the tool reports

The project's whole value is refusing to flatter a result. Everything it prints
inherits that.

- **Never collapse disagreement into reassurance.** When gates disagree, say "mixed"
  and show every check with its value and threshold. A single soothing number is a
  bug, not a summary.
- **Label what a number is.** A search's winning candidate and a validated
  out-of-sample result are identical as numbers and opposite as evidence. Anything
  ranking or comparing results must show the kind.
- **Never rank in-sample results as achievements.** An `optimize` row is best-of-N by
  construction — it *is* the selection bias. Exclude it from leaderboards by default.
- **Reused, memoized, degraded, and partial results say so prominently**, with the
  original's age where one exists.
- **Silence about what was dropped reads as "everything ran."** Report skipped,
  truncated, failed, and excluded counts even when they are zero-worthy.
- **A partial run gets no verdict.** Any failed step means no overall answer,
  whatever the completed sections show.
