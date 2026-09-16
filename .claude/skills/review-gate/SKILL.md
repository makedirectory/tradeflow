---
name: review-gate
description: Review a change against this project's gate before it lands — the two-clocks invariant, layering, parity points, numeric correctness, test quality, surface completeness, durability, and what the tool ends up printing. Use after finishing a change and before opening a PR, whenever asked to review a diff or branch here, and on anything touching the order path, the trial store, the ledger, or a surface. The generic code-review skill hunts bugs; this one is the project gate.
---

# The review gate

Review is a quality gate here, not a courtesy. It runs on every change, including your
own, and the invariant angle runs every time — including when the diff looks unrelated
to it.

Read the diff (`git diff main...`) and work the angles below. Each is a question with a
wrong answer, not a box.

## The angles

**Correctness, line by line.** What does this do on the boundary, on an empty input, on
a repeated call? What silently changes for a caller nobody updated?

**The two clocks.** Can research-clock code reach the order path? Does anything pull a
model, optimizer, LLM or database into `engine/live.py` or `execution/`? Does automation
still only *propose*, with a human promoting? Can the MCP server still not construct a
trading client? A change that blurs this is wrong regardless of how well it works. See
[the trade clock](../../rules/trade-clock.md).

**Divergence.** Is what changed implemented somewhere else too, and did the diff change
only one side? Consult [parity points](../../rules/parity-points.md): if the idea is on
that list, both sides belong in this commit; if it is not, ask whether it belongs there.
Two tests that each pass do not establish parity — the test has to build the thing both
ways and compare.

**Layering.** Dependencies point downward, one concern per module, no vendor SDK above
`tradeflow/brokers/`, no business logic in an entry point or an MCP tool handler.

**Numeric correctness.** NaN handling, index alignment, dtype coercion — in indicators,
metrics, and anything walk-forward.

**Test quality.** Offline and deterministic through `tests/fakes.py`; state pointed at
`tmp_path`; each new test watched failing without the fix. A green test sitting over a
real break is this project's most repeated defect. See [testing](../../rules/testing.md).

**Surfaces.** Is every knob reachable from each surface it applies to, or deliberately
absent with the reason written down? A gate that cannot be configured on a surface is
not stricter there — it does not run, and nothing says so. See
[surfaces](../../rules/surfaces.md).

**What it prints.** Reused, memoized, degraded, partial and excluded results have to say
so; a partial run gets no verdict; in-sample results are never ranked as achievements.
See [honest output](../../rules/honest-output.md).

**Durability.** If the change writes a record, can a reader handle every shape ever
written, and is a new field distinguishable as *absent* rather than defaulted? See
[durable records](../../rules/durable-records.md).

**Consolidation.** Is anything here already solved in `services/`, `utils/` or
`analytics/`? Run the consolidation-pass skill rather than noting it and moving on.

## Output

```md
### Findings

1. **[Severity] [Title]**
   - Problem: what is wrong
   - Impact: why it matters
   - Fix: what changed or should change
   - Coverage: the regression test or verification

### Verified clean
- [an important concern checked and found safe — e.g. "live.py still imports nothing from services/"]

### Deferred
- [follow-up] → [tracked issue]
```

| Severity | Meaning |
|----------|---------|
| Critical | Crosses the two-clocks line, loses data, leaks a credential, or breaks a core flow outright |
| High | Serious bug, broken edge case, bad contract, layering violation |
| Medium | Narrower incorrect behavior, test gap, maintainability |
| Low | Cleanup, clarity, minor doc or UX issue |

Findings are fixed before merge or deliberately deferred against a tracked issue.
Nothing is deferred by going unmentioned. Every bug found here gets a regression test
whose docstring says what the failure actually was.

**Say what you actually verified, and what you did not.** "Verified clean" means you
went and looked; an angle you did not work belongs under a heading that says so.

An independent read is available through `/peer-review`. It is advisory: a peer does not
know this project's invariants, so an answer that violates one is wrong here however
sound it looks in general.
