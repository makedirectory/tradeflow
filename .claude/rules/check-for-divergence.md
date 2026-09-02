# Check for divergence before you finish

Every change that touches a behaviour reachable from more than one place carries the
same question: **is this idea implemented somewhere else too, and did I change only one
of them?** Ask it deliberately, because the failure mode gives no signal — two
implementations of one idea drift apart while both read correctly and every test stays
green.

[Parity points](parity-points.md) is the list, the preferred shape (delegate, don't
re-implement), and how to test one. This rule is the obligation to consult it.

## The check

Before calling a change done:

1. **Is what I changed on the list?** If yes, change both sides in the same commit. A
   fix that lands on one side and not the other is worse than the original defect: the
   two now disagree, and the disagreement is what nothing catches.
2. **If it is not on the list, does it belong there?** The question is not "did I
   duplicate something" but "is this idea *also* expressed somewhere that cannot import
   me". Two clocks that must not import each other, two transports over one service, an
   installed copy and a checkout, a writer and its reader, a value and the identity it
   is keyed under. Any of those is a candidate.
3. **Then converge or record.** Prefer one definition reached two ways. Where that is
   genuinely impossible, add the pair to the list with what it costs when it drifts,
   and guard it with a test that builds the thing both ways and compares — not two
   tests that each pass.

## What makes it worth doing every time

Every entry on that list was written after the divergence shipped, not before. The
book limits reached one dedup key and not the other. `lifecycles()` and `_replay` read
the same `basis` field two different ways. A CLI flag had no MCP equivalent, so a
promotion gate silently never ran on that surface. In each case one side was changed
carefully and correctly, and the other was simply not looked at.

The check costs a minute. Finding a parity bug afterwards means distrusting every
result produced in between, because a green suite never proved they agreed.
