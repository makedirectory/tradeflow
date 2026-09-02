---
paths:
  - "tradeflow/engine/**/*.py"
  - "tradeflow/execution/**/*.py"
  - "tradeflow/strategies/base.py"
  - "tradeflow/cli.py"
---

# Cross-clock parity

The two clocks are deliberately disconnected — the trade clock imports nothing from the
research clock, and that is the invariant. The cost of it is that **the same idea is
implemented twice**, in code that cannot reference itself. A limit, a sizing rule, a
fill assumption, a cost parameter: each exists on both sides, written separately.

So a defect found on one side is a defect on the other until proven otherwise.

## The rule

**When a change touches something that exists on both clocks, check and test both.**
Not "the other side probably does it right" — go and look. A wrong parameter in the
backtest is almost certainly the same wrong parameter live, and the code being
disassociated is exactly why nobody notices.

This applies to a fix, a rename, a default, and a new option alike. If you are adding a
knob to one clock, the question is not whether the other clock needs it too; it is
whether the other clock is now silently different.

## What "both" means

- `tradeflow/engine/backtest.py` and `tradeflow/execution/live_trader.py` — the two
  places a book is sized, admitted, limited and exited.
- `tradeflow/engine/live.py` for anything about the loop, the book, or reconciliation.
- Both CLI surfaces. A flag that only one command accepts is a difference in what can
  be expressed, which becomes a difference in what gets run.

## Evidence this is not hypothetical

Every one of these shipped, passed the full suite, and was found by running the thing:

- `max_total_risk` was enforced across the book in the backtest and per-position live,
  so a config validated at a 5% budget could run unbounded.
- Book limits were settable from the command line on `live` and not on `backtest`, so
  the only way to test a deployable cap was to edit the config the test was about.
- `max_net_exposure` was added to both gates in one change, deliberately, and the
  parity is what makes the derived cap mean anything.

## How to satisfy it

Write the test on both sides, in the same change. Two tests asserting the same property
against two implementations is not duplication here — it is the only thing that can
catch the two drifting, because nothing else in the codebase connects them.

State in the commit which clocks you checked. "Checked live only, backtest has no
equivalent" is a fine answer; silence is not.
