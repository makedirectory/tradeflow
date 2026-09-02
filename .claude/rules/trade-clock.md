---
paths:
  - "tradeflow/engine/**/*.py"
  - "tradeflow/execution/**/*.py"
  - "tradeflow/brokers/**/*.py"
---

# The trade clock

This is the only code that can lose money. Different rules apply.

The separation costs something, and [cross-clock parity](cross-clock-parity.md) is how
that cost is paid: the same idea is implemented twice, so a defect on one side is a
defect on the other until checked.

- **Import nothing from the research clock** — not `services/`, `analytics/`,
  `optimization/`, or `research/`. A structural test asserts this; keep it true.
- **No vendor SDK above `tradeflow/brokers/`.** The broker layer is the only place
  `alpaca` is imported.
- **Guards reject; they never repair.** No interpolation, gap-filling, or correction
  of an input. The moment the live path fixes its inputs it stops being the thing the
  backtest validated, and every historical result quietly stops describing what will
  happen.
- **A guard that rejects real data is worse than no guard.** Thresholds default
  loose, every rejection is logged with the offending values, and an elevated
  rejection rate is reported — a guard silently eating a third of the feed looks
  exactly like a quiet market.
- **Report; never remediate.** Detection is the feature. Automatically correcting a
  financial position is how one gets doubled unattended.
- **Bookkeeping never breaks the order path.** Ledger and reconciliation failures are
  logged and swallowed.
- **Bounded work per bar.** No operation that scales with universe size in the loop.
