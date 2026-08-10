---
sidebar_position: 6
title: Stopping trading
---

# Stopping trading

Three levels of stop, least to most drastic. Pick by how much you trust the system
right now.

Everything here works whether or not the engine is running. Halt state is a file
under the state root, and `flatten` talks straight to the broker — so a wedged
engine, a hung process, or a machine you have already killed does not stand between
you and stopping.

## Level 1 — halt one strategy

Use when a strategy is behaving oddly but you have no reason to distrust the
*positions*.

```bash
tradeflow halt VolumeSpikeStrategy --reason "entries look wrong since 14:00"
```

- New entries for that strategy are refused.
- Existing positions stay open, and their broker-side bracket legs stay protective.
- Exits still work — see below.

## Level 2 — halt everything

Use when something system-wide looks wrong and you want to think before acting.

```bash
make halt REASON="feed quality"        # or: tradeflow halt all --reason "..."
tradeflow halts                        # what is currently in force
```

Positions remain open. Nothing is closed.

## Level 3 — flatten

Use when you do not trust the system, or a strategy is losing in a way you cannot
diagnose quickly.

```bash
make flatten REASON="unexplained drawdown"
# or: tradeflow flatten --confirm --reason "..."
```

In this order, deliberately:

1. **Halt** — recorded first. Cancelling and closing while an engine is still
   streaming bars is a race the engine can win: it re-enters on the next bar and the
   account refills behind you.
2. **Cancel** every open order.
3. **Close** every position.

Every step is attempted even if an earlier one failed, because a partial flatten
beats stopping halfway and leaving positions open. The report says exactly which
steps succeeded, and the command exits non-zero if any did not.

## A halt never blocks an exit

Halts refuse *entries* only. This is deliberate and worth understanding, because it
is the difference between a switch you will pull and one you will hesitate over:

- A halt that also blocked closing orders would trap the book at exactly the moment
  someone decided to stop.
- `flatten` sets a halt and then closes positions. If the halt blocked exits, it
  would deadlock against its own gate.

## Resuming

```bash
tradeflow resume all                   # or a strategy name
```

Then restart the engine. Nothing resumes on its own — that is the point of the state
being durable.

## After a flatten

1. Verify at the broker that you are actually flat. The report is what this system
   believes; the broker is what is true.
2. `tradeflow halts` — confirm the halt is in force.
3. `tradeflow reconcile` — compare the ledger against the account.
4. Diagnose, fix, then `tradeflow resume all`.

## Practise it

Run a flatten drill on paper, roughly monthly, while actually holding a position:

```bash
tradeflow flatten --confirm --reason "monthly drill"
```

Check that the position closed, `tradeflow halts` shows the halt, and `resume`
clears it. An emergency procedure nobody has run is a procedure nobody knows works —
find out on a day you chose, not on the day you need it.

## What stopping does not do

- It does not close positions you opened by hand outside the engine; close those
  yourself.
- It does not stop the engine process. Use `Ctrl-C` for that.
- It does not undo fills. Slippage is real and irreversible.
