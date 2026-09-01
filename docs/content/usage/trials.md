---
sidebar_position: 13
title: Browsing the trial store
---

# Browsing the trial store

The trial store is the project's memory: every backtest, optimization,
walk-forward, and verdict across every session, with provenance, dedup identity,
and each trial's out-of-sample return series. `trials list`, `trials show`, and
`trials best` are how you ask it questions without writing SQL.

It is deliberately a **CLI browser, not a UI** — the store is SQLite, the queries
are cheap, and what was missing was filters, a detail view, and honest
presentation.

Everything here is **read-only**. There is no delete, no edit, no annotate: the
journal is append-only history, and the browser must not make corrupting it
convenient. (`trials rebuild` already exists to reconstruct the index from the
journal.)

## `trials list` — what have we tried?

```bash
python main.py trials list --strategy volume_spike --min-sharpe 1.0 --sort dsr --limit 20
```

```
ID            KIND        STRATEGY           SHARPE     DSR  PROMO  ACCT  TS
a1b2c3d4e5f6  walkforward volume_spike        1.512   0.903    yes     3  2025-03-01T09:14:22
b2c3d4e5f6a1  backtest    volume_spike        2.104   0.201     no     3  2025-02-01T11:02:07
…

Showing 20 of 4318 matching trials (use --limit/--offset for more).
```

| Flag | Meaning |
|---|---|
| `--strategy` / `--kind` | Filter to one strategy, or one kind (`backtest`, `optimize`, `walkforward`, `alpha`, `research`, `verdict`) |
| `--symbols` | Filter to one universe — matched on the **normalized** universe, so `NVDA,MSFT` and `msft, nvda` find the same trials |
| `--since` / `--until` | Filter by when the trial was recorded |
| `--min-sharpe` | Only trials with a recorded OOS Sharpe at or above this |
| `--gates-passed` | Only trials recorded as promotable |
| `--sort date\|sharpe\|dsr` | Sort order (default `date`) |
| `--limit` / `--offset` | Paging — done in SQL, so a store with tens of thousands of rows stays fast |
| `--accounting` / `--all-accounting` | Stay within one accounting version (default), or span them deliberately |
| `--json` | Emit the rows as JSON |

Two rules the listing follows:

- **Absent is not zero.** A trial from before a field existed, or of a kind that
  never produces one, renders as `—`. A pre-return-series trial did not *fail* to
  persist returns, and a forecast did not score a Sharpe of 0.00.
- **Truncation is never silent.** The footer always says how many rows matched
  versus how many were shown.

`trials query` still works as an alias for `trials list` for one release.

## `trials show` — what did trial X actually run?

```bash
python main.py trials show a1b2c3d4e5f6
```

Prints everything the store knows about one trial: its full params (including the
folded `_cost` and data-vintage keys that make up its dedup identity), provenance
(git SHA, timestamp, accounting version), headline metrics, and what was stored
alongside it —

```
  Stored alongside this trial:
    return series : 361 periods, 2024-01-27 → 2025-01-21
    proposed book : 4 names, with factor exposures
    trade table   : — (not recorded — pass --record-trades on the run)

  Reused by 2 later trial(s) with the same identity:
    c3d4e5f6a1b2  2025-04-02T08:11:40  (backtest)
```

That last section is the reverse of the memoization lookup: a number served from
the store is traceable in both directions, from the reuse back to its origin and
from the origin forward to everything it was reused for.

An unknown id exits non-zero with a plain message.

## Promoting a validated trial

`walkforward --save-config` writes the chosen config *after* a validation, so saving a
config you have already validated means validating it again — and the memo only serves
an identical recipe, which it is not once a seed has changed to ask a different
question.

`trials promote` reads the recorded trial instead:

```bash
tradeflow trials promote 50fd06209f49 --save-config configs/alpha.json
```

```
Promoted trial 50fd06209f49 -> configs/alpha.json
  strategy 'ma_crossover'  universe 61 symbols resolved from 85 candidates
  Saving a config never trades it - a human promotes it to live.
```

**This is not a fast path through validation.** The trial store holds what a config
needs *because a real validation put it there*, so reading from it cannot bless state
that was never validated. A `--skip-validation` flag could, and would be reached for
exactly when someone is in a hurry — which is when it matters most that a saved config
means what it says.

A trial that did not clear its gates is **refused**, because promoting one silently
would put a config on disk whose own provenance says it failed. `--force` saves it
anyway with that verdict recorded in the file.

The universe comes from the **journal**, not the store: the store records a universe
*hash*, not the symbols. That keeps the store passive and derived, and means a trial
journaled before candidate lists were recorded reads as *less complete* rather than
being backfilled into looking more authoritative than it is — such a config says so,
and `--re-resolve-universe` will tell you it has no candidates to re-scan.

## `trials best` — the honest leaderboard

```bash
python main.py trials best --strategy volume_spike
```

```
  Top 5 by deflated Sharpe:
    #  ID            STRATEGY             DSR   SHARPE    FAMILY n_trials
    1  a1b2c3d4e5f6  volume_spike       0.903    1.512                 87
    …

  Ranked by DEFLATED Sharpe, which already discounts for how many configs the
  family tried. Each row's family n_trials is shown: the larger it is, the more
  of the leader's edge is selection.

  This family has tried 87 configs. At that count the best raw Sharpe is largely
  selection — read the deflated column, not the rank.
```

**A leaderboard is the most dangerous view in the project.** Sorting a research
campaign's trials by raw Sharpe and showing the winner is precisely the
selection-bias trap the evaluation machinery exists to fight — and it would be
worse coming from our own tooling, which lends it authority. So:

- Ranking is by **deflated** Sharpe by default.
- **In-sample rows are excluded.** An `optimize` row is the winner of a search —
  best-of-N by construction — so ranking one ranks the selection bias rather than
  any skill. `--include-in-sample` opts back in and says plainly what that means;
  either way the count of excluded rows is reported, so the exclusion is never
  silent.
- **Every row names its kind**, because a validated walk-forward and a search's
  winning candidate look identical as numbers and mean opposite things as evidence.
- Every row shows its family's `n_trials`, and a large count gets its own warning
  line.
- `--rank-by sharpe` is available and prints a caveat saying, in as many words,
  that the ordering does not correct for how many configs were tried.

These rules live in the **payload**, not only in the terminal formatting: `--json`
carries the family counts and the caveat too, so an agent reading this over MCP
sees the same context a human reads on screen.

## Keeping trade tables (`--record-trades`)

`backtest` and `walkforward` accept `--record-trades`, which journals the run's
trade table (entries, exits, per-trade P&L, costs, MAE/MFE) with its trial so
`trials show` can render it.

```bash
python main.py backtest --symbols NVDA,AAPL --start 2024-01-01 --end 2024-12-31 --record-trades
```

It is **opt-in** on purpose: a long optimization campaign multiplying thousands of
candidates by hundreds of trades each is exactly the storage nobody asked for. Use
it for the runs you intend to open again. Without the flag, the trial row is
identical to what it would otherwise have been, and `show` reports the trade table
as *not recorded* — never as zero trades.

Persistence follows the same journal-first pattern as the return series and the
proposed book: the journal is the source of truth, and `trials rebuild`
reconstructs the table from it alone. Very large tables are capped, and hitting
the cap is recorded on the payload rather than silently truncating.

## Not covered here

Retention and pruning are deliberately out of scope — append-only history is the
point. `trials status` and `trials rebuild` handle index health; see the
[walk-forward](walk-forward) page for what the store's campaign count feeds.
