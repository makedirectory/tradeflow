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
python main.py trials list --strategy demo_trend --min-sharpe 1.0 --sort dsr --limit 20
```

```
ID            KIND        STRATEGY           SHARPE     DSR  PROMO  ACCT  TS
a1b2c3d4e5f6  walkforward demo_trend        1.512   0.903    yes     3  2025-03-01T09:14:22
b2c3d4e5f6a1  backtest    demo_trend        2.104   0.201     no     3  2025-02-01T11:02:07
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
folded `_cost`, `_limits` and data-vintage keys that make up its dedup identity —
`_limits` is the book the run was given, so two otherwise identical runs at different
position caps are different trials rather than one served twice), provenance
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
  strategy 'demo_trend'  universe 61 symbols resolved from 85 candidates
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
python main.py trials best --strategy demo_trend
```

```
  Top 5 by deflated Sharpe:
    #  ID            STRATEGY             DSR   SHARPE    FAMILY n_trials
    1  a1b2c3d4e5f6  demo_trend       0.903    1.512                 87
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

## `trials analyze` — what did those trades actually do?

```bash
tradeflow trials analyze a1b2c3d4e5f6
tradeflow trials analyze a1b2c3d4e5f6 --json
```

Exit-reason P&L, win and loss by reason, holding period, and per-trade excursion for
one recorded run — the questions that otherwise mean opening SQLite.

```
=== Trades of a1b2c3d4e5f6 (backtest) ===
  strategy demo_trend | accounting v5 | recorded 2026-03-01T09:14:22
  table    : 40 trades

  Overall:
    Trades          40
    Wins / losses   28 / 12
    Win rate        70.0%
    Net P&L         $8,483
    Average win     $409
    Average loss    -$247
    Expectancy      $212
    Profit factor   3.86

  By exit reason:
    exit              trades   share       net P&L  win rate     avg win    avg loss
    TAKE_PROFIT           26   65.0%        $4,528     65.4%        $431       -$312
    STOP_LOSS             14   35.0%        $3,955     78.6%        $374        -$54

  Held     : median 6.5 days, p25 5.8, p75 7.2, max 8.0 (40 measured, 0 not)
  Excursion: per-trade excursion — not the book's aggregate open drawdown
    adverse     median 2.84%, p90 3.85%, max 5.23% (40 measured, 0 not)
    favourable  median 4.88%, p90 7.06%, max 9.04% (40 measured, 0 not)
```

*(Illustrative figures.)* It grades nothing — the register is the one
[`execution-report`](validation-diagnostics) sets: report the number, say what it does
not cover, leave the judgement alone.

Three things it will not do:

- **Sum a capped table.** If the run's trades were truncated at the storage ceiling,
  there are no totals and the command exits non-zero saying why. `--allow-partial`
  computes them anyway, labels every number as covering the stored rows only, and
  still draws no concentration verdict — which exit carried a run is a claim about all
  of its trades.
- **Report a missing column as a zero distribution.** A table with no `entry_time`
  says the holding period is not computable from it, rather than reporting that trades
  lasted no time.
- **Confuse per-trade excursion with the book's.** A position deep underwater that is
  a small fraction of the book did not put the book that far underwater. The label is
  on every excursion figure because that conflation is what makes people rewrite a
  result they had measured correctly.

## `trials compare` — are these two results one result?

```bash
tradeflow trials compare a1b2c3d4e5f6 b2c3d4e5f6a1
```

```
=== Return-series comparison ===
  a1b2c3d4e5f6     300 periods, 2024-01-02 → 2024-10-27  (v5, demo_trend)
  b2c3d4e5f6a1     300 periods, 2024-01-02 → 2024-10-27  (v5, demo_trend)

  a1b2c3d4e5f6 vs b2c3d4e5f6a1  overlap 300 periods, 2024-01-02..2024-10-27
      correlation +0.98 [+0.98, +0.99]

  1 of 1 pair(s) compared; 0 refused. Minimum overlap 60 periods.
  Highest: a1b2c3d4e5f6 vs b2c3d4e5f6a1 at +0.98.
  At that level they are one bet held twice, however differently they were
  parameterised — promoting both counts a single result as two.
```

*(Illustrative figures.)* Two candidates that correlate near 1.0 are one candidate. A
campaign that promotes both believes it has two findings and has one.

**Pairs get refused, not caveated.** A correlation is a claim about a relationship and
there is no partial version of one:

| Refusal | Why |
| --- | --- |
| Fewer than `--min-overlap` shared dates (default 60) | A correlation over a handful of dates is an error bar wearing two decimals |
| Different accounting versions | The two series came from engines that compute different things. `--across-accounting` computes it anyway and marks the pair incomparable |
| Either series not recorded | Not every trial kind persists one |
| Either series flat over the overlap | Nothing to correlate |

Every correlation carries a 95% interval, so one resting on a thin overlap arrives
visibly wide rather than merely short of decimals. In `--json`, read `pairs` alongside
`matrix`: the matrix holds `null` where nothing was computed, because a zero there is
the strong claim that two results move independently — exactly what a refusal cannot
say.

Both commands are also MCP tools (`analyze_trial`, `compare_trials`), and both journal
nothing.

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
reconstructs the table from it alone.

### A stored table says whether it is all of them

Very large tables are capped at a storage ceiling, and the cap travels with the rows.
A trade table has four states and they never render alike:

| What `trials show` prints | What it means |
| --- | --- |
| `— (not recorded — pass --record-trades on the run)` | The run did not opt in. Nothing is known about its trades |
| `0 trades` | It opted in and made none |
| `1,204 trades` | All of them |
| `5,000 of 18,432 trades — TRUNCATED at the storage cap…` | A prefix. Any total over these rows is short by the rest |

A fifth reads as *whether that was all of them was not recorded*: a table stored
before the count was kept. It did not prove completeness, so it is never rendered as
though it had.

The distinction is not cosmetic — everything that aggregates a stored table (exit-reason
P&L, win rates, duration) sums exactly these rows, and a capped table that looks complete
turns a partial sum into a confident wrong number. `--trades-limit` truncates the *view*
and reports itself separately (`display_truncated` in the JSON); that one is undone by
raising the flag, and the storage cap is not undone by anything.

## Maintenance: quarantine a subset, or retire an era

Two commands, deliberately not one. They answer different questions and differ in how
reversible they are, and collapsing them is a mistake in both directions — quarantining
an era leaves thousands of rows carrying a caveat nobody can act on, archiving a subset
throws away the evidence around it.

### `trials mark-contaminated` — a suspect subset

For when something is learned about *some* trials after they were recorded: a scanner
that used a stale universe, a vendor's data correction, a config that turned out not to
be the one you thought.

```bash
tradeflow trials mark-contaminated --reason "vendor split adjustment was wrong"     --strategy demo_trend --before 2026-08-01 --dry-run
tradeflow trials mark-contaminated --reason "vendor split adjustment was wrong" --id abc123def456
```

`--reason` is required. Rows excluded from every leaderboard with nothing saying why
cannot be judged later, which is worse than not excluding them.

**Nothing is rewritten.** The quarantine is an appended event naming the affected
trials; the trial records stay exactly as written. `trials rebuild` replays the event
and reaches the same rows, so the quarantine survives the store being thrown away — which
it is, routinely, at every schema bump.

What it changes, and what it deliberately does not:

| | Quarantined trial |
|---|---|
| Served as a memo for an identical run | **No** — a fresh run is always safe, a suspect number never is |
| Ranked in `trials best` | **No**, and the excluded count is reported |
| Counted toward the family's multiple-testing total | **Yes** |

That last row is the one that could most easily have gone the flattering way. The search
still happened — you did look at that configuration — so dropping it would *lower* the
deflated-Sharpe bar for the family, and this store never moves that bar down on its own.

### `trials archive` — a whole era

For when everything recorded becomes incommensurable rather than suspect. An accounting
bump changes what the engine computes, so every stored metric was measured with a
different instrument; that is not a subset and no annotation fixes it.

```bash
tradeflow trials archive --reason "accounting v5 invalidated v4 metrics" --label pre-v5 --dry-run
tradeflow trials archive --reason "accounting v5 invalidated v4 metrics" --label pre-v5
tradeflow trials archives     # what has been retired, and why
```

**Both files move together, always.** The store is an index over the journal, so moving
one alone leaves the other describing something that is not there — or an era's rows
sitting beside a fresh journal, still reporting a multiple-testing count for evidence
that is gone, with nothing erroring because both files are individually valid. That is
the hand-rolled state this replaces.

Each archive keeps a manifest (what, when, why, which accounting version and commit), and
the **new** journal opens with a record of the archive, so its emptiness reads as "an era
was retired here" rather than "nothing has ever been run".

### There is no `reset`

Deliberately. The destructive version of archive is what the append-only rule forbids:
the journal has nothing behind it to rebuild from, so a record removed is a record gone.
Archive moves the files; they are still on disk under a name that says when and why.

### Both are CLI-only

Neither is an MCP tool, and that is a decision rather than an omission. Quarantining
evidence and retiring an era are operator decisions about a campaign's record, not run
configuration: one changes what every later leaderboard and memo reports, the other moves
your files. An agent that believes a trial is contaminated should say so and let you act.

## Index health, and how it repairs itself

The database is **derived**. The journal is the record; the index is a cache over it, and
deleting the file loses nothing that `trials rebuild` cannot put back.

That is what makes the repair safe. Every time the store is opened it compares the tables
actually in the file against the schema this build declares, and rebuilds from the journal
if they differ — so a store written by an older version gets the columns a newer one
needs, rather than reporting the new version over the old shape. A version stamp alone
could not catch that: the stamp is written by the rebuild, and the old rebuild emptied the
tables instead of recreating them.

One case it will not repair, on purpose:

```
$ tradeflow trials rebuild
418 recorded trial(s) are indexed here but the journal they came from
(~/.tradeflow/logs/research_journal.jsonl) cannot be read. Rebuilding would replace
them with an empty index and the journal is the only other copy. Restore the journal,
or point --journal at the one this store indexes.
```

Replaying an absent journal produces an empty index, and an empty index reads as a
campaign that tried nothing — which would quietly lower the deflation bar every one of
those trials paid for. `trials status` reports any schema mismatch it could not fix, and
reports a quarantine count it cannot read as unknown rather than as zero.

## Not covered here

Retention and pruning are deliberately out of scope — append-only history is the
point. `trials status` and `trials rebuild` handle index health, and the maintenance
commands above never delete; see the [walk-forward](walk-forward) page for what the
store's campaign count feeds.
