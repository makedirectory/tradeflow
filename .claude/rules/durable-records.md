---
paths:
  - "tradeflow/store/**/*.py"
  - "tradeflow/services/audit.py"
  - "tradeflow/execution/ledger.py"
---

# Durable records

Three things on disk outlive the code that wrote them, and they do not get the same
treatment.

## Derived: rebuild, never migrate

**The trial store** (`store/trials.py`) is a cache of the journal. `SCHEMA_VERSION`
carries a per-version note, and a mismatch **rebuilds from the journal** rather than
running migration code — which is cheap precisely because the journal is the truth.

Bump it when the schema changes shape. Never write anything into the store that is not
recoverable from the journal, or the rebuild silently loses it and nothing errors.

### Retiring and quarantining, without rewriting

Two maintenance operations exist and neither edits a record.
`services.maintenance.mark_contaminated` appends a `trials:contaminated` event naming the
suspect trials; the derived index sets a column by replaying it, so a rebuild reproduces
the quarantine from the journal like every other column. `services.maintenance.archive`
moves the journal and its index **together** for a whole-era break, and writes a manifest
plus a first line in the new journal saying what was retired and why.

Together is the requirement. Moving one file alone leaves a store indexing a file that is
not there, or an era's rows beside a fresh journal reporting a multiple-testing count for
evidence that is gone — and nothing errors, because both files are individually valid.

There is deliberately no `reset`. A record removed is a record gone, which is what this
whole section forbids. Archive moves; the files are still on disk under a name that says
when and why.

A quarantined trial is never served as a memo and never ranked, and **still counts**
toward its family's multiple-testing total. The search happened; dropping it would lower
the deflation bar, and that is the one direction this store must never move on its own.

## Authoritative: every shape you ever wrote, forever

**The research journal** and **the position ledger** are append-only and have nothing
behind them to rebuild from. A record written badly in March is still there in
September, and no migration can reach it — so the *reader* has to handle every shape the
writer has ever produced, for as long as the file exists.

That is a much harder contract than a schema version, and it is where this project has
actually been bitten:

- Fill quantities were written cumulatively and read as if incremental. An order that
  filled 8 arrived as 21, reconciliation diverged on every symbol, and the wrong number
  had already been serving a live session.
- Refusals were grouped by their message, which embeds the amounts that caused them, so
  one throttle read as sixteen distinct events.

Both were fixed by adding a field. Neither could fix the records already written.

### The rules that follow from that

**A new field must be distinguishable as absent.** Not defaulted — *absent*. When
`basis` was added to fills, a missing `basis` had to mean "written before we knew", not
"incremental", because the second is a guess that reads exactly like a fact.

**Absence gets reported, never silently reinterpreted.** A ledger holding pre-fix
records says so at reconciliation instead of quietly producing a number. A reader that
guesses is worse than one that refuses, because nothing downstream can tell.

**Mark the shape when you change what a field means.** A value whose interpretation
changed and whose name did not is unreadable afterwards by anyone, including you.

**A best-effort backfill is a read-path concern and must be partial on purpose.**
Pre-code refusals are recognised by message prefix; a message the map does not know
keeps its own text rather than being forced into a family it may not belong to.

### The version is the thing that stops the improvising

`LEDGER_VERSION` is stamped on every record the ledger writes. A reader asks what shape
it is looking at rather than inferring it from which keys happen to be present, and
`version_summary()` counts the shapes in a file so an operator can decide whether to
archive rather than guess.

Bump it when the meaning of a field changes or a reader would need to behave differently.
Absent means "written before this existed" and stays permanently readable as that — the
pre-version records cannot be recovered and are reported, never reinterpreted.

## Paths are part of the format

`state_root()`, always — never a relative path. The multiple-testing correction rests on
one journal, and a campaign split across two roots deflates against half its evidence
while nothing errors. The journal's location is currently defined in two modules and
defined once in ``settings.trial_journal_path()``, which is the layer both the writer
and the indexer already depend on — defining it in either of them would be a cycle.
