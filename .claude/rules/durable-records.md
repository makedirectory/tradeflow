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

### Open weakness

The position ledger has **no version marker of any kind**. Every compatibility decision
in it is per-field improvisation, which has worked twice and is not a system. If a third
field needs it, add a record-level version first.

## Paths are part of the format

`state_root()`, always — never a relative path. The multiple-testing correction rests on
one journal, and a campaign split across two roots deflates against half its evidence
while nothing errors. The journal's location is currently defined in two modules and
kept in sync by a comment; see [parity points](parity-points.md).
