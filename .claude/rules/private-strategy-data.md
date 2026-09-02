# Private strategy data stays out of the repository

This repository is public. The strategies and scanners it is run against are not, and
they live in a separately installed package. Nothing about them belongs in anything
committed here.

## What must never be committed

- **Names.** Private strategy and scanner identifiers, and the package they come from.
  Examples in docs use a clearly fictional package.
- **Numbers from a real candidate.** Its capital, universe size, position count, caps,
  returns, Sharpe, or the symbols it traded. A sample output that reproduces a real
  run's contract describes that candidate to anyone who reads it.
- **The combination.** Individually harmless figures — a capital, a universe size, a
  set of caps — identify a candidate when printed together as one block, which is
  exactly the form sample output takes.

This covers code, tests, docs, fixtures and commit messages equally. `specs/` is
gitignored and local-only, so it may hold real numbers; nothing else may.

## What to write instead

Illustrative values that resemble no actual book, labelled as illustrative where a
reader might otherwise take them for measured. Keep the *shape* — the point of a sample
is the format and the reasoning, never the figures.

If a real number is doing genuine explanatory work, describe the effect without the
value: "a per-position ceiling far above the capital in play" rather than the pair.

## Why it is worth the friction

The reasoning in these docs is the valuable part and is meant to be public. The
candidate is the part that is not. Fabricating a decay curve to sit under a real
headline number, which happened once, is worse than either alone: it publishes the
candidate *and* presents invented data as measured.

Before committing, grep the diff for the private package name and for any figure you
recognise from a run.
