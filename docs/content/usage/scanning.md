---
sidebar_position: 3
title: Scanning the universe
---

# Scanning the universe

The scanner filters a list of *candidate* symbols down to the ones worth trading
at a specific research clock. It runs a `ScannerStrategy` over each symbol's
recent bars and keeps those that produce an actionable scan signal.

```bash
quantred scan --scanner demo_volume --symbols NVDA,META,TSLA,AMD
```

A scan reads recent daily bars, so it needs market data: Alpaca keys from
[`quantred init`](first-run), or a warmed local cache and `--offline` (see
[configuration](configuration)). Nothing is traded, journaled or written.

| Flag | Meaning |
|---|---|
| `--scanner NAME` | Which scanner to run (default `demo_volume`). An unknown name lists the ones installed |
| `--symbols A,B,C` | Candidates to filter (default: the built-in demonstration universe) |
| `--as-of DATE` | Resolve at this date or datetime instead of now (a date is read in New York time) |
| `--drift` | Also resolve at nearby clocks and report how much the universe moves |
| `--cache` / `--offline` / `--cache-dir` | Serve bars from the local cache; `--offline` never touches the network |

Standalone scans default to wall-clock now. Pin them to a historical clock with
`--as-of`:

```bash
quantred scan --scanner demo_volume --symbols NVDA,META,TSLA,AMD --as-of 2024-06-01
```

Example output:

```
SYMBOL    SIGNAL
NVDA      SCANNER_BUY
TSLA      SCANNER_SELL
```

**Reading it.** Each row is a symbol the scanner flagged at that clock. `SCANNER_BUY` and
`SCANNER_SELL` are the scanner's read of the flagged bar (for `demo_volume`, a close above
or at-or-below the open); they select the universe and are not trade orders. A symbol that
is not listed was not flagged. When nothing is, the command prints `No symbols flagged.`
That is a normal answer on a quiet day, and the other commands handle it: `backtest`,
`optimize` and `walkforward` log a warning and use your candidate list as given
(see [below](#how-it-feeds-the-other-commands)). With `--offline` a line above the table
says the universe was resolved from cached bars only, so it is as current as the cache.

**Common errors.**

| Message | What to do |
|---|---|
| `Missing Alpaca credentials: APCA_API_KEY_ID, APCA_API_SECRET_KEY.` | Run `quantred init`, or add `--offline` against a warmed cache |
| `unknown scanner 'x'. Available: ...` | Use a listed name, or `none` to skip scanning |
| `--offline: no cached 1day bars for ...` | Run the `quantred cache warm ...` command the message prints, then retry |

## The demo scanner

The shipped `demo_volume` scanner flags a symbol when its latest bar shows **unusually
high volume** (relative to its moving average) **and** a meaningful price move. It
is pure pandas/numpy — see [Scanners](../engineering/scanners) for the internals.

It is a demonstration, and it arrives the way your own scanner does: declared as an
entry point by the engine's own package rather than hard-coded into a registry. On a
bare install it is the only scanner there is.

## Is the universe stable?

A config records the universe its scanner **resolved**, not the scanner. So if the scan
moves, the book a deployment gets is not the book that was validated — and no promotion
gate would notice, because the gates never see the scan twice.

```bash
quantred scan --scanner demo_volume --drift --as-of 2026-08-22
```

```
=== Scanner drift (18 flagged at 2026-08-22) ===
       clock  flagged  overlap  turnover
         -1d       17       16     11.1%
         -2d       19       14     22.2%
         -5d       15        9     50.0%
  Universe turnover peaks at 50.0% across these clocks.
```

(Illustrative figures.) Turnover is measured against the baseline universe, so "50.0%" reads as *this share of
the universe I am comparing from is not in the other one*. A selection that turns over
half its names across a week is a different object from one that is stable, and which
you have decides how much a single validated run tells you.

This needs no new machinery: `--as-of` already resolves a scanner at an arbitrary
historical clock, so drift is that seam asked several times and differenced.

## How it feeds the other commands

`backtest`, `optimize`, `walkforward`, `research`, `cache warm` and `live` all accept
a `--scanner` option. When set, they run the scanner first and trade only the flagged
symbols; if the scanner flags nothing, they log a warning and fall back to the candidate
list. Pass `--scanner none` to skip scanning and use the symbols as-is.

`allocate` is the exception: it does not fall back. When the scanner flags none of the
candidates it prints that and stops, suggesting `--scanner none` to allocate across the
candidates as given.

The historical commands resolve the scanner at `--end` by default, so a validation
window does not mix in today's universe, and `--scan-as-of` pins a different scanner
clock. **`live` is the exception, and deliberately so:** it resolves at wall-clock
now, because a live book is selected from the universe as it stands, not as it stood
at the end of some window. It therefore takes no `--scan-as-of`.

Every scan reports the clock it resolved at, in the exchange zone — including the ones
that resolved to "now", so a payload never leaves which universe it selected from as
something to be inferred.

```bash
quantred backtest --scanner none --symbols AAPL,MSFT --start 2024-01-02 --end 2024-04-01
```

## Next

Scan, then judge the universe: [`verdict`](verdict) asks whether the signal has any
information in it, [`backtest`](backtesting) replays it, and
[`walkforward`](walk-forward) validates it out of sample. Scanner internals and how to
write your own are in [Scanners](../engineering/scanners).

## From a source checkout (contributors)

The commands above are the installed `quantred` command. In a checkout, `uv run quantred <command> ...` runs the same thing (`uv run python main.py <command> ...` also works). The Makefile shortcut for this page is `make scan`. Extras are installed with `uv sync --inexact --extra <name>`
(see [optional extras](extras)).
