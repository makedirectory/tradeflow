---
sidebar_position: 11
title: Validation diagnostics
---

# Validation diagnostics

Statistical validation and tradeability are separate questions, and clearing one says
nothing about the other. A walk-forward can hand you a deflated Sharpe that clears every
gate on a result that is entirely an artefact of one fill assumption.

These are the tools for interrogating a result you already believe. **None of them
gates anything.** They report numbers and leave the judgement where it belongs.

## Where the P&L came from

Printed by every backtest, no flag required.

```
=== Where the P&L came from ===
  exit              trades   share       net P&L
  TAKE_PROFIT         1130   57.9%$      581,923
  SIGNAL               787   40.3%$     -118,444
  STOP_LOSS             30    1.5%$      -47,112
  Nearly all of the gain comes from TAKE_PROFIT. The result is a bet on that exit's
  fill assumption - stress it before believing the headline.
```

A headline return says nothing about where it came from. A book whose entire gain
arrives through one exit path is a bet on that path's fill assumption, and nothing in
the summary metrics distinguishes it from one whose edge is spread across exits.

## Take-profit fill stress

```bash
tradeflow backtest --config configs/breakout.json --start ... --end ... --fill-stress
```

The engine closes a position at its target the moment a bar's high reaches it — a
single print at the level is enough. That models a resting limit order always first in
the queue, which is the most generous reading available.

`--fill-stress` re-runs requiring the price to trade progressively further *through*
the target before the exit counts as filled:

```
=== Take-profit fill stress ===
    through by    Sharpe    return   trades
    touch only      1.84     62.40%      420
         5 bps      1.61     51.80%      398
        10 bps      1.44     43.20%      377
        25 bps      0.96     22.10%      321
        50 bps      0.31      4.70%      248
  Edge survives requiring 50 bps through the target.
```

The trigger tightens; the fill price does not — a limit order that fills, fills at its
limit, and the question is whether it filled at all. A curve that decays gently is a
different proposition from one that goes negative by 10 bps, and both are "profitable"
under the default.

## Directional tilt

Printed after a backtest of a book that traded both sides. A long-only book's net *is*
its gross, so a cap on it would be a second name for a limit that already exists.

```
=== Directional tilt actually carried (400 steps) ===
  |net| / equity        median 19.2%  p90 29.5%  p95 32.1%  p99 37.1%  max 42.7%
  signed mean           +18.4% — the book leans long by construction
  gross max             80.0% of equity

  A cap and what it would have done:
    --max-net-exposure 0.32   would have bound on 5.0% of steps — a different book from the validated one
    --max-net-exposure 0.47   never binds — documents the intent, enforces nothing new

  Smallest cap that leaves the validated book intact: 0.47
```

`max_gross_exposure` bounds long **plus** short and cannot see direction, so a book
sitting inside a gross cap can be entirely one-directional. This is how you choose
`max_net_exposure` from the strategy's own history rather than by picking a number.

The trade-off is the point: **any cap below the observed maximum would have changed the
book that was validated.** It either never binds — documenting an intent rather than
enforcing one — or it binds, and the thing running is no longer the thing that was
tested.

## Execution quality

Read-only, over the live ledger rather than a backtest. This is what a paper session
produces.

```bash
tradeflow execution-report --orders
```

```
=== Execution quality ===
  orders                2 submitted, 0 never filled, 0 ended short, 2 filled across several prints
  slippage              2 of 2 fills measured
                        median +4.3 bps, mean +4.3 bps  (positive = worse)
  decision to fill      2 measured, median 1,845 ms, worst 2,239 ms
  modelled cost         $0.16 over 2 orders (commission $0.05 + spread $0.11; excludes impact)
  broker fees           $0.03 over 2 fills

  Signals that produced no order:
       4  gross_exposure_capped
       1  book_full
```

Slippage is signed so **positive is always worse**, whichever way the trade went. The
modelled cost comes from the same cost model the backtest charges, so the two are
comparable — and it is never added to the venue's own fee, because one is a prediction
and the other an observation.

The refusal block is the half people forget. A strategy that produced no orders all
session left no other trace, and "why did nothing happen" is answerable only here.

## Cost stress

```bash
tradeflow backtest --config configs/breakout.json --start ... --end ... --cost-stress
```

Re-runs under progressively worse cost assumptions. An edge that clears at 1bp and
evaporates at 2bp is a different proposition from one that survives five times its
assumed cost, and a single cost assumption reports both as a pass.

## Portfolio excursion

```bash
tradeflow backtest --config configs/breakout.json --start ... --end ... --excursion
```

The equity curve is marked at each bar's **close**. Everything inside the bar — the
tick where three positions were simultaneously at their worst — is invisible to it, so
a shallow drawdown can sit over a period the book actually spent in more trouble.

Per-trade MAE cannot settle it. A position 40% underwater that was a twentieth of the
book did not put the book 40% underwater, and the per-trade figure looks identical
either way. This is the book-level version.

```
=== Portfolio excursion ===
  Worst the book ever looked : -21.12% from its peak
    at 2024-01-03 00:00:00 — 2 open, gross 96.0%, net 96.0% of equity
    the equity curve showed -0.96% at that same instant
  Deepest closing drawdown   : -6.47% at 2024-01-05 00:00:00
  Best the book ever looked  : +2.88% above its peak
  The curve did NOT sample the same pain: 14.65pp deeper intra-bar than the
  closing marks ever showed.
  Basis: every open position marked at its own worst (and best) tick within
  the same bar, which assumes they all got there at once — an upper bound on
  the pain, never a measurement of it.
  The realized worst lies between the two: the closing mark is a lower bound and
  the simultaneous-extremes figure an upper one. Neither is the answer alone.
```

*(Illustrative figures.)*

**Read it as a pair, not as a number.** The two lines bracket the truth from opposite
sides:

| | What it is | Which way it errs |
| --- | --- | --- |
| Deepest closing drawdown | Prices the book actually printed at | Too shallow — it never sees inside a bar |
| Worst the book ever looked | Every open position at its own worst tick in the same bar | Too deep — they did not all get there at once |

Quoting only the second is the pessimistic reading the per-trade figures already
invite; quoting only the first is the complacent one the equity curve already gives.
The gap between them is the finding, and where it is small the diagnostic says so —
that answer is as useful as the alarming one and rather more common.

Both excursions are measured against the running peak of the **closing** curve, since a
peak the book only touched intra-bar would inflate the drawdown with the same noise
this exists to separate out. Nothing is gated on any of it, and the figures are in the
`excursion` block of the JSON payload and over MCP whether or not the flag was passed.

## Causality probes

```bash
tradeflow backtest --config configs/breakout.json --start ... --end ... --causality
# or
make causality
```

Asks one question about the *mechanism* rather than about the strategy: **could each
decision have been made when it was made?**

Four probes, all by perturbation — withhold something that only becomes knowable after
a fill, re-run, and require the decision at that instant to be unchanged:

| Probe | What must hold |
|-------|----------------|
| `execution_clock` | Every input to a decision was available strictly before the price it transacted at |
| `same_bar_ranking` | When candidates outnumber slots, the ordering does not consult the bar it transacts on |
| `benchmark_alignment` | The benchmark series is not paired out of step with the strategy's |
| `as_of_scanner` | Universe selection does not read past the clock it claims |

### This is not the leakage probe

The [walk-forward](walk-forward)'s leakage probe shifts the whole feed forward and
checks the result changes. It tests for **future data** — a strategy reading bars it
should not have.

**It cannot test intra-bar causality, and a passing one says nothing about it.** A feed
shift moves signal and price together, so the relationship "signal from bar *i*'s close,
filled at bar *i*'s open" survives the shift completely intact. That probe ran against a
candidate and passed while the engine was executing every signal one bar before it could
have known.

Treat them as different classes. Neither substitutes for the other.

### Read `incomplete` as incomplete

Each probe reports pass, fail, or **not exercised** — and the third is not the first. A
run that never traded has not been cleared by a probe about trading; a book that never
filled its last slot has not been cleared by a probe about ranking. When any probe could
not be exercised the overall verdict is `incomplete`, and `not_exercised` names which.

```
  [PASS] execution_clock       (intra-bar causality)
  [PASS] same_bar_ranking      (intra-bar causality)
  [PASS] benchmark_alignment   (intra-bar causality)
  [n/a ] as_of_scanner         (as-of clock)

  verdict               INCOMPLETE
  not exercised         as_of_scanner — not the same as passing
```

Probes sample: each examines the first few decision instants, because each one costs a
re-run. The report says how many it looked at, so a pass is never read as coverage it
did not have.

## Inspecting the trades

```bash
tradeflow trials show <trial-id> --trades-limit 25
tradeflow trials show <trial-id> --json --trades-limit 25
```

Trade tables are stored only for runs recorded with them — a search evaluating thousands
of candidates would otherwise store storage nobody asked for. When present, this is
where exit reasons, durations and per-trade excursions live, and it is how the
concentration block above gets turned into a specific question.

`trials analyze <trial-id>` answers that question directly — the same exit-reason split
printed above, plus win and loss by reason, holding period and excursion distributions,
over a run recorded weeks ago. It is the same computation: the block above and the
command below go through one function, so a live result and a recorded one cannot come
to disagree about where a run's P&L came from. See [the trial store](trials) for both
commands.

## None of this is journaled

The stress runs and the causality probes record nothing. They are one candidate under stated assumptions, not new
candidates, and journaling them would inflate the multiple-testing count that the
deflated Sharpe deflates against.

That also means you can run them freely. Exploring the parameter neighbourhood with
`--no-journal` costs nothing statistically; a walk-forward search costs family budget
permanently.
