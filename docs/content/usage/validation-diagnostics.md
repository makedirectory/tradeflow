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

## Inspecting the trades

```bash
tradeflow trials show <trial-id> --trades-limit 25
tradeflow trials show <trial-id> --json --trades-limit 25
```

Trade tables are stored only for runs recorded with them — a search evaluating thousands
of candidates would otherwise store storage nobody asked for. When present, this is
where exit reasons, durations and per-trade excursions live, and it is how the
concentration block above gets turned into a specific question.

## None of this is journaled

The stress runs record nothing. They are one candidate under stated assumptions, not new
candidates, and journaling them would inflate the multiple-testing count that the
deflated Sharpe deflates against.

That also means you can run them freely. Exploring the parameter neighbourhood with
`--no-journal` costs nothing statistically; a walk-forward search costs family budget
permanently.
