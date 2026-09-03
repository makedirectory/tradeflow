---
sidebar_position: 5
title: Screening a parameter space
---

# Screening a parameter space

`tradeflow screen` sweeps a strategy's parameters cheaply to answer one question:
**is there anything in this family at all?** It runs many configurations in one
process against one data fetch, and it **journals nothing**.

That last part is the whole feature. Every journaled trial raises the deflated-Sharpe
bar for its `(strategy, universe, accounting)` family permanently — see
[trials](trials). A researcher who cannot ask a cheap question without spending that
budget will either spend it carelessly or stop asking, and both are worse than the
question being free.

```bash
make screen
# or
tradeflow screen --strategy demo_trend --symbols AAPL,MSFT,NVDA \
    --start 2019-01-01 --end 2023-06-30 \
    --method random --max-evals 60 --max-positions 5
```

## Read the distribution, not the winner

The report leads with the distribution and puts the best point last, deliberately.

**The best of N is the maximum of N draws.** That is a positive number even when
nothing you searched has any edge, and it grows as you search harder. A leaderboard
printed without a null beside it is exactly the selection bias the
[deflated Sharpe](../engineering/evaluation-metrics) exists to prevent — one layer up,
with no deflation applied.

So the screen prints, in this order:

| Block | What it answers |
|-------|-----------------|
| Distribution | n, median, quartiles, spread, positive rate. This is the finding. |
| Noise baseline | What the best of *that many* draws is worth if none had any edge. |
| Gradients | How the result moves across each parameter's axis. |
| Best point | Last, and only after the two numbers that say how to read it. |

Illustrative: a run whose best Sharpe is `+0.41` against an expected noise maximum of
`+0.55` has found nothing, however good `+0.41` looks on its own.

### What the baseline assumes

Two things travel with the number, and both matter:

- **It assumes the points are independent.** Neighbouring grid points share most of
  their parameters and most of their trades, so the effective number of independent
  trials is smaller than the count — and this bar is correspondingly high.
- **The spread it uses is measured on your results**, which may contain real structure,
  so it is not a pure-noise dispersion either.

It is a reference for reading the table, not a test. For any objective whose null is
not zero — a profit factor is null-centred on 1 and heavily skewed — no baseline is
computed at all, and the report says so instead of printing a number that would be
quoted.

## Gradients: the finding a leaderboard cannot show

A positive rate that falls monotonically as a filter tightens is structure. The same
count of positive points scattered at random across the axis is not — and the two
produce identical winners.

Illustrative figures, for the shape rather than the values:

```
--- lookback: how the result moves across the axis ---
       value  points  positive    median     best
          10      12       58%    +0.044   +0.612
          20      12       33%    -0.187   +0.395
          30      12       25%    -0.298   +0.221
          40      12        8%    -0.461   +0.074
```

That reads as a real relationship, and it points *off the edge of the searched space* —
toward lookbacks shorter than the shortest one tried. A best-point report structurally
cannot tell you that.

## Narrowing the search

`--range` narrows one axis and is repeatable. Each override supplies any of
`min`/`max`/`step` and inherits the rest, so narrowing cannot silently change a
parameter's type or drop its default:

```bash
tradeflow screen --range fast_ema_period=5:12:1 --range stop_loss=0.02:0.06:0.01
```

A name the strategy does not declare is **refused**, not ignored: a typo that quietly
screened the full range would report a distribution for a space you never asked about,
and finding nothing there means nothing.

Combinations a strategy declares invalid are never drawn — see
[constraints between parameters](../engineering/optimization#constraints-between-parameters).

## The book you are screening

`--max-positions` (or a `--config` carrying `position_limits`) sets the book each point
is evaluated against. Without it, every point runs at whatever the strategy class
declares, which is usually one position — a different strategy from the one you intend
to deploy, reported under the same name.

## Turning a screen into evidence

`--confirm` re-runs **exactly one** point as a proper journaled trial:

```bash
tradeflow screen ... --confirm best     # or --confirm 3, the rank in the table
```

Exactly one is the constraint that matters. A confirm that could take a set would be a
screen that journals, which puts the budget problem straight back in through the door
the screen exists to open — and it would record the best of N, the one selection a
sweep cannot support.

A confirmed point is identical to running [`backtest`](backtesting) with those
parameters: same dedup identity, same memoization, same journal record. It counts once.

## Screen, optimize, validate

Three different questions, in order:

| Command | Question | Journals |
|---------|----------|----------|
| `screen` | Is there anything in this family? | nothing |
| [`optimize`](optimization) | Which configuration looks best in-sample? | one trial per config |
| [`walkforward`](walk-forward) | Does the chosen configuration survive out-of-sample? | one trial |

Screening is not a cheaper optimize — it is the step before deciding whether an
optimize is worth its statistical budget at all.
