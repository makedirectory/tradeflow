---
sidebar_position: 4
title: Backtesting
---

# Backtesting

A backtest replays historical bars through a strategy, simulates fills, and
reports performance. It answers "would this have worked?" — a useful question, as
long as you remember it's not the same as "will this work?"

```bash
make backtest
# or
uv run python main.py backtest \
    --strategy volume_spike --scanner volume \
    --symbols NVDA,META,TSLA --start 2024-01-02 --end 2024-04-01 --capital 100000
```

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--strategy` | `volume_spike` | Strategy to run |
| `--scanner` | `volume` | Universe scanner (`none` to skip) |
| `--symbols` | a 10-name list | Comma-separated candidates |
| `--start` / `--end` | last 30 days | Backtest window (`YYYY-MM-DD`) |
| `--capital` | `100000` | Starting capital |

## Reading the report

The engine prints a metrics block, for example:

```
=== Backtest Results ===
Capital                 $100,000.00 -> $103,420.00
Total Return            3.42%
Buy & Hold Return       2.10%
Sharpe Ratio            1.24
Max Drawdown            4.80%
Total Trades            37
Win Rate                54.05%
Profit Factor           1.61
...
```

- **Total Return** vs **Buy & Hold** shows whether the strategy beat simply
  holding the symbols.
- **Sharpe / Max Drawdown / Profit Factor** describe risk-adjusted quality.

How fills and P&L are simulated is documented in **[The Engine](../engineering/engine)**;
how each metric is computed is in **[Indicators & Analytics](../engineering/indicators)**.

## Universe provenance

Every backtest says where its universe came from:

```
=== Universe provenance ===
  candidates      85 names from --symbols
  scanner         volume as of 2026-08-22T00:00:00-04:00
  resolved        61 of 85 names
  universe        resolved this run
  survivorship    a hand-supplied list is today's names applied to history; membership
                  was not point-in-time
```

A 61-name large-cap list is not "the market", and a report that leaves the universe in
the background invites it to be read as one. The `universe` line distinguishes a
[replayed config](walk-forward.md#reusing-a-saved-config) from a fresh resolution, so
you never have to infer which book you are looking at.

**The survivorship line is a statement, not a measurement.** Anything that left a
hand-supplied list — delisted, acquired, collapsed — is already absent from every
backtest run over it. Quantifying that needs point-in-time membership data this project
does not ingest, so the report says the bias exists by construction rather than
computing a number a static list cannot support.

## Three verdicts, not one

A backtest answers one of the three questions worth asking, and says so:

```
=== Verdicts ===
  Statistical validation  not assessed here - was the edge real, and not overfit (`walkforward`)
  Execution viability     FAIL - unfillable_entries beyond limit at this capital
  Evidence completeness   not assessed here - what has actually been checked (`walkforward --bootstrap-skill`)
  Three separate facts. Clearing one says nothing about the others.
```

These were always three separate verdicts — `promotable`, `executable` and the
promotion prerequisites — that never collapse into one another. What was missing is
that each was printed by a different command at a different moment, so nothing showed
you all three. That is how a backtest replay can read as *approved* when it only means
"this saved config runs, and its history looks good".

A verdict this command cannot assess is printed as **not assessed here**, naming the
command that would assess it. An unknown left blank is an unknown a reader fills in
optimistically.

## Net of transaction cost

Backtest metrics are **net of transaction cost by default** — commission + half-spread
+ square-root market impact, charged on both legs of every trade (see
**[Transaction costs](../engineering/transaction-costs)**). The report prints the total
cost and the gross final capital alongside. Pass `--gross` to disable the charge (for
attribution — "how much did costs cost me?"), and tune `--commission-bps` / `--impact-eta`.
High-turnover strategies degrade sharply once costs are on; that's the point.

## Long/short legs

When a run trades both sides, each leg is reported separately:

```
--- Legs (diagnostic; no thresholds) ---
  leg       return     vol   max DD    beta   corr  trades       cost
  long     +18.00%  0.310%    9.40%   0.920   0.88      54      1,200
  short    -11.00%  0.290%   12.10%  -0.890  -0.85      61      3,400
  Both legs carry real market exposure - a small net beta here is two exposures
  cancelling, not an absence of them.
```

**A near-zero net beta has two completely different causes** — genuinely small exposure
on both sides, or a large long beta cancelling a large short one. They are the same
number and opposite risks, and no net-level figure can tell them apart. The example
above nets to a beta of 0.03 while each leg carries close to a full unit of market
exposure.

The columns answer the questions a headline cannot: whether both legs make money or one
subsidises the other, whether drawdown comes from one side, and what the short side
costs to carry.

Leg curves are **marked, not realized**: they follow the book as held rather than
recording P&L at exit, because a position held through a large excursion and closed
flat would otherwise look as though it never moved — and volatility, drawdown and beta
are exactly the figures that distortion ruins.

This is diagnostic. There are no thresholds and it gates nothing; the point is to make
the risk visible before deciding whether any of it deserves to become a prerequisite.

## Cost stress — where the edge dies

A single cost assumption produces a single number, and no way to tell how much of the
result was the assumption. `--cost-stress` re-runs the same config under scaled costs:

```bash
uv run python main.py backtest --strategy ma_crossover --symbols NVDA,META,TSLA \
    --cost-stress
```

```
=== Cost stress (all axis) ===
    multiple    Sharpe    return          cost
        1.0x      0.31     +1.47%        $  412
        2.0x      0.26     +1.26%        $  824
        3.0x      0.22     +1.06%        $1,236
        5.0x      0.14     +0.66%        $2,060
  Edge survives to 5x its assumed cost.
```

That is a different proposition from a config that reads `+0.05%` at 1x and turns
negative at 2x — and both are "profitable at 1bp". The curve is the point: *where* an
edge dies matters more than whether it clears at one assumed cost.

On `backtest` this is opt-in, because the loop is interactive and each point is a full
re-run. On [`walkforward`](walk-forward.md#promotion-prerequisites) it is on by default:
that run is already expensive and it is where a promotion decision gets made.

`--cost-stress borrow` scales only the borrow rate. Worth asking separately because a
long-short book is exposed to it differently — borrow is carry on inventory, so it
grows with holding period rather than with turnover, and a long-only book is flat under
it while the combined axis still bites.

**Nothing is journaled.** Each point is one candidate under a stated assumption, not a
new candidate; counting them would inflate the multiple-testing total the
[deflated Sharpe](../engineering/evaluation-metrics.md) deflates against, punishing you
for asking how robust your strategy is.

## Trial journaling

Each run records one **trial** — the config it evaluated, on this universe and
window — to `logs/research_journal.jsonl`, and is dual-written into the
[trial store](../engineering/walk-forward#the-trial-store) (`logs/trials.db`) so
`trials query` can report the real campaign-wide trial count on demand — the
gate itself still counts only the current run (see the
[open item](../engineering/walk-forward#n_trials-still-counts-a-run-at-gate-time-not-a-campaign)).
Pass `--no-journal` to keep a throwaway or reproducibility run out of that total.

## Stating the book limits

`--max-positions`, `--max-position-size`, `--max-gross-exposure`, `--max-net-exposure`,
`--max-total-risk` and `--min-notional` set the limits the backtest's book is held to,
overriding what a strategy declares and what a config carries. The same names, units and
meanings the [live path](live-trading.md#stating-the-book-limits-for-this-run) uses.

They exist because asking "what would this look like under a cap I could actually
deploy" otherwise meant editing the saved config, which is exactly where the validated
book and the tested one drift apart. A limit far above the capital in play — a $100,000
per-position ceiling on an $8,000 book — is not a limit; it is a default nobody chose,
and it will not resemble the contract a live run is given.

Only a flag you type applies, and typed limits are part of a run's cache identity. They
were not before: limits are not tunable params, so they went through no identity at all,
and two runs differing only in `max_gross_exposure` hashed alike — the second answered
from the first.

## Where the P&L came from, and the assumption under it

A headline return says nothing about where it came from. Every backtest now prints the
net P&L by exit reason, because a book whose entire gain arrives through one exit path
is a bet on that path's fill assumption, and nothing in the summary metrics
distinguishes it from one whose edge is spread across exits. When one winning exit
accounts for over 90% of the gain, the report says so.

That matters most for take-profit exits, because of how they fill. The engine closes a
position at its target as soon as a bar's high reaches it — **a single print at the
level is enough**. That models a resting limit order that is always first in the queue,
which is the most generous reading available. For a strategy whose gain is concentrated
in target exits, that is not a modelling detail; it is the result.

`--fill-stress` makes the assumption a number you can move:

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

(Shape only — the numbers are illustrative.) A curve that decays gently is a different
proposition from one that goes negative by 10 bps, and both are "profitable" under the
default.

Each row requires the price to trade that far *through* the target before the exit
counts as filled. The trigger tightens; the fill price does not — a limit order that
fills, fills at its limit, and the question being asked is whether it filled at all.
`touch only` is the historical assumption and every result this project has recorded.

Nothing is journaled: these are one candidate under stated assumptions, not new
candidates, and recording them would inflate the multiple-testing count the deflated
Sharpe deflates against.

## Tuning the strategy

Once a backtest runs, search for better parameters with
**[Optimization](optimization)**.
