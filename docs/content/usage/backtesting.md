---
sidebar_position: 4
title: Backtesting
---

# Backtesting

A backtest replays historical bars through a strategy, simulates fills, and
reports performance. It answers "would this have worked?" — a useful question, as
long as you remember it's not the same as "will this work?"

```bash
quantred backtest \
    --strategy demo_trend --scanner demo_volume \
    --symbols NVDA,META,TSLA --start 2024-01-02 --end 2024-04-01 --capital 100000
```

A backtest reads daily bars for the symbols and a benchmark (`SPY` unless you change it),
so it needs Alpaca keys from [`quantred init`](first-run) or a warmed cache with
`--offline`. The minimal form is `quantred backtest --symbols AAPL,MSFT --scanner none
--start 2024-01-02 --end 2024-12-31`. Next to try: [`verdict`](verdict) first if you
have not asked whether the signal has any information, and [`walkforward`](walk-forward)
after, because a backtest alone cannot tell a real edge from a fit.

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--strategy` | `demo_trend` | Strategy to run; an unknown name lists the installed ones |
| `--scanner` | `demo_volume` | Universe scanner (`none` to skip); `--scan-as-of` pins its clock (default `--end`) |
| `--symbols` | a 10-name list | Comma-separated candidates |
| `--start` / `--end` | last 30 days | Window: a date or ISO datetime, read in New York time |
| `--capital` | `100000` | Starting capital |
| `--config PATH` | — | Take strategy, params, universe, capital, costs and limits from a saved [run config](run-configs); flags you type win |
| `--benchmark` | `SPY` | Benchmark symbol for beta and the benchmark rows |
| `--gross`, `--commission-bps`, `--impact-eta`, `--borrow-bps` | net, `1.0`, `0.3`, `50.0` | The cost model ([below](#net-of-transaction-cost)) |
| `--max-positions` and the other `--max-*` / `--min-notional` limits | strategy's own | The book the run is held to ([below](#stating-the-book-limits)) |
| `--beta-sizing` | off | Scale position size inversely by each symbol's beta |
| `--cost-stress`, `--fill-stress` | off | Re-run under harsher assumptions ([validation diagnostics](validation-diagnostics#cost-stress)) |
| `--excursion`, `--causality` | off | Extra diagnostics; see [validation diagnostics](validation-diagnostics) |
| `--chart PATH`, `--html PATH` | — | Write an equity-curve image (needs `viz`) or a [self-contained report](html-reports) |
| `--record-trades` | off | Keep the trade table with the trial so `trials show` / `trials analyze` can read it |
| `--no-journal`, `--force` / `--rerun` | off | Skip recording this run; or re-run instead of serving an identical earlier one |
| `--cache`, `--offline`, `--cache-dir` | off | Serve bars from the local [bar cache](configuration) |

## Reading the report

The engine prints a metrics block, for example:

```
=== Backtest Results ===
Capital                     $100,000.00 -> $103,420.00
(!) low sample              fewer than 30 trades - treat ratios with caution

--- Execution & cost (i) ---
  ...
--- Returns ---
Total Return                3.42%
CAGR                        3.42%
Buy & Hold (universe)       2.10%
Buy & Hold (benchmark)      1.85%
--- Risk-adjusted ---
Sharpe Ratio                1.24
--- Drawdown & tail ---
Max Drawdown                4.80%
--- Trades ---
Total Trades                37
Win Rate                    54.05%
Profit Factor               1.61
...
```

(Illustrative figures; the full report has more rows in each section, and the
`Execution & cost` block is [described below](#execution-and-cost).)

- **Total Return** against the two **Buy & Hold** rows shows whether the strategy beat
  simply holding the symbols it traded (`universe`) and holding the benchmark
  (`benchmark`). The benchmark row is absent when no benchmark series was available.
- **Sharpe / Max Drawdown / Profit Factor** describe risk-adjusted quality.
- **`(!) low sample`** appears under 30 trades. A few trades tell you almost nothing:
  read the ratios as anecdote, and lengthen the window or widen the universe.
- **A single position.** The demo strategy declares a one-position book, so unless a
  config or `--max-positions` says otherwise a many-name universe still trades one name at
  a time. The `book_breadth` check below fails for that reason.
- **Run the same backtest twice** and the second is served from the
  [trial store](trials) under a `REUSED` banner, with fewer rows (a reused result keeps
  no cost, execution or trade detail and says which). `--force` recomputes it.

How fills and P&L are simulated is documented in **[The Engine](../engineering/engine)**;
how each metric is computed is in **[Indicators & Analytics](../engineering/indicators)**.

## Universe provenance

Every backtest says where its universe came from:

```
=== Universe provenance ===
  candidates      120 names from --symbols
  scanner         demo_volume as of 2026-08-22T00:00:00-04:00
  resolved        40 of 120 names
  universe        resolved this run
  survivorship    a hand-supplied list is today's names applied to history; membership
                  was not point-in-time
```

A 40-name large-cap list is not "the market", and a report that leaves the universe in
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

## Execution and cost

Between the capital line and the metrics, a backtest prints an `Execution & cost` block
whenever the book that could be traded differs from the one the sizer asked for - whole
shares rounded away, entries that never opened, or a check failing:

```
--- Execution & cost (!) ---
Intended notional           $34,583.55
Filled notional             $32,721.56
  [PASS] rounding_drag         5.38% vs 10.00%
  [PASS] unfillable_entries    0.00% vs 5.00%
  [FAIL] book_breadth          1 vs 2
  [PASS] cost_share_of_gross  12.40% vs 40.00%
  entries never opened      4 (4 rounded to zero, 0 below min notional)
  Not the book that was validated - see the failing check above.
```

(Illustrative figures.) Each check is `PASS` or `FAIL` with the measured value beside its
threshold, in that check's own unit: `rounding_drag` is the share of intended notional
whole-share rounding removed, `book_breadth` the position cap (it fails at 1, the shipped default, when the universe is larger), `unfillable_entries` the
share of entries that never opened. `(!)` marks a failing check and `(i)` an informational
block. A `FAIL` here is what the Verdicts block above reports as `Execution viability`.
See [configuration](configuration.md) for how these arise at small capital.

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
quantred backtest --strategy demo_trend --symbols NVDA,META,TSLA \
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

The figures are illustrative. What the table does promise is the shape of the `cost`
column: every term the stress scales - commission, the quoted spread, market impact and
borrow - is linear in the multiple, so the cost of the 3x row is three times the 1x row
(to within the small drift from a slightly poorer book sizing off less cash). The spread
used to stay at its default through the stress, which made "3x" cost about 1.6x.

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
window — to `logs/research_journal.jsonl` under the state root (`~/.quantred` unless
`QUANTRED_HOME` is set), and is dual-written into the
[trial store](../engineering/walk-forward#the-trial-store) (`logs/trials.db`) so
`trials list` can report the real campaign-wide trial count on demand — the
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
book and the tested one drift apart. A limit far above the capital in play — a $1,000,000
per-position ceiling on a $10,000 book — is not a limit; it is a default nobody chose,
and it will not resemble the contract a live run is given.

Only a flag you type applies, and typed limits are part of a run's cache identity. They
were not before: limits are not tunable params, so they went through no identity at all,
and two runs differing only in `max_gross_exposure` hashed alike — the second answered
from the first.

See [validation diagnostics](validation-diagnostics.md) for the full set — fill
stress, directional tilt, cost stress, and the execution report a paper session
produces.

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

The block under the Verdicts shows the same split for every backtest:

```
=== Where the P&L came from ===
  exit              trades   share       net P&L  win rate
  SIGNAL                15   93.8%          $129     33.3%
  END_OF_PERIOD          1    6.2%           $22    100.0%
```

(Illustrative figures.) `win rate` is the share of that exit's trades that made money,
which separates an exit that earns by being right often from one that earns on a few
large winners.

## Common errors

| Message | What to do |
|---|---|
| `Missing Alpaca credentials ...` | Run `quantred init`, or use `--offline` against a warmed cache |
| `--offline: no cached 1day bars for ...` | Run the `quantred cache warm ...` line it prints; the benchmark needs bars too |
| `the window ends ... before it starts` | `--end` must come after `--start` |
| `invalid choice: 'x'` for `--strategy` | The message lists the installed strategies |

## Tuning the strategy

Once a backtest runs, search for better parameters with
**[Optimization](optimization)**, and validate the result with
**[walk-forward](walk-forward)**. Recorded runs are browsed with [trials](trials). The
engine's fill and cost model is in [The Engine](../engineering/engine).

## From a source checkout (contributors)

The commands above are the installed `quantred` command. In a checkout, `uv run quantred <command> ...` runs the same thing (`uv run python main.py <command> ...` also works). The Makefile shortcut for this page is `make backtest`. Extras are installed with `uv sync --inexact --extra <name>`
(see [optional extras](extras)).
