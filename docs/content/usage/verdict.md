---
sidebar_position: 5
title: One-command verdict
---

# One-command verdict

`quantred verdict` runs the whole cross-sectional pipeline once — scan →
alphas → portfolio → information — and prints **one** consolidated report ending in
a single verdict line. It is **read-only**: it proposes a book, it never places an
order.

```bash
quantred verdict \
  --strategy demo_trend --scanner none \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN,AVGO,JPM,XOM,UNH \
  --start 2024-01-01 --end 2024-12-31
```

```
=== Research verdict: 'demo_trend' 2024-01-01..2024-12-31 ===
  universe: 12 names (NVDA, AAPL, META, AMD, TSLA, GOOG, MSFT, AMZN, +4 more)
  timeframe 1Day | benchmark SPY | cost 1.0bps commission, impact η=0.3, borrow 50.0bps
  provenance: git 4f2c1ab | campaign trials 14 | bar requests: 3 of 7 reached the data client, the rest shared within this run

  Steps:
    - scan: skipped — no scanner — candidates used as-is
    - combination: skipped — one signal — nothing to combine

  Alphas (case1 scaling, assumed IC +0.0300): 12 names
    NVDA     alpha +0.0412
    AMD      alpha +0.0188
    ...

  Portfolio (proposal, not an order): 4 names, TE 3.71% (target 4.00%)
    expected active return +1.42% gross / +0.61% net
    predicted IR +0.38 | transfer coefficient +0.71
    ...

  Information (24 rebalances, horizon 5 bars):
    IC +0.0184  t-stat +0.74  rank-IC +0.0210
    breadth 142 effective (ρ̄ 0.41, 12 names)
    IR predicted +0.21 vs realized +0.18 ± 0.61
    P(any |t|>2 across 14 campaign trials) = 0.51

  VERDICT: mixed — passed: ir_above_noise, sample_size, sanity_ceiling; failed: ic_tstat
    [FAIL] ic_tstat: 0.74 vs 2 — IC t-stat below 2 is not distinguishable from luck
    ...
```

## Why one command

Running `scan`, `alphas`, `allocate`, and `info` by hand gives four reports that
*look* joined-up but are not: each command re-resolves its own universe and applies
its own defaults, so the flags have to agree by hand or the four sections quietly
describe four different things. `verdict` resolves the universe once, fetches each
distinct set of bars once, and hands every step the same window, universe, and cost
model. The provenance line reports how many of its bar requests actually reached the
provider, so "one shared fetch" is a measured claim rather than a promise.

It answers **"what does the pipeline say about this universe as of `--end`"** — a
forecast and a proposed book. The scanner is resolved at `--end`, not at wall-clock
now, so an older window does not accidentally inherit today's universe. For **"did
this ever work"**, that is
[`backtest`](backtesting) and [`walkforward`](walk-forward); `verdict` does not
replace them, and it is not a historical simulation.

## Reading the verdict line

The verdict is one of `promotable`, `not promotable`, `needs more data`, `mixed`,
`infeasible`, or `incomplete`, and every check behind it is printed with its value and threshold —
passes included. Nothing is averaged: when checks disagree the answer is `mixed` and
you are shown which side each one fell on.

| Check | Passes when | Why it is there |
|---|---|---|
| `ic_tstat` | \|t\| ≥ 2 | Below 2, the mean IC is a few lucky periods, not skill |
| `ir_above_noise` | \|realized IR\| > its own standard error | An IR inside its band is indistinguishable from zero |
| `sanity_ceiling` | realized IR ≤ 2 | An IR above 2 on public data means suspect a bug or a leak |
| `sample_size` | enough measured rebalances | Too few rebalances cannot measure an IC at all |
| `net_of_cost_alpha` | expected active return after cost > 0 | Gross alpha that the cost of trading eats is not alpha |

Two results are deliberately not a pass:

- **A failed step means no verdict at all.** The report shows what ran and what did
  not, and the verdict is `incomplete` however good the completed sections look — a
  book whose information analysis never ran is not a book to act on. The command
  also exits non-zero, so a script can tell a partial run from a complete one.
- **An information step that measured nothing** (too short a window, too few
  sampleable rebalances) reads as `needs more data`, never as a partial pass on the
  strength of the portfolio alone.

## An infeasible book

A long-only book that is fully invested needs enough names to carry the whole
portfolio under the per-name cap: `max_names × max_weight` (or the universe size, if
smaller) must reach 1. At the default 25% cap that is four names. When it does not,
no amount of data can help, so the verdict says so first and does not ask for more
dates:

```
  Steps:
    - scan: skipped — no scanner — candidates used as-is
    - alphas: skipped — allocation preflight: max_weight*3 < 1 (cardinality/weight cap can't fund the book)
    - combination: skipped — one signal — nothing to combine
    - portfolio: skipped — allocation preflight: max_weight*3 < 1 (cardinality/weight cap can't fund the book)
    - information: skipped — allocation preflight: max_weight*3 < 1 (cardinality/weight cap can't fund the book)

  Portfolio: NOT FEASIBLE — max_weight*3 < 1 (cardinality/weight cap can't fund the book)

  VERDICT: infeasible — no book satisfies the constraints: max_weight*3 < 1 (cardinality/weight cap can't fund the book). More dates will not change this; widen the universe or raise max_weight/max_names explicitly.
    [FAIL] portfolio_feasible: False vs True — max_weight*3 < 1 (cardinality/weight cap can't fund the book)
    The constraints, not the data, are what failed: change the universe or the caps (--max-weight, --max-names) yourself - nothing was relaxed for you.
```

The check runs right after the universe is resolved - after the scanner, so a scanner
that flags only a few of your candidates is the usual cause - and the alphas,
portfolio and information steps are reported as skipped. The caps are never relaxed
for you; the way out is a wider `--symbols` list, `--scanner none`, or an explicit
`--max-weight` / `--max-names`. It is a necessary check only: a universe that passes
it can still fail to solve, and a solve that names the constraint it hit is reported
as `infeasible` too. A solve that fails only for want of history (too few bars to
estimate a covariance matrix) names no constraint, so it is `needs more data` instead:
that one more dates do fix.

The documented starter (twelve names, no scanner) passes. It is a usability example,
not a validated universe.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--strategy` | `demo_trend` | The signal to refine into alphas |
| `--combine a,b,c` | — | Combine several strategies' signals into one alpha (measures + shrinks their ICs); with one or none, the single-signal path runs |
| `--scanner` | `demo_volume` | Universe scanner; `none` analyzes the candidates as given |
| `--symbols` | default universe | Candidate universe |
| `--start` / `--end` | last 365 days | The one window every step uses |
| `--source` | `strategy` | Alpha score origin (`strategy` / `signal` / `scanner`) |
| `--benchmark` | `SPY` | Benchmark for residual vol and beta |
| `--timeframe` | `1Day` | Bar timeframe |
| `--horizon` | `5` | Forward-return horizon, in bars |
| `--risk-model` | `shrinkage` | Covariance model (`shrinkage` / `sample` / `factor`) |
| `--target-te` / `--max-weight` / `--max-names` | `0.04` / `0.25` / — | Portfolio constraints |
| `--capital` | `100000` | Sizes the impact term and the holdings table |
| `--lookback-days` | `365` | History used to fit the covariance and the alphas |
| `--participation-cap` | `0.10` | Largest fraction of a name's daily volume the book may trade into |
| `--n-points` | `24` | Rebalance dates sampled by the information step |
| `--neutralize-factors` | — | Regress the listed risk-model exposures out of the alphas |
| `--gross` / `--commission-bps` / `--impact-eta` / `--borrow-bps` | net, `1.0` / `0.3` / `50.0` | The one cost model every step prices with |
| `--cache` / `--offline` / `--cache-dir` | off | Serve bars from the local [bar cache](configuration) |
| `--json PATH` | — | Also write the structured result object the report renders |
| `--html PATH` | — | Also write a [self-contained HTML report](html-reports) of the run |
| `--force` / `--rerun` | off | Re-run instead of serving an identical prior run |
| `--no-journal` | off | Do not record this run as a trial |

`verdict` keeps the *shared* knobs only. Step-specific tuning stays on the
individual commands — it is the honest default path through the pipeline, not a
superset of every flag.

## Journaling and reuse

The whole composite records **one** trial (kind `verdict`), not one per step: five
rows per command would inflate the campaign's multiple-testing total fivefold and
corrupt the deflated-Sharpe accounting. That single trial's count is what the
report's `P(any |t|>2 across N campaign trials)` line is computed against.

An `incomplete` run in which no step measured an IC — the information step failed,
typically on a cold cache or a dropped connection, and no signal combination was
measured — records **no** trial, and the report says `Not recorded as a trial` with the
reason; its provenance line reads `campaign trials N (this run not counted)`. It looked at nothing, so counting it would make every failed attempt deflate
the next real result. An incomplete run that did measure an IC (either step) still
counts, because it was a look at the data whatever else failed. `journaled` in the
structured result says whether a trial was actually written.

An identical rerun is served from the trial store rather than re-run, under the same
prominent `REUSED` banner every other command uses, with the original run's
timestamp and age. A run that came back `incomplete` is never reused: a failed step
answered nothing, so repeating the command after fixing the cause runs it again.
`--force` re-verifies and appends a new trial instead. The run's
identity covers every input any step reads — including knobs like `--target-te` that
live inside a step — so two materially different composites can never collide as
"the same trial".

The proposed book is journaled with the trial: the weights, the active weights when
a benchmark portfolio was given, and the factor-exposure vector. Trials recorded
before that existed have no book stored, which reads as *not recorded* rather than
as an empty one.

## Over MCP

The same composite is available to an agent as the `run_verdict` tool, returning the
identical structured object — see the [MCP server](../engineering/mcp-server) page.
The tool takes the same allocation knobs (`max_weight`, `max_names`, `lookback_days`,
`neutralize_factors`, `participation_cap`, `n_points`), so an agent that is told its
book is `infeasible` can widen the universe or change a cap explicitly. Omitted, each
keeps the service's default.

## From a source checkout (contributors)

The commands above are the installed `quantred` command. In a checkout, run them as
`uv run python main.py <command> ...`. The Makefile shortcut for this page is `make verdict`. Extras are installed with `uv sync --inexact --extra <name>`
(see [optional extras](extras)).
