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

## Tuning the strategy

Once a backtest runs, search for better parameters with
**[Optimization](optimization)**.
