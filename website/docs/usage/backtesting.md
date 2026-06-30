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

## Net of transaction cost

Backtest metrics are **net of transaction cost by default** — commission + half-spread
+ square-root market impact, charged on both legs of every trade (see
**[Transaction costs](../engineering/transaction-costs)**). The report prints the total
cost and the gross final capital alongside. Pass `--gross` to disable the charge (for
attribution — "how much did costs cost me?"), and tune `--commission-bps` / `--impact-eta`.
High-turnover strategies degrade sharply once costs are on; that's the point.

## Tuning the strategy

Once a backtest runs, search for better parameters with
**[Optimization](optimization)**.
