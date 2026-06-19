---
slug: /
sidebar_position: 1
title: Overview
---

# TradeFlow

A small, **layered**, **broker-agnostic** algorithmic-trading engine. It ships
with an [Alpaca](https://alpaca.markets) adapter, but everything above the broker
layer is vendor-neutral. It scans a universe of symbols, runs a strategy over
them, and either **backtests** on history or **trades live** (paper by default) —
with optional **parameter optimization** and **constraint-solver portfolio
allocation**.

It is built to be **easy to try** and **easy to read**:

- **No TA-Lib, no native build step.** Indicators are pure pandas/numpy, so
  `uv sync` is the whole install and the Docker image carries no compiler.
- **Broker-agnostic.** Every layer is written against a `Broker` /
  `MarketDataProvider` interface; Alpaca is just the first implementation.
- **Strict separation of concerns.** Each package does exactly one job.

Two doc tracks:

- **[Usage](usage/installation)** — install, configure, and run the four
  workflows (scan, backtest, live, optimize) plus portfolio allocation.
- **[Engineering Wiki](engineering/architecture)** — how it's built and why, and
  how to extend it with a new strategy, scanner, or broker.

:::warning Educational software
This project is for learning. Trading carries real financial risk. Keep
`PAPER_TRADE = True` unless you fully understand the consequences. No warranty.
:::

## 60-second tour

```bash
cp config_example.py config.py    # add your Alpaca paper keys
make install                      # uv sync
make scan                         # what's flagged right now?
make backtest                     # scan -> strategy -> performance report
```

See **[Installation](usage/installation)** to get set up.
