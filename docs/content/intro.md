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
with optional **parameter optimization**, **walk-forward validation**, and
**constraint-solver portfolio allocation**.

> Beating the market is hard — embarrassingly hard. TradeFlow's real value isn't a
> money printer; it's a rigorous skeptic that makes it harder to mistake luck for
> skill. If your strategy survives walk-forward and the deflated Sharpe, *maybe*
> you've got something. If it doesn't, you just saved yourself some tuition.

It is built to be **easy to try** and **easy to read**:

- **No TA-Lib, no native build step.** Indicators are pure pandas/numpy, so
  `uv sync` is the whole install and the Docker image carries no compiler.
- **Broker-agnostic.** Every layer is written against a `Broker` /
  `MarketDataProvider` interface; Alpaca is just the first implementation.
- **Strict separation of concerns.** Each package does exactly one job.

**New here? [Getting started](usage/getting-started)** walks from install to a real
result with Claude connected, in six steps. The first two need no keys and no
network.

Two doc tracks:

- **[Usage](usage/installation)** — install, configure, and run the workflows
  (scan, backtest, live, optimize, walk-forward) plus portfolio allocation and the
  optional AI agents.
- **[Engineering Wiki](engineering/architecture)** — how it's built and why, and
  how to extend it with a new strategy, scanner, or broker.

:::warning Educational software
This project is for learning. Trading carries real financial risk. Keep
`PAPER_TRADE = True` unless you fully understand the consequences. No warranty.
:::

## 60-second tour

No keys, no network — see the whole pipeline (and its honest verdict) first:

```bash
make install                      # uv sync
make demo                         # backtest + walk-forward on synthetic data
```

Then point it at real data with free Alpaca paper keys:

```bash
cp .env.example .env              # add your Alpaca paper keys
make scan                         # what's flagged right now?
make backtest                     # scan -> strategy -> performance report
```

See **[Installation](usage/installation)** to get set up.
