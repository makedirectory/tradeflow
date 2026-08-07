---
slug: /
sidebar_position: 1
title: Overview
---

# TradeFlow

A layered, broker-agnostic algorithmic-trading **research engine** — and an honest
one, which mostly means it is very good at telling you your brilliant strategy is
actually noise.

## Try it in 30 seconds

No keys, no account, no network:

```bash
uv tool install tradeflow-engine
tradeflow demo
```

That runs the entire pipeline on synthetic data: it backtests every bundled
strategy, picks the best-looking one, walk-forward validates it — and **refuses to
promote it**.

The refusal is the point. The synthetic series is a seeded random walk with no edge
in it, so a strategy that looks profitable in-sample gets called noise
out-of-sample. If that had *not* happened, the tool would be broken.

**→ [Getting started](usage/getting-started)** walks the rest: real market data with
free paper keys, your first verdict, and connecting Claude — six steps, and you can
stop at any of them.

No `uv`? [Install it](https://docs.astral.sh/uv/getting-started/installation/), or
use `pipx install tradeflow-engine`.

## What it actually does

Scans a universe, turns signals into comparable return forecasts, builds a
cost-aware portfolio, and — the part that matters — tells you whether any of it is
distinguishable from luck.

```bash
tradeflow verdict --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31
```

One command runs scan → alphas → portfolio → information analysis over one universe,
one window, and one cost model, ending in a single verdict with every gate shown:

```
  VERDICT: mixed — passed: sample_size, sanity_ceiling; failed: ic_tstat, net_of_cost_alpha
    [FAIL] ic_tstat: 0.70 vs 2 — IC t-stat below 2 is not distinguishable from luck
    [FAIL] net_of_cost_alpha: -0.034 vs 0 — expected active return after the cost of trading
    [PASS] sample_size: 24 vs 12 — too few rebalances to measure an IC with any confidence
```

> Beating the market is hard — embarrassingly hard. TradeFlow's real value isn't a
> money printer; it's a rigorous skeptic that makes it harder to mistake luck for
> skill. If your strategy survives walk-forward and the deflated Sharpe, *maybe*
> you've got something. If it doesn't, you just saved yourself some tuition.

## Why it's built this way

- **Two clocks that never touch.** Research (backtest, optimize, walk-forward, the
  AI agent) is slow, exploratory, and only ever *proposes*. The live order path is
  deterministic and imports none of it. Promotion between them is a manual human
  step. See [the architecture](engineering/architecture).
- **AI-assisted research without AI-controlled trading.** The
  [MCP server](engineering/mcp-server) builds only a market-data client, so an agent
  is *structurally* incapable of placing an order — there is no order tool to
  prompt-inject around.
- **Broker-agnostic.** Every layer is written against a `Broker` /
  `MarketDataProvider` interface; Alpaca is just the first implementation.
- **No TA-Lib, no native build step.** Indicators are pure pandas/numpy, so there is
  no compiler in the install path and none in the Docker image.

## Where to go

- **[Getting started](usage/getting-started)** — install → keys → first result →
  Claude, end to end
- **[Usage](usage/installation)** — every command: verdict, backtest, walk-forward,
  optimization, portfolio construction, the trial store, the AI agents
- **[Engineering wiki](engineering/architecture)** — how it is built and why, and how
  to add a strategy, scanner, or broker
- **[Using it as a library](engineering/embedding)** — `tradeflow.services.*` from
  your own code
- **[Changelog](changelog)** — every release, and the project's history since 2023

:::warning Educational software
This project is for learning. Trading carries real financial risk. Keep
`PAPER_TRADE=true` unless you fully understand the consequences. No warranty, and
nothing here is investment advice.
:::
