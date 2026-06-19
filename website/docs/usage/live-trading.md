---
sidebar_position: 5
title: Live (paper) trading
---

# Live (paper) trading

Live mode warms up the strategy with recent history, subscribes to the Alpaca
real-time bar stream, and routes signals to the broker as bracket orders.

```bash
make live
# or
uv run python main.py live --strategy volume_spike --scanner volume --symbols NVDA,META,TSLA
```

:::warning
With `PAPER_TRADE = True` (the default) this trades the **paper** account. Set it
to `False` only when you intend to trade real money.
:::

## What happens on start

1. The scanner picks the universe from your candidate symbols.
2. The engine fetches enough history to make every indicator valid, and seeds the
   strategy's rolling buffers (**warm-up**).
3. It subscribes to live bars. For each bar the strategy emits a signal.
4. Actionable signals go to the `LiveTrader`, which sizes the position from the
   strategy config and submits a **bracket order** (entry + stop-loss +
   take-profit) through the broker.

Press `Ctrl-C` to stop.

## Managing the account

```bash
make cancel-orders      # cancel all open orders
make close-positions    # liquidate all positions (also cancels orders)
```

The full real-time path is described in **[The Engine](../engineering/engine)** and
**[Broker Abstraction](../engineering/broker-abstraction)**.
