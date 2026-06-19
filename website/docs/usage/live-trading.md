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
3. It subscribes to the **live bar stream** for every monitored symbol. The
   stream auto-reconnects with backoff if the socket drops.
4. Each streamed OHLCV bar updates the strategy, which emits a signal.
5. Actionable signals go to the `LiveTrader`, which sizes the position and submits
   a **bracket order** (entry + stop-loss + take-profit) through the broker.

Press `Ctrl-C` to stop.

## Position sizing

By default each entry is sized by the strategy's risk-per-trade / stop-loss
config (`RiskBasedSizer`). Add `--portfolio` to instead size positions by
**portfolio weights** computed with the OR-Tools allocator — capital is shared
across the universe rather than sized per trade:

```bash
make live-portfolio
# or
uv run python main.py live --scanner volume --symbols NVDA,META,TSLA \
    --portfolio --max-positions 5 --max-weight 0.25
```

With `--portfolio`, only the symbols the allocator funds are traded; if OR-Tools
isn't installed or nothing is funded, it falls back to risk-based sizing. See
[Portfolio allocation](portfolio).

## Managing the account

```bash
make cancel-orders      # cancel all open orders
make close-positions    # liquidate all positions (also cancels orders)
```

The full real-time path is described in **[The Engine](../engineering/engine)** and
**[Broker Abstraction](../engineering/broker-abstraction)**.
