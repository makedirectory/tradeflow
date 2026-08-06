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
6. In parallel, the **trade-update stream** logs fills/cancels/rejects so you can
   see what the account is actually doing.

Press `Ctrl-C` to stop.

### Order safety

- Entries are **skipped while an order is pending** for that symbol, so a repeated
  signal can't double-submit before the first fills.
- A discretionary close **cancels the resting bracket legs first**, so you're never
  left with an orphaned stop/take order.
- Orders are only sent during **market hours** (the clock is checked, with a short
  cache; disable with `respect_market_hours=False` if you need extended-hours).

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

Or use `--beta-sizing` to scale each position inversely by its **beta** vs a
benchmark (default `SPY`) — higher-beta names get smaller positions, evening out
risk:

```bash
make live-beta
# or
uv run python main.py live --scanner volume --symbols NVDA,META,TSLA --beta-sizing --benchmark SPY
```

## Managing the account

```bash
make cancel-orders      # cancel all open orders
make close-positions    # liquidate all positions (also cancels orders)
```

The full real-time path is described in **[The Engine](../engineering/engine)** and
**[Broker Abstraction](../engineering/broker-abstraction)**.

## Bar-quality guards

The live loop validates every bar before the strategy sees it. Guards are **on by
default** — the live path is the only place a corrupt bar costs money.

| Check | Rejects |
|---|---|
| OHLC consistency | `high < low`, open/close outside the range, non-positive prices, negative volume |
| Ordering | a timestamp at or before the last accepted bar for that symbol |
| Staleness | a bar arriving more than ~3 intervals late |
| Spike | a single-bar move beyond `--max-bar-return` (default 35%) |
| Zero volume | no volume on a symbol that has traded before |

**A guard rejects; it never repairs.** Nothing is interpolated, gap-filled, or
corrected. The moment the live path fixes its inputs it stops being the thing the
backtest validated, and every historical result quietly stops describing what will
happen. A rejected bar is skipped and logged with the offending values; the strategy
simply never sees it.

**The threshold is deliberately loose.** A 35% single-bar move is news, and the
strategy should act on it. The spike check exists to catch a decimal-point error or
a crossed quote, not a violent day — a guard tight enough to catch every bad tick
also removes the strategy's best opportunities.

At shutdown the loop reports what it discarded, and flags an elevated rejection rate
loudly. A guard quietly eating a third of the feed looks, from the strategy's side,
exactly like a quiet market.

```bash
python main.py live --symbols NVDA,AAPL             # guards on
python main.py live --symbols NVDA --max-bar-return 0.15   # stricter
python main.py live --symbols NVDA --no-bar-checks         # off (not recommended)
```

## Position reconciliation

Orders used to be submitted and forgotten, so a partial fill, a rejection, or a
position closed by hand in the broker's UI was discovered by reading the P&L and
being surprised.

The live loop now keeps an append-only ledger of **intent** (what was submitted) and
**observation** (what the broker reported), and sweeps it against the broker's actual
account state on a timer. Check it any time:

```bash
python main.py reconcile          # or: reconcile --json
```

```
RECONCILIATION FOUND 2 DIVERGENCE(S):
  [quantity_drift] NVDA: ledger expects +10, broker holds +4 — likely a partial fill
  [unexpected] TSLA: broker holds +7 that this ledger never ordered — opened manually
The broker's state is authoritative. Nothing has been corrected automatically.
```

Three rules govern it, and the first two are what keep it safe:

- **The broker is authoritative, always.** The ledger records what we believed so a
  difference can be *noticed*. When they disagree, the broker is right and the
  ledger is a question for a human.
- **It reports; it never remediates.** No corrective order is ever placed. An
  automated system that notices a missed fill and fixes it is one that can double a
  position at 3am while nobody is watching.
- **Append-only.** Entries are never edited or deleted, and the file *is* the state —
  a restarted process recovers its expectation by replaying it.

The sweep costs one `list_positions` call, never one per symbol, because it runs
inside the trade-clock loop. Exit code is non-zero when divergence is found, so a
scheduled `reconcile` can page you.

`--no-ledger` disables recording; `--reconcile-every 0` disables the in-loop sweep.
