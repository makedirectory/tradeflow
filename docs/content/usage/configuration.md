---
sidebar_position: 2
title: Configuration
---

# Configuration

:::tip No keys needed to explore
Want to see TradeFlow work before signing up for anything? Run `make demo` — it
runs the whole pipeline (backtest + walk-forward) on synthetic data, offline.
:::

Credentials live in a `.env` file, which is **gitignored** so your keys are never
committed. Create it from the template:

```bash
cp .env.example .env
```

```bash title=".env"
APCA_API_KEY_ID=<your key id>
APCA_API_SECRET_KEY=<your secret>

# Keep this true until you are absolutely sure you want to trade real money.
PAPER_TRADE=true
```

Get paper keys from the [Alpaca dashboard](https://app.alpaca.markets/) under
**Paper Account → API Keys**.

## Where settings come from

Every credential is resolved through one place (`tradeflow/settings.py`) in a fixed
order, so there's never any ambiguity:

1. **Environment variables** — the standard, 12-factor way (`export APCA_API_KEY_ID=…`).
2. **`.env`** in the project root — loaded automatically; real environment
   variables already set always win.

The same chain resolves the optional LLM provider keys for the research agent
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`) — set them in `.env`
or export them, whichever you prefer.

## Paper vs. live

`PAPER_TRADE` selects the Alpaca endpoint. Leave it `True` while learning — every
`live` run then trades against the paper account with no real money at risk.

## Strategy config keys

A strategy's config carries its tunable parameters plus a few behavioral keys the
base class recognizes:

| Key | Meaning |
|---|---|
| `risk_per_trade` | Fraction of capital risked per trade |
| `stop_loss` / `take_profit` | Fractional distances from entry price |
| `position_limits` | Portfolio limits the [engine](../engineering/engine.md) enforces across the book: `max_positions` (open positions), `max_position_size` (notional per position, in dollars), `max_total_risk` (risk budget as a fraction of equity), `max_gross_exposure` (deployed notional as a fraction of equity; unset by default). See [what `max_total_risk` caps](#what-max_total_risk-caps). |
| `reaffirm_entries` | Live only. Open a position the score implies even when its entry edge was missed — a rejected bar, a dropped stream, a restart, or a crossing inside the warm-up history. Default `true`, so a strategy started mid-trend takes the position rather than waiting for the next crossing. `--no-reaffirm-entries` on `live` turns it off. Exits are never gated by it. |

## What `max_total_risk` caps

`max_total_risk` is a **risk budget**, not an exposure cap, and the two read alike
enough to be worth stating plainly.

Each open position contributes `notional × stop_loss` — what it gives up if it stops
out — and the engine refuses an entry that would push the sum past
`max_total_risk × equity`. So it bounds *loss if everything stops out*. It says
nothing about how much notional is deployed, and the relationship between the two
runs through the stop distance:

| `stop_loss` | Notional admitted by `max_total_risk: 0.05` |
|---|---|
| 5% | 1× equity |
| 1% | 5× equity |
| 0.5% | 10× equity |

A tighter stop buys more notional for the same budget. Reading `max_total_risk: 0.05`
as "at most 5% of the book is deployed" is wrong in every row but the first.

`max_gross_exposure` is the cap on deployed notional: marked gross notional over
equity, shorts counted by magnitude. It is unset by default, which leaves free cash
as the only bound — and because [shorts are fully
cash-collateralized](../engineering/engine.md), that holds a backtest near 1× on its
own. Set it when a config should sit deliberately below that:

```python
"position_limits": {
    "max_positions": 5,
    "max_position_size": 1500.0,
    "max_total_risk": 0.05,     # lose at most 5% if every stop fills
    "max_gross_exposure": 0.60,  # never deploy more than 60% of equity
}
```

Both are enforced across the whole book — by the backtest engine on the research
clock, and by the live trader on the trade clock (see [portfolio limits are enforced
live](live-trading.md#portfolio-limits-are-enforced-live), which notes where the two
counts differ). `Strategy.calculate_position_size` also clamps a *single* position
against `max_total_risk`, because sizing has no view of the open book; that clamp
asks only whether one position could exhaust the whole budget.

## Run-time options

Everything else (symbols, dates, capital, strategy, scanner) is passed per run on
the command line or via Make variables — nothing else needs editing:

```bash
make backtest SYMBOLS=AAPL,MSFT START=2024-06-01 END=2024-09-01 CAPITAL=50000
```

See the individual workflow pages for the full option list.
