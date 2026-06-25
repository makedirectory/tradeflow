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

Every credential is resolved through one place (`src/settings.py`) in a fixed
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

## Run-time options

Everything else (symbols, dates, capital, strategy, scanner) is passed per run on
the command line or via Make variables — nothing else needs editing:

```bash
make backtest SYMBOLS=AAPL,MSFT START=2024-06-01 END=2024-09-01 CAPITAL=50000
```

See the individual workflow pages for the full option list.
