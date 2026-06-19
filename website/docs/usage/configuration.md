---
sidebar_position: 2
title: Configuration
---

# Configuration

Credentials live in `config.py`, which is **gitignored** so your keys are never
committed. Create it from the template:

```bash
cp config_example.py config.py
```

```python title="config.py"
APCA_API_KEY_ID = "<your key id>"
APCA_API_SECRET_KEY = "<your secret>"

# Keep this True until you are absolutely sure you want to trade real money.
PAPER_TRADE = True
```

Get paper keys from the [Alpaca dashboard](https://app.alpaca.markets/) under
**Paper Account → API Keys**.

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
