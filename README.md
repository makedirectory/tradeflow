# Alpaca Trading Engine

A small, **layered** algorithmic-trading engine on the [Alpaca](https://alpaca.markets) API.
It scans a universe of symbols, runs a strategy over them, and either **backtests**
on history or **trades live** (paper by default) — with optional **parameter
optimization** and **constraint-solver portfolio allocation**.

Designed to be easy to try and easy to read:

- **No TA-Lib / no native build step** — indicators are pure pandas/numpy, so
  `uv sync` is all you need and the Docker image carries no compiler toolchain.
- **Broker-agnostic** — everything is written against a `Broker` /
  `MarketDataProvider` interface. Alpaca is the first implementation; dropping in
  another venue means writing one adapter, nothing else.
- **Strict separation of concerns** — each layer does one job (see below).

> ⚠️ Educational software. Trading is risky; use paper trading. No warranty.

## Quickstart

```bash
# 1. Install uv:  https://docs.astral.sh/uv/
# 2. Add your Alpaca paper keys
cp config_example.py config.py        # then edit config.py

# 3. Install dependencies
make install                          # uv sync

# 4. Try it (preconfigured combos)
make scan                             # which symbols are flagged right now?
make backtest                         # scan -> volume_spike strategy -> report
make live                             # paper-trade the scanned universe
```

Run `make help` to see every target. Anything is overridable inline:

```bash
make backtest SYMBOLS=AAPL,MSFT,NVDA START=2024-06-01 END=2024-09-01 CAPITAL=50000
```

Or call the CLI directly:

```bash
uv run python main.py backtest --strategy volume_spike --scanner volume \
    --symbols NVDA,META,TSLA --start 2024-01-02 --end 2024-04-01
```

## What it does

| Command | What happens |
|---------|--------------|
| `scan` | Run the universe scanner and print flagged symbols |
| `backtest` | Scan → run `volume_spike` over history → performance report |
| `live` | Scan → warm up indicators → stream bars → place paper/live orders |
| `optimize` | Search strategy parameters by backtest objective (grid / random / Bayesian) |
| `allocate` | Weight a portfolio across scanned symbols (OR-Tools constraint solver) |

### Optional features

Two capabilities are optional extras so the base install stays lean:

```bash
make install-optimize     # scikit-learn, for `optimize --method bayesian`
make install-portfolio    # Google OR-Tools, for `allocate`
```

## Architecture

The codebase is organised into single-responsibility layers. Nothing above the
broker layer imports a vendor SDK.

```
brokers/        Broker interface + domain types  ── alpaca/ (AlpacaBroker, AlpacaMarketData)
marketdata/     MarketDataProvider interface, Timeframe, MarketDataClient
indicators/     Pure pandas/numpy technical indicators
strategies/     Strategy base + signals + volume_spike (signals, sizing, risk)
scanners/       ScannerStrategy base + volume scanner + SymbolScanner (universe)
execution/      LiveTrader (signals -> broker orders)
analytics/      Performance metrics + reporting
engine/         BacktestEngine + LiveEngine (orchestration only)
optimization/   ParameterSpace + ParameterOptimizer (tune params via backtest)
portfolio/      PortfolioAllocator (OR-Tools MIP position weighting)
utils/          logging, numeric, time helpers
```

Data flows the same way in both modes:

```
marketdata → strategy.process_data → strategy.generate_signals
           → engine (simulate fills | route to execution) → analytics
```

See the docs site for the full engineering wiki and usage guide:

```bash
make docs        # serve the Docusaurus site at http://localhost:3000
```

## Docker

```bash
make docker-build
make docker-run            # paper live-trading; mounts your config.py
```

## Tests

```bash
make test                  # offline suite — no API keys or network required
```

The whole stack is testable offline because every layer depends on the
broker/data abstractions; tests inject in-memory fakes.

## Account utilities

```bash
make cancel-orders         # cancel all open orders
make close-positions       # liquidate all positions (and cancel orders)
```
