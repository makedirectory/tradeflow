# TradeFlow

[![CI](https://github.com/makedirectory/tradeflow/actions/workflows/ci.yml/badge.svg)](https://github.com/makedirectory/tradeflow/actions/workflows/ci.yml)

A small, **layered**, **broker-agnostic** algorithmic-trading engine. It ships with
an [Alpaca](https://alpaca.markets) adapter, but everything above the broker layer
is vendor-neutral.
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

## Requirements

You need **either** of these — not both:

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — the Python
  package manager used to run everything locally, **or**
- **[Docker](https://docs.docker.com/get-docker/)** — to build and run the app in
  a container (no local Python or uv needed).

The `Makefile` targets run through **uv** (e.g. `make backtest` → `uv run …`), and
there are separate **Docker** targets (`make docker-build`, `make docker-run`).
Pick whichever you prefer.

Either way you'll need free Alpaca **paper-trading** API keys from the
[Alpaca dashboard](https://app.alpaca.markets/) → *Paper Account → API Keys*.

## Quickstart (uv)

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

## Quickstart (Docker)

No local Python or uv required — just Docker:

```bash
cp config_example.py config.py        # add your Alpaca paper keys
make docker-build                     # build the image (uv runs inside it)
make docker-run                       # paper live-trading; mounts your config.py

# or run any command in the container directly:
docker run --rm -v $(pwd)/config.py:/app/config.py tradeflow \
    uv run python main.py backtest --symbols NVDA,META --start 2024-01-02 --end 2024-04-01
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
| `walkforward` | Out-of-sample validation: optimize in-sample, score out-of-sample across folds, with a sacred holdout and promotion gates |
| `mcp` | Serve TradeFlow over MCP so an agent (Claude Code / Desktop) can drive scan/backtest/optimize/walk-forward — read-only, no live trading |

### Optional features

Capabilities are optional extras so the base install stays lean:

```bash
make install-optimize     # scikit-learn, for `optimize --method bayesian`
make install-portfolio    # Google OR-Tools, for `allocate`
uv sync --extra mcp       # the MCP SDK, for `python main.py mcp`
```

## Agent integration (MCP)

`python main.py mcp` exposes TradeFlow's deterministic capabilities to any MCP
client (Claude Code / Claude Desktop) as tools: discovery, `run_scan`,
`run_backtest`, `run_optimization`, `run_walk_forward`, `get_metrics_glossary`,
`summarize_bars`, and `save_config`/`load_config`/`list_configs`. Every call is
logged to `logs/mcp_audit.jsonl` for replay.

**The safety model is structural.** The server constructs only a *data* client —
no trading client, no broker — so it is incapable of placing an order. There is
no `place_order`, `start_live`, `cancel`, or `set_paper_trade` tool; promoting a
config to live is a manual human step outside MCP. The capability simply isn't
wired in, so it can't be prompt-injected around. The agent works on the
*research clock* (offline, exploratory); the live order path stays LLM-free.

Register it with a client (Claude Desktop / Claude Code `mcpServers`):

```json
{ "mcpServers": { "tradeflow": {
    "command": "uv",
    "args": ["run", "--extra", "mcp", "python", "main.py", "mcp"],
    "cwd": "/path/to/tradeflow" } } }
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

Lint and format with ruff (what CI runs):

```bash
uv run ruff check .          # lint
uv run ruff format --check . # format
```

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup, the pre-push checks, and the
coding standards (layering rules, separation of concerns, no vendor SDK above the
broker layer). CI runs ruff lint + format + the test suite on every PR.

## Account utilities

```bash
make cancel-orders         # cancel all open orders
make close-positions       # liquidate all positions (and cancel orders)
```

## ☕ Coffee?

If this code somehow makes you money, I'd genuinely love to hear about it. If it
*loses* you money — we've never met, and this is the first you're hearing of it.

Either way, if it saved you some time, you can buy me a coffee:

**[buy me a coffee →](https://venmo.com/u/Andrew-Schwartz-92)**

