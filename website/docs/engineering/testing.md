---
sidebar_position: 11
title: Testing
---

# Testing

```bash
make test     # uv run --extra dev pytest
```

The whole suite runs **offline** — no API keys, no network, no `alpaca` import.
That's a direct benefit of the [broker abstraction](broker-abstraction): tests
inject in-memory fakes where production injects Alpaca.

## Test doubles (`tests/fakes.py`)

| Fake | Implements | Used for |
|------|-----------|----------|
| `FakeMarketData` | `MarketDataProvider` | deterministic synthetic OHLCV (random walk + volume spikes) |
| `DictMarketData` | `MarketDataProvider` | serves exact fixture frames for precise engine assertions |
| `FakeBroker` | `Broker` | records submitted orders / closes for execution tests |

## What's covered

| File | Focus |
|------|-------|
| `test_units.py` | timeframe parsing, numeric helpers, metric primitives, parameter space |
| `test_strategy.py` | sizing caps, exit conditions, signal validation, indicator columns |
| `test_scanner.py` | volume-scanner signal logic, config validation, forward scoring |
| `test_backtest.py` | engine fills — take-profit, stop-loss, signal exit, end-of-period P&L (exact) |
| `test_live_trader.py` | entry → bracket order, dup-position skip, exit close, insufficient funds |
| `test_optimizer.py` | grid/random ranking, max-evals cap, Bayesian path |
| `test_portfolio.py` | cardinality, weight caps, score preference (skipped if OR-Tools absent) |
| `test_offline.py` | full end-to-end backtest / scan / optimize on synthetic data |

## A real bug the tests caught

`grid_search` originally materialised the entire parameter grid before capping —
billions of combinations for the bundled strategy, which OOM-killed the process
(`exit 137`). The optimizer test surfaced it immediately; the fix computes
`grid_size()` and samples instead. See [Optimization](optimization).

Engine fill logic is asserted with a `ScriptedStrategy` that emits a fixed signal
per bar, so stop/take/exit/end-of-period behaviour and P&L are checked **exactly**,
independent of any indicator quirk.
