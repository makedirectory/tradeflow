---
sidebar_position: 7
title: Indicators & analytics
---

# Indicators & analytics

## Why no TA-Lib

The original project's reference design pulled in TA-Lib, a C library that needs a
compiler and system headers to install. That conflicts with the "easy to try"
goal. Everything TA-Lib offered here is a few lines of pandas/numpy, so the
project ships its own.

The payoff: `uv sync` is the entire install, the Dockerfile has **no** build
toolchain, and the math is right there to read.

## `src/indicators/indicators.py`

Pure, side-effect-free functions:

| Function | Notes |
|----------|-------|
| `calculate_sma(series, period)` | simple moving average |
| `calculate_ema(series, period)` | exponential moving average |
| `calculate_rsi(series, period=14)` | RSI in `[0, 100]` |
| `calculate_atr(df, period=14)` | average true range |
| `calculate_volume_spike(volume, price, ...)` | boolean: volume > MA×threshold **and** price move > threshold |

Each takes Series/DataFrames and returns new ones, leaving inputs untouched.

## `src/analytics/`

Performance accounting lives here, **not** in the strategy or engine.

### `metrics.py` — primitives

`returns_from_equity`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`,
`win_rate`, `profit_factor`, `calmar_ratio`. Every function is defensive: empty or
degenerate inputs return `0.0` (or `inf` for a ratio with a zero denominator)
rather than raising. These are shared by the backtester **and** the scanner.

### `performance.py` — backtest metrics

`compute_backtest_metrics(trades_df, equity_curve, initial_capital,
final_capital, market_data)` turns a trade log into a flat metrics dict
(total/buy-hold return, Sharpe/Sortino/Calmar, drawdown, win rate, profit factor,
average/largest win & loss). `build_equity_curve` accumulates daily P&L.

### `reporting.py` — rendering

Formats a metrics dict into the aligned text block you see after a backtest. Kept
separate from computation so the numbers can be consumed programmatically (the
optimizer reads `metrics`, it doesn't parse text).
