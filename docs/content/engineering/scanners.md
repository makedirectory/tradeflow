---
sidebar_position: 6
title: Scanners
---

# Scanners

A scanner answers a different question than a strategy: *which symbols are worth
trading right now?* It scores a symbol's recent bars and emits a per-symbol scan
signal.

## `ScannerStrategy` (`tradeflow/scanners/base.py`)

Scanners operate on **one symbol's OHLCV frame at a time** — the `SymbolScanner`
iterates the universe — which keeps the interface simple and avoids MultiIndex
bookkeeping.

| Method | Purpose |
|--------|---------|
| `process_data(df)` | add scan indicators |
| `generate_signals_df(df)` | per-bar frame with `signal` + `signal_strength` |
| `latest_signal(signals_df)` | the most recent actionable signal, or `SCANNER_HOLD` |
| `evaluate_forward(...)` | score signals against subsequent returns (used by tuning) |
| `_validate_config(...)` | defaults + range-checks against `PARAM_RANGES` |

Scan signals (`SCANNER_BUY` / `SCANNER_SELL` / `SCANNER_HOLD`) are deliberately
**distinct** from trade signals so the two can never be confused.

## `SymbolScanner` (`tradeflow/scanners/symbol_scanner.py`)

The driver. It holds a registry of scanner strategies, fetches a lookback window
sized to the scanner's `required_data_points()`, runs the scanner per symbol, and
returns the flagged `(symbol, signal)` pairs.

```python
SCANNERS = {"volume": VolumeScannerStrategy}
```

Only **TA-Lib-free** scanners are registered, consistent with the no-compiled-
dependencies goal.

## The volume scanner

`VolumeScannerStrategy` flags a symbol when its latest bar shows volume well above
its moving-average baseline **and** a meaningful price move — a simple
liquidity/attention filter. All pure pandas/numpy.

## Reusing metric primitives

`evaluate_forward` computes hit rate, average return, Sharpe, and profit factor
using the **same** `analytics.metrics` primitives the backtester uses — so
"Sharpe" means the same thing in a scan score and a backtest report.
