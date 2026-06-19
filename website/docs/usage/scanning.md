---
sidebar_position: 3
title: Scanning the universe
---

# Scanning the universe

The scanner filters a list of *candidate* symbols down to the ones worth trading
right now. It runs a `ScannerStrategy` over each symbol's recent bars and keeps
those that produce an actionable scan signal.

```bash
make scan
# or
uv run python main.py scan --scanner volume --symbols NVDA,META,TSLA,AMD
```

Example output:

```
SYMBOL    SIGNAL
NVDA      SCANNER_BUY
TSLA      SCANNER_SELL
```

## The volume scanner

The bundled `volume` scanner flags a symbol when its latest bar shows **unusually
high volume** (relative to its moving average) **and** a meaningful price move. It
is pure pandas/numpy — see [Scanners](../engineering/scanners) for the internals.

## How it feeds the other commands

`backtest`, `live`, and `optimize` accept a `--scanner` option. When set, they run
the scanner first and trade only the flagged symbols; if the scanner flags
nothing, they fall back to the candidate list. Pass `--scanner none` to skip
scanning and use the symbols as-is.

```bash
uv run python main.py backtest --scanner none --symbols AAPL,MSFT --start 2024-01-02 --end 2024-04-01
```
