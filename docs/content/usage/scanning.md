---
sidebar_position: 3
title: Scanning the universe
---

# Scanning the universe

The scanner filters a list of *candidate* symbols down to the ones worth trading
at a specific research clock. It runs a `ScannerStrategy` over each symbol's
recent bars and keeps those that produce an actionable scan signal.

```bash
make scan
# or
uv run python main.py scan --scanner volume --symbols NVDA,META,TSLA,AMD
```

Standalone scans default to wall-clock now. Pin them to a historical clock with
`--as-of`:

```bash
uv run python main.py scan --scanner volume --symbols NVDA,META,TSLA,AMD --as-of 2024-06-01
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

`backtest`, `optimize`, `walkforward`, `research`, `cache warm` and `live` all accept
a `--scanner` option. When set, they run the scanner first and trade only the flagged
symbols; if the scanner flags nothing, they fall back to the candidate list. Pass
`--scanner none` to skip scanning and use the symbols as-is.

The historical commands resolve the scanner at `--end` by default, so a validation
window does not mix in today's universe, and `--scan-as-of` pins a different scanner
clock. **`live` is the exception, and deliberately so:** it resolves at wall-clock
now, because a live book is selected from the universe as it stands, not as it stood
at the end of some window. It therefore takes no `--scan-as-of`.

Every scan reports the clock it resolved at, in the exchange zone — including the ones
that resolved to "now", so a payload never leaves which universe it selected from as
something to be inferred.

```bash
uv run python main.py backtest --scanner none --symbols AAPL,MSFT --start 2024-01-02 --end 2024-04-01
```
