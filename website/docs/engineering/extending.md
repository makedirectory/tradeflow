---
sidebar_position: 12
title: Extending
---

# Extending

Three common extension points. Each touches one layer.

## Add a strategy

1. Subclass `Strategy` in `src/strategies/`:

   ```python
   class MyStrategy(Strategy):
       TIMEFRAME = "5Min"
       PARAM_RANGES = {  # min/max/step/default/type per tunable param
           "lookback": {"type": "int", "min": 5, "max": 50, "step": 5, "default": 20},
           "risk_per_trade": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01, "default": 0.02},
           "stop_loss": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01, "default": 0.02},
           "take_profit": {"type": "float", "min": 0.02, "max": 0.10, "step": 0.02, "default": 0.04},
       }

       def calculate_required_lookback(self): return self.config["lookback"] + 1
       def initialize(self): ...
       def process_data(self, df): ...                 # add indicator columns
       def generate_signals(self, df): ...             # -> {timestamp: signal}
   ```

2. Register it in `main.STRATEGIES`. It now works in `backtest`, `live`, and
   `optimize` — sizing, fills, execution, and metrics come for free because they
   only depend on the base interface.

Use the pure [indicators](indicators); don't reach for a compiled TA library.

## Add a scanner

1. Subclass `ScannerStrategy` in `src/scanners/` — implement `process_data` and
   `generate_signals_df` (emit `SCANNER_BUY`/`SCANNER_SELL`/`SCANNER_HOLD` plus a
   `signal_strength`).
2. Register it in `SymbolScanner.SCANNERS`. Keep it TA-Lib-free.

## Add a broker

1. Implement `Broker` (and optionally `MarketDataProvider`) for the venue in a new
   `src/brokers/<vendor>/` package, mapping the SDK to the domain types.
2. Construct it in `main.build_data_and_broker()`.

Nothing in `engine/`, `execution/`, `strategies/`, `scanners/`, or the optimizer
changes — they only ever knew the interface. Prove it the same way the suite does:
run against your adapter, or against `FakeBroker` first.

## Add an optimization objective

Any key in the metrics dict (`sharpe_ratio`, `total_return`, `calmar_ratio`, ...)
is a valid `--objective`. To add a new one, compute it in
`analytics.performance.compute_backtest_metrics` and it's immediately selectable.
