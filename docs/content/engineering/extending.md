---
sidebar_position: 12
title: Extending
---

# Extending

Three common extension points. Each touches one layer.

## Add a strategy

1. Subclass `Strategy` in `tradeflow/strategies/`:

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
       def calculate_scores(self, df): ...             # -> {timestamp: signed score}
   ```

   You implement `calculate_scores` (one signed conviction per bar) and nothing
   else for decisions: the base class derives `BUY/SELL/HOLD` from the score, and
   the [alpha layer](alphas) scales the same score. Set `LONG_ONLY = False` to allow
   shorts, and override `signal_thresholds()` for asymmetric entry/exit bands.

2. Register it in `STRATEGIES` in `tradeflow/services/registry.py`. It now works in
   `backtest`, `live`, `optimize`, the MCP server, and the research agent — sizing,
   fills, execution, and metrics come for free because they only depend on the base
   interface. (`create_with_defaults()` is inherited from `Strategy`; no need to
   write it.)

Use the pure [indicators](indicators); don't reach for a compiled TA library.

## Add a scanner

1. Subclass `ScannerStrategy` in `tradeflow/scanners/` — implement `process_data` and
   `generate_signals_df` (emit `SCANNER_BUY`/`SCANNER_SELL`/`SCANNER_HOLD` plus a
   `signal_strength`).
2. Register it in `SymbolScanner.SCANNERS`. Keep it TA-Lib-free.

## Add a broker

1. Implement `Broker` (and optionally `MarketDataProvider`) for the venue in a new
   `tradeflow/brokers/<vendor>/` package, mapping the SDK to the domain types.
2. Construct it in `main.build_data_and_broker()`.

Three parts of the contract are easy to get wrong, and all three are load-bearing:

- **Map failures onto `BrokerError`**, don't return `None`. Callers act differently
  on a rate limit, revoked credentials, and a rejected order; collapsing them into
  one non-answer removes the only basis for choosing.
- **`list_positions` raises rather than returning `[]`** when the account cannot be
  read. An empty list is the claim that the account is flat, and the strategy's
  position book is rebuilt from it.
- **Honor `client_order_id`.** A venue that has already accepted one must reject the
  duplicate as `DuplicateOrderError` rather than placing a second order — that
  rejection is the idempotency guarantee, and the only one that survives a restart.

Nothing in `engine/`, `execution/`, `strategies/`, `scanners/`, or the optimizer
changes — they only ever knew the interface. Prove it the same way the suite does:
run against your adapter, or against `FakeBroker` first. `FakeBroker` models all
three behaviors above, and `FailingBroker` fails any named method with a chosen
error, so an adapter can be exercised against the same expectations.

## Add an optimization objective

Any key in the metrics dict (`sharpe_ratio`, `total_return`, `calmar_ratio`, ...)
is a valid `--objective`. To add a new one, compute it in
`analytics.performance.compute_backtest_metrics` and it's immediately selectable.
