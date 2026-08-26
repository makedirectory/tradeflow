---
sidebar_position: 12
title: Extending
---

# Extending

Three common extension points. Each touches one layer. Strategies and scanners can
live either in this repository or in a separate private Python package; the private
package route is the intended shape for proprietary signal IP.

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

2. For public/example strategies, register it in `BUILTIN_STRATEGIES` in
   `tradeflow/services/registry.py`. For proprietary strategies, expose it from a
   separate installed package through the `tradeflow.strategies` entry-point group:

   ```toml
   [project.entry-points."tradeflow.strategies"]
   private_trend = "tradeflow_alpha_pack.strategies:PrivateTrendStrategy"
   ```

   Once installed in the same environment, it works in `backtest`, `live`,
   `optimize`, the MCP server, and the research agent — sizing, fills, execution,
   and metrics come for free because they only depend on the base interface.
   (`create_with_defaults()` is inherited from `Strategy`; no need to write it.)

Use the pure [indicators](indicators); don't reach for a compiled TA library.

## Add a scanner

1. Subclass `ScannerStrategy` in `tradeflow/scanners/` — implement `process_data` and
   `generate_signals_df` (emit `SCANNER_BUY`/`SCANNER_SELL`/`SCANNER_HOLD` plus a
   `signal_strength`).
2. For public/example scanners, register it in `BUILTIN_SCANNERS` in
   `tradeflow/services/registry.py`. For proprietary scanners, expose it from a
   separate installed package through the `tradeflow.scanners` entry-point group:

   ```toml
   [project.entry-points."tradeflow.scanners"]
   private_volume = "tradeflow_alpha_pack.scanners:PrivateVolumeScanner"
   ```

   Keep it TA-Lib-free.

## Private alpha packs

Keep the engine boring and open; keep the signal IP elsewhere. A private package
can depend on `tradeflow-engine`, define strategies/scanners in its own modules,
and expose them with entry points. TradeFlow loads entry points at startup, but
built-in names are reserved, so a private package cannot silently replace
`ma_crossover`, `mean_reversion`, or `volume`.

A private package can also return several contributions from one entry point:

```toml
[project.entry-points."tradeflow.strategies"]
private_pack = "tradeflow_alpha_pack.registry:strategies"
```

```python
def strategies():
    return {
        "private_trend": PrivateTrendStrategy,
        "private_reversal": PrivateReversalStrategy,
    }
```

The MCP server includes draft validation tools for the workbench phase:

- `validate_draft_strategy_code` checks generated/private strategy source against
  the sandbox and base-class contract without registering or running it.
- `validate_draft_scanner_code` does the same for scanner source and verifies the
  scanner output schema.
- `run_draft_walk_forward` validates strategy source in-memory, runs the normal
  walk-forward validator, and records the result under
  `draft:<ClassName>:<code_hash>` when journaling is enabled.

That gives an agent a safe loop for proposing and modifying code without putting
the proprietary implementation in this repository. Once a candidate survives
validation, move it into the private package and expose it by entry point so future
runs can refer to it by name and share the normal registry/memoization path.

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
