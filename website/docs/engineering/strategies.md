---
sidebar_position: 5
title: Strategies
---

# Strategies

A `Strategy` (`src/strategies/base.py`) is responsible for three things and no
more:

1. **Indicators** — `process_data(df)` returns the OHLCV frame plus the columns
   it needs.
2. **Signals** — `generate_signals(df)` maps each timestamp to a signal string.
3. **Sizing & risk** — `calculate_position_size`, `check_exit_conditions`,
   `validate_signal`.

It does **not** fetch data, place orders, or compute portfolio metrics.

## The signal vocabulary

Signals are plain strings, defined once in `src/strategies/signals.py` so every
layer agrees:

```python
BUY, SELL          # open a position
CLOSE_BUY, CLOSE_SELL   # close a position
HOLD               # do nothing
```

Keeping them central avoids the classic bug where one layer emits `"buy"` and
another checks for `"BUY"`.

## Lifecycle hooks

| Method | When | Purpose |
|--------|------|---------|
| `calculate_required_lookback()` | construction | bars needed before indicators are valid |
| `initialize()` | start of a run | validate parameters / relationships |
| `process_data(df)` | each batch/bar | add indicator columns |
| `generate_signals(df)` | after processing | produce `{timestamp: signal}` |
| `process_real_time_data(...)` | live, per bar | fold a streamed bar into a rolling buffer and emit the latest signal |

## Position sizing

`calculate_position_size(capital, price)` derives a size from `risk_per_trade`
and `stop_loss`, then clamps it to the configured limits (`max_position_size`,
`max_total_risk`). It is the smallest of the three constraints — risk target, per-
position notional cap, and total-risk cap.

## Parameters & validation

Each strategy declares `PARAM_RANGES` with `min`/`max`/`step`/`default`/`type`.
The base class coerces and range-checks every supplied value at construction, so
an out-of-range parameter fails fast. `step` also lets the
[optimizer](optimization) search the space.

## The bundled strategies

Three ship today, all built on the pure [indicators](indicators) — chosen to
*disagree*, so the [walk-forward](walk-forward) scorecard has something to
discriminate between:

| `--strategy` | Style | In one line |
|--------------|-------|-------------|
| `volume_spike` | Trend + volume (5-minute) | Enter with the short/long EMA trend when a volume spike confirms a move out of an RSI extreme. |
| `ma_crossover` | Trend (daily) | Long-only EMA crossover: buy the golden cross, exit the death cross. |
| `mean_reversion` | Contrarian (daily) | Long-only RSI mean reversion: buy oversold dips, exit on the rebound. |

`ma_crossover` and `mean_reversion` are deliberately minimal (a handful of
parameters each) — honest baselines and clean worked examples. See
[Extending](extending) to add your own; it's a one-file change.
