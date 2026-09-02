---
sidebar_position: 5
title: Strategies
---

# Strategies

A `Strategy` (`tradeflow/strategies/base.py`) is responsible for three things and no
more:

1. **Indicators** — `process_data(df)` returns the OHLCV frame plus the columns
   it needs.
2. **Conviction** — `calculate_scores(df)` returns one continuous, signed **score**
   per bar (positive = bullish, negative = bearish, magnitude = strength). This is
   the strategy's single source of truth.
3. **Sizing & risk** — `calculate_position_size`, `check_exit_conditions`,
   `validate_signal`.

It does **not** fetch data, place orders, or compute portfolio metrics.

## One score, two consumers

A strategy defines **only** the score. Everything else is derived from it, so there
is no parallel decision path to keep in sync:

- The **trade clock** gets its discrete `BUY/SELL/HOLD` from `generate_signals(df)`,
  which the base class implements by walking the score with hysteresis (see below).
  Strategies do not override it.
- The **alpha layer** ([continuous alphas](alphas)) reads the same score as a
  cross-sectional conviction and scales it into a residual-return forecast.

### Deriving the signal from the score

`generate_signals` tracks the desired position direction implied by the score:

```
enter long   when score crosses above  enter_long
exit long    when score falls to/below exit_long      → CLOSE_BUY
(short side mirrors it, for strategies with LONG_ONLY = False)
```

The bands come from `signal_thresholds()` (default: pure sign — long while score >
0). A strategy with asymmetric entry/exit — e.g. `mean_reversion` enters when RSI is
oversold but holds until it's overbought — overrides `signal_thresholds()` to set a
wide hold band. Entries are edge-triggered (emitted on the crossing bar); while a
direction is held the bar emits `HOLD`, and the engine dedupes against the open
position.

### Edges, and what live mode adds

An edge says *change*. That is enough for a backtest, where the book is derived from
the same signals and so can never disagree with them. Live it is not: the crossing
bar can be missed — rejected by a quality guard, lost to a dropped stream, consumed
by a restart, or simply inside the warm-up history — after which the score still says
"should be long" while every bar emits `HOLD`, and the position is never opened. A
missed exit is worse: a real position that nothing will close.

So `process_bar` also compares the direction the score *implies* against the position
book (which `LiveTrader` keeps synced with broker truth) and re-states any difference.
Where an edge says *change*, this says *what should be true now*, and the loop
converges on the intended book instead of depending on having caught one bar.

Entries are gated by `reaffirm_entries` (default on — a trend-follower started
mid-trend should hold the trend). Exits never are: declining to open a position is a
preference, declining to close one the strategy no longer wants is a stuck position.
See [live trading](../usage/live-trading.md) for the operational side.

## The signal vocabulary

The derived signals are plain strings, defined once in `tradeflow/strategies/signals.py`
so every layer agrees:

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
| `calculate_scores(df)` | after processing | produce `{timestamp: score}` (the one thing you implement) |
| `signal_thresholds()` | per signal derivation | optional: asymmetric entry/exit bands |
| `generate_signals(df)` | after scoring (base class) | derive `{timestamp: signal}` from the score |
| `process_real_time_data(...)` | live, per bar | fold a streamed bar into a rolling buffer and emit the latest signal |

## Position sizing

`calculate_position_size(capital, price)` derives a size from `risk_per_trade`
and `stop_loss`, then clamps it to the configured limits (`max_position_size`,
`max_total_risk`). It is the smallest of the three constraints — risk target, per-
position notional cap, and total-risk cap.

The total-risk clamp here applies the *whole book's* budget to one position, since
sizing has no view of what is already open: it only answers "could this position
alone exhaust the budget?". Enforcing the budget across the book — and the separate
`max_gross_exposure` notional cap, which no single-position sizing call can
meaningfully apply — is the [engine](engine)'s job. What each fraction actually
measures is spelled out under
[what `max_total_risk` caps](../usage/configuration.md#what-max_total_risk-caps).

## Parameters & validation

Each strategy declares `PARAM_RANGES` with `min`/`max`/`step`/`default`/`type`.
The base class coerces and range-checks every supplied value at construction, so
an out-of-range parameter fails fast. `step` also lets the
[optimizer](optimization) search the space.

## The bundled strategies

Three ship today, all built on the pure [indicators](indicators) — chosen to
*disagree*, so the [walk-forward](walk-forward) scorecard has something to
discriminate between:

| `--strategy` | Style | Score (the one decision) |
|--------------|-------|--------------------------|
| `volume_spike` | Trend + volume (5-minute, long/short) | Signed EMA-trend strength `× volume / volume_ma` — lean with the trend, conviction amplified by volume confirmation. |
| `ma_crossover` | Trend (daily, long-only) | Normalized EMA gap `(fast − slow) / slow` — sign crossings are the golden / death cross. |
| `mean_reversion` | Contrarian (daily, long-only) | Oversold-ness `50 − rsi`, with asymmetric bands: enter below `oversold`, exit above `overbought`. |

`ma_crossover` and `mean_reversion` are deliberately minimal (a handful of
parameters each) — honest baselines and clean worked examples. See
[Extending](extending) to add your own; it's a one-file change.
