---
sidebar_position: 12
title: Feature panel & scan
---

# Feature panel & scan

`src/data/` is the cross-sectional data substrate — **all the data in one place, as
of one moment**. Every research-clock module that reasons across names at a rebalance
(alphas today; risk, costs, portfolio construction, information analysis next) reads
from and writes to the same table, instead of each assembling its own.

## The scan seam

`BarSource.scan(universe, timeframe, as_of, lookback)` returns the bars for a
universe up to `as_of`. It is the **single home of the leakage guard**: the
`<= as_of` slice lives here and nowhere else, so no caller can accidentally let a
future bar through. `ClientBarSource` backs it with the existing `MarketDataClient`
today.

The signature is deliberately the contract an out-of-core columnar source implements:
growing the storage tier is a new adapter behind `scan()`, not a rewrite of the layers
above.

### Out-of-core storage (the `store` extra)

`ParquetBarStore` is the first storage tier that scales past RAM. Bars live in
**symbol-partitioned Parquet**, and `scan()` reads them through **Arrow with predicate
pushdown** — the `as_of` window is a Parquet filter, so a point-in-time scan
*physically never reads rows after `as_of`*. It implements the same `BarSource`
contract as the in-memory `ClientBarSource`, so it's a **drop-in**: the alpha, risk,
and portfolio layers never learn where the data lives, and pandas appears only at the
edge (the returned per-symbol frames). It's opt-in (`uv sync --extra store`, pure
`pyarrow` wheels — the no-compiler promise holds).

This is the concrete meaning of *"design for scale, start small"*: the seam exists from
the start, so reaching Parquet (and later partitioned/columnar tiers, or migrating the
compute layer to lazy Polars/DuckDB) is an adapter swap, not a rewrite.

### Streaming backtest (bounded memory)

`BacktestEngine.run_streaming(source, symbols, …)` backtests from any `BarSource`
**one symbol at a time** — each symbol's bars are scanned, simulated, and retired
before the next, so peak memory is bounded by a single symbol's history rather than
the whole panel. Because the per-symbol simulation is independent and cash carries
across symbols in order, the result is identical to the in-memory `run()` (a
regression test asserts this against a Parquet store). The remaining phase — migrating
the panel-wide *compute* (indicators, cross-sectional ops) to lazy Polars/DuckDB — is
deferred until data volume forces it; the seam makes it incremental.

## The panel

A `FeaturePanel` is a symbol-indexed table for one universe at one `as_of`:

- **Rows** are symbols; **columns** are features.
- Producers call `panel.set(name, values)` (aligned to the universe — a missing name
  is `NaN`, not an error); consumers call `panel.get(name)`.
- `panel.meta` carries cross-sectional flags (`low_confidence`, `benchmark_available`).

Stacked over time, a sequence of panels is the `(time × symbol × feature)` data that
signal-combination and information-analysis search over — *"which factors matter
right now"* is a query against the panel.

## Producers and consumers

Each module is a column producer, a consumer, or both:

| Producer | Writes |
|----------|--------|
| `add_risk_features` | `beta`, `residual_vol` (annualised; falls back to total vol with no benchmark) |
| `add_factor_exposure_features` | `exp_<factor>` (the [risk model's](./risk-model) standardized exposures, for [factor-neutral alphas](./alphas#neutralization); reuses the panel's `beta`, writes nothing if `<2` names qualify) |
| `add_score_feature` | `score` (applies any `scorer`: a strategy, a scanner, …) |

| Consumer | Reads → writes |
|----------|----------------|
| `refine_alpha` ([alphas](./alphas)) | `score`, `residual_vol`, `beta`, `exp_*` → `z`, `alpha` (+ `meta.neutralized_against`) |

New producers (transaction-cost params, liquidity) slot in the same way, and
consumers downstream don't change — the factor-exposure producer arrived exactly
this way.

## Why it's shaped this way

The first cut of the alpha service computed β and σ inline and threaded them around
in ad-hoc dicts (`AlphaContext` carried a `residual_vol` map). That is a private copy
of a structure that wants to be shared. Making the panel the explicit currency means
the leakage guarantee, the as-of assembly, and the feature columns live in one place
— and the next modules (a real risk model, portfolio construction) plug into a
defined shape rather than re-deriving the cross-section each time.

The `set`/`get` API is column-at-a-time on purpose: the pandas frame can become a
lazy columnar panel at scale without any consumer changing. *Start small, but nothing
above the data layer assumes the panel fits in RAM.*
