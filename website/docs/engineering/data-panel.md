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

The signature is deliberately the contract an out-of-core Arrow/Polars/DuckDB source
would implement: growing the storage tier is a new adapter behind `scan()`, not a
rewrite of the layers above.

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
| `add_score_feature` | `score` (applies any `scorer`: a strategy, a scanner, …) |

| Consumer | Reads → writes |
|----------|----------------|
| `refine_alpha` ([alphas](./alphas)) | `score`, `residual_vol`, `beta` → `z`, `alpha` |

New producers (factor exposures, transaction-cost params, liquidity) slot in the same
way, and consumers downstream don't change.

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
