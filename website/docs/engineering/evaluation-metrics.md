---
sidebar_position: 14
title: Evaluation metrics
---

# Evaluation metrics

The analytics layer turns a table of closed trades plus an equity curve into a
metrics dict honest enough to **rank thousands of configs by a single number**.
The layering is deliberate:

- `src/analytics/metrics.py` — pure, defensive primitives. One canonical
  definition per metric, shared by the backtest analytics and the scanner base
  class so "Sharpe" or "max drawdown" mean the same thing everywhere. Empty or
  degenerate inputs return `0.0` (or `inf` for a zero-denominator ratio) rather
  than raising.
- `src/analytics/performance.py` — `compute_backtest_metrics(...)` composes those
  primitives over the trade table and equity curve into a flat, JSON-serializable
  dict keyed by `METRIC_KEYS` (+ the `FLAG_KEYS` flags).
- `src/analytics/reporting.py` — human-readable, sectioned rendering, kept
  separate from computation so the numbers stay machine-consumable.

## Metric tiers

**Statistical robustness (the most important additions).** With grid/Bayesian
search in play, the system *will* find a high-Sharpe config by chance. Two metrics
correct for that:

- `probabilistic_sharpe_ratio` (PSR) — P(true Sharpe > 0) given sample length,
  skew, and kurtosis. Corrects a Sharpe estimated from a short, fat-tailed sample.
- `deflated_sharpe_ratio` (DSR) — PSR against the *expected best* Sharpe of
  `n_trials` configs. The anti-overfitting metric: the more configs tried, the
  higher the bar. The optimizer / walk-forward thread `n_trials` in.

Both use a hand-rolled normal CDF (`math.erf`) and inverse-CDF (Acklam's
approximation), so the base install needs no SciPy.

**Returns / growth.** `cagr` (the cross-window comparison number),
`annualized_volatility`, `downside_deviation`.

**Risk-adjusted.** `sharpe_ratio`, `sortino_ratio`, `calmar_ratio` (CAGR / max
drawdown), `treynor_ratio`, `information_ratio`, `martin_ratio`, `sterling_ratio`.

**Drawdown & tail.** `max_drawdown`, `max_drawdown_duration`, `ulcer_index`,
`recovery_factor`, `var_95` / `var_99`, `cvar_95`, `tail_ratio`, `omega_ratio`.

**Trade-level.** `win_rate`, `profit_factor`, `payoff_ratio`, `expectancy`,
`gain_to_pain_ratio`, `kelly_criterion`, `sqn`, `max_consecutive_wins/losses`,
`avg_trade_duration`, `turnover`, and **MAE/MFE** (max adverse / favorable
excursion, tracked intra-trade by the simulator).

**Benchmark-relative.** `alpha`, `beta`, `r_squared`, `information_ratio`,
`treynor_ratio` — computed when a benchmark return series is supplied; otherwise
zeroed and `benchmark_available` is `False`.

## Annualization

Headline ratios take a `periods_per_year`. The backtest equity curve is resampled
to daily P&L, so its returns are daily and `TRADING_DAYS_PER_YEAR` (252) is the
correct factor. `Timeframe.periods_per_year()` exists for callers that work on
bar-frequency returns directly.

## Known limitation (flagged in the glossary)

`build_equity_curve` accumulates **closed-trade** P&L resampled to daily, so
intra-trade (mark-to-market) drawdown is invisible — `max_drawdown`,
`ulcer_index`, and volatility *understate* true risk during long holds. The metric
glossary surfaces this so agents don't over-trust the numbers. A mark-to-market
equity curve from bar data is a future enhancement.

## Sample-size guard

Any result with fewer than ~30 trades sets the `low_sample` flag; ratios from a
handful of trades are noise, not edge.

## Bootstrap skill inference

PSR/DSR are parametric: they assume the Sharpe estimator's asymptotic
distribution, and DSR needs an *assumed* effective trial count and trial-Sharpe
variance to approximate "the best of K configs tried." `src/analytics/bootstrap.py`
is the heavier, definitive check behind `--bootstrap-skill` — it **simulates the
null instead of assuming it** — not a replacement for PSR/DSR, always shown
alongside them.

Two questions, two functions:

1. **Is this one track record skill?** `bootstrap_null` imposes the null (demean
   the return series by its own estimated alpha), then resamples the residual
   with a **stationary (Politis–Romano) block bootstrap** — never i.i.d., since
   active returns are autocorrelated and i.i.d. resampling would understate how
   likely a long lucky streak is. Reports the realized IR's rank in the
   empirical null distribution: an *own* p-value with no assumption about the
   return distribution's shape.
2. **Is the best of the configs actually tried skill?** `reality_check`
   (White's Reality Check) resamples the **same block indices across every
   trial's return series at once**, so the cross-trial correlation structure is
   preserved by construction — DSR has to *assume* a trial-Sharpe variance to
   approximate this; this replays the actual trials recorded in the
   [trial store](./walk-forward#the-trial-store). The null distribution is "the
   best IR among K correlated tries," compared against the actually observed
   best.

Both report a Monte-Carlo standard error on the p-value and a block-length
sensitivity check (p at half and double the chosen block length) — a p-value
that flips across that range is not a result, it's noise dressed as a verdict.

**Own and family are always reported together, never one alone.** A great own
p and a terrible family p is exactly the selection-luck signature this test
exists to catch — publishing only the own-track-record number would hide that
this config was the best of many tried. Family scoring is **advisory**, not
(yet) a hard promotion gate; see
[the trial store](./walk-forward#the-trial-store) for why wiring campaign-wide
counts into a hard gate is a separate, deliberately deferred decision.

`--bootstrap-skill` on `walkforward` and on `info --attribution` both surface
this; see the [usage guide](../usage/walk-forward.md#nonparametric-skill-check).
