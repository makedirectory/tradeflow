---
sidebar_position: 10
title: Information report (IC / IR)
---

# Information report (IC / IR)

`python main.py info` measures whether a strategy has **skill** — its information
coefficient and effective breadth — and reconciles the predicted information ratio
with the realized one. It is **read-only**: a diagnostic, never a control input.

```bash
python main.py info \
  --strategy volume_spike \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN \
  --start 2024-01-01 --end 2024-12-31 \
  --n-trials 10
```

```
Information report: 'volume_spike' 2024-01-01..2024-12-31
  measured over 24 rebalances (horizon 5 bars)
  IC mean +0.018  t-stat +0.74  rank-IC +0.021
  breadth: 142 effective vs 504 naive (ρ̄ 0.41, 8 names)
  IR: predicted +0.21  realized +0.18 ± 0.61 (SE)
  guardrails: P(any |t|>2 in 10 trials) = 0.40

  Verdict: skill is NOT distinguishable from luck (IC t-stat +0.74).
```

Read it as: the **IC t-stat** is the honesty gate — below ~2 the mean IC is a few
lucky periods, not skill. **Effective breadth** deflates the name count by how
correlated the bets are (`ρ̄`); **predicted IR** = `mean_IC · √BR_eff`. The realized
IR comes with a **standard-error band** — a 1-year window has `SE(IR) ≈ 1`, so almost
any IR is indistinguishable from zero on a single short window. `--n-trials` reports
how inflated a "significant" result is once you account for everything you tried.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--strategy` | `volume_spike` | The strategy whose alpha is measured. |
| `--source` | `strategy` | Alpha score origin (`strategy` / `signal` / `scanner`). |
| `--symbols` | demo universe | The cross-section. |
| `--start` / `--end` | last year | Measurement window. |
| `--benchmark` | `SPY` | Used to strip beta (residual returns). |
| `--horizon` | `5` | Forward-return horizon, in bars. |
| `--n-trials` | `1` | Configs tried, for the multiple-testing inflation. |
| `--neutralize-factors` | off | Measure the **factor-neutral** alpha (bare flag = `market,volatility,size`) — use the same setting you deploy with, so the measured IC/IR describes the forecast you actually trade. Also on `horizon`. |
| `--scaling-ab` | off | Research mode: walk-forward the realized IR under **Case-1** (`σ·IC·z`) vs **Case-2** (`IC·c_g·z`) scaling and compare against the regression's pick — the ground-truth tiebreak for the [Case test](alphas#case-scaling). |

## The level shrink and the risk-bucket monitor

The report also carries two more diagnostics:

- **Level shrink.** The measured IC is itself estimated; the report prints
  what fraction of the naive level survives that estimation error —
  `keep 13% of the naive level (T_eff 60, IC 0.05)` — and the shrunk IC to deploy.
  `T_eff` deflates the rebalance count for horizon overlap, so a daily-sampled monthly
  horizon isn't credited 21× the observations it really has.
- **Risk buckets.** Under correct scaling every residual-vol bucket
  contributes ~equally to active variance; a **monotone gradient** flags a mis-scaled
  alpha (usually a Case mis-choice). Suppressed on universes too thin for reliable
  buckets rather than reporting noise.

## What it does (and doesn't) tell you

- **Skill vs luck.** A high IC with a low t-stat, or a realized IR inside its SE band
  of zero, is not skill — the report says so plainly.
- **No look-ahead.** The IC pairs each forecast with strictly *later* residual returns.
- **Feedback to alphas.** When skill is distinguishable, the report recommends the
  measured IC to replace the prior used in [alpha scaling](alphas) — a human applies it;
  nothing auto-tunes the trade clock.

The same report is available to agents as the read-only MCP tool `compute_information`.

## Performance attribution (`--attribution`)

`info --attribution` replaces the pooled IC/IR report with a **per-period,
per-source** breakdown of realized active return: systematic benchmark timing,
each risk factor, each signal (your strategy's own alpha, plus any
`--attribution-signals` you name), and stock-picking — every row exact by
construction (they sum to the realized active return) and Bayesian-blended for
a short-sample-honest t-stat.

```bash
python main.py info --attribution \
  --strategy volume_spike --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN \
  --start 2024-01-01 --end 2024-12-31
```

```
Performance attribution: 'volume_spike' 2024-01-01..2024-12-31
  measured over 24 rebalances (horizon 5 bars)
  row                       mean/yr      IR       t  share ψ²
  active beta · expected          —       —       —  (not skill)
  active beta · surprise          —       —       —  (not skill)
  timing (δβ·δr)               0.12    0.08    0.31       0.4%
  market factor                -0.31   -0.22   -0.84       2.1%
  specific (stock-picking)      1.02    0.71    2.76      41.2%

  cumulative active return: +2.14% (top-down parts +2.09% + δ_CP +0.05%)
  Verdict: IR +0.68 ± 0.31 — distinguishable from luck (t +2.19)
```

Read "specific" as genuine stock-picking skill and the factor rows as cheap
tilts you could have gotten from an ETF. `--attribution-signals
ma_crossover,mean_reversion` attributes other strategies' combined scores as
additional signal columns, so a [combined-alpha](alphas#combining-several-signals)
weight can be checked against its realized counterpart. Add `--conditional
ewma|har` for a predicted-vs-realized tracking-error table split by volatility
regime, and `--bootstrap-skill` for a nonparametric own p-value next to the
parametric verdict (see [below](#nonparametric-skill-check---bootstrap-skill)). See
[Performance attribution (engineering)](../engineering/attribution) for the
regression identity and why signals and risk factors are fit jointly.

## Conditional-risk A/B (`--conditional-ab`)

The net-of-cost decision for [conditional risk](risk#conditional-risk): walks
the window forward constructing the **same** alpha book against a conditional
and an unconditional Σ, carries each variant's weights forward, and compares
realized net IR — not TE-tracking alone.

```bash
python main.py info --conditional-ab --conditional ewma \
  --strategy volume_spike --symbols NVDA,AAPL,META,AMD,TSLA --start 2024-01-01 --end 2024-12-31
```

```
Conditional-risk net-of-cost A/B: 'volume_spike' 2024-01-01..2024-12-31
  measured over 15 rebalances (horizon 21 bars, method ewma)
  variant       net IR  realized TE   pred TE  turnover
  unconditional  +1.25        12.5%     11.3%      6.7%
  conditional    +1.23        12.2%     11.2%      6.7%
  winner (net IR): unconditional
  (net of the real transaction cost — a Σ that tracks TE better but churns the book loses here)
```

## Multi-period-policy A/B (`--policy-ab`)

The equivalent net-of-cost decision for
[multi-period trading](../engineering/multi-period-trading): the myopic policy
vs the aim (`--policy aim`) policy on the same alpha book.

```bash
python main.py info --policy-ab --strategy volume_spike \
  --symbols NVDA,AAPL,META,AMD,TSLA --start 2024-01-01 --end 2024-12-31
```

```
Multi-period policy net-of-cost A/B: 'volume_spike' 2024-01-01..2024-12-31
  measured over 12 rebalances (horizon 5 bars)
  variant      net IR  realized TE   pred TE  turnover
  myopic        -1.07         1.5%      1.2%      8.3%
  aim           -1.07         1.5%      1.2%      8.3%
  winner (net IR): myopic
```

`--trade-rate` overrides the derived `κ` for the aim leg. `over_damped` (shown
as a warning line when it fires) means the aim policy traded *less* and still
scored a *lower* net IR — not an improvement, over-damping.

## Nonparametric skill check (`--bootstrap-skill`)

Adds a stationary block-bootstrap own p-value (no distributional assumption on
returns) next to the parametric `SE{IR}≈1/√Y` verdict — the heavier,
definitive check behind the report's skill-vs-luck call. See
[Evaluation metrics — bootstrap skill inference](../engineering/evaluation-metrics#bootstrap-skill-inference)
for the family (Reality Check) half of this, which needs the
[trial store](../engineering/walk-forward#the-trial-store) and lives on
`walkforward --bootstrap-skill` instead.

## Not yet covered

Capacity analysis lives on the [portfolio construction](portfolio) report
(`capacity_capital`), not here.
