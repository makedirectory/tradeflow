---
sidebar_position: 10
title: Information report (IC / IR)
---

# Information report (IC / IR)

`python main.py info` measures whether a strategy has **skill** — its information
coefficient and effective breadth — and reconciles the predicted information ratio
with the realised one. It is **read-only**: a diagnostic, never a control input.

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
correlated the bets are (`ρ̄`); **predicted IR** = `mean_IC · √BR_eff`. The realised
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

## What it does (and doesn't) tell you

- **Skill vs luck.** A high IC with a low t-stat, or a realised IR inside its SE band
  of zero, is not skill — the report says so plainly.
- **No look-ahead.** The IC pairs each forecast with strictly *later* residual returns.
- **Feedback to alphas.** When skill is distinguishable, the report recommends the
  measured IC to replace the prior used in [alpha scaling](alphas) — a human applies it;
  nothing auto-tunes the trade clock.
- **Not yet covered.** Factor/specific attribution and capacity analysis are deferred
  (they need the factor risk model and the cost model).

The same report is available to agents as the read-only MCP tool `compute_information`.
