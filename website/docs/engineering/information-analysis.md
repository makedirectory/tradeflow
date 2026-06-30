---
sidebar_position: 16
title: Information analysis
---

# Information analysis

The rest of the spine *assumes* an information coefficient and *predicts* an
information ratio. `src/analytics/information.py` (+ `services.compute_information`)
**measures** them and confronts the prediction with reality — the most honest
diagnostic in the system.

> A backtest tells you *what* happened. Information analysis tells you *why*, and
> whether it was skill or luck — the only thing that predicts whether it recurs.

:::note Research clock only
A measurement, never a control input to the order path. The measured IC is surfaced
as a *recommendation* for [alpha](./alphas) scaling that a human applies.
:::

## Information coefficient

At each rebalance `t`, the IC is the cross-sectional correlation between the alpha
known *at* `t` and the realised **residual** return over `(t, t+horizon]`:

```
IC_t = corr( α_{·,t} , r^res_{·,t+1} )
```

Two disciplines are load-bearing:

- **Residual, not raw, return.** Correlating alpha with raw forward return rewards
  beta, not skill — so the benchmark/`β` component is stripped first, consistent with
  how the alpha was neutralised.
- **Strict forward alignment.** `IC_t` pairs `α` at `t` with returns realised *after*
  `t`. An off-by-one (using `r_t` instead of `r_{t+1}`) is a leak that manufactures
  spectacular fake IC — a regression test pins that the IC function is alignment-sensitive.

The summary is `mean_IC`, `IC_vol`, and the honesty gate **`IC_tstat =
mean_IC / (IC_vol/√P)`**: a fine mean IC with a t-stat below ~2 is a few lucky
periods, not skill. Both Pearson and rank-IC are reported.

## Breadth — not the name count

```
predicted_IR = mean_IC · √BR_eff
BR_eff ≈ (rebalances/yr) · N / (1 + (N−1)·ρ̄)
```

`BR_naive = (rebalances/yr)·N` treats every name as an independent bet. `BR_eff`
deflates it by the average pairwise correlation `ρ̄` (taken from the [risk model](./risk-model)'s
Σ): perfectly correlated bets (`ρ̄→1`) collapse to one bet per rebalance. Reporting
`BR_eff` vs `BR_naive` exposes the single most common self-deception in quant —
*thinking you have 500 bets when you have 3.*

## The reconciliation

The report puts the predicted IR next to a **realised IR** — computed from the actual
period returns of the standardized-alpha portfolio over the same forward windows — so
the gap is visible. Against it sit the research-integrity guardrails:

```
IR standard error      SE(IR) ≈ √((1 + IR²/2) / T_years)
multiple-testing        P(any |t| > 2 in m trials) = 1 − 0.95^m
sanity ceiling          realised IR > 2 on public data ⇒ suspect a bug/leak
```

A 3-year backtest has `SE(IR) ≈ 0.6`, so an IR of 0.5 is statistically
indistinguishable from 0 — the report shows IR with that band, not as a point
estimate, and counts how many configs were tried (`--n-trials`) so one lucky backtest
doesn't read as skill.

## Deferred

Factor/specific **attribution** (was the return from cheap factor tilts or genuine
name selection?) needs the structural factor risk model; **capacity** (where net IR
crosses zero as capital scales) needs the transaction-cost model. Both are noted in
the report rather than faked.

## Where it runs

`services.compute_information` scans the window, samples rebalances, measures the IC
series and realised portfolio returns, and assembles the report. The CLI
(`python main.py info`) and the read-only MCP tool `compute_information` route through
it. See the [usage guide](../usage/information.md).
