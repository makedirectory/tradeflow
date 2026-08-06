---
sidebar_position: 11
title: Information horizon (decay & cadence)
---

# Information horizon (decay & cadence)

`python main.py horizon` measures how fast a strategy's alpha **decays** and
turns that into a recommended rebalance cadence and a current/lagged blend. It
is **read-only** — a diagnostic over an existing alpha's decay structure, never
a control input.

```bash
python main.py horizon \
  --strategy volume_spike \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN \
  --start 2024-01-01 --end 2024-12-31
```

```
Information horizon: 'volume_spike' 2024-01-01..2025-02-05
  IC by lag: 1:-0.084  2:+0.090  3:+0.012  4:-0.024  5:-0.029  6:+0.007  7:+0.027  8:+0.072  9:+0.113  10:+0.126
  decay δ 1.177  half-life ∞ (no decay detected)  fit R² 0.18
    CI (±1.96 SE on the fit): [4.8, ∞] periods — the multi-period trading policy
    discounts using the upper bound (conservative against killing a good signal)
  recommended cadence: every 2 periods  (best return horizon ≈ inf)
  lagged blend [diversify]: w_now +0.39  w_lagged +0.61  (ρ 0.28, cost 2.70%/yr → not worth the turnover cost)
```

Read it as: `δ` is the per-period IC decay and `half-life` the periods for IC to
halve — with a confidence band (`CI`), since short histories give noisy decay
fits and a point estimate alone invites over-trusting it. `recommended cadence`
is the rebalance interval that maximizes `IC(Δt)·√(1/Δt)` — trading faster than
that pays cost for noise, slower throws away breadth. The **lagged blend**
mixes the current signal with a lagged copy to raise the IR when the two
diversify (`ρ` below `δ`) or hedge (`ρ` above `δ`); it's only *recommended* when
it diversifies and its turnover cost is modest — a high-turnover blend that
would lose net of cost is flagged, not recommended.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--strategy` | `volume_spike` | The strategy whose alpha decay is measured. |
| `--source` | `strategy` | Alpha score origin (`strategy` / `signal` / `scanner`). |
| `--symbols` | demo universe | The cross-section. |
| `--start` / `--end` | last year | Measurement window. |
| `--benchmark` | `SPY` | Used to strip beta (residual returns). |
| `--max-lag` | `10` | Longest lag (periods) the IC-vs-lag profile is fit over. |
| `--timeframe` | `1Day` | Bar timeframe. |
| `--neutralize-factors` | off | Measure the decay of the **same** factor-neutral alpha you'd deploy — see [Ranking by alpha](alphas#factor-neutral-alphas). |

## Superseded (as a recommendation, not as a report)

The lagged blend here is the original, ad-hoc answer to "should I slow down
trading into a decaying signal?" [Multi-period trading](portfolio#multi-period-trading)
(`allocate --objective utility --policy aim`) generalizes it into a continuous
decay discount composed with the optimizer's own no-trade band. This report's
numbers stay accurate either way; prefer `--policy aim` for new construction
work. See
[Multi-period trading (engineering)](../engineering/multi-period-trading) for
why the discount formula needs the confidence band this report now surfaces.

The same computation is available to agents as the read-only MCP tool
`compute_horizon`.
