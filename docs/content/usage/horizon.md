---
sidebar_position: 11
title: Information horizon (decay & cadence)
---

# Information horizon (decay & cadence)

`quantred horizon` measures how fast a strategy's alpha **decays** and
turns that into a recommended rebalance cadence and a current/lagged blend. It
is **read-only** — a diagnostic over an existing alpha's decay structure, never
a control input.

```bash
quantred horizon \
  --strategy demo_trend \
  --symbols NVDA,AAPL,META,AMD,TSLA,GOOG,MSFT,AMZN \
  --start 2024-01-01 --end 2024-12-31
```

```
Information horizon: 'demo_trend' 2024-01-01..2025-02-05
  IC by lag: 1:-0.084  2:+0.090  3:+0.012  4:-0.024  5:-0.029  6:+0.007  7:+0.027  8:+0.072  9:+0.113  10:+0.126
  decay δ 1.177  half-life ∞ (no decay detected)  fit R² 0.18
    CI (±1.96 SE on the fit): [4.8, ∞] periods — the aim policy
    discounts using the upper bound (conservative against killing a good signal)
  recommended cadence: every 2 periods  (best return horizon ≈ inf)
  lagged blend [diversify]: w_now +0.39  w_lagged +0.61  (ρ 0.28, cost 2.70%/yr → not worth the turnover cost)
```

*(Illustrative figures.)*

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
| `--strategy` | `demo_trend` | The strategy whose alpha decay is measured. |
| `--source` | `strategy` | Alpha score origin (`strategy` / `signal` / `scanner`). |
| `--symbols` | demo universe | The cross-section. |
| `--start` / `--end` | 365 days before `--end` / now | Measurement window, read in New York time. |
| `--benchmark` | `SPY` | Used to strip beta (residual returns). |
| `--max-lag` | `10` | Longest lag (periods) the IC-vs-lag profile is fit over. |
| `--timeframe` | `1Day` | Bar timeframe. |
| `--scanner` | `demo_volume` | Scanner used as the metric when `--source scanner`. |
| `--config` / `--re-resolve-universe` | none | Replay a saved [run config](run-configs)'s universe and settings. |
| `--cache` / `--offline` / `--cache-dir` | off | Reuse a local bar cache; `--offline` never touches the network. |
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
`compute_horizon` (strategy, symbols, window, source, benchmark, `max_lag`).

## What to do next

Use the recommended cadence as a starting point for how often to re-run
[`allocate --objective utility`](portfolio), and check it with the net-of-cost A/B in
[`info --policy-ab`](information#multi-period-policy-ab---policy-ab) rather than adopting
the lagged blend on this report alone.

## Common errors

- **`Missing Alpaca credentials`** - this command fetches bars, so it needs the paper key
  pair from `quantred init`. To work with no keys, warm a cache once and add `--offline`.
- **`--offline: no cached 1day bars for ... Run quantred cache warm ...`** - the cache has
  nothing for that range. The message prints the exact `cache warm` command to run (it
  needs the keys once, and the `store` extra, see [optional extras](extras)).
- **`half-life ∞ (no decay detected)`** - not an error: the fitted decay was flat or
  negative over this window, so there is no half-life to report. Read the `CI` line; a
  band reaching infinity means the window cannot tell.

## From a source checkout (contributors)

The commands above are the installed `quantred` command. In a checkout, run them as
`uv run python main.py <command> ...`. The Makefile shortcut for this page is `make horizon`. Extras are installed with `uv sync --inexact --extra <name>`
(see [optional extras](extras)).
