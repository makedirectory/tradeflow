---
sidebar_position: 17
title: Transaction costs
---

# Transaction costs

Until a cost model charges trading, every Sharpe and equity curve is **gross** — and
gross results are the most reliable way to fool yourself, because the strategies that
look best in-sample are very often the highest-turnover ones that costs destroy.
`tradeflow/costs/` prices the cost of changing a position so research metrics are **net by
default**.

:::note Research clock only
This models cost for *simulation*. The live path gets real fills from the broker; it
imports no cost model.
:::

## The decomposition

For a trade of `q` shares at price `p` in a name with average daily volume `ADV` and
quoted spread `s`:

```
cost($) = commission + (s/2)·|q|·p + impact_rate·|q|·p
```

- **Commission** — a flat per-notional broker fee (default 1 bp).
- **Half-spread** — `(s/2)·|q|·p`, the immediate cost of crossing (default 5 bp spread).
- **Market impact** — the **square-root law** (Almgren et al., the robust default):

  ```
  impact_rate = η · σ_daily · √(|q| / ADV)
  ```

  `|q|/ADV` is participation (the fraction of a day's volume you demand), `σ_daily`
  the name's daily volatility, `η` a coefficient (~0.3 default). Impact is concave
  per share (√) but **convex in total cost** (`|q|·impact_rate ∝ |q|^{3/2}`) — exactly
  the property that makes spreading a target across names cheaper than dumping it on
  one. A linear-participation fallback is offered for a convex-quadratic optimizer.

## As an alpha haircut

Amortizing the round-trip cost over the holding period gives a cost rate per unit
time, which is the honest input to portfolio construction:

```
α_net = α − round_trip_cost_rate / holding_period_years
```

A +4%/yr alpha with a 2% round-trip held for a month is *not* a +4% opportunity — it's
deeply negative. This is what stops a high-IC, high-turnover signal from looking
profitable when it isn't.

## Integration

- **Backtest** (`engine/backtest.py`): every fill is charged on **both legs** (entry
  and exit), using a **trailing** (as-of) ADV and volatility so a past trade's cost
  never depends on future volume. Metrics become net; `total_cost` and the gross
  final capital are reported alongside for the haircut attribution. **Net is the
  default** at the service/CLI layer; `--gross` disables the charge.
- **Portfolio construction** ([engineering](./portfolio-construction#cost-inside-the-objective-cost-aware-by-default)):
  the cost is priced **inside** the optimizer's objective, not just as an ex-post
  drag — `turnover_cost_rate` (the linear `cᵢ`) and `impact_coefficient` (the
  conic `kᵢ`) are the same functions the objective's proximal solve uses, so the
  optimizer trades a name's alpha against *that name's* cost and a no-trade band
  emerges from it. This needed no new solver dependency: the cost term's exact
  proximal operator is closed-form, solved by the same 1-D budget bisection the
  cost-free projection already used.
- **Participation cap**: a trade demanding more than ~10% of a day's volume is flagged
  — the seed of a [capacity analysis](./information-analysis#attribution-and-capacity)
  (how much capital before cost eats the edge).

## Borrow (short financing)

Holding a **short** accrues a borrow cost over time: `borrow_rate · notional ·
holding_years` (default 50 bp/yr, `--borrow-bps`). The backtest charges it per short
position by how long it was held, so a short-heavy strategy isn't silently flattered.
Long-side margin financing and leverage costs remain out of scope.

## Where it runs

`ParametricCostModel` in `tradeflow/costs/` — the single source of the √-impact
coefficient, shared by the backtest and the [cost-aware
optimizer](./portfolio-construction#cost-inside-the-objective-cost-aware-by-default)
so the two price the same model. Surfaced as backtest flags (`--gross`,
`--commission-bps`, `--impact-eta`), `allocate --objective utility`'s
`--gross-objective`/`--holding-period`/`--capital`, and folded into
`compute_risk`-adjacent flows; the backtest report and the MCP `run_backtest`
tool both return net metrics + `total_cost`.
