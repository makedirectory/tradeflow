---
sidebar_position: 2
title: Design philosophy
---

# Design philosophy

TradeFlow makes a handful of opinionated bets that explain almost every decision
in the codebase. They are worth stating plainly, because the hard part of an
algorithmic-trading system is not the trading — it is *not fooling yourself*, and
these principles are how the design defends against that.

## 1. Two clocks that never touch

The single idea that organizes everything: the system runs on two clocks with
different rules.

- **The research clock** is offline, slow, and exploratory — scanning, backtesting,
  optimizing, out-of-sample validation, and (optionally) an AI research assistant.
  Non-determinism is allowed here. It only ever *proposes*: it writes
  provenance-stamped candidate configurations to disk.
- **The trade clock** is live and deliberately dull — `live bar → signal → order`.
  No model, no optimizer, and no database sits in the order path, so there is
  nothing to prompt-inject and nothing non-deterministic to debug when real money
  is at stake.

Keeping these apart is a structural guarantee, not a convention. See
[Architecture](architecture) for how the layers enforce it.

## 2. Promote by hand — automation never flips the switch

Nothing automated ever moves money. The research clock emits a config and a
rationale; a human reads it and chooses to promote it to live trading. No process
flips paper-to-live or places an order on its own. Where the system exposes itself
to tooling, that boundary is enforced *structurally* — the integration layer is
built with a data client only, so it physically cannot trade.

## 3. Honest fitness, or nothing

It is trivially easy to manufacture a beautiful backtest; an engine that optimizes
and reports on the *same* slice of history is just a machine for discovering noise.
The defense is built in, not bolted on:

- **Optimize on one slice of time, score on a slice the optimizer never saw**, and
  reserve a final holdout that is touched exactly once.
- **Correct for the number of things you tried.** Statistical-robustness metrics
  (deflated/probabilistic Sharpe and friends) deflate results by how many
  configurations were tested to find them.
- **Gate, don't cherry-pick.** A candidate is "promotable" only if it clears every
  threshold — out-of-sample Sharpe, profit factor, drawdown ratio, a leakage probe,
  parameter-sensitivity, a minimum trade count — so neither a human nor an agent can
  pick the one lucky fold.

The result: more automation makes the engine *more* honest, not faster at
overfitting.

## 4. Strict separation of concerns, behind interfaces

Each layer does one job, and everything above the broker layer is written against
a `Broker` / `MarketDataProvider` interface rather than a vendor SDK. Adding a new
venue is one adapter; nothing else changes. Because the boundary is an interface,
the whole system is testable offline with in-memory fakes — no keys, no network.
See [Separation of Concerns](separation-of-concerns).

## 5. Everything heavy is opt-in

The base install is small and has no native build step. Anything expensive — the
optimizer's ML backend, the constraint solver, AI providers — sits behind an
optional extra and is imported lazily. You can explore the whole research surface
with `uv sync` and `make demo`, no keys required.

## 6. Built for scale, from the start

A trading system lives or dies on time-series data volume. The data layer is
designed so that growing from a handful of symbols to a broad, high-frequency,
multi-year universe is a change of *storage tier*, not a rewrite: data is exchanged
in a columnar, zero-copy form and processed lazily, so larger-than-memory panels
stream rather than blow up. Nothing above the data layer assumes the panel fits in
RAM.

## Where it's headed

The current engine runs one strategy at a time and judges it on absolute return.
The direction of travel is toward a genuine *active portfolio management* system —
the same discipline used by quantitative equity desks:

- **Forecasts as numbers, not labels.** Turn a strategy's signal into a
  comparable estimate of expected risk-adjusted return, so positions reflect
  conviction and many signals can be weighed against each other.
- **A real risk model.** Move portfolio construction from "weight the
  best-scoring names" to trading expected return off against *covariance* —
  diversification and tracking error become first-class, not afterthoughts.
- **Costs as first-class citizens.** Charge realistic transaction costs in the
  backtest and let them shape both position sizing and how often you trade, so
  reported performance is honest *net* of friction.
- **Measuring skill, not luck.** Decompose returns into the part that came from
  cheap factor exposure versus genuine selection, and compare the skill you
  *predicted* against the skill you *realized*.

Every one of these lives on the research clock. The trade clock stays exactly as
simple as it is today — that is the whole point.
