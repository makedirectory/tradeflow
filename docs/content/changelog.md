---
sidebar_position: 99
title: Changelog
---

# Changelog

TradeFlow started in June 2023 as a personal project and has been rebuilt more than
once since. It is not a product, and it has never managed real money — it exists
partly for the pleasure of building it, and partly to demonstrate something worth
demonstrating: **the machinery serious quantitative research runs on is buildable by
one person.** Structural risk models, cost-aware portfolio construction, walk-forward
validation with honest multiple-testing correction, point-in-time data discipline —
none of it requires a hedge fund's resources. It requires knowing which parts matter
and being unwilling to fool yourself about the results.

That second goal is why so many entries below describe *refusals* rather than
features: a gate that would not pass, a default left off because its evidence did not
clear, a number that turned out to be an artifact. The hard part of this domain is
not computing a Sharpe ratio. It is building something that will tell you your idea
does not work.

Every release, newest first. The project's status is **Experimental** — interfaces
and gate thresholds may still change, and it has no production users or live capital.

Dates are release dates. Versions before 2.0.0 were never published to a package
index; they are tagged in the repository.

---

## 2.0.1 — 2026-08-07

- **Fixed the broken screenshot on the PyPI project page.** The README referenced the
  demo image by a repository-relative path, which GitHub resolves and PyPI cannot —
  PyPI renders a README standalone, with no repository context. It is now an absolute
  URL, which works in both places.
- Bumped the GitHub Actions used by CI and the release workflow off Node 20, which
  GitHub is deprecating.

No functional changes.

---

## 2.0.0 — 2026-08-06

The first release published to PyPI, as **`tradeflow-engine`**. Also the release that
made the research machinery usable by someone who did not write it: before this,
every capability existed but assumed you had the repository, the flags, and the
context in your head.

### Breaking

- **The package is now `tradeflow`, not `src`.** Every import moves:
  `from src.services.analysis import …` → `from tradeflow.services.analysis import …`.
  A package named `src` cannot be installed — it would collide with every other
  project that shipped one — so this was the precondition for distributing anything
  at all.
- **The distribution is `tradeflow-engine`.** The bare `tradeflow` name on PyPI
  belongs to an unrelated project that also imports as `tradeflow`; installing both
  into one environment would collide. The command and the importable package remain
  `tradeflow`.
- **The Docker image no longer defaults to `live`.** `docker run tradeflow` with no
  arguments used to start a paper-trading loop; it now prints help. Turning the
  machine on must not turn trading on.

### Added

- **`tradeflow` as an installed command.** `uv tool install tradeflow-engine`, no
  clone required. State resolves to `TRADEFLOW_HOME`, else a checkout, else
  `~/.tradeflow`; `tradeflow --version` and `init --check` both print which copy is
  running and where its state lives.
- **`verdict`** — the whole cross-sectional pipeline as one command over one
  universe, one window, and one cost model, ending in a single gate-derived verdict
  with every check shown. Running the steps by hand produced reports that looked
  joined-up while each re-resolved its own universe.
- **`init`** — guided first-run setup: hidden prompts, fully masked output, credential
  validation through the data-only client, and `--check`, a doctor that writes
  nothing and makes no network call.
- **`--html`** on `verdict`, `backtest`, `walkforward`, and `info` — one
  self-contained report file that issues zero network requests when opened, with
  provenance as a mandatory header and honesty labels as banners rather than
  footnotes.
- **`trials list` / `show` / `best`** — the campaign's memory, browsable: filters,
  SQL-level paging, per-trial detail (including which later trials reused it), and a
  leaderboard ranked by *deflated* Sharpe with every row's family trial count
  attached.
- **`--workers N`** on `optimize` and `walkforward` — parallel candidate evaluation
  with the single-writer invariant intact. Measured 1.85× on four workers, with an
  identical winner and an identical trial count.
- **MCP research surface completed** — `run_verdict`, `render_report`, and read-only
  trial-store tools, plus an audit of every tool description against current
  behavior, anchored to the shared metric glossary.
- **Live-path hardening** — bar-quality guards that reject (never repair) a bad bar,
  an append-only position ledger reconciled against the broker, a `reconcile` verb,
  and the loop-level test fence the live engine never had.
- **`docker compose`** for a local development stack, with state on named volumes
  that survives container replacement.
- **`tradeflow.services.*` documented as the supported library API** for embedding.

### Fixed

- An HTML-escaping hole in the report renderer: a symbol list could inject script
  through the provenance header.
- A crash in the live loop when a data feed switched timestamp awareness mid-stream.
- Optimizer ranking was not a total order, so a tie's winner depended on evaluation
  order — in the sequential path too, not only under parallelism.
- `construct_portfolio` hardcoded its cost model, so cost flags could never reach it.
- The engineering docs claimed the portfolio tool over MCP solved a cost-blind book;
  it had been cost-aware since the default changed.

### Notes

- The compatibility posture remains **clean-slate**. Live-path hardening is the
  *precondition* for trading real capital, built ahead of that decision; the decision
  itself is a separate, deliberate change.
- Every research run journals a trial and counts toward the campaign's
  multiple-testing total. Embedding this in a loop that runs thousands of backtests
  will correctly make every later result harder to clear.

---

## 1.0.1 — 2026-08-05

- **The Alpaca SDK is confined to the broker layer**, for real. A broker factory
  (`build_broker` / `build_market_data`) means entry points and services construct
  vendor-backed objects through the abstraction instead of importing the SDK
  directly — making the architecture page's "the only place `import alpaca` appears"
  claim true rather than aspirational.
- Regenerated the dependency lockfile, which still recorded a stale version.

---

## 1.0.0 — 2026-08-05

The portfolio layer, the research infrastructure that makes a campaign's history
count, and an honest capital model.

### Portfolio construction

- **Benchmark as a held portfolio** — active weights, true tracking error in active
  space, alpha neutralized against the benchmark, and reverse optimization for the
  consensus returns that make the benchmark itself optimal.
- **Black–Litterman posterior** blending consensus with our views, so coverage holes
  get a real propagated posterior instead of an implicit zero-view. Defaults off
  until validated out-of-sample.
- **Long/short books** — market-neutral construction with a mandatory gross-leverage
  cap and short-side borrow carry, plus a report pricing what the long-only
  constraint actually costs.
- **Multi-period trading policy** — aim in front of the target: alphas discounted by
  their measured decay, then a partial adjustment composed with the cost-aware
  no-trade band. Defaults off; the net-of-cost A/B does not clear on this
  repository's own demo data, which is a legitimate outcome rather than unfinished
  work.

### Risk and analytics

- **Regime-conditioned volatilities** on slow correlations, with a built-in
  predictive-accuracy gate and a net-of-cost A/B that decide adoption rather than
  preference. Default off — the gate does not clear on the demo data.
- **Return attribution** splitting realized active return into benchmark timing,
  risk factors, signals, and stock-picking, by an exact regression identity.
- **Bootstrap inference** — a block bootstrap for a single track record's own
  zero-alpha null, and a joint reality check across every trial ever run.

### Research infrastructure

- **The trial store.** A SQLite index over the research journal, so the
  multiple-testing correction counts every configuration tried across a whole
  campaign rather than resetting each session. Identical trials are served from it
  instead of re-run, unmistakably labeled with the original run's age.
- **A local bar cache** with gap-filling and an offline mode, so a repeated study
  reuses data instead of re-fetching it.
- **`demo-agent`** — a narrated research session on real market data, deterministic
  by default, with the LLM proposer able to author strategy code that a sandbox
  admits or rejects.

### The capital model

**One clock, one capital pool.** Every symbol now simulates on a merged timeline
against shared cash with portfolio-level position limits and a per-bar
mark-to-market equity curve. Previously each symbol ran its own full-capital
backtest and the results were summed — which meant return and Sharpe scaled with
universe size, an artifact rather than a finding. Transaction costs are charged
through optimization and validation, and an unrunnable backtest is now
distinguishable from a strategy with no edge.

---

## 0.3.0 — 2026-07-03

The active-management spine: the classical quantitative-equity framework, built end
to end on a point-in-time data substrate.

- **Continuous alphas.** A per-name signal becomes a cross-sectionally comparable
  residual-return forecast (`α = σ·IC·z`) through a pure pipeline — winsorize,
  standardize, neutralize, scale, cap. Strategies migrated to score-first: each
  defines one continuous conviction score, and both the discrete signal and the
  alpha derive from it.
- **The feature panel** — a cross-sectional, point-in-time data spine with an
  as-of seam that everything above runs on, so leakage is prevented structurally
  rather than by discipline.
- **Risk models** — Ledoit–Wolf shrinkage, then a structural factor model with a
  factor/specific split.
- **Transaction costs** as a first-class model (commission, spread, square-root
  impact, borrow carry), with backtests net of cost by default.
- **Mean-variance portfolio construction** in pure numpy, with the transfer
  coefficient and risk aversion calibrated to a target tracking error — then the
  cost term moved *inside* the objective, so a no-trade band emerges from the cost
  itself.
- **Information analysis** — measured information coefficient, effective breadth
  deflated by average correlation, and predicted-versus-realized information ratio,
  with the research-integrity guardrails that keep a lucky backtest honest.
- **Multi-signal combination** by information coefficient and correlation, so
  redundant signals split a weight rather than double-counting.
- **Information horizon** — signal decay and half-life, driving rebalance cadence.
- **Out-of-core storage and lazy compute** — a date-partitioned columnar bar store
  behind the as-of seam, with streaming covariance bounded by universe size rather
  than by history length.
- **Forecast refinement** — score-scaling case selection, an
  estimation-uncertainty haircut on the alpha level, and an equal-risk diagnostic.

---

## 0.2.0 — 2026-06-24

- **Surfaced the two-clocks model.** The research-clock / trade-clock split is the
  single idea that makes the safety story cohere, and it was buried. Promoted, with
  a diagram, to the top of the README and the architecture documentation.
- Quickstarts now lead with `make demo` — try it before signing up for anything.

---

## 0.1.0 — 2026-06-19

The rebuild into a layered, broker-agnostic engine — and a deliberate discarding of
most of what came before it (see below). The earlier code could place orders; it had
no way to tell you whether it should.

- **Strict separation of concerns** across brokers, market data, indicators,
  strategies, scanners, execution, analytics, engine, optimization, portfolio, and
  utilities — with dependencies pointing downward and the engine orchestrating
  rather than owning.
- **Broker and market-data interfaces**, isolating vendor SDK usage inside adapters
  so another venue means writing one adapter and nothing else.
- **No TA-Lib.** Indicators are pure pandas and numpy, so installation needs no
  compiler and the Docker image carries no build toolchain.
- **Honest evaluation from the start** — risk, tail, and trade-level metrics
  including the probabilistic and deflated Sharpe ratios; walk-forward validation
  with a sacred holdout and promotion gates; an MCP server that is structurally
  incapable of trading; and an autonomous research loop where only candidates that
  survive out-of-sample, leakage-checked, gated validation ever reach a human.

---

## Before the rebuild — 2023 to 2024

The first two and a half years, kept here because the shape of the project came out
of what these versions got wrong.

**June to October 2023 — the original.** Market-data endpoints, order placement, the
first attempts at a scanner, and a README rewritten more times than the code. It
could talk to a broker and do something in response to a price. What it could not do
was answer whether that something was a good idea, which turned out to be the entire
problem.

**October 2024 — the streaming rewrite.** Live market-data and account-action
streams, websocket lifecycle handling that closed cleanly on exit, subscription
management, a file watchdog for live reloads, and paper trading moved behind
configuration rather than a code edit. This is where the live path grew up, and
several of its ideas survived the rebuild intact — the streaming reconnect logic in
particular.

**What did not survive:** everything above the venue. There was no separation between
deciding and doing, no way to test a strategy without a network, and no evaluation at
all — so every result was in-sample by construction, and nothing could distinguish a
real edge from a lucky window. The 2026 rebuild kept the broker abstraction's
lessons, threw out the rest, and started from the question the earlier versions could
not answer.
