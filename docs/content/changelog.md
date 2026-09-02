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

## 2.2.0 — 2026-09-02

Live-path validation against a real paper account, and the defect it eventually found.
Everything here came from *running* the thing — the preflight, the ledger, the execution
report and finally a trade table each surfaced something the full test suite agreed was
fine.

### The headline

**The backtest transacted one bar before it could have known.** A signal at bar `i` comes
from scores computed on bar `i`'s *close*, and the engine executed it against bar `i`'s
*open* — for entries and signal exits alike. That is a one-bar look-ahead applied to
every trade in every result this project had recorded.

It survived because nothing could see it. A feed shift moves signal and price together,
so the leakage probe passed over it; and it made the backtest structurally impossible to
match live, where a closed bar produces a signal and a market order fills afterwards.
Three days of narrowing — a fill-assumption stress, a position-size cap, a per-trade
excursion study — each eliminated a hypothesis and left the anomaly intact, until the
question became "is this information causally available at all".

`ACCOUNTING_VERSION` is 4. **Every number recorded under 1–3 overstates what a deployment
could achieve**, and the trial store keeps the two apart. Causality is now asserted on
both clocks, in one file, because the two implement it separately and nothing in the
codebase connects them.

### Added

- **`tradeflow execution-report`** — what the live path actually did, reconstructed from
  the ledger: slippage in basis points, decision-to-fill latency, submitted-versus-filled
  notional, modelled cost beside observed fees, and refusals grouped by kind. Ungraded on
  purpose: what counts as bad slippage is not knowable from one session.
- **A live preflight**, printed on every run and exiting early under `--preflight`. It
  states the contract before any order logic: broker mode, account balance beside the
  capital this run may deploy, data feed, every book limit in the units it is enforced in,
  telemetry destinations, and how many symbols actually warmed up — and whether they
  warmed up *enough*, which is a different question.
- **`--fill-stress`** re-runs a backtest requiring the price to trade progressively
  further *through* each take-profit before it counts as filled. The default fills a
  target the moment a bar touches it, which models a resting limit always first in the
  queue; for a strategy whose gain concentrates in target exits, that assumption is the
  result rather than a detail.
- **Book limits on both clocks** — `--max-positions`, `--max-position-size`,
  `--max-gross-exposure`, `--max-net-exposure`, `--max-total-risk`, `--min-notional`.
  `max_net_exposure` is new and bounds directional tilt: gross bounds long + short and
  cannot see direction, so a book inside a gross cap can be entirely one-directional. A
  backtest of a long/short book now derives a cap from the tilt it actually carried, and
  says when the gross cap already subsumes it.
- **`--capital`**, so a run deploys what a config was validated at rather than whatever
  equity a paper account was handed. **`--feed`**, pinning the historical and streaming
  halves to one data feed. **`walkforward --config`**, so the run type that produces a
  config can consume one.
- Every backtest prints net P&L by exit reason, and says so when one winning exit carries
  over 90% of the gain.

### Fixed

- **The ledger counted fills twice and recorded every short as a long.** A venue
  re-reports an order's cumulative filled quantity on each partial fill and again on the
  final one, and those events were summed; separately the trade-update type had no `side`
  field, so a defaulted `"buy"` reached every record. An order that filled 8 arrived as
  21, and a 31-share short as +31.
- **A resumed session disagreed with its own ledger**, and a reconciliation sweep landing
  between an entry's submission and its fill could erase a position the strategy held —
  leaving it flat in a symbol it owned, and a strategy that believes it is flat cannot
  emit an exit.
- **Shutdown hung until a second interrupt, and `SIGTERM` was unhandled entirely.** Four
  passes, each correct and each still wrong one level up, ending at a synchronous SDK
  `stop()` that blocks the very loop that would have to run its close. A blocked loop
  defeats every loop-scheduled bound, signal handlers included, so the one that must
  always hold now lives on a daemon thread.
- **Blocking broker calls ran on the event loop**, stalling other symbols' bars, fill
  delivery and reconciliation for the length of every entry. They run in a worker thread
  now, with the order path serialized behind one semaphore — a correctness requirement,
  not a speed-up.
- **A failed market-data fetch returned "no bars" instead of raising**, so an unreachable
  provider produced a zero-trade backtest that was journaled as an evaluated trial and
  then served from cache to the next identical run. Absent is not zero, and unreachable
  is not absent.
- **Risk-limit flags never reached the live book**, and a flag that cannot reach anything
  now stops the run instead of being silently ignored.
- Position limits went through no cache identity at all, so two runs differing only in
  `max_gross_exposure` hashed alike and the second was answered from the first.

---

## 2.1.0 — 2026-08-10

Trade-clock hardening. The live path is the smallest and least-tested part of this
project, which is backwards — it is the only part that can lose money. Most of what
follows was found by *running* it rather than reading it.

### Fixed

- **The live path could open a position but never close one.** A strategy decides
  whether an exit is legitimate by looking itself up in its own position book, and
  nothing in live mode ever wrote that book — so every `CLOSE_BUY`/`CLOSE_SELL` was
  rewritten to `HOLD` before execution saw it, and a position could only be closed by
  its broker-side bracket legs, if it had any. The book is now rebuilt from broker
  truth: before the first bar, on every sweep, and when an entry is placed. The unit
  tests had been passing because they fed `handle_signal` directly, exercising the
  layer below the break.
- **A strategy could go permanently silent with nothing logged.** Warm-up history is
  timezone-aware and a feed may stream naive timestamps; mixing them made every later
  comparison raise inside `process_bar`, whose blanket `except` turned that into a
  strategy that emitted nothing at all, indefinitely. Timestamps are aligned before
  append, and the `except` now says what it swallowed.
- **Warm-up fetched a fraction of the history it asked for.** The lookback converted
  bars to wall-clock time directly, which counts the overnight gap, the weekend and
  every holiday as tradeable: fifty one-minute bars became "100 minutes ago", so a
  09:35 start warmed a fifty-bar indicator with five bars and ran anyway. Daily
  under-fetched too, less visibly. The window is now measured in sessions, and a
  short warm-up is reported rather than absorbed.
- **Missed signal edges were lost for good.** Entries fire on the crossing bar and
  never again, so a bar rejected by a quality guard, a dropped stream, or a restart
  left the score saying "should be long" while every bar emitted `HOLD`. Live mode
  now compares the direction the score implies against the position book and
  re-states the difference. Exits always; entries under `reaffirm_entries`, on by
  default.

### Added

- **A kill switch.** `tradeflow halt | resume | halts` records a durable decision to
  stop, and `tradeflow flatten --confirm` halts, cancels every order, and closes
  every position — going straight to the broker, so it works when the engine is
  wedged. Halt state is a file, not a database: the order path must hold no
  connection to anything that can be down. Halts block entries and never exits, so
  the switch cannot trap the book and flatten cannot deadlock against its own gate.
- **Deterministic client order ids.** The guard against double-submitting was a
  question asked of the broker immediately before placing an order — a check-then-act
  race with no memory across a restart. Each order now carries an id derived from the
  decision behind it, so a venue that has already accepted it rejects the duplicate.
- **Typed broker failures.** Every broker call used to fail as `None` or `False`, so
  a rate limit, an expired token, insufficient funds and a deliberate rejection all
  produced the same non-answer. Two cases were actively wrong: a duplicate order read
  as a failed submission, and `list_positions` returning `[]` — the claim that the
  account is flat — when the broker could not be reached.
- **A decision record for every signal.** "No order" had a dozen causes and one
  representation. Execution now returns what it decided, why, and which guards it
  consulted — the guards that *ran*, not just the one that fired — and the ledger
  records the declines, which are the case that leaves no other trace.

### Changed

- **An unreadable market clock no longer uniformly means "assume open".** That is
  right for a transient blip, but it was also being applied to revoked credentials,
  which are never transient. Authentication failures now fail closed.
- **A strategy started while the score already implies a position takes it** on the
  first live bar, rather than waiting for the next crossing. Turn it off with
  `--no-reaffirm-entries`.
- **Internal trade-clock interfaces changed shape.** `Broker.submit_*` take a
  `client_order_id`; the order-path methods raise `BrokerError` rather than
  returning `None`/`False`; `LiveTrader.handle_signal` returns a `Decision` rather
  than an optional order. No supported API moved — `tradeflow.services.*` is the
  supported surface and is untouched — but anyone who had reached past it into the
  broker or execution layer will need these. Hence a minor, not a major.

## 2.0.4 — 2026-08-07

### Fixed

- **`trials best` ranked in-sample search results as if they were track records.**
  An `optimize` row is the winner of a parameter search — best-of-N by construction,
  which is the selection bias the whole evaluation stack exists to correct for — and
  the leaderboard ranked them alongside validated results without even showing the
  kind. On a real campaign, four of the top five rows were search artifacts and
  nothing said so. In-sample kinds are now excluded by default (with the count of
  exclusions reported, and `--include-in-sample` to opt back in), and every row names
  its kind. Found by using the tool rather than reading it.

### Added

- `CLAUDE.md` and `.claude/rules/` — the conventions this project is actually built
  to, as path-scoped rules that load only when a matching file is touched: the
  two-clocks invariant, the honesty rules for anything the tool prints, the trade
  clock's reject-never-repair discipline, and the verify-don't-assert habit that has
  caught most of the real defects here.

## 2.0.3 — 2026-08-07

Found by walking the getting-started flow end to end against real paper credentials,
using the published package rather than a checkout.

### Fixed

- **`tradeflow-engine[mcp]` installed a broken server.** The MCP SDK released 2.0.0,
  which removed the server class this integration is built on, and the dependency was
  an unconstrained `mcp>=1.0` — so a fresh install resolved to a version that could
  not be imported. It was invisible locally because the lockfile held a working 1.x.
  Now constrained to `<2`, with tests that check the *resolved* SDK rather than
  trusting the lockfile.
- **`init --check` suggested `uv sync --extra …`**, which an installed copy cannot
  run — the third place this same checkout-only assumption turned up.
- **`PAPER_TRADE` was masked as `****`.** It is not a secret, and hiding it cost the
  reader the one value they most want to confirm.
- **Reports said `git unknown`** when run from an installed copy. A provenance block
  exists so a result can say what produced it; it now falls back to the package
  version, which is the honest answer when there is no repository.

### Changed

- **The documentation site now deploys itself** on every push to `main`. It was
  published by hand, so it silently fell behind the repository — a page could be
  written, reviewed, merged, and linked from the README while 404ing for every
  reader.
- **The home page leads with `uv tool install tradeflow-engine` and `tradeflow
  demo`**, rather than the clone-and-`uv sync` instructions it still carried from
  before the package existed.

## 2.0.2 — 2026-08-07

Onboarding fixes, all found by installing the published package and following the
path a new user would.

- **The missing-credentials message sent installed users to dead ends.** It told them
  to copy a `.env.example` they do not have and run a `make` target that does not
  exist, and never mentioned `tradeflow init` — the one command that fixes it. It is
  now phrased for the copy that is running, and names the exact path the `.env` goes
  to.
- **`tradeflow mcp` crashed with a traceback** when the `mcp` extra was not
  installed, instead of saying so. The guard wrapped only the import, but the SDK is
  pulled in lazily further down, so the real failure escaped — at exactly the moment
  someone is first trying to connect Claude. The install hint it prints is now
  correct for an installed copy too.
- **Added [Getting started](usage/getting-started)** — one walkthrough from install
  to a real result with Claude connected, in six steps, replacing a set of correct
  pages with no thread between them.
- The README's MCP registration snippet assumed a checkout; it now shows the
  installed form first, and its tool list is no longer two releases stale.

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
