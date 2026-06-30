# Build Process

**Project:** `TradeFlow` — a simple, layered, broker-agnostic algorithmic trading
engine (Alpaca adapter included; backtest + live).

This is the durable **how we work** guide for the project: it defines the build
cycle, conventions, the two-clocks invariant, architecture discipline, and the
review gate.

> Read this before starting a feature or change. A change is not done until
> implementation, offline tests, docs, and review all pass. How much you plan or
> write down before coding is up to whoever's building.

---

## 0. The one invariant: two clocks

Everything in TradeFlow respects one line, and **no convenience is worth crossing
it**:

- **Research clock** — offline, slow, exploratory. Backtest, optimize,
  walk-forward, the optimizer's GP surrogate, the MCP server, and the AI research
  agent live here. Non-determinism and LLMs are allowed; **nothing here may place
  an order.**
- **Trade clock** — live, deterministic, LLM-free: `live bar → signal → order`.
  No model, no optimizer, no database, and no network call to an LLM sits in the
  order path (`src/engine/live.py`, `src/execution/`).

Two structural guarantees enforce it, and every change must preserve them:

- **Propose-don't-apply.** Automation (the optimizer, the research agent) only ever
  *proposes* a config. A human promotes it. See `src/optimization/config_store.py`
  and `src/research/`.
- **The MCP server builds only a data client** (`src/mcp/server.py`) — it has no
  trading client wired in, so it *physically cannot* trade.

If a change would let research-clock code reach the order path, or let the trade
clock depend on a model/optimizer/LLM/DB, it is wrong regardless of how clean it
looks. Flag it in review.

---

## 1. Build cycle

Every change follows this cycle. Steps that don't apply are noted with a one-line
reason in the PR description.

```text
1. Plan (as needed)
   For anything non-trivial, think through the problem, the honest constraints,
   the math, and the failure modes before writing code — and sketch a test plan.
   We tend to do this as a short design spec, but how much you write down (a spec,
   a design note, an issue) is your call.

2. Core implementation
   Build in the correct src/ layer (see §4). Domain meaning lives in a strategy /
   scanner / engine / service; reusable mechanics live in utils/ or analytics/.
   No vendor SDK above the broker layer.

3. Offline tests
   Add deterministic pytest coverage using the fakes in tests/fakes.py — no API
   keys, no network. Cover the new mechanic (unit) and the product flow
   (integration). Add a regression test for any bug found in review.

4. Surfaces
   Wire the capability into the surfaces that apply: the CLI (main.py), an MCP
   tool (src/mcp/server.py, research-clock only), and a Makefile target. Keep CLI
   flags and tool schemas explicit and typed.

5. Docs
   Update the engineering wiki (website/docs/engineering/) for architecture/
   behavior, the usage guide (website/docs/usage/) for how to run it, and the
   README if the headline workflow changed.

6. Review
   Review the diff against §7. Treat review as a quality gate, not a courtesy —
   the two-clocks invariant is a review angle every time.

7. Land
   Open a focused PR (template + green CI). On merge, record what changed in the
   PR description (§8).
```

### Completion rule

A change is not complete until every applicable step passes. For a
research-clock-only feature, step 4 may be "MCP tool + Makefile target, no live
surface" — state that in the PR rather than skipping silently.

### Recommended sequencing

- Build in dependency order: stabilize what sits underneath before what builds on
  top of it.
- Stabilize a layer's interface (`Broker`, `MarketDataProvider`, `Strategy`,
  `Scanner`) before building consumers on top of it.
- Build the research-clock machinery and prove it offline before considering any
  trade-clock surface — most research-clock work never touches the trade clock at
  all.

---

## 2. Definition of done

Default checklist for every change.

### Implementation

- Logic lives in the correct layer (§4); dependencies point **downward**.
- No vendor SDK (`alpaca`, ...) is imported above `src/brokers/`.
- Public APIs carry type hints and docstrings; reuse `utils/` and `analytics/`
  helpers instead of re-deriving logic.
- Indicators stay pure pandas/numpy (no TA-Lib or compiled deps).
- Errors are intentional and readable; configuration is explicit (`src/settings.py`,
  validated from `.env`).
- The two-clocks invariant and propose-don't-apply guarantees are intact.

### Tests

- Unit tests cover the new mechanic; integration tests cover the product flow.
- Everything runs **offline and deterministically** via `tests/fakes.py`.
- Regression tests cover any bug found during review.
- `uv run ruff check .`, `uv run ruff format --check .`, and `make test` all pass.
- When automation is insufficient (e.g. a live paper-trading path), the manual
  smoke test is recorded in the PR.

### Documentation

- Public behavior is documented in the engineering wiki and/or usage guide.
- The README reflects any change to the headline workflow or `make` targets.
- The PR states what changed, what was verified, what was reviewed, and what was
  deferred.

### Review

- Review covers correctness, the two-clocks invariant, layering, security/data
  safety, test quality, and docs.
- Findings are fixed before merge, unless explicitly deferred to a tracked issue.

---

## 3. Cross-cutting conventions

### Documentation source of truth

- The **engineering wiki** (`website/docs/engineering/`) explains architecture and
  behavior; the **usage guides** (`website/docs/usage/`) explain how to run things.
- Once a change lands, its behavior belongs in the wiki/usage docs. Any design
  note or spec you wrote up front stays a historical rationale, not the
  source of truth for built behavior.
- Docs (and code comments) must be **self-contained**: describe the built behavior
  directly and never reference a planning spec by file or number (`spec 005`,
  `§3.2`, `specs/planning/...`). A reader should never need a planning doc to
  understand shipped behavior. Link to the relevant wiki/usage page instead, and
  for not-yet-built work name the capability generically (e.g. "a future
  information-analysis step", "the portfolio optimiser") rather than its spec.
- **Code must be self-contained: never reference specs or planning docs in code.**
  A comment should explain the behavior on its own terms; if it needs to point
  somewhere, link to the engineering wiki — never to a planning spec.

### Review after every change

Run a review before landing. It must verify correctness, the two-clocks
invariant, test/regression coverage, surface/contract completeness (CLI flags,
MCP schemas), docs, security/data safety, layering discipline, and consolidation
opportunities. Don't merge until findings are fixed or deliberately deferred.

### Compatibility policy

**Current project mode: `Clean-slate`.**

TradeFlow has no production users or stored production data; live trading is
paper-trading against Alpaca. Prefer simple, direct changes; do not add
compatibility shims unless they protect a local developer or a paper account.

| Mode | Use when | Rule |
|------|----------|------|
| Clean-slate *(current)* | No users, no production data | Prefer simple schema/code changes. No back-compat shims unless needed for local/paper safety. |
| Production | Real users, live capital, external integrations | Preserve backward compatibility; add deprecation paths, rollout/rollback plans, and feature flags as needed. |

If TradeFlow ever manages real capital, flip this section to Production
deliberately and update the order-path discipline accordingly.

### Status vocabulary

- ⬜ Not started   🔄 In progress   ✅ Complete   ⚠️ Blocked   — Not applicable

---

## 4. Architecture and layer discipline

TradeFlow is a layered engine: **dependencies point downward**, **one concern per
module**, and **no vendor SDK lives above the broker layer**. Product policy
(what a strategy/scan/engine *means*) stays separate from reusable mechanics.

### The layers (`src/`)

| Layer | Modules | Owns |
|-------|---------|------|
| Brokers | `brokers/base.py`, `brokers/alpaca/` | The **only** place a vendor SDK (`alpaca`) is imported. `Broker` / `MarketDataProvider` interfaces. |
| Market data | `marketdata/` (`client`, `synthetic`, `timeframe`) | Bar fetching and the keyless synthetic feed behind `make demo`. |
| Indicators | `indicators/` | Pure pandas/numpy signal math. |
| Strategies | `strategies/` (`base`, `volume_spike`, `ma_crossover`, `mean_reversion`, `signals`) | `bar → score` policy: each strategy defines one continuous score; the base class derives the discrete signal. One strategy per file. |
| Scanners | `scanners/` (`base`, `volume`, `symbol`) | Universe selection. One scanner per file. |
| Data | `data/` (`scan`, `panel`, `features`) | The cross-sectional substrate: the point-in-time `scan()` seam (the leakage guard) and the `FeaturePanel` every research module reads/writes. |
| Alphas | `alphas/` (`refine`, `base`, `scorers`, `combine`, `horizon`) | Continuous-alpha refinement (`α = σ·IC·z`); multi-signal combination (IC + correlation + shrinkage); alpha-decay / half-life and the lagged blend. Research-clock. |
| Risk | `risk/` (`base`, `sample`, `exposures`, `factor`) | The covariance matrix Σ — Ledoit–Wolf shrinkage *or* a structural factor model (`XFXᵀ+Δ`, attributable into factor/specific) — plus tracking error / MCR. Research-clock; never in the order path. |
| Costs | `costs/` (`base`, `parametric`) | Transaction-cost model (commission + half-spread + √-impact) charged in the backtest so metrics are net. Research-clock; the live path uses real fills. |
| Engine | `engine/backtest.py`, `engine/live.py` | The backtest loop and the **sacred trade-clock** live loop. |
| Execution | `execution/` (`live_trader`, `sizing`) | Order placement and position sizing — trade-clock. |
| Analytics | `analytics/` (`metrics`, `performance`, `reporting`, `charts`, `information`) | Honest evaluation metrics, reports, result charts, and information analysis (IC / breadth / IR reconciliation). |
| Optimization | `optimization/` (`optimizer`, `param_space`, `walk_forward`, `config_store`) | Research-clock parameter modeling + the walk-forward fitness function. |
| Portfolio | `portfolio/` (`allocator`, `optimizer`) | OR-Tools scalar-score sizing (operational); mean-variance utility construction `αᵀw − λ·wᵀΣw` from alpha + Σ (research proposal). |
| Research | `research/` (`agent`, `proposer`, `llm`, `sandbox`) | The autonomous, LLM-driven, **propose-only** research loop. |
| MCP | `mcp/server.py` | Read-only agent surface — builds a data client only; cannot trade. |
| Services | `services/` | The domain-service layer that the CLI and surfaces call into (see inventory). |
| Settings | `settings.py` | Validated credentials/config from `.env`. |
| Utils | `utils/` (`numeric`, `timeutils`, `streaming`, `logging_config`) | Neutral reusable mechanics. |

Entry points (`main.py` CLI commands, `mcp/server.py` tools, the `*_all_*.py`
account scripts) handle transport only and delegate into services/strategies/
engine. Keep business logic out of `main.py`.

### Shared service inventory

`src/services/` is the consolidation layer the CLI and other surfaces route
through. Keep this table current as services are added.

| Module | Responsibility | Why it's shared |
|--------|----------------|-----------------|
| `services/registry.py` | Resolve strategies/scanners by name | One lookup used by CLI, MCP, optimizer, research |
| `services/data.py` | Build the right market-data provider (synthetic vs. live) | Every command needs bars from a consistent source |
| `services/analysis.py` | Run backtest/scan analysis flows | Shared by CLI and MCP without duplicating wiring |
| `services/sizing.py` | Position-sizing policy entry point | Backtest and live both size positions the same way |
| `services/configs.py` | Load/store proposed strategy configs | The propose-don't-apply boundary |
| `services/audit.py` | Auditable record of runs/decisions | Traceability across surfaces |
| `services/glossary.py` | Shared terminology/metric definitions | One source for docs and tool descriptions |

### When to extract a shared service / util

Extract a reusable mechanic when more than one caller needs it, the same
operational code is being duplicated, or a bug fix should propagate everywhere
doing the same thing. **Skip** extraction for one-off, single-caller domain logic —
premature abstraction is also debt. Keep neutral modules (`utils/`, `analytics/`)
free of imports from heavy domain graphs to avoid cycles.

### Anti-patterns to avoid

- Importing `alpaca` (or any vendor SDK) above the broker layer.
- Business logic in `main.py` or in an MCP tool handler.
- Any model/optimizer/LLM/DB dependency reaching `engine/live.py` or `execution/`.
- An MCP tool or research module that constructs a trading client.
- Strategies/scanners that span more than one concern, or share state sideways.
- Import cycles from putting shared mechanics inside a domain package.

---

## 5. Refactor and consolidation discipline

After implementing a change, before merging, do a consolidation pass:

1. Scan the new code for duplication against `services/`, `utils/`, `analytics/`.
2. Route through an existing service/helper when one already exists.
3. Extract a new service/util only when the mechanic will clearly be reused.
4. Confirm the import graph still points downward with no new cycle.
5. Run `make test` (and `ruff check`/`format --check`) — all green.
6. Record what was extracted/consolidated, which callers migrated, and which
   tests prove behavior held, in the PR.
7. Log intentional deferrals against a tracked issue.

---

## 6. Dependency source of truth

When behavior depends on how a third-party dependency actually works, inspect the
source rather than guessing. This matters most for:

- `alpaca-py` order/data-client behavior and the trade-clock order path.
- pandas/numpy edge cases (NaN handling, alignment, dtype coercion) in indicators
  and metrics.
- `scikit-learn` GP surrogate behavior in the Bayesian optimizer, `ortools`
  solver semantics, and the `mcp` / `anthropic` / `openai` client contracts.

### Tools (Python / uv)

```bash
uv pip show <pkg>                 # version + location
python -c "import x, inspect; print(inspect.getsource(x.fn))"
rg "<symbol>" .venv/lib/python*/site-packages/<pkg>   # read installed source
```

### Rule

If a correctness or security decision hinges on dependency behavior, cite the
dependency file, symbol, and version in the PR.

---

## 7. Review guide

### Review angles

- Line-by-line correctness and regression risk.
- **Two-clocks invariant** — does anything let research-clock code trade, or pull
  a model/optimizer/LLM/DB into the order path? Is propose-don't-apply intact?
- **Layering** — dependencies downward, one concern per module, no vendor SDK
  above the broker layer.
- Numeric correctness (NaN/alignment/dtype) in indicators, metrics, walk-forward.
- Test coverage gaps and whether tests stay offline/deterministic.
- Surface/contract completeness (CLI flags, MCP tool schemas) and docs.
- Consolidation opportunities against `services/`/`utils/`/`analytics/`.

### Review output format

```md
### Findings

1. **[Severity] [Title]**
   - Problem: [What is wrong]
   - Impact: [Why it matters]
   - Fix: [What changed or should change]
   - Coverage: [Regression test or verification]

### Verified clean
- [Important concern checked and found safe — e.g. "live.py still LLM-free"]

### Deferred
- [Follow-up] → [Tracked issue]
```

### Severity levels

| Severity | Meaning |
|----------|---------|
| Critical | Crosses the two-clocks line, data loss, leaked credential, or guaranteed-broken core flow |
| High | Serious bug, broken important edge case, bad contract, layering violation |
| Medium | Narrower incorrect behavior, test gap, maintainability issue |
| Low | Cleanup, clarity, minor doc or UX issue |

---

## 8. Recording what changed

Merge only after applicable implementation, offline tests, docs, and review are
complete and review fixes are merged or deferred. Capture what changed in the PR
description so the history stays legible:

```md
> **[Change name].**
> **Scope:** [What was built and in which layers.]
> **Verification:** [make test result, ruff, manual paper smoke test if any.]
> **Docs:** [Wiki/usage pages, README.]
> **Two clocks:** [How the invariant + propose-don't-apply stayed intact.]
> **Fixed during review:** [Findings fixed before merge.]
> **Deferred:** [Follow-ups → tracked issue.]
```

---

## 9. Project-specific overrides

### Stack

- Core: Python ≥3.10, pandas/numpy, pytz.
- Broker/data: `alpaca-py` (broker layer only).
- Optional extras (opt-in, base install stays light): `optimize` (scikit-learn),
  `portfolio` (ortools), `mcp`, `ai` (anthropic), `openai`, `viz` (matplotlib).
- Env/tooling: `uv` (app, `package=false`), `.env` via `src/settings.py`.
- Tests: pytest, offline & deterministic via `tests/fakes.py`.
- Lint/format: ruff (line length 110, E501 off, isort `known-first-party=["src"]`).
- Docs: Docusaurus site in `website/`.

### Commands

```bash
# Install (deps + ruff + pytest + optimize/portfolio/viz extras)
uv sync --extra dev

# Try it now — full pipeline on synthetic data, no keys, no network
make demo

# Run the offline test suite
make test            # == uv run --extra dev pytest -q

# Lint / format (exactly what CI runs)
uv run ruff check .
uv run ruff format --check .

# Docs site
make docs            # serve at http://localhost:3000
make docs-build      # static build

# Other preconfigured flows: make backtest | scan | optimize | live | allocate
make help            # list every target
```

CI (`.github/workflows/ci.yml`) runs ruff check, ruff format --check, and pytest
on every PR and on push to `main`. Work on a focused branch
(`feat/* | improve/* | refactor/* | chore/*`), open a PR with the template
filled out, and merge only on green CI.

### Non-negotiables

- **The two clocks never cross** (§0): no model/optimizer/LLM/DB in the order
  path; automation only proposes, a human promotes; the MCP server can't trade.
- **No vendor SDK above the broker layer.** Everything else goes through `Broker`
  / `MarketDataProvider`.
- **Tests stay offline and deterministic** via `tests/fakes.py` — no keys, no
  network in the suite.
- **Indicators stay pure pandas/numpy** — no TA-Lib or compiled deps.
- **One concern per module; dependencies point downward.**
- **Think before you build:** non-trivial work starts with a plan (a design
  spec, note, or issue — your call) that names the failure modes and a test plan.

---

*Conventions cross-reference the engineering wiki under
[`website/docs/engineering/`](website/docs/engineering/coding-standards.md).*
