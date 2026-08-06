# Changelog

Notable changes per release. Dates are release dates; the project's status remains
**Experimental** — interfaces and gate thresholds may still change.

## 2.0.0 — 2026-08-06

The first release published to PyPI, as **`tradeflow-engine`**. Also the release that
made the whole research machinery usable by someone who did not write it.

### Breaking

- **The package is now `tradeflow`, not `src`.** Every import moves:
  `from src.services.analysis import …` → `from tradeflow.services.analysis import …`.
  A package named `src` cannot be installed — it would collide with every other
  project that shipped one — so the rename was the precondition for distributing
  anything at all.
- **The distribution is `tradeflow-engine`.** The bare `tradeflow` on PyPI belongs to
  an unrelated project that also imports as `tradeflow`; installing both into one
  environment would collide. The command and the importable package remain
  `tradeflow`.
- **The Docker image no longer defaults to `live`.** `docker run tradeflow` with no
  arguments used to start a paper-trading loop; it now prints help. Turning the
  machine on must not turn trading on.

### Added

- **`tradeflow` as an installed command** — `uv tool install tradeflow-engine`, no
  clone required. State resolves to `TRADEFLOW_HOME`, else a checkout, else
  `~/.tradeflow`; `tradeflow --version` and `init --check` both print which copy is
  running and where its state lives.
- **`verdict`** — the whole cross-sectional pipeline as one command over one
  universe, one window, and one cost model, ending in one gate-derived verdict with
  every check shown.
- **`init`** — guided first-run setup with hidden prompts, masked output, data-only
  credential validation, and `--check`, a doctor that writes nothing.
- **`--html`** on `verdict`/`backtest`/`walkforward`/`info` — one self-contained
  report file that makes zero network requests when opened.
- **`trials list` / `show` / `best`** — the campaign's memory, browsable: filters,
  SQL-level paging, per-trial detail, and a leaderboard ranked by *deflated* Sharpe
  with every row's family trial count attached.
- **`--workers N`** on `optimize`/`walkforward` — parallel candidate evaluation.
  Measured 1.85× on 4 workers with an identical winner and trial count.
- **MCP research surface completed** — `run_verdict`, `render_report`, and read-only
  trial-store tools, plus a description audit anchored to the shared glossary.
- **Live-path hardening** — bar-quality guards that reject (never repair) a bad bar,
  an append-only position ledger reconciled against the broker, a `reconcile` verb,
  and the loop-level test fence the live engine never had.
- **`docker compose`** for a local dev stack with state on named volumes.
- **`tradeflow.services.*` documented as the supported library API** for embedding.

### Fixed

- An HTML-escaping hole in the report renderer: a symbol list could inject script
  through the provenance header.
- A crash in the live loop when a feed switched timestamp awareness mid-stream.
- Optimizer ranking was not a total order, so a tie's winner depended on evaluation
  order — in the sequential path too, not only under parallelism.
- `construct_portfolio` hardcoded its cost model, so cost flags could never reach it.

### Notes

- The compatibility posture remains **Clean-slate**. Live-path hardening is the
  *precondition* for trading real capital, built ahead of that decision; the decision
  itself is a separate, deliberate change.
- Every research run journals a trial and counts toward the campaign's
  multiple-testing total. Embedding this in a loop that runs thousands of backtests
  will correctly make every later result harder to clear.

## 1.0.1 and earlier

Pre-publication. See the git history and `specs/complete/` for the research-engine
arc: honest evaluation and walk-forward validation, the active-management spine
(alphas, risk, costs, cost-aware construction, information analysis), and the
post-book work that followed it.
