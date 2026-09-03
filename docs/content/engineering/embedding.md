---
sidebar_position: 17
title: Using TradeFlow as a library
---

# Using TradeFlow as a library

TradeFlow is primarily a command-line tool, but the layer the CLI and the MCP server
both call is a clean, JSON-returning boundary — and it is usable from your own code.

```bash
uv add tradeflow-engine        # or: pip install tradeflow-engine
```

```python
from datetime import datetime

from tradeflow.services.analysis import run_verdict
from tradeflow.services.data import build_data_client

result = run_verdict(
    build_data_client(),
    "demo_trend",
    ["NVDA", "AAPL", "META"],
    datetime(2024, 1, 1),
    datetime(2024, 12, 31),
)

print(result["verdict"]["summary"])
for name, check in result["verdict"]["checks"].items():
    print(name, check["value"], "vs", check["threshold"], "->", check["passed"])
```

## What is supported, and what is not

**Supported: `tradeflow.services.*`.** Every function there takes a data-only client
and returns a plain, JSON-serializable dict. This is the same code path the CLI
renders and the MCP server exposes, so anything you can do from the terminal you can
do from Python, and you get identical numbers by construction rather than by
agreement.

| Module | What it gives you |
|---|---|
| `services.analysis` | `run_verdict`, `run_backtest`, `run_optimization`, `run_walk_forward`, `compute_alphas`, `construct_portfolio`, `compute_information`, and the rest of the research pipeline |
| `services.data` | `build_data_client` (data-only, never a broker), `resolve_universe` |
| `services.registry` | Available strategies and scanners, and their tunable ranges |
| `services.glossary` | Canonical definitions and pitfalls for every metric reported |
| `services.configs` | The promoted-config store |
| `services.setup` | Credential inspection and validation |
| `store.trials` | The campaign's trial history (`TrialStore`) |
| `analytics.htmlreport` | `render_html` — a result dict to a self-contained report |

**Not supported: everything else.** `engine`, `execution`, `strategies`,
`optimization`, `portfolio`, `risk`, `alphas`, `data`, and the rest are internal.
They are importable — nothing stops you — but they change without notice, and a
refactor that moves a class between them is not treated as a breaking change.

**`tradeflow.cli` is not an API.** It is a transport. If you find yourself importing
from it, the thing you want should probably move into `services/` first; that is the
same rule the MCP server follows.

## Things worth knowing before you embed it

**State is shared, and that is the point.** Anything that runs a trial writes to the
research journal, and the campaign's trial count is what the Deflated Sharpe deflates
against. Embedding TradeFlow in a loop that runs thousands of backtests will — 
correctly — make every subsequent result harder to clear. Set `TRADEFLOW_HOME` to
keep a project's campaign separate from your own, or pass `journal=False` /
`no_journal` where a function offers it if a run genuinely should not count.

**Results are memoized.** An identical prior run is served from the trial store
rather than recomputed, labeled `memoized` with the original timestamp. That is
usually what you want; pass `force=True` when it is not.

**Nothing here can trade.** `build_data_client` constructs only a market-data
client, never a broker — the same structural guarantee that makes the MCP server
safe. Live trading lives behind `tradeflow.engine.live` and is deliberately not part
of the supported surface.

**Optional capabilities are extras.** `tradeflow-engine[store]` for the bar cache,
`[viz]` for charts, `[optimize]` for Bayesian search, `[portfolio]` for the
constraint solver. A missing extra raises an actionable message at the point of use,
not at import.

## Stability

The project's status is **Experimental**: interfaces and gate thresholds may still
change, and there are no production users. Practically that means `services/` is
where changes are made carefully and announced, and everything else may move at any
time.

A formal commitment — curated exports, semantic versioning, internal refactors
treated as breaking — is on the roadmap but deliberately not made yet. It constrains
every future change, and that constraint should be paid for by real dependents
rather than in anticipation of them.
