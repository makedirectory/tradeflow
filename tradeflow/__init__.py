"""TradeFlow — a layered, broker-agnostic algorithmic-trading research engine.

The package root deliberately imports nothing. Every subpackage pulls in something
heavy or optional (pandas, the Alpaca SDK, matplotlib, the MCP SDK), and a root that
eagerly imported them would make ``tradeflow --help`` pay for capabilities the
command is not using — and would make an absent optional extra an import error at
startup rather than an actionable message at the point of use.

The layers, in dependency order (each may import from below, never above):

    brokers -> marketdata -> indicators -> strategies/scanners -> engine
                                        -> alphas/risk/portfolio/analytics
                                        -> services -> cli / mcp

The one invariant that outranks all of them: **two clocks**. The research clock
(backtest, optimize, walk-forward, the whole analytics stack) may be slow,
exploratory, and LLM-assisted; it only ever *proposes*. The trade clock
(``engine.live``) is deterministic and imports none of it. Promotion between them is
a manual human step.
"""

#: The single source of truth for the version — packaging reads it from here, so a
#: published artifact and what `tradeflow --version` reports cannot drift apart.
__version__ = "2.0.0"
