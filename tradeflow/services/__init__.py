"""Shared service core.

Plain, side-effect-light functions that wrap the existing engine / optimizer /
walk-forward / analytics layers and return JSON-serializable dicts. This is the
*one* orchestration code path: the CLI (``main.py``), the MCP server
(:mod:`tradeflow.mcp.server`), and the research agent (:mod:`tradeflow.research`) all call
these - no business logic lives in any of those adapters.

Nothing here can place an order: service functions take a
:class:`~tradeflow.marketdata.client.MarketDataClient` (data only) and never a broker.
"""

from tradeflow.services.analysis import (
    run_backtest,
    run_optimization,
    run_scan,
    run_walk_forward,
    summarize_bars,
)
from tradeflow.services.configs import list_configs, load_config, save_config
from tradeflow.services.glossary import metrics_glossary
from tradeflow.services.registry import (
    SCANNERS,
    STRATEGIES,
    get_param_ranges,
    list_scanners,
    list_strategies,
)

__all__ = [
    "STRATEGIES",
    "SCANNERS",
    "list_strategies",
    "list_scanners",
    "get_param_ranges",
    "run_scan",
    "run_backtest",
    "run_optimization",
    "run_walk_forward",
    "summarize_bars",
    "metrics_glossary",
    "save_config",
    "load_config",
    "list_configs",
]
