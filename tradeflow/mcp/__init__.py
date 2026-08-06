"""TradeFlow MCP server package.

Opt-in: requires the ``mcp`` extra (``uv sync --extra mcp``). The ``mcp`` import
lives only inside :mod:`tradeflow.mcp.server` and is imported lazily by the ``mcp``
subcommand, so the base install and other commands never pull it in.
"""
