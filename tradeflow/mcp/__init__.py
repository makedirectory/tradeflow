"""QuantRed MCP server package.

Opt-in: requires the ``mcp`` extra (``tradeflow init --check`` prints the install command for
the copy you are running). The ``mcp`` import
lives only inside :mod:`tradeflow.mcp.server` and is imported lazily by the ``mcp``
subcommand, so the base install and other commands never pull it in.
"""
