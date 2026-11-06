"""QuantRed MCP server package.

Opt-in: requires the ``mcp`` extra (the command that needs it prints the install line for the copy you
are running). The ``mcp`` import lives only inside :mod:`tradeflow.mcp.server` and is imported lazily by the ``mcp``
subcommand, so the base install and other commands never pull it in.
"""
