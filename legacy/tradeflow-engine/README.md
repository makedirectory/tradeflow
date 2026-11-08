# tradeflow-engine

This project is now **[quantred](https://pypi.org/project/quantred/)**.

This final release (2.3.1) contains no engine. It installs `quantred` and keeps two old
surfaces working, so an existing environment, script or MCP client configuration does not
break on upgrade:

- `import tradeflow` (and `tradeflow.<submodule>`) resolves to the same modules as
  `quantred`, and emits a `DeprecationWarning` (hidden by default filters unless raised from `__main__`; shown under `-W default` or in test runs).
- the `tradeflow` command runs the same program as `quantred`.

Your state directory (`~/.tradeflow`, `TRADEFLOW_HOME`) is unchanged, and packs that
register under the old `tradeflow.strategies` / `tradeflow.scanners` entry-point groups are
still discovered. The extras (`tradeflow-engine[mcp]` and the rest) forward to the same
extras of `quantred`.

## Moving fully to quantred

```bash
pip uninstall tradeflow-engine && pip install quantred
# or, for a uv tool install:
uv tool uninstall tradeflow-engine && uv tool install quantred
```

Then change a pack's imports from `tradeflow` to `quantred`, and its dependency from
`tradeflow-engine` to `quantred`. MCP client configs should launch `quantred mcp`.

The full migration notes are at
<https://quantred.mk-dir.com/usage/migrating-to-quantred>.
