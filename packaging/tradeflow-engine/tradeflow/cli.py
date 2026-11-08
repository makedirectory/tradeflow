"""The old ``python -m tradeflow.cli`` entry, for MCP client configs written before the rename.

This is a real file, unlike every other ``tradeflow.*`` module (which the import shim in
``tradeflow/__init__.py`` maps onto ``quantred``): ``python -m`` runs through ``runpy``, which
needs a loader that can hand back code for the module, and an alias loader has none.
"""

import sys

from quantred import cli as _cli
from quantred.cli import main

__all__ = ["main"]


def __getattr__(name):
    # Anything else a caller reached for on the old module is quantred's.
    return getattr(_cli, name)


if __name__ == "__main__":
    sys.exit(main())
