#!/usr/bin/env python
"""``python main.py <command>`` — the checkout entry point.

A shim, deliberately. The CLI itself lives in :mod:`tradeflow.cli`, which is also
what the installed ``tradeflow`` command points at, so the two are the same code
reached two ways rather than two things that have to be kept in agreement.

Kept at the repo root because every Makefile target, doc, and habit refers to it.
"""

from tradeflow.cli import main

if __name__ == "__main__":
    main()
