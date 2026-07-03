"""Centralized logging configuration.

A single ``setup_logging`` entry point keeps log formatting consistent across
every module (engine, scanner, strategies) without each one re-running
``logging.basicConfig`` and clobbering the others.
"""

import logging

_CONFIGURED = False

# An aligned, greppable, pipe-delimited line:
#   2026-06-19 14:03:22 | INFO     | src.engine.backtest      | message
# The fixed-width level and logger columns keep multi-line runs easy to scan.
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once per process.

    Safe to call from module import time in many modules; only the first call
    actually configures handlers.

    Args:
        level: Minimum log level for the root logger.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    _CONFIGURED = True
