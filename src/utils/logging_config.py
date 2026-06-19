"""Centralised logging configuration.

A single ``setup_logging`` entry point keeps log formatting consistent across
every module (engine, scanner, strategies) without each one re-running
``logging.basicConfig`` and clobbering the others.
"""

import logging

_CONFIGURED = False

# A compact, greppable line format: time - logger - LEVEL - message
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


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

    logging.basicConfig(level=level, format=_LOG_FORMAT)
    _CONFIGURED = True
