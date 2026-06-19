"""Reconnecting async stream runner.

Live WebSockets drop. Both the market-data bar stream and the trade-update
stream want the same behaviour: run until the connection errors, then reconnect
with capped exponential backoff, while letting cancellation (Ctrl-C) through
cleanly. That loop lives here so both share one implementation.
"""

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


async def run_with_reconnect(
    name: str,
    connect: Callable[[], Awaitable[None]],
    *,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
) -> None:
    """Run ``connect()`` repeatedly, reconnecting on error with backoff.

    Args:
        name: Label for log messages (e.g. "market-data").
        connect: Zero-arg coroutine function that establishes the stream and
            runs until it ends or raises. It is responsible for its own cleanup
            (e.g. in a ``finally``).
        base_delay / max_delay: Backoff bounds in seconds.
    """
    delay = base_delay
    while True:
        try:
            await connect()
            logger.info("%s stream ended normally.", name)
            return
        except asyncio.CancelledError:
            logger.info("%s stream cancelled; shutting down.", name)
            raise
        except Exception as exc:  # noqa: BLE001 - reconnect on any stream error
            logger.error("%s stream error (%s); reconnecting in %.0fs", name, exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
