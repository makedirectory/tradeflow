"""Reconnecting async stream runner.

Live WebSockets drop. Both the market-data bar stream and the trade-update
stream want the same behavior: run until the connection errors, then reconnect
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
            logger.info("%s stream canceled; shutting down.", name)
            raise
        except Exception as exc:  # noqa: BLE001 - reconnect on any stream error
            logger.error("%s stream error (%s); reconnecting in %.0fs", name, exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)


def run_until_stopped(coro, *, teardown_timeout: float = 5.0):
    """Run ``coro`` on a private loop, guaranteeing the process can exit afterwards.

    ``asyncio.run`` cannot be used for a streaming session. On the way out it cancels
    whatever is still running and then *awaits all of it with no bound*, so a stream
    slow to close its socket holds the process open until a second interrupt — and the
    second one lands during cleanup, where it can interrupt anything.

    This drains instead: cancel, wait a bounded moment, and leave. Stragglers have
    their results collected either way, because a task carrying an unretrieved
    exception prints one at exit — which is how a clean shutdown ends up looking like
    a crash.

    Trade-clock safe: stdlib only, no unbounded waits.
    """
    loop = asyncio.new_event_loop()
    stopping = []  # non-empty once teardown has begun
    loop.set_exception_handler(_make_exception_handler(stopping))
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        stopping.append(True)
        try:
            _drain(loop, teardown_timeout)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _make_exception_handler(stopping: list):
    """Quiet the two exceptions that only ever mean "we are stopping".

    A task that finished before the drain began is already gone from
    ``asyncio.all_tasks``, so its result can no longer be collected by hand - and an
    unretrieved one is dumped to stderr with a full traceback at interpreter exit.
    After a Ctrl-C that is pure noise, and it reads as a crash on the way out of a
    clean shutdown.

    A task still pending when the loop closes is reported a second time by its
    destructor. That one is genuinely worth knowing, which is why :func:`_drain` logs
    a plain line counting them; repeating it as a traceback adds nothing.

    Narrow on purpose. Only ``KeyboardInterrupt`` and ``CancelledError``, and only the
    destructor notice, and only once teardown has begun. Everything else reaches the
    default handler, because a shutdown that hides real failures is worse than one
    that prints something ugly.
    """

    def handler(loop, context) -> None:
        exception = context.get("exception")
        if isinstance(exception, (KeyboardInterrupt, asyncio.CancelledError)):
            logger.debug("Ignoring %s surfaced during shutdown", type(exception).__name__)
            return
        if stopping and "was destroyed but it is pending" in str(context.get("message", "")):
            logger.debug("Pending task destroyed during shutdown: %s", context.get("task"))
            return
        loop.default_exception_handler(context)

    return handler


def _retrieve(task) -> None:
    """Collect a finished task's outcome so nothing is reported as never retrieved."""
    if task.done() and not task.cancelled():
        task.exception()


def _drain(loop, timeout: float) -> None:
    """Cancel everything still running and give it a bounded moment to unwind."""
    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if not pending:
        return
    for task in pending:
        task.cancel()
    try:
        loop.run_until_complete(asyncio.wait(pending, timeout=timeout))
    except Exception:  # noqa: BLE001 - shutdown must not raise over the real outcome
        logger.debug("Exception while draining pending tasks", exc_info=True)
    stuck = 0
    for task in pending:
        if task.done():
            _retrieve(task)
        else:
            stuck += 1
            # It will never be awaited, so claim its outcome in advance.
            task.add_done_callback(_retrieve)
    if stuck:
        logger.warning("%d background task(s) did not stop within %gs; exiting anyway.", stuck, timeout)
