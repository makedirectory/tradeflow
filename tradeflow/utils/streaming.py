"""Reconnecting async stream runner.

Live WebSockets drop. Both the market-data bar stream and the trade-update
stream want the same behavior: run until the connection errors, then reconnect
with capped exponential backoff, while letting cancellation (Ctrl-C) through
cleanly. That loop lives here so both share one implementation.
"""

import asyncio
import logging
import signal
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


#: Signals that mean "stop this session". SIGTERM matters as much as SIGINT: it is
#: what a supervisor, a container runtime, and plain ``kill`` send, and a shutdown
#: reachable only by a human at a keyboard does not exist in any of those.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def run_until_stopped(coro, *, teardown_timeout: float = 5.0, stop_signals=STOP_SIGNALS):
    """Run ``coro`` on a private loop, guaranteeing the process can exit afterwards.

    ``asyncio.run`` cannot be used for a streaming session. On the way out it cancels
    whatever is still running and then *awaits all of it with no bound*, so a stream
    slow to close its socket holds the process open until a second interrupt — and the
    second one lands during cleanup, where it can interrupt anything.

    This drains instead: cancel, wait a bounded moment, and leave. Stragglers have
    their results collected either way, because a task carrying an unretrieved
    exception prints one at exit — which is how a clean shutdown ends up looking like
    a crash.

    **Stopping is signal-driven, not exception-driven.** A bare ``KeyboardInterrupt``
    unwinds from wherever the interpreter happened to be, which is not necessarily a
    point where the running coroutine's own cleanup can complete; a loop signal handler
    delivers cancellation at an await instead, so every ``finally`` on the way out
    actually runs. The first signal cancels; a second restores the default handler, so
    an operator who wants out immediately still gets out immediately.

    The cancellation is re-raised as ``KeyboardInterrupt`` so callers keep one thing to
    catch however the stop arrived.

    Trade-clock safe: stdlib only, no unbounded waits.
    """
    loop = asyncio.new_event_loop()
    stopping = []  # non-empty once teardown has begun
    signalled = []  # non-empty once a stop signal has been seen
    loop.set_exception_handler(_make_exception_handler(stopping))
    try:
        asyncio.set_event_loop(loop)
        task = loop.create_task(coro)
        abandoned = asyncio.Event()
        installed = _install_stop_handlers(loop, task, signalled, stop_signals, abandoned, teardown_timeout)
        try:
            return loop.run_until_complete(_await_bounded(task, abandoned))
        except asyncio.CancelledError:
            # Only translate a stop we asked for. A CancelledError from anywhere else
            # is a real outcome and must not be disguised as an interrupt.
            if signalled:
                raise KeyboardInterrupt from None
            raise
        except _Abandoned:
            logger.warning(
                "The session did not finish unwinding within %gs; abandoning it.",
                teardown_timeout,
            )
            raise KeyboardInterrupt from None
        finally:
            _remove_stop_handlers(loop, installed)
    finally:
        stopping.append(True)
        try:
            _drain(loop, teardown_timeout)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


class _Abandoned(Exception):
    """The stopped session outlasted its teardown budget and was left behind."""


async def _await_bounded(task, abandoned: asyncio.Event):
    """Await ``task``, unless it is abandoned first.

    The budget has to cover the session's *own* unwinding, not just the tasks it
    started. ``run_until_complete`` waits for a cancelled coroutine to finish its
    ``finally``, with no bound - so a cleanup that blocks holds the process open
    exactly as the unbounded gather did, one level further in.
    """
    waiter = asyncio.ensure_future(abandoned.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        raise _Abandoned
    finally:
        waiter.cancel()


def _install_stop_handlers(
    loop, task, signalled: list, stop_signals, abandoned: asyncio.Event, teardown_timeout: float
) -> list:
    """Route stop signals to a cancel of ``task``. Returns the signals actually hooked.

    ``add_signal_handler`` is Unix-only, and unavailable off the main thread; where it
    is missing the process keeps whatever behaviour it had rather than losing the
    ability to stop at all.
    """
    installed = []
    for sig in stop_signals:
        try:
            loop.add_signal_handler(
                sig, _request_stop, loop, task, signalled, sig, abandoned, teardown_timeout
            )
        except (NotImplementedError, RuntimeError, ValueError, AttributeError, OSError):
            logger.debug("No loop signal handler available for %s", sig)
            continue
        installed.append(sig)
    return installed


def _remove_stop_handlers(loop, installed) -> None:
    for sig in installed:
        try:
            loop.remove_signal_handler(sig)
        except (NotImplementedError, RuntimeError, ValueError, OSError):
            logger.debug("Could not remove the signal handler for %s", sig)


def _request_stop(loop, task, signalled: list, sig, abandoned, teardown_timeout: float) -> None:
    """First stop signal cancels; a second hands the signal back to the default."""
    if signalled:
        logger.warning("Second %s — exiting immediately.", sig.name)
        _remove_stop_handlers(loop, [sig])
        signal.raise_signal(sig)
        return
    signalled.append(sig)
    logger.info("%s received — stopping.", sig.name)
    task.cancel()
    # Starts the clock on the session's own unwinding, not just on the tasks it owns.
    loop.call_later(teardown_timeout, abandoned.set)


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
        if stopping and isinstance(exception, (KeyboardInterrupt, asyncio.CancelledError)):
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
    except KeyboardInterrupt:
        # Deliberately not caught: a second interrupt during the drain means "now",
        # and skipping straggler retrieval to honour that is the right trade. The
        # loop is still closed by the caller's finally.
        raise
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
