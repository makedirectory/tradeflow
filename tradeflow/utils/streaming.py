"""Reconnecting async stream runner.

Live WebSockets drop. Both the market-data bar stream and the trade-update
stream want the same behavior: run until the connection errors, then reconnect
with capped exponential backoff, while letting cancellation (Ctrl-C) through
cleanly. That loop lives here so both share one implementation.
"""

import asyncio
import contextlib
import inspect
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

#: How long to wait for a websocket to close before moving on.
STREAM_CLOSE_TIMEOUT = 2.0


async def close_stream(stream) -> None:
    """Best-effort websocket shutdown; never raises, never blocks the loop.

    An SDK's synchronous ``stop()`` must not be called from inside the event loop.
    alpaca-py's does::

        asyncio.run_coroutine_threadsafe(self.stop_ws(), self._loop).result(timeout=5)

    which submits work to the loop and then blocks the calling thread waiting for it.
    Called from a coroutine, the caller *is* that loop, so the work can never run and
    the loop stays frozen for the whole timeout. Nothing scheduled on the loop happens
    meanwhile — including the signal callbacks that stop the process, which is how a
    shutdown ends up ignoring a second SIGTERM and a Ctrl-C, and why the only bound
    that survives a blocked loop is the watchdog thread below.

    ``stop_ws()`` is the coroutine that wrapper wraps, so awaiting it directly is the
    same shutdown without the deadlock.
    """
    closer = getattr(stream, "stop_ws", None) or getattr(stream, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if not inspect.isawaitable(result):
            return
        # Bounded, and via wait(): wait_for cancels what it waits on and then awaits
        # it, which cannot bound a close slow to answer cancellation.
        task = asyncio.ensure_future(result)
        done, _ = await asyncio.wait({task}, timeout=STREAM_CLOSE_TIMEOUT)
        if task not in done:
            logger.warning("Stream close did not finish within %gs", STREAM_CLOSE_TIMEOUT)
            task.cancel()
        else:
            task.exception()  # collect, so nothing is reported as never retrieved
    except Exception:  # noqa: BLE001 - cleanup must not mask the real error
        logger.debug("Stream close raised during shutdown", exc_info=True)


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

#: Indirection so tests can observe a forced exit instead of dying. Nothing else
#: should call ``os._exit`` directly.
_exit_process = os._exit


class _Abandoned(Exception):
    """The stopped session outlasted its teardown budget and was left behind."""


@dataclass
class _Stop:
    """The stop sequence for one :func:`run_until_stopped` call.

    Holds the state the signal handler, the watchdog and the caller all need, so none
    of them has to be threaded through as a growing list of positional arguments.
    """

    loop: asyncio.AbstractEventLoop
    task: "asyncio.Task"
    abandoned: asyncio.Event
    teardown_timeout: float
    hard_exit_grace: float
    signalled: bool = False
    watchdog: Optional[threading.Timer] = None
    installed: tuple = ()

    # -- the three ways a session can end -------------------------------------
    def request(self, sig) -> None:
        """First stop signal cancels gracefully; a second leaves immediately."""
        if self.signalled:
            self.force(f"Second {sig.name} — exiting immediately.")
            return
        self.signalled = True
        logger.info("%s received — stopping.", sig.name)
        self.task.cancel()
        # Bounds the session's own unwinding, not just the tasks it owns.
        self.loop.call_later(self.teardown_timeout, self.abandoned.set)
        # Armed here, at the moment the stop begins — not after the loop returns.
        # If the loop blocks during teardown it never returns, which is exactly the
        # case the watchdog exists for.
        self.arm_watchdog()
        self.hand_back_signals()

    def force(self, reason: str, code: int = 130) -> None:
        """Leave now, without running interpreter shutdown.

        ``sys.exit`` is not enough: interpreter exit joins the non-daemon worker
        threads ``asyncio.to_thread`` uses, and a worker blocked in a broker call
        keeps the process alive with no way left to interrupt it.
        """
        logger.error("%s", reason)
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        _exit_process(code)

    def arm_watchdog(self) -> None:
        """A last resort that does not depend on the event loop being alive.

        Every other bound here is scheduled on the loop, so all of them fail together
        the moment third-party code blocks it — which a synchronous SDK call does. A
        daemon timer thread is the only one that still fires, and so the only thing
        that makes "a stop signal always ends the process" true rather than usually
        true.
        """
        if self.watchdog is not None:
            return
        grace = self.teardown_timeout + self.hard_exit_grace
        self.watchdog = threading.Timer(
            grace,
            self.force,
            args=(
                f"Shutdown did not complete within {grace:g}s — forcing exit. Any open "
                "positions are untouched at the broker; nothing was flattened.",
            ),
        )
        self.watchdog.daemon = True
        self.watchdog.start()

    def disarm(self) -> None:
        if self.watchdog is not None:
            self.watchdog.cancel()
            self.watchdog = None

    # -- signal plumbing ------------------------------------------------------
    def install_signals(self, stop_signals) -> None:
        """Route stop signals to :meth:`request`.

        ``add_signal_handler`` is Unix-only and unavailable off the main thread; where
        it is missing the process keeps whatever behaviour it had rather than losing
        the ability to stop at all.
        """
        installed = []
        for sig in stop_signals:
            try:
                self.loop.add_signal_handler(sig, self.request, sig)
            except (NotImplementedError, RuntimeError, ValueError, AttributeError, OSError):
                logger.debug("No loop signal handler available for %s", sig)
                continue
            installed.append(sig)
        self.installed = tuple(installed)

    def hand_back_signals(self) -> None:
        """Swap the loop's handlers for plain ones that force an exit.

        Deliberately ``signal.signal`` rather than ``loop.add_signal_handler``: the
        loop delivers its signal callbacks as loop callbacks, so once something blocks
        the loop a second interrupt routed through it is never seen — which is exactly
        the moment an operator is sending one.
        """
        self.remove_signals()
        for sig in STOP_SIGNALS:
            with contextlib.suppress(Exception):
                signal.signal(sig, self._forced)

    def _forced(self, signum, _frame) -> None:
        self.force(f"Second {signal.Signals(signum).name} — exiting immediately.")

    def remove_signals(self) -> None:
        for sig in self.installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError, OSError):
                self.loop.remove_signal_handler(sig)
        self.installed = ()


async def _await_bounded(task, abandoned: asyncio.Event):
    """Await ``task``, unless it is abandoned first.

    The budget has to cover the session's *own* unwinding, not just the tasks it
    started. ``run_until_complete`` waits for a cancelled coroutine to finish its
    ``finally`` with no bound — so a cleanup that blocks holds the process open
    exactly as an unbounded gather did, one level further in.
    """
    waiter = asyncio.ensure_future(abandoned.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        raise _Abandoned
    finally:
        waiter.cancel()


def run_until_stopped(
    coro, *, teardown_timeout: float = 5.0, stop_signals=STOP_SIGNALS, hard_exit_grace: float = 10.0
):
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
    point where the running coroutine's cleanup can complete; a loop signal handler
    delivers cancellation at an await instead, so every ``finally`` on the way out
    actually runs.

    **Nothing here trusts the loop to stay alive.** Third-party code called from a
    coroutine can block it, and every loop-scheduled bound fails together when that
    happens. A daemon watchdog thread, armed the moment a stop begins, is what makes
    the guarantee hold anyway.

    The cancellation is re-raised as ``KeyboardInterrupt`` so callers keep one thing to
    catch however the stop arrived.

    Trade-clock safe: stdlib only, no unbounded waits.
    """
    loop = asyncio.new_event_loop()
    stopping = []  # non-empty once teardown has begun
    loop.set_exception_handler(_make_exception_handler(stopping))
    stop = None
    try:
        asyncio.set_event_loop(loop)
        task = loop.create_task(coro)
        stop = _Stop(
            loop=loop,
            task=task,
            abandoned=asyncio.Event(),
            teardown_timeout=teardown_timeout,
            hard_exit_grace=hard_exit_grace,
        )
        stop.install_signals(stop_signals)
        try:
            return loop.run_until_complete(_await_bounded(task, stop.abandoned))
        except asyncio.CancelledError:
            # Only translate a stop we asked for. A CancelledError from anywhere else
            # is a real outcome and must not be disguised as an interrupt.
            if stop.signalled:
                raise KeyboardInterrupt from None
            raise
        except _Abandoned:
            logger.warning(
                "The session did not finish unwinding within %gs; abandoning it.",
                teardown_timeout,
            )
            raise KeyboardInterrupt from None
        finally:
            stop.remove_signals()
    finally:
        stopping.append(True)
        try:
            _drain(loop, teardown_timeout)
        finally:
            asyncio.set_event_loop(None)
            loop.close()
            if stop is not None:
                # Everything that had to happen has happened; the watchdog would only
                # fire on a process that is already leaving.
                stop.disarm()


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
