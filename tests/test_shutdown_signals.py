"""Stopping a streaming session, by either signal.

The shutdown reasoning in the engine — bounded stream teardown, the report of what was
still open — was reachable only by a human pressing Ctrl-C in a terminal. Under
systemd, docker, `kill`, or any supervisor, the process gets SIGTERM and none of it
ran. These pin that both signals take the same path.
"""

import asyncio
import os
import signal
import time

import pytest

from tradeflow.utils.streaming import run_until_stopped


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """Stop sequences install process-wide handlers on purpose.

    Leaving one behind lets a later test's signal reach a previous test's `_Stop`,
    whose loop is closed and whose exit hook is no longer patched — so the leak shows
    up as an unrelated hang rather than as a failure here.
    """
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGTERM"), reason="POSIX signals unavailable on this platform"
)


def _spawn(tmp_path, body, name):
    """Start a child session and wait, with a deadline, until it says it is running.

    A file handshake rather than reading a pipe: a blocking `readline` on a child that
    dies early, or whose output is buffered, hangs the suite instead of failing it.
    """
    import subprocess
    import sys
    import textwrap

    ready = tmp_path / f"{name}.ready"
    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(body).replace("READY_FILE", repr(str(ready))))
    child = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + 30
    while not ready.exists():
        assert child.poll() is None, f"child exited early with {child.returncode}"
        assert time.monotonic() < deadline, "child never became ready"
        time.sleep(0.05)
    return child


def _session(record, hold=0.0):
    """A coroutine whose cleanup must run, however the stop arrives."""

    async def session():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            record.append("cancelled")
            raise
        finally:
            if hold:
                await asyncio.sleep(hold)
            record.append("cleanup")

    return session()


def _stop_with(sig, delay=0.05):
    async def fire():
        await asyncio.sleep(delay)
        os.kill(os.getpid(), sig)

    return fire()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_either_stop_signal_runs_the_session_cleanup(sig):
    """The bug, for SIGTERM: the stream's finally never ran, the engine's finally never
    ran, the open-position report never printed, exit code 143."""
    record = []

    async def both():
        asyncio.ensure_future(_stop_with(sig))
        await _session(record)

    with pytest.raises(KeyboardInterrupt):
        run_until_stopped(both(), teardown_timeout=1.0)

    assert record == ["cancelled", "cleanup"]


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_stop_signal_surfaces_as_keyboardinterrupt(sig):
    """One thing for callers to catch, so the report of what was left open prints
    whether the stop came from a keyboard or a supervisor."""

    async def both():
        asyncio.ensure_future(_stop_with(sig))
        await asyncio.sleep(30)

    with pytest.raises(KeyboardInterrupt):
        run_until_stopped(both(), teardown_timeout=1.0)


def test_cleanup_that_overruns_the_budget_does_not_hold_the_process():
    """Cleanup is given a bounded moment, not an unbounded one."""
    record = []

    async def both():
        asyncio.ensure_future(_stop_with(signal.SIGTERM))
        await _session(record, hold=30)

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        run_until_stopped(both(), teardown_timeout=0.2)

    assert time.monotonic() - started < 2.0


def test_a_cancellation_we_did_not_ask_for_is_not_disguised_as_an_interrupt():
    """Only a stop we signalled becomes KeyboardInterrupt. Anything else is a real
    outcome and must not be reported as though somebody pressed Ctrl-C."""

    async def cancels_itself():
        task = asyncio.current_task()
        task.cancel()
        await asyncio.sleep(1)

    with pytest.raises(asyncio.CancelledError):
        run_until_stopped(cancels_itself(), teardown_timeout=0.2)


def test_a_session_that_finishes_on_its_own_returns_its_value():
    """Both directions: the stop path must not be the only path out."""

    async def quick():
        return "done"

    assert run_until_stopped(quick(), teardown_timeout=0.2) == "done"


def test_a_failing_session_still_raises_its_own_error():
    async def fails():
        raise RuntimeError("stream died")

    with pytest.raises(RuntimeError, match="stream died"):
        run_until_stopped(fails(), teardown_timeout=0.2)


def test_the_signal_handlers_are_removed_afterwards():
    """A handler left installed would silently swallow the next stop, and the process
    that owns the terminal afterwards is not ours to reconfigure."""
    before = signal.getsignal(signal.SIGTERM)

    async def quick():
        return None

    run_until_stopped(quick(), teardown_timeout=0.2)

    assert signal.getsignal(signal.SIGTERM) == before


def test_a_second_signal_exits_immediately(tmp_path):
    """The first stop is graceful; an operator who sends another wants out now and must
    not be held by a cleanup taking its time."""
    child = _spawn(
        tmp_path,
        """
        import asyncio, pathlib, sys
        from tradeflow.utils.streaming import run_until_stopped

        async def session():
            pathlib.Path(READY_FILE).write_text("go")
            try:
                await asyncio.sleep(60)
            finally:
                await asyncio.sleep(60)   # cleanup that will not finish

        try:
            run_until_stopped(session(), teardown_timeout=30.0, hard_exit_grace=300.0)
        except KeyboardInterrupt:
            sys.exit(7)
        """,
        "second_signal",
    )

    child.send_signal(signal.SIGTERM)  # graceful; cleanup will hang
    time.sleep(0.5)
    assert child.poll() is None, "the first signal should not have exited yet"
    child.send_signal(signal.SIGTERM)  # and now: out

    assert child.wait(timeout=15) != 0


# --- when the loop itself is blocked ----------------------------------------------
# Every bound in the shutdown path except one is scheduled on the event loop, so all
# of them fail together the moment third-party code blocks it. A synchronous SDK call
# does exactly that: alpaca-py's `stop()` submits a coroutine to the running loop and
# then blocks the caller waiting for it, which from inside that loop can never
# complete. These cover the one bound that survives.


@pytest.fixture
def forced(monkeypatch):
    """Capture a forced exit instead of taking the interpreter down with it."""
    calls = []
    monkeypatch.setattr("tradeflow.utils.streaming._exit_process", lambda code: calls.append(code))
    return calls


def _stop(**kwargs):
    from tradeflow.utils.streaming import _Stop

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(0))
        defaults = dict(
            loop=loop,
            task=task,
            abandoned=asyncio.Event(),
            teardown_timeout=0.05,
            hard_exit_grace=0.05,
        )
        stop = _Stop(**{**defaults, **kwargs})
        try:
            yield stop
        finally:
            # Before close(): asyncio registers a wakeup fd for its signal handlers,
            # and closing the loop without removing them leaves that fd dangling on a
            # closed pipe. The damage lands on whatever reuses the descriptor next,
            # which is how this showed up as an unrelated subprocess test hanging.
            stop.remove_signals()
            # And disarm: a watchdog outliving its test fires into a later one, after
            # the patched exit hook has been restored to the real os._exit.
            stop.disarm()
    finally:
        loop.close()


def test_the_watchdog_is_armed_when_the_stop_begins_not_after_it_finishes(forced):
    """The bug: it was armed after `run_until_complete` returned — which is precisely
    what does not happen when the loop is blocked during teardown."""
    for stop in _stop():
        stop.request(signal.SIGTERM)

        assert stop.watchdog is not None and stop.watchdog.is_alive()
        stop.watchdog.join(2.0)
        assert forced == [130], "the watchdog did not force an exit"


def test_a_healthy_session_is_never_put_on_a_timer(forced):
    """Both directions: arming unconditionally would kill a slow but working session."""
    for stop in _stop():
        assert stop.watchdog is None
        stop.disarm()

    assert forced == []


def test_disarming_stops_the_watchdog_firing(forced):
    """A stop that completes must not then be shot by its own backstop."""
    for stop in _stop(teardown_timeout=0.2, hard_exit_grace=0.2):
        stop.request(signal.SIGTERM)
        stop.disarm()

    time.sleep(0.6)
    assert forced == []


def test_a_second_signal_forces_an_exit(forced):
    for stop in _stop():
        stop.request(signal.SIGTERM)
        stop.request(signal.SIGTERM)

    assert forced == [130]


def test_the_second_handler_does_not_go_through_the_loop(forced):
    """The loop delivers its signal callbacks as loop callbacks, so a second interrupt
    routed through a blocked loop is never seen — exactly when one is being sent."""
    for stop in _stop():
        stop.install_signals((signal.SIGTERM,))
        stop.request(signal.SIGTERM)

        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), "SIGTERM was left on the loop"

        handler(signal.SIGTERM, None)
        assert forced == [130]


def test_one_signal_kills_a_process_whose_loop_is_deadlocked(tmp_path):
    """End to end, in a child process, against the real failure.

    A short paper session logged "SIGTERM received", then "Shutting down live
    streams", and stayed alive for over a minute; a second SIGTERM and a Ctrl-C were
    both ignored, and it took SIGKILL. The loop was blocked, so every loop-scheduled
    bound — including the signal handlers themselves — was unreachable.
    """
    child = _spawn(
        tmp_path,
        """
        import asyncio, pathlib, sys
        from tradeflow.utils.streaming import run_until_stopped

        async def session():
            pathlib.Path(READY_FILE).write_text("go")
            try:
                await asyncio.sleep(60)
            finally:
                # The SDK's shape: submit to this loop, then block this loop on it.
                loop = asyncio.get_running_loop()
                fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop)
                try:
                    fut.result(timeout=300)
                except Exception:
                    pass

        try:
            run_until_stopped(session(), teardown_timeout=1.0, hard_exit_grace=1.0)
        except KeyboardInterrupt:
            sys.exit(0)
        """,
        "deadlocked",
    )

    child.send_signal(signal.SIGTERM)  # one signal, and only one

    assert child.wait(timeout=20) != 0  # forced, not a clean return
