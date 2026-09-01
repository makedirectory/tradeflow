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

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGTERM"), reason="POSIX signals unavailable on this platform"
)


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
    not be held by a cleanup taking its time.

    Run in a child process: the second signal restores the default handler and
    re-raises, which terminates whatever process receives it.
    """
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "session.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio, signal, sys
            from tradeflow.utils.streaming import run_until_stopped

            async def session():
                print("ready", flush=True)
                try:
                    await asyncio.sleep(60)
                finally:
                    await asyncio.sleep(60)   # cleanup that will not finish

            try:
                run_until_stopped(session(), teardown_timeout=30.0)
            except KeyboardInterrupt:
                sys.exit(7)
            """
        )
    )
    child = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    assert child.stdout.readline().strip() == "ready"

    child.send_signal(signal.SIGTERM)  # graceful; cleanup will hang for 60s
    time.sleep(0.3)
    assert child.poll() is None, "the first signal should not have exited yet"
    child.send_signal(signal.SIGTERM)  # and now: out

    # Well inside the 30s teardown budget the first signal started.
    assert child.wait(timeout=10) != 0
