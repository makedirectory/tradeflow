"""Stopping a terminal-launched MCP server: one Ctrl-C ends it, repeats cannot abort it.

The SDK reads stdin on a worker thread blocked in ``read(2)``. Nothing can cancel or
wake that thread, so before the server owned its signals a single Ctrl-C left
``anyio.run`` waiting forever (nothing printed), and a second one forced the interpreter
into finalization, where the stuck reader holds the stdin buffer lock - a third
interrupt then aborted the process with a fatal error. Observed on 3.12 and 3.14 alike,
with a pipe and with a pseudo-terminal as stdin.

These spawn the real server. Nothing in-process can show this: the failure is a thread
that outlives the interpreter's willingness to wait for it.
"""

import json
import os
import select
import signal
import subprocess
import sys
import time

import pytest

pytest.importorskip("mcp", reason="the MCP server needs the 'mcp' extra")

from tradeflow import settings  # noqa: E402
from tradeflow.mcp.selftest import default_command  # noqa: E402

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals"),
]

#: Generous against a loaded CI box, small against the unbounded hang being guarded.
BOUND = 10

_INIT = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "shutdown-test", "version": "0"},
            },
        }
    )
    + "\n"
)


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A running server whose stdin is an open pipe that never sends another byte.

    Ready means it answered ``initialize``, i.e. it is serving - not merely started, so
    a signal sent next cannot arrive before the server has installed its handlers.
    """
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    monkeypatch.setenv("APCA_API_KEY_ID", "PKDUMMYDUMMYDUMMY")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "dummydummydummydummydummydummydummydummy")
    child = subprocess.Popen(
        default_command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        child.stdin.write(_INIT.encode())
        child.stdin.flush()
        deadline = time.monotonic() + 60
        reply = b""
        while b"\n" not in reply:
            assert child.poll() is None, f"server exited early: {child.returncode}"
            assert time.monotonic() < deadline, "server never answered initialize"
            if select.select([child.stdout], [], [], 0.5)[0]:
                chunk = os.read(child.stdout.fileno(), 65536)
                assert chunk, "server closed stdout before answering"
                reply += chunk
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        for pipe in (child.stdin, child.stdout, child.stderr):
            try:
                pipe.close()
            except Exception:  # noqa: BLE001
                pass


def _finish(child, *, bound=BOUND):
    """Wait for exit within the bound; return (returncode, stderr text)."""
    try:
        child.wait(timeout=bound)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        pytest.fail(f"server still running {bound}s after being asked to stop")
    return child.returncode, child.stderr.read().decode(errors="replace")


def _assert_clean(stderr):
    for symptom in ("Fatal Python error", "Traceback", "Exception ignored", "no trial recorded"):
        assert symptom not in stderr, stderr


def test_one_interrupt_stops_the_server_within_the_bound(server):
    """Before the server owned its signals this hung with nothing printed at all."""
    server.send_signal(signal.SIGINT)

    code, stderr = _finish(server)

    assert code == 130
    _assert_clean(stderr)
    assert "MCP server stopped (interrupted)" in stderr


def test_the_stop_line_does_not_claim_nothing_was_recorded(server):
    """The CLI's generic text is true of an interrupted command, false of a server."""
    server.send_signal(signal.SIGINT)

    _, stderr = _finish(server)

    assert "0 tool call(s)" in stderr
    assert "journaled are kept in" in stderr
    assert "research_journal.jsonl" in stderr
    assert "no trial recorded" not in stderr


def test_repeated_interrupts_cannot_abort_the_interpreter(server):
    """The reported crash: further Ctrl-C during shutdown ended in a fatal abort."""
    for _ in range(4):
        server.send_signal(signal.SIGINT)
        time.sleep(0.05)

    code, stderr = _finish(server)

    assert code == 130  # an abort would be -6
    _assert_clean(stderr)


def test_terminate_stops_the_server_too(server):
    server.send_signal(signal.SIGTERM)

    code, stderr = _finish(server)

    assert code == 143
    assert "MCP server stopped (terminated)" in stderr


def test_closing_stdin_is_a_normal_shutdown(server):
    """How a client ends a session it launched: no signal, exit 0, nothing printed."""
    server.stdin.close()

    code, stderr = _finish(server)

    assert code == 0
    _assert_clean(stderr)
    assert "MCP server stopped" not in stderr


def test_the_startup_line_says_no_chat_is_attached(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    monkeypatch.setenv("APCA_API_KEY_ID", "PKDUMMYDUMMYDUMMY")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "dummydummydummydummydummydummydummydummy")
    child = subprocess.Popen(
        default_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        child.stdin.close()  # end the session at once; the startup text is already out
        _, stderr = _finish(child, bound=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert f"{settings.BRAND} MCP server (stdio) is waiting for an MCP client on stdin/stdout" in stderr
    assert "attached to no chat" in stderr
    assert "mcp --self-test" in stderr


def test_a_client_that_vanishes_does_not_leave_a_traceback(server):
    """A client that exits without closing stdin first: the next write hits a dead pipe."""
    server.stdout.close()
    ping = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}) + "\n"
    for _ in range(2):
        try:
            server.stdin.write(ping.encode())
            server.stdin.flush()
        except OSError:
            break
        time.sleep(0.3)

    code, stderr = _finish(server)

    assert code == 0
    _assert_clean(stderr)
    assert "client disconnected" in stderr


def test_the_stop_path_exits_even_when_a_log_handler_cannot_flush(monkeypatch):
    """In-process: ``stop_now`` ignores stop signals first, so a flush that raised
    before the exit would leave a process nothing but SIGKILL could end."""
    import logging

    from tradeflow.mcp import server as mcp_server
    from tradeflow.utils import streaming

    class Broken(logging.Handler):
        def emit(self, record):
            pass

        def flush(self):
            raise RuntimeError("reentrant call")

    exited = []
    monkeypatch.setattr(streaming, "_exit_process", exited.append)
    monkeypatch.setattr("signal.signal", lambda *a, **k: None)
    handler = Broken()
    logging.getLogger().addHandler(handler)
    try:
        mcp_server.stop_now("interrupted", 130)
    finally:
        logging.getLogger().removeHandler(handler)

    assert exited == [130]


def test_a_keyless_server_starts_and_says_it_is_waiting(tmp_path, monkeypatch):
    """Missing credentials used to stop the server before it answered, so it did not claim
    to be waiting. Bars are built lazily now: a keyless server really is waiting for a
    client, serves the tools that need no market data, and ends cleanly when stdin closes."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    monkeypatch.setenv("APCA_API_KEY_ID", "")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "")
    child = subprocess.Popen(
        default_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        child.stdin.close()
        _, stderr = _finish(child, bound=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert child.returncode == 0
    assert "Missing" not in stderr
    assert "waiting for an MCP client" in stderr


def test_the_stop_line_counts_the_tool_calls_the_session_handled(server):
    """The count is the one number a person reads to learn whether anything ran."""
    server.stdin.write(
        (json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode()
    )
    for i in (2, 3):
        request = {
            "jsonrpc": "2.0",
            "id": i,
            "method": "tools/call",
            "params": {"name": "list_strategies", "arguments": {}},
        }
        server.stdin.write((json.dumps(request) + "\n").encode())
    server.stdin.flush()
    seen = b""
    deadline = time.monotonic() + 30
    while not all(f'"id":{n}' in seen.decode(errors="replace").replace(" ", "") for n in (2, 3)):
        assert time.monotonic() < deadline, "tool calls were never answered"
        if select.select([server.stdout], [], [], 0.5)[0]:
            seen += os.read(server.stdout.fileno(), 65536)
    server.send_signal(signal.SIGINT)

    _, stderr = _finish(server)

    assert "handled 2 tool call(s)" in stderr
