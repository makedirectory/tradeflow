"""The MCP server driven by the real SDK client over stdio.

The in-process ``call_tool`` used elsewhere skips the output-schema validation a real
client performs, so a tool whose declared return type disagreed with its rows passed
every test while failing for everyone who connected. These tests spawn the server as
a child process and talk to it the way a chat client does.

The registry tools make no network call, so the dummy credentials below (needed only
because the server builds a data client at startup) never leave the machine.
"""

import json
import re
import shlex
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the MCP server needs the 'mcp' extra")

import anyio  # noqa: E402
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from tradeflow import cli, settings  # noqa: E402
from tradeflow.mcp.selftest import DISCLAIMER, default_command  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture
def child_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    monkeypatch.setenv("APCA_API_KEY_ID", "PKDUMMYDUMMYDUMMY")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "dummydummydummydummydummydummydummydummy")
    import os

    return dict(os.environ)


def test_a_real_client_can_discover_strategies_and_scanners_over_stdio(child_env):
    """Both discovery tools must survive client-side output validation.

    ``demo`` is a boolean in the rows; the tools used to declare string-to-string
    maps, which every validating client rejected on the first row.
    """
    command = default_command()

    async def drive():
        params = StdioServerParameters(command=command[0], args=command[1:], env=child_env)
        with anyio.fail_after(60):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    names = {t.name for t in (await session.list_tools()).tools}
                    strategies = await session.call_tool("list_strategies")
                    scanners = await session.call_tool("list_scanners")
        return names, strategies, scanners

    names, strategies, scanners = anyio.run(drive)

    assert {"list_strategies", "list_scanners"} <= names
    assert not strategies.isError and not scanners.isError
    by_name = {r["name"]: r for r in strategies.structuredContent["result"]}
    assert by_name["demo_trend"]["demo"] is True
    assert by_name["demo_trend"]["timeframe"]
    scanner_rows = {r["name"]: r for r in scanners.structuredContent["result"]}
    assert scanner_rows["demo_volume"]["demo"] is True


def test_the_server_announces_the_app_version_not_the_sdks(child_env):
    """`initialize` carries the name and version a client shows beside the server. The
    SDK fills the version in with its own when the server gives none, so the app was
    displayed under a number that belongs to a different package."""
    import tradeflow

    command = default_command()

    async def drive():
        params = StdioServerParameters(command=command[0], args=command[1:], env=child_env)
        with anyio.fail_after(60):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    return (await session.initialize()).serverInfo

    info = anyio.run(drive)

    assert info.name == settings.MCP_SERVER_NAME
    assert info.version == tradeflow.__version__


def test_self_test_reports_discovery_and_disclaims_chat_attachment(child_env, capsys):
    args = cli.parse_cli(["mcp", "--self-test"])

    args.func(args)  # returns normally on success; sys.exit(1) would raise

    out = capsys.readouterr().out
    assert "self-test passed" in out
    # The name the server announced in `initialize`, read from the child - what a client
    # displays - rather than the constant the server was built from.
    assert f"server:     {settings.MCP_SERVER_NAME}" in out
    # Not just the words: a run that found nothing prints "0 strategies (0 demo, ...)".
    assert re.search(r"[1-9]\d* strategies \([1-9]\d* demo", out), out
    assert re.search(r"[1-9]\d* scanners \([1-9]\d* demo", out), out
    assert DISCLAIMER in out
    assert shlex.join(default_command()) in out  # says which command it checked


def test_self_test_fails_rather_than_pass_on_a_server_with_no_structured_output(
    child_env, monkeypatch, capsys
):
    """Structured content first exists in mcp 1.10. A client without it hands back no
    rows, and a check that reports `0 strategies` as a pass has verified nothing."""
    from tradeflow.mcp import selftest

    monkeypatch.setattr(selftest, "_rows", lambda result: [])
    args = cli.parse_cli(["mcp", "--self-test"])
    with pytest.raises(SystemExit) as raised:
        args.func(args)
    out = capsys.readouterr().out
    assert raised.value.code == 1
    assert "no structured rows" in out and "FAILED" in out


def test_self_test_fails_when_the_builtin_demo_entries_are_missing(child_env, monkeypatch, capsys):
    from tradeflow.mcp import selftest

    monkeypatch.setattr(selftest, "_rows", lambda result: [{"name": "x", "demo": False}])
    args = cli.parse_cli(["mcp", "--self-test"])
    with pytest.raises(SystemExit):
        args.func(args)
    assert "did not list the built-in demo entries" in capsys.readouterr().out


def test_init_tells_a_reader_to_configure_the_command_the_self_test_runs(capsys, monkeypatch):
    monkeypatch.delenv("TRADEFLOW_HOME", raising=False)
    """Two definitions of "the server command" - one validated, one printed - is how a
    check passes on something no client was ever told to launch. The printed JSON and
    the Claude Code line are read back and compared with the command under test."""
    cli._print_next_steps()
    out = capsys.readouterr().out
    block = out[out.index("{") : out.rindex("}") + 1]
    server = json.loads(block)["mcpServers"][settings.MCP_SERVER_NAME]
    assert [server["command"], *server["args"]] == default_command()
    assert f"claude mcp add {settings.MCP_SERVER_NAME} -- {shlex.join(default_command())}" in out
    assert Path(server["command"]).is_absolute()


def test_self_test_names_the_step_and_exits_nonzero_when_the_server_cannot_start(
    child_env, monkeypatch, capsys
):
    """A server that dies at startup looks like a closed pipe to a client.

    The check must say where it stopped and show what the child printed, and it must
    still carry the disclaimer - a failure is no more a chat connection than a pass.
    """
    from tradeflow.mcp import selftest

    monkeypatch.setattr(
        selftest,
        "default_command",
        lambda: [sys.executable, "-c", "import sys; sys.stderr.write('boom: no credentials'); sys.exit(3)"],
    )
    args = cli.parse_cli(["mcp", "--self-test"])

    with pytest.raises(SystemExit) as raised:
        args.func(args)

    out = capsys.readouterr().out
    assert raised.value.code == 1
    assert "FAILED at step 'initialize'" in out
    assert "boom: no credentials" in out
    assert DISCLAIMER in out


def test_the_printed_client_config_carries_a_non_default_state_root_and_never_keys(
    tmp_path, monkeypatch, capsys
):
    """A desktop client does not inherit the shell that exported TRADEFLOW_HOME, so a
    config without it journals somewhere else than the root the self-test just used."""
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    monkeypatch.setenv("APCA_API_KEY_ID", "PKDUMMYDUMMYDUMMY")
    cli._print_next_steps()
    out = capsys.readouterr().out
    block = out[out.index("{") : out.rindex("}") + 1]
    server = json.loads(block)["mcpServers"][settings.MCP_SERVER_NAME]
    assert server["env"] == {"TRADEFLOW_HOME": str(tmp_path.resolve())}
    assert "PKDUMMY" not in out
    assert f"--env TRADEFLOW_HOME={shlex.quote(str(tmp_path.resolve()))} {settings.MCP_SERVER_NAME} --" in out


def test_a_default_state_root_adds_no_env_block(monkeypatch):
    from tradeflow.services import setup

    monkeypatch.delenv("TRADEFLOW_HOME", raising=False)
    assert "env" not in setup.mcp_client_config()["mcpServers"][settings.MCP_SERVER_NAME]


def test_self_test_says_so_when_credentials_came_only_from_the_shell(
    child_env, tmp_path, monkeypatch, capsys
):
    """The child inherits the shell's exports; a client launching it would not."""
    from tradeflow.services import setup

    monkeypatch.setattr(setup, "ENV_PATH", tmp_path / "absent.env")
    args = cli.parse_cli(["mcp", "--self-test"])
    args.func(args)
    out = capsys.readouterr().out
    assert "self-test passed" in out
    assert "APCA_API_KEY_ID" in out and "not the .env" in out
