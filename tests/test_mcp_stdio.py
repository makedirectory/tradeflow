"""The MCP server driven by the real SDK client over stdio.

The in-process ``call_tool`` used elsewhere skips the output-schema validation a real
client performs, so a tool whose declared return type disagreed with its rows passed
every test while failing for everyone who connected. These tests spawn the server as
a child process and talk to it the way a chat client does.

The registry tools make no network call, so the dummy credentials below (needed only
because the server builds a data client at startup) never leave the machine.
"""

import sys

import pytest

pytest.importorskip("mcp", reason="the MCP server needs the 'mcp' extra")

import anyio  # noqa: E402
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from tradeflow import cli  # noqa: E402
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


def test_self_test_reports_discovery_and_disclaims_chat_attachment(child_env, capsys):
    args = cli.parse_cli(["mcp", "--self-test"])

    args.func(args)  # returns normally on success; sys.exit(1) would raise

    out = capsys.readouterr().out
    assert "self-test passed" in out
    assert "strategies" in out and "scanners" in out
    assert DISCLAIMER in out


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
