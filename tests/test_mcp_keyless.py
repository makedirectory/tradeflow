"""The MCP server starts without broker keys and says so only where keys are needed.

It used to build its data client at startup, so a machine with no keys exited before
answering anything and the client showed "Connection closed" - taking down the tools
that need no market data at all. Everything here is offline.
"""

import asyncio
import json

import pytest

from quantred import cli, settings
from quantred.brokers.errors import AuthenticationError
from quantred.marketdata.client import MarketDataClient
from quantred.mcp import server as mcp_server
from quantred.services import analysis, audit, setup
from tests.fakes import FakeMarketData

pytest.importorskip("mcp", reason="the MCP surface needs the 'mcp' extra")


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(audit, "DEFAULT_AUDIT_PATH", tmp_path / "mcp_audit.jsonl")
    return tmp_path


@pytest.fixture
def no_keys(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    # The .env is read once per process; without this a developer's real keys leak in.
    monkeypatch.setattr(settings, "_load_dotenv_once", lambda: None)


def _call(built, name, **kwargs):
    content = asyncio.run(built.call_tool(name, kwargs))
    if isinstance(content, tuple):
        content = content[0]
    return json.loads(content[0].text)


def test_a_server_with_no_keys_starts_and_answers_the_catalogue(no_keys):
    """`build_server()` raised the missing-keys error at construction, so nothing ran."""
    built = mcp_server.build_server()
    names = {t.name for t in asyncio.run(built.list_tools())}
    assert {"list_strategies", "list_trials", "get_metrics_glossary"} <= names
    rows = asyncio.run(built.call_tool("list_strategies", {}))
    assert rows  # answered, rather than refused for want of keys


def test_a_tool_needing_bars_says_the_keys_are_missing_and_how_to_add_them(no_keys):
    built = mcp_server.build_server()
    with pytest.raises(Exception) as raised:
        _call(built, "summarize_bars", symbols=["AAA"])
    text = str(raised.value)
    assert "no broker keys" in text
    assert settings.invocation("init") in text
    assert "--offline" in text
    # The CLI's own message points at `make init` for a checkout; this one is for a person
    # whose client launched the server, and names the command the process was started by.
    assert "make init" not in text or settings.invocation("init") == "make init"


def test_offline_serves_without_keys_and_a_miss_names_the_warm_command(no_keys, tmp_path):
    built = mcp_server.build_server(offline=True, cache_dir=tmp_path / "empty-cache")
    with pytest.raises(Exception) as raised:
        _call(built, "summarize_bars", symbols=["AAA"])
    text = str(raised.value)
    assert "cache warm" in text
    assert "no broker keys" not in text


def test_a_failed_build_is_retried_not_remembered(no_keys, monkeypatch):
    """Keys added after a refusal must be picked up by the next call, not refused forever."""
    built = mcp_server.build_server()
    with pytest.raises(Exception):
        _call(built, "summarize_bars", symbols=["AAA"])
    from quantred.services import data

    monkeypatch.setattr(
        data,
        "build_data_client",
        lambda **_: MarketDataClient(FakeMarketData(["AAA"], n=200, freq="1D")),
    )
    assert _call(built, "summarize_bars", symbols=["AAA"])


def test_rejected_keys_reach_the_agent_as_a_tool_error_naming_init(monkeypatch):
    """The fix text is the invocation of init for this copy, not a hard-coded form."""

    class Rejecting(FakeMarketData):
        def get_bars(self, symbols, timeframe, start, end):
            raise AuthenticationError(
                f"Alpaca rejected these credentials (HTTP 401). Then run `{settings.invocation('init')}`."
            )

    built = mcp_server.build_server(data_client=MarketDataClient(Rejecting(["AAA"], n=50, freq="1D")))
    with pytest.raises(Exception) as raised:
        _call(built, "summarize_bars", symbols=["AAA"])
    assert "rejected these credentials" in str(raised.value)
    assert settings.invocation("init") in str(raised.value)


def test_the_launch_command_carries_offline_and_the_self_test_and_printed_config_agree(tmp_path, capsys):
    """One command for the check and the instruction: a self-test of `--offline` that
    printed a config without it would vouch for a server nobody was told to launch."""
    plain = setup.mcp_launch_command()
    offline = setup.mcp_launch_command(offline=True, cache_dir=tmp_path)
    assert offline[: len(plain)] == plain
    assert offline[len(plain) :] == ["--offline", "--cache-dir", str(tmp_path.resolve())]

    server = setup.mcp_client_config(offline=True, cache_dir=tmp_path)["mcpServers"][settings.MCP_SERVER_NAME]
    assert [server["command"], *server["args"]] == offline

    block = cli._mcp_client_block(True, tmp_path)
    printed = json.loads(block[block.index("{") : block.rindex("}") + 1])["mcpServers"]
    assert [
        printed[settings.MCP_SERVER_NAME]["command"],
        *printed[settings.MCP_SERVER_NAME]["args"],
    ] == offline
    assert "--offline" in block.split("Claude Code")[1]


def test_an_offline_self_test_without_the_store_extra_fails_naming_it(monkeypatch, capsys):
    """`mcp --offline --self-test` passed on a copy without `store`: the server builds its
    cache-only client lazily, so discovery worked and the first tool that read a bar
    failed - after the reader had pasted the printed config into a client."""
    from quantred.mcp import selftest
    from quantred.utils import extras

    real = extras.installed
    monkeypatch.setattr(extras, "installed", lambda name: name != "store" and real(name))
    monkeypatch.setattr(
        selftest, "_drive", lambda *a, **k: pytest.fail("the server was launched without the store extra")
    )
    args = cli.parse_cli(["mcp", "--offline", "--self-test"])
    with pytest.raises(SystemExit) as caught:
        args.func(args)
    out = capsys.readouterr().out
    assert caught.value.code == 1
    assert "self-test FAILED" in out and "'store' extra" in out
    assert "passed" not in out
