"""Drive the MCP server with the real SDK client, the way a chat client does.

``tradeflow mcp --self-test`` exists so a user can tell "the server starts and
answers" apart from "my chat is connected". It launches a *child* copy of the server
over stdio, initializes, lists the tools and calls the two discovery tools. Going
through the SDK client matters: the in-process ``call_tool`` shortcut skips the
output-schema validation a real client performs, which is how a mistyped discovery
row shipped.

Importable only with the ``mcp`` extra; the CLI imports it lazily.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

#: Said on every run, pass or fail: a passing self-test is easy to over-read.
DISCLAIMER = (
    "This checks that the server starts and answers. It does not connect it to any chat "
    "- your MCP client launches its own copy from its config."
)

#: The discovery tools the check calls. Names here are the tool names, not services.
DISCOVERY_TOOLS = ("list_strategies", "list_scanners")


def default_command() -> List[str]:
    """The server command for *this* copy - the one ``init`` tells a reader to configure.

    Delegates to the setup service so the check and the instruction cannot name two
    different commands (the check passing on one nobody was told to use).
    """
    from tradeflow.services.setup import mcp_launch_command

    return mcp_launch_command()


def _rows(result: Any) -> List[Dict[str, Any]]:
    structured = getattr(result, "structuredContent", None) or {}
    rows = structured.get("result")
    return list(rows) if isinstance(rows, list) else []


async def _drive(command: List[str], env: Dict[str, str], errlog, report: Dict[str, Any]) -> None:
    """Fill ``report`` as it goes, so a failure still knows which step it reached."""
    params = StdioServerParameters(command=command[0], args=command[1:], env=env)
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            report["step"] = "initialize"
            await session.initialize()
            report["step"] = "list_tools"
            report["tools"] = sorted(t.name for t in (await session.list_tools()).tools)
            for tool in DISCOVERY_TOOLS:
                report["step"] = tool
                result = await session.call_tool(tool)
                if result.isError:
                    text = " ".join(getattr(c, "text", "") for c in result.content)
                    raise RuntimeError(f"{tool} returned an error: {text}")
                rows = _rows(result)
                if not rows:
                    raise RuntimeError(
                        f"{tool} returned no structured rows - the server's MCP SDK may predate "
                        "structured output (the 'mcp' extra needs >=1.10), or discovery found nothing"
                    )
                if not any(r.get("demo") for r in rows):
                    raise RuntimeError(f"{tool} did not list the built-in demo entries")
                report[tool] = rows
    report["step"] = "close"


def run_self_test(
    command: Optional[List[str]] = None,
    *,
    timeout: float = 60,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run the discovery check; never raises for a server-side failure.

    Returns ``{"ok", "step", "elapsed", ...}``. On success it also carries ``tools``,
    ``list_strategies`` and ``list_scanners`` rows. On failure ``step`` names where it
    stopped and ``error`` / ``server_stderr`` say why: a server that exits before
    answering (a missing credential, a missing extra) reaches a client as a closed
    pipe and nothing else, so what the child printed is the useful part.

    ``env`` defaults to a copy of the caller's environment - the SDK would otherwise
    hand the child a minimal one, losing ``TRADEFLOW_HOME`` and credentials.
    """
    command = list(command) if command else default_command()
    child_env = dict(os.environ if env is None else env)
    started = time.monotonic()
    report: Dict[str, Any] = {"step": "start"}

    async def go() -> Dict[str, Any]:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errlog:
            try:
                with anyio.fail_after(timeout):
                    await _drive(command, child_env, errlog, report)
                return {"ok": True, **report}
            except BaseException as exc:  # noqa: BLE001 - reported, not raised
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                errlog.flush()
                errlog.seek(0)
                return {
                    "ok": False,
                    "step": report["step"],
                    "error": _describe(exc),
                    "server_stderr": errlog.read().strip()[-2000:],
                }

    out = anyio.run(go)
    out["command"] = command
    out["elapsed"] = time.monotonic() - started
    return out


def _describe(exc: BaseException) -> str:
    # Task-group failures arrive wrapped; the first member is the cause worth showing.
    members = getattr(exc, "exceptions", None)
    if members:
        return _describe(members[0])
    if isinstance(exc, TimeoutError):
        return "timed out waiting for the server"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
