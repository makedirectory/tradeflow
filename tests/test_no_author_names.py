"""User-facing help names the concept, never the author a technique came from.

The writing rules forbid citing provenance. The strings a reader sees are the part a
review reads least: argparse help, and the MCP tool descriptions an agent acts on.
Both lists are enumerated from the real parser and the real server, so a new flag or
tool is covered without anyone remembering to add it here.
"""

import argparse
import asyncio
import re

import pytest

from quantred.cli import build_parser

AUTHORS = re.compile(
    r"black.?litterman|ledoit|white.s|politis|romano|almgren|chriss|grinold|kahn"
    r"|reality check|markowitz|fama|patton|lopez de prado|bailey",
    re.IGNORECASE,
)


def _help_strings(parser, path=()):
    """Yield (location, text) for every help/description/choice text in a parser tree."""
    where = " ".join(path) or "quantred"
    for text in (parser.description, parser.epilog):
        if text:
            yield where, text
    for action in parser._actions:
        if action.help and action.help is not argparse.SUPPRESS:
            yield f"{where} {action.dest}", action.help
        if isinstance(action, argparse._SubParsersAction):
            for choice_action in action._choices_actions:
                if choice_action.help:
                    yield f"{where} {choice_action.dest}", choice_action.help
            for name, sub in action.choices.items():
                yield from _help_strings(sub, (*path, name))


def test_no_cli_help_names_an_author():
    offenders = [
        (where, m.group(0))
        for where, text in _help_strings(build_parser())
        for m in [AUTHORS.search(text)]
        if m
    ]
    assert not offenders, offenders


def _walk(schema):
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in ("description", "title") and isinstance(value, str):
                yield value
            else:
                yield from _walk(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _walk(item)


def test_no_mcp_tool_description_or_parameter_doc_names_an_author():
    pytest.importorskip("mcp", reason="the MCP surface needs the 'mcp' extra")
    from quantred.marketdata.client import MarketDataClient
    from quantred.mcp import server as mcp_server
    from tests.fakes import FakeMarketData

    built = mcp_server.build_server(
        data_client=MarketDataClient(FakeMarketData(["AAA", "SPY"], n=50, freq="1D"))
    )
    tools = asyncio.run(built.list_tools())
    assert tools
    offenders = []
    for tool in tools:
        texts = [tool.description or "", *_walk(tool.inputSchema), *_walk(tool.outputSchema)]
        offenders += [(tool.name, m.group(0)) for t in texts for m in [AUTHORS.search(t)] if m]
    assert not offenders, offenders
