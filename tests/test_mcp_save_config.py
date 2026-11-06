"""`save_config` and `load_config` over MCP: a name is a plain file name, and a config
that could not run is refused before it is written.

`save_config(name="../evil")` wrote outside the configs directory, `load_config` read any
JSON file by absolute path, and a config could name a strategy, parameter or book limit
that does not exist. All offline, driven through the real tool call.
"""

import asyncio
import json

import pytest

from tests.fakes import FakeMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.mcp import server as mcp_server
from tradeflow.optimization import config_store
from tradeflow.services import analysis, audit

pytest.importorskip("mcp", reason="the MCP surface needs the 'mcp' extra")


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(audit, "DEFAULT_AUDIT_PATH", tmp_path / "mcp_audit.jsonl")
    monkeypatch.setattr(config_store, "DEFAULT_CONFIG_DIR", tmp_path / "configs")


@pytest.fixture
def built():
    return mcp_server.build_server(data_client=MarketDataClient(FakeMarketData(["AAA"], n=50, freq="1D")))


def _call(built, tool, **kwargs):
    content = asyncio.run(built.call_tool(tool, kwargs))
    if isinstance(content, tuple):
        content = content[0]
    return json.loads(content[0].text)


def _refused(built, tool, **kwargs) -> str:
    with pytest.raises(Exception) as raised:
        _call(built, tool, **kwargs)
    return str(raised.value)


# --- C: save_config hardening --------------------------------------------------------
GOOD = dict(strategy="demo_trend", params={"fast_ema_period": 9})


@pytest.mark.parametrize("name", ["../evil", "a/b", "/abs/path", "..", ".", "", ".json", "sub\\dir"])
def test_a_config_name_that_is_not_a_plain_file_name_writes_nothing(built, tmp_path, name):
    """`save_config(name="../evil")` wrote outside the configs directory."""
    assert "plain file name" in _refused(built, "save_config", name=name, **GOOD)
    assert not (tmp_path / "evil.json").exists()
    assert not list(tmp_path.rglob("evil*"))


def test_a_plain_name_is_written_directly_under_the_configs_directory(built, tmp_path):
    saved = _call(built, "save_config", name="candidate", **GOOD)
    assert (tmp_path / "configs" / "candidate.json").is_file()
    assert saved["name"] == "candidate"
    assert _call(built, "load_config", name="candidate")["params"] == GOOD["params"]


def test_loading_cannot_read_a_file_outside_the_configs_directory(built, tmp_path):
    outside = tmp_path / "secret.json"
    outside.write_text('{"strategy": "x", "params": {}}')
    assert "plain file name" in _refused(built, "load_config", name=str(outside))


@pytest.mark.parametrize(
    "override, expect",
    [
        (dict(strategy="no_such_strategy"), "Unknown strategy"),
        (dict(params={"fast_emaperiod": 9}), "no parameter"),
        (dict(params={"fast_ema_period": 10_000}), "outside valid range"),
        (dict(position_limits={"max_posiitons": 8}), "No position limit"),
        (dict(scanner="no_such_scanner"), "unknown scanner"),
    ],
)
def test_a_config_that_could_not_run_is_refused_before_it_is_written(built, tmp_path, override, expect):
    assert expect in _refused(built, "save_config", name="bad", **{**GOOD, **override})
    assert not (tmp_path / "configs" / "bad.json").exists()


def test_an_unknown_provenance_field_lists_the_accepted_ones(built, tmp_path):
    """The tool told agents to put the trial id in provenance, and the dataclass has no
    such field, so the call failed with a raw constructor TypeError."""
    message = _refused(built, "save_config", name="p", provenance={"trial_id": "abc"}, **GOOD)
    assert "'trial_id'" in message and "notes" in message and "campaign" in message
    assert "__init__" not in message
    assert not (tmp_path / "configs" / "p.json").exists()

    saved = _call(built, "save_config", name="p", provenance={"notes": "trial abc"}, **GOOD)
    assert saved["name"] == "p"


def test_the_description_names_provenance_fields_that_exist(built):
    description = {t.name: t for t in asyncio.run(built.list_tools())}["save_config"].description
    assert "its trial id" not in description
    import dataclasses

    for field in dataclasses.fields(config_store.Provenance):
        assert f"`{field.name}`" in description
