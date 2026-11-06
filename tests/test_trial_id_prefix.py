"""A trial id may be given as a unique prefix, on every surface, resolved one way.

Ids are long hashes; asking readers to paste all of one to look at a trial made
`trials show <first few characters>` answer "no trial", which reads as the trial being
missing. The properties: a unique prefix finds it, an ambiguous one is refused with the
candidates listed (never resolved to one of them), and the CLI and the MCP tools agree.
"""

import asyncio
import json

import pytest

from tradeflow.cli import main
from tradeflow.services import analysis, audit, maintenance
from tradeflow.services.campaign import campaign_material
from tradeflow.store import trials as trials_module
from tradeflow.store.trials import AmbiguousTrialId, TrialStore, db_path_for_journal

AA = "aaaa1111deadbeef"
AB = "aaaa2222deadbeef"
BB = "bbbb3333deadbeef"


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "research_journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", path)
    monkeypatch.setattr(trials_module, "DEFAULT_JOURNAL_PATH", path)
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_AUDIT_PATH", tmp_path / "mcp_audit.jsonl")
    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    with TrialStore(db_path_for_journal(path), journal_path=path) as store:
        for n, tid in enumerate((AA, AB, BB)):
            store.record(
                id=tid,
                kind="walkforward",
                strategy="demo_trend",
                symbols=["AAA"],
                params={"fast": n},
                accounting=3,
                ts=f"2025-0{n + 1}-01T00:00:00",
            )
    return path


def _store(journal):
    return TrialStore(db_path_for_journal(journal), journal_path=journal)


def _cli(monkeypatch, journal, *argv):
    # `--db` explicitly: the CLI's default store is the real state root, not the redirect.
    monkeypatch.setattr("sys.argv", ["cli", *argv, "--db", str(db_path_for_journal(journal))])
    main()


def test_a_unique_prefix_resolves_to_the_full_id(journal):
    """`trials show bbbb` said there was no trial; the prefix names exactly one."""
    with _store(journal) as store:
        assert store.resolve_id("bbbb") == BB
        assert store.resolve_id(BB) == BB


def test_an_ambiguous_prefix_is_refused_with_the_candidates(journal):
    """Picking either of two matches would act on a trial the reader may not have meant."""
    with _store(journal) as store:
        with pytest.raises(AmbiguousTrialId) as caught:
            store.resolve_id("aaaa")
    assert AA in str(caught.value) and AB in str(caught.value)
    assert BB not in str(caught.value)


def test_a_prefix_shorter_than_four_characters_names_nothing(journal):
    """Three characters match too much to be a deliberate reference."""
    with _store(journal) as store:
        assert store.resolve_id("bbb") is None
        assert store.resolve_id("zzzz") is None


def test_wildcard_characters_in_a_prefix_are_literal(journal):
    """A typed `%` must not match every trial."""
    with _store(journal) as store:
        assert store.resolve_id("%%%%") is None


def test_cli_and_mcp_resolve_the_same_prefix_the_same_way(journal, monkeypatch, capsys):
    """One resolver behind both transports: the same hit for a unique prefix and the
    same refusal, naming the same candidates, for an ambiguous one."""
    pytest.importorskip("mcp")
    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.mcp import server as mcp_server

    built = mcp_server.build_server(data_client=MarketDataClient(FakeMarketData(["AAA"], n=50, freq="1D")))

    def call(name, **kwargs):
        content = asyncio.run(built.call_tool(name, kwargs))
        if isinstance(content, tuple):
            content = content[0]
        return json.loads(content[0].text)

    for prefix, expected in (("bbbb", BB), ("aaaa1", AA), (AB, AB)):
        _cli(monkeypatch, journal, "trials", "show", prefix, "--json")
        assert json.loads(capsys.readouterr().out)["id"] == expected
        assert call("get_trial", trial_id=prefix)["id"] == expected

    with pytest.raises(Exception) as refused:
        call("get_trial", trial_id="aaaa")
    mcp_error = str(refused.value)
    with pytest.raises(SystemExit) as stop:
        _cli(monkeypatch, journal, "trials", "show", "aaaa")
    # The tool adds its own next step (list_trials) after the shared refusal.
    assert str(stop.value) in mcp_error
    assert "ambiguous" in mcp_error and AA in mcp_error and AB in mcp_error


def test_analyze_campaign_and_compare_accept_a_prefix(journal):
    """These go through the services, which resolve exactly as `show` does."""
    with _store(journal) as store:
        assert analysis.analyze_trial(store, "bbbb")["trial_id"] == BB
        assert campaign_material(store, "bbbb")["trial_id"] == BB
        assert analysis.compare_trials(store, ["bbbb", "aaaa1"])["unknown_trial_ids"] == []
        assert "ambiguous" in analysis.analyze_trial(store, "aaaa")["error"]
        assert "ambiguous" in campaign_material(store, "aaaa")["reason"]
        assert "ambiguous" in analysis.compare_trials(store, ["aaaa", "bbbb"])["error"]


def test_mark_contaminated_accepts_a_prefix_and_still_reports_a_miss(journal):
    """The unmatched-id report compared typed text to full ids, so a prefix looked like a typo."""
    report = maintenance.mark_contaminated(reason="x", journal_path=journal, trial_ids=["bbbb"], dry_run=True)
    assert report["trial_ids"] == [BB] and report["unmatched_ids"] == []
    report = maintenance.mark_contaminated(reason="x", journal_path=journal, trial_ids=["zzzz"], dry_run=True)
    assert report["unmatched_ids"] == ["zzzz"]
    with pytest.raises(ValueError, match="ambiguous"):
        maintenance.mark_contaminated(reason="x", journal_path=journal, trial_ids=["aaaa"], dry_run=True)
