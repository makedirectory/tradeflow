"""The composite research verdict: composition, coherence, and honesty.

All offline against FakeMarketData. The properties that matter here are not the
individual numbers (their own steps' tests own those) but the guarantees that only
exist because the steps ran together: one universe and one window everywhere, one
journaled trial rather than five, a verdict assembled from the steps' own gates,
and a partial run that refuses to render a verdict at all.
"""

import json
from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tradeflow.analytics.reporting import format_verdict_report
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.marketdata.session import SessionBarCache, session_client
from tradeflow.marketdata.timeframe import Timeframe
from tradeflow.services import analysis, audit
from tradeflow.store.trials import TrialStore, db_path_for_journal
from tradeflow.utils.timeutils import NEW_YORK

# Six names, not three: the default weight cap can't fund a book from three, and
# the information sampler needs a cross-section wide enough to correlate.
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
START, END = datetime(2024, 1, 2), datetime(2025, 6, 1)


class CountingMarketData(FakeMarketData):
    """A fake that remembers every ``get_bars`` request it was asked to serve."""

    def __init__(self, symbols, **kwargs):
        super().__init__(symbols, **kwargs)
        self.calls = []

    def get_bars(self, symbols, timeframe, start, end):
        self.calls.append((tuple(symbols), str(timeframe), str(start), str(end)))
        return super().get_bars(symbols, timeframe, start, end)


def _provider():
    return CountingMarketData([*SYMBOLS, "SPY"], n=600, freq="1D")


def _client(provider=None):
    return MarketDataClient(provider or _provider())


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Keep the journal, the trial store it dual-writes, and the verdict artifacts
    out of the repo tree - a test that memoizes against the real journal is a test
    whose result depends on what the developer ran yesterday."""
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    return tmp_path


def _run(client=None, **kwargs):
    return analysis.run_verdict(client or _client(), "demo_trend", SYMBOLS, START, END, **kwargs)


def _journal_rows(tmp_path):
    path = tmp_path / "journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- composition ------------------------------------------------------------
def test_every_section_is_present_and_serializable():
    result = _run()
    assert result["schema"] == analysis.VERDICT_SCHEMA
    for section in ("scan", "alphas", "portfolio", "information", "verdict"):
        assert section in result
    assert json.loads(json.dumps(result))  # the object 030/032 consume must round-trip


def test_one_universe_and_one_window_across_every_section():
    """The whole value of the composite: the sections describe the same thing."""
    result = _run()
    universe = result["inputs"]["universe"]
    assert universe

    assert result["portfolio"]["as_of"] == END.isoformat()
    assert result["alphas"]["as_of"] == END.isoformat()
    # The scan names its clock in the exchange zone rather than echoing the argument,
    # so this is the same instant spelled differently - which is what the section has
    # to agree on. Comparing the strings would make the section coherent only while
    # every section happened to render a datetime the same way.
    assert datetime.fromisoformat(result["scan"]["as_of"]) == NEW_YORK.localize(END)
    assert result["information"]["window"] == {"start": START.isoformat(), "end": END.isoformat()}
    # Every step scored the universe the scan resolved, not the candidate list and
    # not a universe of its own.
    assert {row["symbol"] for row in result["alphas"]["alphas"]} <= set(universe)
    assert set(result["portfolio"]["weights"]) <= set(universe)


def test_repeated_bar_requests_are_served_once():
    provider = _provider()
    result = _run(_client(provider))
    distinct = {call for call in provider.calls}
    # The steps legitimately want different lookbacks, so this is not one fetch
    # total - it is one fetch per distinct request, with every repeat served from
    # the session cache.
    assert len(provider.calls) == len(distinct)
    stats = result["provenance"]["bar_requests"]
    assert stats["fetches"] < stats["requests"]
    assert stats["fetches"] == len(provider.calls)


def test_combination_step_runs_only_with_more_than_one_signal():
    single = _run()
    assert single["combination"] is None
    assert single["steps"]["combination"]["status"] == "skipped"

    combined = _run(signals=["demo_trend", "demo_trend"])
    assert combined["steps"]["combination"]["status"] == "ok"
    assert combined["combination"]["combined_ic"] is not None


def test_scanner_none_skips_the_scan_and_uses_the_candidates():
    result = _run(scanner="none")
    assert result["steps"]["scan"]["status"] == "skipped"
    assert result["inputs"]["universe"] == SYMBOLS


# --- the verdict line -------------------------------------------------------
def test_verdict_checks_match_the_steps_that_produced_them():
    """The verdict is a gate over existing numbers, never a fresh heuristic."""
    result = _run()
    checks = result["verdict"]["checks"]
    inf = result["information"]
    assert checks["ic_tstat"]["value"] == pytest.approx(inf["ic_tstat"])
    assert checks["ic_tstat"]["passed"] == (abs(inf["ic_tstat"]) >= 2.0)
    assert checks["sanity_ceiling"]["passed"] != inf["sanity_ceiling_breached"]
    assert checks["sample_size"]["value"] == inf["periods"]
    if result["portfolio"]["feasible"]:
        expected = result["portfolio"]["diagnostics"].get("expected_active_return_net")
        assert checks["net_of_cost_alpha"]["value"] == pytest.approx(expected)


def test_verdict_vocabulary_is_one_of_the_house_terms():
    result = _run()
    assert result["verdict"]["verdict"] in {
        "promotable",
        "not promotable",
        "needs more data",
        "mixed",
        "incomplete",
    }


def test_a_failed_step_yields_no_verdict_and_a_named_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic information failure")

    monkeypatch.setattr(analysis, "compute_information", boom)
    result = _run()

    assert result["steps"]["information"]["status"] == "failed"
    assert "synthetic information failure" in result["steps"]["information"]["error"]
    assert result["verdict"]["verdict"] == "incomplete"
    assert result["verdict"]["promotable"] is None
    assert result["verdict"]["failed_steps"] == ["information"]
    # The steps that did run are still reported - the run is partial, not lost.
    assert result["portfolio"] is not None
    report = format_verdict_report(result)
    assert "do not act on the sections above" in report


def test_mixed_verdict_shows_both_sides_rather_than_averaging():
    result = _run()
    result["verdict"] = analysis._verdict_gates(
        {
            "steps": {"scan": {"status": "ok"}},
            "information": {
                "periods": 40,
                "ic_tstat": 3.0,
                "realized_ir": 0.5,
                "ir_standard_error": 0.2,
                "low_sample": False,
                "sanity_ceiling_breached": False,
            },
            "portfolio": {"feasible": True, "diagnostics": {"expected_active_return_net": -0.01}},
        }
    )
    assert result["verdict"]["verdict"] == "mixed"
    assert "ic_tstat" in result["verdict"]["summary"]
    assert "net_of_cost_alpha" in result["verdict"]["summary"]


def test_an_information_step_that_measured_nothing_never_reads_as_a_pass():
    """A feasible book with no skill evidence behind it is not a partial pass."""
    gates = analysis._verdict_gates(
        {
            "steps": {"information": {"status": "ok"}},
            "information": {"periods": 0},
            "portfolio": {"feasible": True, "diagnostics": {"expected_active_return_net": 0.05}},
        }
    )
    assert gates["verdict"] == "needs more data"
    assert gates["promotable"] is False


def test_low_sample_reads_as_needs_more_data():
    gates = analysis._verdict_gates(
        {
            "steps": {},
            "information": {
                "periods": 3,
                "ic_tstat": 9.0,
                "realized_ir": 1.0,
                "ir_standard_error": 0.1,
                "low_sample": True,
                "sanity_ceiling_breached": False,
            },
            "portfolio": {},
        }
    )
    # A spectacular IC over three rebalances is not a promotion, whatever it says.
    assert gates["verdict"] == "needs more data"
    assert gates["promotable"] is False


# --- journaling and memoization ---------------------------------------------
def test_one_run_journals_exactly_one_trial(_isolated_state):
    _run()
    rows = [r for r in _journal_rows(_isolated_state) if r.get("tool", "").startswith("trial:")]
    assert len(rows) == 1
    assert rows[0]["kind"] == "verdict"


def test_identical_rerun_is_memoized_without_touching_the_provider(_isolated_state):
    first = _run()
    assert first["memoized"] is False

    provider = _provider()
    second = _run(_client(provider))
    assert second["memoized"] is True
    assert second["trial_id"] == first["trial_id"]
    assert provider.calls == []
    assert second["verdict"] == first["verdict"]

    rows = [r for r in _journal_rows(_isolated_state) if r.get("tool", "").startswith("trial:")]
    assert len(rows) == 1  # a memoized run adds no trial


def test_force_appends_a_second_trial(_isolated_state):
    _run()
    forced = _run(force=True)
    assert forced["memoized"] is False
    rows = [r for r in _journal_rows(_isolated_state) if r.get("tool", "").startswith("trial:")]
    assert len(rows) == 2


def test_differing_cost_flags_are_different_trials(_isolated_state):
    _run()
    other = _run(commission_bps=25.0)
    assert other["memoized"] is False
    rows = [r for r in _journal_rows(_isolated_state) if r.get("tool", "").startswith("trial:")]
    assert len(rows) == 2


def test_differing_step_knobs_are_different_trials(_isolated_state):
    """A knob that lives inside a step still changes the run's identity."""
    _run()
    other = _run(target_te=0.10)
    assert other["memoized"] is False


def test_journal_false_records_nothing(_isolated_state):
    _run(journal=False)
    assert _journal_rows(_isolated_state) == []


# --- persisted weights ------------------------------------------------------
def test_weights_are_journaled_and_rebuildable_from_the_journal_alone(_isolated_state):
    result = _run()
    if not result["portfolio"]["feasible"]:
        pytest.skip("no feasible book on this fixture; nothing to persist")

    journal = _isolated_state / "journal.jsonl"
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        stored = store.weights_for(result["trial_id"])
        assert stored is not None
        assert stored["weights"] == pytest.approx(result["portfolio"]["weights"])

        # The journal is the source of truth: a full rebuild must reconstruct it.
        store.rebuild()
        rebuilt = store.weights_for(result["trial_id"])
        assert rebuilt == stored


def test_a_journal_with_no_weights_rebuilds_with_an_empty_table(_isolated_state):
    """Every trial recorded before the book was persisted must still rebuild, with
    'not recorded' rather than an invented empty book."""
    analysis.run_backtest(_client(), "demo_trend", SYMBOLS, START, END)
    journal = _isolated_state / "journal.jsonl"
    with TrialStore(db_path_for_journal(journal), journal_path=journal) as store:
        store.rebuild()
        rows = [r for r in _journal_rows(_isolated_state) if r.get("tool", "").startswith("trial:")]
        assert store.weights_for(rows[0]["run_id"]) is None


# --- the session bar cache --------------------------------------------------
def test_session_cache_serves_repeats_and_never_hands_out_its_own_frames():
    provider = _provider()
    cache = SessionBarCache(provider)
    tf = Timeframe.parse("1Day")

    first = cache.get_bars(SYMBOLS, tf, START, END)
    second = cache.get_bars(SYMBOLS, tf, START, END)
    assert len(provider.calls) == 1
    assert cache.stats() == {"requests": 2, "fetches": 1, "distinct": 1}

    # A caller mutating what it received must not corrupt the next caller's copy.
    first["AAA"].loc[:, "close"] = 0.0
    assert not (second["AAA"]["close"] == 0.0).all()
    assert not (cache.get_bars(SYMBOLS, tf, START, END)["AAA"]["close"] == 0.0).all()


def test_session_cache_does_not_serve_a_different_window_from_a_wider_fetch():
    """Exact-match only: a narrower window is a different request, never a slice of
    a wider one, so a step's frames never depend on what ran before it."""
    provider = _provider()
    cache = SessionBarCache(provider)
    tf = Timeframe.parse("1Day")
    cache.get_bars(SYMBOLS, tf, START, END)
    cache.get_bars(SYMBOLS, tf, datetime(2024, 6, 1), END)
    assert len(provider.calls) == 2


def test_session_client_passes_through_a_client_with_no_provider():
    class Bare:
        pass

    client, cache = session_client(Bare())
    assert cache is None
    assert isinstance(client, Bare)


# --- rendering --------------------------------------------------------------
def test_report_shows_provenance_the_verdict_and_every_check():
    result = _run()
    report = format_verdict_report(result)
    assert "Research verdict" in report
    assert "provenance:" in report
    assert "campaign trials" in report
    assert "VERDICT:" in report
    for name in result["verdict"]["checks"]:
        assert name in report


def test_report_names_the_cost_model_it_priced_with():
    assert "GROSS" in format_verdict_report(_run(gross=True))
    assert "commission" in format_verdict_report(_run(commission_bps=7.0, force=True))


# --- the CLI surface --------------------------------------------------------
def _cli(monkeypatch, provider, *argv):
    from tradeflow import cli as main

    monkeypatch.setattr(main, "build_data_and_broker", lambda **kwargs: (None, MarketDataClient(provider)))
    args = main.build_parser().parse_args(
        [
            "verdict",
            "--symbols",
            ",".join(SYMBOLS),
            "--start",
            START.strftime("%Y-%m-%d"),
            "--end",
            END.strftime("%Y-%m-%d"),
            *argv,
        ]
    )
    args.func(args)


def test_cli_writes_a_json_object_that_round_trips(monkeypatch, tmp_path, capsys):
    out = tmp_path / "verdict.json"
    _cli(monkeypatch, _provider(), "--json", str(out))

    payload = json.loads(out.read_text())
    assert payload["schema"] == analysis.VERDICT_SCHEMA
    assert payload["verdict"]["summary"]
    assert set(payload["inputs"]["universe"]) <= set(SYMBOLS)
    assert "VERDICT:" in capsys.readouterr().out


def test_cli_writes_a_self_contained_html_report(monkeypatch, tmp_path):
    out = tmp_path / "report.html"
    _cli(monkeypatch, _provider(), "--html", str(out))

    document = out.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert "VERDICT" in document.upper()
    assert "http://" not in document and "https://" not in document


def test_cli_exits_non_zero_when_a_step_failed(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic portfolio failure")

    monkeypatch.setattr(analysis, "construct_portfolio", boom)
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, _provider())
    assert exc.value.code == 1
    # The exit code is the machine-readable half; the report still says it plainly.
    assert "incomplete" in capsys.readouterr().out


def test_provenance_identifies_the_code_even_without_a_git_repository(monkeypatch):
    """A report that outlives its context must say what made it. An installed copy
    has no repository, and "unknown" there defeats the whole provenance block."""
    import tradeflow
    from tradeflow.optimization import config_store

    monkeypatch.setattr(config_store, "current_git_sha", lambda: None)
    assert analysis.code_version() == f"tradeflow {tradeflow.__version__}"

    monkeypatch.setattr(config_store, "current_git_sha", lambda: "abc1234")
    assert analysis.code_version() == "abc1234"


def test_the_provenance_line_does_not_claim_provider_access_it_cannot_vouch_for():
    """It read "N of M bar requests hit the provider".

    The number comes from the per-run bar memo: `fetches` counts requests that missed
    it and reached the data client underneath. On an `--offline` run that client is
    the local cache, so the line reported a network round trip on a run that made
    none. Provenance is the one thing a reader cannot check for themselves, so
    overstating where data came from is worse than saying nothing.
    """
    from tradeflow.analytics.reporting import _fetch_summary

    summary = _fetch_summary({"requests": 12, "fetches": 3, "distinct": 3})

    assert "provider" not in summary
    assert "3" in summary and "12" in summary
    assert _fetch_summary(None) == "not measured"


def test_a_passing_gate_does_not_print_the_failure_that_did_not_happen():
    """Reported from a real verdict: `[PASS] ir_above_noise ... realized IR inside its
    own standard-error band is indistinguishable from zero`.

    Every gate's note states the *failing* condition, so printing it beside a PASS
    produced a line that contradicted its own verdict - and a reader who trusts the
    prose over the mark reads a pass as a failure.
    """
    from tradeflow.analytics.reporting import _verdict_banner_lines

    lines = _verdict_banner_lines(
        {
            "verdict": {
                "summary": "mixed",
                "checks": {
                    "ir_above_noise": {
                        "value": 1.78,
                        "threshold": 0.62,
                        "passed": True,
                        "note": "realized IR inside its own standard-error band is indistinguishable from zero",
                    },
                    "ic_tstat": {
                        "value": 0.4,
                        "threshold": 2.0,
                        "passed": False,
                        "note": "IC t-stat below 2 is not distinguishable from luck",
                    },
                },
            }
        }
    )
    rendered = "\n".join(lines)

    passing = next(line for line in lines if "ir_above_noise" in line)
    assert "PASS" in passing
    assert "indistinguishable" not in passing  # the failure text is not on the pass

    # And the failing gate keeps its explanation, or the note would be useless.
    failing = next(line for line in lines if "ic_tstat" in line)
    assert "FAIL" in failing and "not distinguishable from luck" in failing
    assert "1.78 vs 0.62" in rendered  # values still shown on a pass
