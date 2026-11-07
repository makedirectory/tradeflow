"""The trial history says what each number is, and what its filters left out.

A walk-forward's Sharpe is out-of-sample evidence; an optimize row's is a search's winner
and a backtest's is one pass over the window. They sit in one column, so a listing that
ranks or labels them alike presents the selection bias as the finding. All offline,
through the real store, CLI and MCP tool calls.
"""

import asyncio
import json
from datetime import datetime

import pytest

from tradeflow.analytics.reporting import format_campaign_material, format_trial_detail, format_trials_table
from tradeflow.cli import build_parser
from tradeflow.engine.backtest import ACCOUNTING_VERSION
from tradeflow.services.audit import journal_trial
from tradeflow.store.trials import TrialStore, db_path_for_journal


@pytest.fixture
def store(tmp_path):
    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "journal.jsonl") as s:
        yield s


def _record(store, trial_id, **kwargs):
    row = {
        "id": trial_id,
        "kind": "walkforward",
        "strategy": "demo_trend",
        "symbols": ["AAA", "BBB"],
        "params": {"fast": 5},
        "accounting": ACCOUNTING_VERSION,
        "ts": "2025-01-01T15:00:00+00:00",
        "window_start": "2024-01-02T00:00:00",
        "window_end": "2024-12-31T00:00:00",
    }
    row.update(kwargs)
    store.record(**row)


def _mixed(store):
    _record(store, "wf", oos_sharpe=0.4, deflated_sharpe=0.2, promotable=False)
    _record(store, "opt", kind="optimize", oos_sharpe=2.5, deflated_sharpe=0.95, params={"fast": 6})
    _record(store, "bt", kind="backtest", oos_sharpe=1.8, deflated_sharpe=0.7, params={"fast": 7})


# --- ranking leaves in-sample kinds out ------------------------------------------------
def test_ranking_by_sharpe_does_not_put_a_search_winner_first(store):
    """The failure: `--sort sharpe` listed the optimize row (best of N, in-sample) above
    every validated run with nothing marking it."""
    _mixed(store)
    ranked = store.listing(sort="sharpe", accounting=ACCOUNTING_VERSION)
    assert [r["id"] for r in ranked["rows"]] == ["wf"]
    assert ranked["in_sample_excluded"] == 2  # reported, not silent

    opted_in = store.listing(sort="sharpe", include_in_sample=True, accounting=ACCOUNTING_VERSION)
    assert [r["id"] for r in opted_in["rows"]][0] == "opt"
    assert opted_in["in_sample_excluded"] == 0


def test_a_sharpe_floor_is_a_ranking_too(store):
    """`--min-sharpe` described itself as an OOS floor and returned in-sample rows."""
    _mixed(store)
    assert [r["id"] for r in store.listing(min_sharpe=0.1, accounting=ACCOUNTING_VERSION)["rows"]] == ["wf"]


def test_a_date_ordered_listing_is_history_and_shows_everything(store):
    _mixed(store)
    listed = store.listing(accounting=ACCOUNTING_VERSION)
    assert {r["id"] for r in listed["rows"]} == {"wf", "opt", "bt"}
    assert listed["in_sample_excluded"] == 0


def test_asking_for_an_in_sample_kind_by_name_is_honoured_and_still_marked(store):
    _mixed(store)
    listed = store.listing(sort="sharpe", kind="optimize", accounting=ACCOUNTING_VERSION)
    assert [r["id"] for r in listed["rows"]] == ["opt"]
    assert listed["rows"][0]["in_sample"] is True
    assert "IN-SAMPLE" in format_trials_table(listed["rows"], total=1)


def test_the_table_says_what_each_sharpe_measures(store):
    _mixed(store)
    table = format_trials_table(store.listing(accounting=ACCOUNTING_VERSION)["rows"])
    by_id = {line.split()[0]: line for line in table.splitlines() if line.startswith(("wf", "opt", "bt"))}
    assert "oos agg" in by_id["wf"]
    assert "IN-SAMPLE" in by_id["opt"] and "IN-SAMPLE" in by_id["bt"]


def test_the_mcp_listing_ranks_the_same_way_as_the_cli(tmp_path, monkeypatch):
    """Two transports, one rule: the tool leaves in-sample rows out of a ranking and
    says how many, exactly as the store's single definition does for the CLI."""
    pytest.importorskip("mcp")
    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.mcp import server as mcp_server
    from tradeflow.services import analysis, audit

    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal)
    monkeypatch.setattr(audit, "DEFAULT_AUDIT_PATH", tmp_path / "audit.jsonl")
    for kind, sharpe in (("walkforward", 0.4), ("optimize", 2.5)):
        journal_trial(
            kind,
            strategy="demo_trend",
            symbols=["AAA"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 4, 30),
            params={"fast_ema_period": 5 if kind == "optimize" else 6},
            metrics={"sharpe_ratio": sharpe},
        )
    built = mcp_server.build_server(data_client=MarketDataClient(FakeMarketData(["AAA"], n=50, freq="1D")))

    def call(**kwargs):
        content = asyncio.run(built.call_tool("list_trials", kwargs))
        content = content[0] if isinstance(content, tuple) else content
        return json.loads(content[0].text)

    ranked = call(sort="sharpe")
    assert [r["kind"] for r in ranked["rows"]] == ["walkforward"]
    assert ranked["in_sample_excluded"] == 1
    assert ranked["rows"][0]["sharpe_basis"] == "out-of-sample, aggregate over folds"
    assert call(sort="sharpe", include_in_sample=True)["rows"][0]["kind"] == "optimize"


# --- the leaderboard ranks only what has the metric, and says what a family count is --
def test_a_row_with_no_metric_is_not_on_the_board_at_all(store):
    """A verdict records neither a Sharpe nor a DSR. Sorted last it still appeared at #5
    of a five-row board, as if ranked."""
    from tradeflow.analytics.reporting import format_leaderboard

    _record(store, "wf1", oos_sharpe=0.4, deflated_sharpe=0.3)
    _record(store, "wf2", oos_sharpe=0.2, deflated_sharpe=0.1, params={"fast": 8})
    _record(store, "vd", kind="verdict", params={"fast": 9})
    for rank_by in ("dsr", "sharpe"):
        board = store.best(rank_by=rank_by, accounting=ACCOUNTING_VERSION)
        assert [r["id"] for r in board["rows"]] == ["wf1", "wf2"]
        assert board["unranked_excluded"] == 1
    text = format_leaderboard(store.best(accounting=ACCOUNTING_VERSION))
    assert "cannot be ranked" in text
    assert "FAMILY n_trials =" in text and "oos agg" in text


# --- the detail view labels the number, with units ------------------------------------
def test_show_does_not_call_a_backtests_full_window_sharpe_out_of_sample(store):
    _mixed(store)
    for trial_id, wanted in (
        ("bt", "full window, in-sample"),
        ("opt", "best of N candidates, in-sample"),
        ("wf", "out-of-sample, aggregate over folds"),
    ):
        text = format_trial_detail(store.get_trial(trial_id))
        assert "OOS Sharpe" not in text
        assert wanted in text


def test_show_prints_drawdown_in_the_unit_campaign_uses(store):
    """`Max drawdown 1.674` beside `1.67%` in `campaign` was one number in two units."""
    _record(store, "wf", oos_sharpe=0.4, oos_max_dd=1.674, oos_trades=88)
    text = format_trial_detail(store.get_trial("wf"))
    assert "1.67%" in text
    assert "Trades            88" in text


# --- the campaign heading and the promote warning, per kind ---------------------------
def _journal_backtest(journal):
    return journal_trial(
        "backtest",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast_ema_period": 5, "_cost": {}},
        metrics={"sharpe_ratio": 1.0},
        path=journal,
    )


def test_a_backtests_campaign_is_not_headed_as_a_validation():
    assert "NOT validated" in _heading("backtest")
    assert "What validated" in _heading("walkforward")


def _heading(kind):
    return format_campaign_material({"available": True, "trial_id": "x", "kind": kind, "recipe": {}})


def test_promoting_an_ungated_run_does_not_claim_it_failed_its_gates(tmp_path, monkeypatch, capsys):
    from tradeflow.optimization.config_store import DEFAULT_CONFIG_DIR  # noqa: F401

    monkeypatch.setattr("tradeflow.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr("tradeflow.services.audit.DEFAULT_TRIAL_JOURNAL", journal, raising=False)
    trial_id = _journal_backtest(journal)
    base = [
        "trials",
        "promote",
        trial_id,
        "--save-config",
        str(tmp_path / "p.json"),
        "--db",
        str(db_path_for_journal(journal)),
        "--journal",
        str(journal),
    ]
    args = build_parser().parse_args(base)
    with pytest.raises(SystemExit) as refused:
        args.func(args)
    message = str(refused.value)
    assert "never faced the promotion gates" in message
    assert "did not clear" not in message

    args = build_parser().parse_args([*base, "--force"])
    args.func(args)
    out = capsys.readouterr().out
    assert "never faced the promotion gates" in out and "did not clear" not in out


def test_the_refusal_for_a_failed_run_points_at_a_command_that_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("tradeflow.optimization.config_store.DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr("tradeflow.services.audit.DEFAULT_TRIAL_JOURNAL", journal, raising=False)
    trial_id = journal_trial(
        "walkforward",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
        params={"fast_ema_period": 5},
        metrics={"sharpe_ratio": -1.0},
        extra={"promotable": False},
        path=journal,
    )
    args = build_parser().parse_args(
        ["trials", "promote", trial_id, "--save-config", str(tmp_path / "p.json")]
        + ["--db", str(db_path_for_journal(journal)), "--journal", str(journal)]
    )
    with pytest.raises(SystemExit) as refused:
        args.func(args)
    message = str(refused.value)
    assert f"trials campaign {trial_id}" in message
    assert "Re-run `trials show`" not in message


# --- show prints the proposed book, not only its size ---------------------------------
def test_show_lists_the_proposed_book_not_just_how_many_names(store):
    """A verdict's next step sends the reader to `trials show` for the proposed book, which
    printed "8 names"."""
    _record(store, "vd", kind="verdict")
    book = {f"S{i}": 0.1 + i * 0.01 for i in range(8)}
    store.record_weights("vd", {"as_of": "2024-12-31", "weights": book, "active_weights": {"S7": 0.02}})
    text = format_trial_detail(store.get_trial("vd"))
    assert "8 names" in text
    assert "S7" in text and "17.00%" in text  # largest first, with its weight
    assert text.index("S7") < text.index("S0")
    assert "total" in text

    many = {f"X{i:02d}": 0.01 for i in range(40)}
    _record(store, "wide", kind="verdict", params={"fast": 2})
    store.record_weights("wide", {"as_of": "2024-12-31", "weights": many})
    wide = format_trial_detail(store.get_trial("wide"))
    assert "15 more name(s) not shown" in wide  # truncated loudly


# --- comparing unlike series says so ----------------------------------------------------
def test_comparing_a_stitched_oos_series_with_a_full_window_one_is_flagged():
    import numpy as np

    from tradeflow.analytics.reporting import format_series_comparison
    from tradeflow.analytics.series_comparison import compare_series

    dates = [f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(120)]
    rng = np.random.default_rng(3)

    def entry(trial_id, kind):
        return {
            "trial_id": trial_id,
            "dates": dates,
            "values": list(rng.normal(0, 0.01, 120)),
            "accounting": 6,
            "strategy": "demo_trend",
            "kind": kind,
        }

    unlike = compare_series([entry("wf", "walkforward"), entry("bt", "backtest")])
    assert unlike["kinds_differ"] is True and unlike["pairs"][0]["unlike_basis"] is True
    text = format_series_comparison(unlike)
    assert "walkforward: stitched out-of-sample folds" in text
    assert "backtest: full-window, in-sample" in text
    assert "UNLIKE SERIES" in text

    like = compare_series([entry("a", "walkforward"), entry("b", "research")])
    assert like["kinds_differ"] is False and like["pairs"][0]["unlike_basis"] is False
    assert "UNLIKE SERIES" not in format_series_comparison(like)


# --- a date bound discloses the rows it could not place -------------------------------
def test_a_date_filter_counts_the_rows_with_no_recorded_time(store):
    """`--since` said "12 matching" of 14: the two rows with an unknown timestamp
    satisfied neither bound and vanished. Unknown dates are disclosed, never guessed."""
    _record(store, "dated")
    _record(store, "undated", ts=None, params={"fast": 9})
    listed = store.listing(since=datetime(2024, 1, 1), accounting=ACCOUNTING_VERSION)
    assert [r["id"] for r in listed["rows"]] == ["dated"]
    assert listed["undated_excluded"] == 1
    assert "no recorded timestamp" in format_trials_table(listed["rows"], undated_excluded=1)
    # No date bound, nothing to disclose - and the row is there.
    everything = store.listing(accounting=ACCOUNTING_VERSION)
    assert everything["undated_excluded"] == 0 and len(everything["rows"]) == 2
