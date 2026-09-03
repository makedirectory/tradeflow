"""Concurrent trial execution: determinism, dedup, and the single-writer invariant.

All offline and deterministic. The headline property is that parallelism changes
throughput and **nothing else**: the same seed produces the same trials, the same
winner, and the same campaign count whether one process or four did the work. The
one test that must run a genuine process pool does so, because spawn semantics
(macOS, Windows) are exactly what a mocked pool would hide.
"""

import json
from datetime import datetime

import pytest

from tradeflow.marketdata.client import MarketDataClient
from tradeflow.marketdata.synthetic import SyntheticMarketData
from tradeflow.optimization import parallel
from tradeflow.optimization.optimizer import ParameterOptimizer
from tradeflow.optimization.parallel import DataSpec, EvalRequest, candidate_key, resolve_workers, run_pool
from tradeflow.services import analysis, audit
from tradeflow.services.registry import STRATEGIES

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2024, 9, 1)
STRATEGY = "demo_trend"

#: The keyless deterministic feed — reconstructible in any process, which is what
#: makes a genuinely parallel test both offline and reproducible.
SPEC = DataSpec(kind="synthetic", seed=7)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    return tmp_path


def _client():
    return MarketDataClient(SyntheticMarketData(seed=7))


def _optimizer(**kwargs):
    return ParameterOptimizer(
        STRATEGIES[STRATEGY],
        _client(),
        initial_capital=100_000.0,
        seed=42,
        strategy_name=STRATEGY,
        **kwargs,
    )


def _full(params):
    """Searched values over the strategy's full defaults — what the search itself
    dispatches. A strategy is constructed from a complete config, not a partial one."""
    return {**STRATEGIES[STRATEGY].create_with_defaults().config, **params}


def _request(params, **kwargs):
    complete = _full(params)
    return EvalRequest(
        key=candidate_key(STRATEGY, complete, SYMBOLS, START, END),
        strategy=STRATEGY,
        params=complete,
        symbols=tuple(SYMBOLS),
        start=START,
        end=END,
        data=SPEC,
        **kwargs,
    )


# --- worker resolution ------------------------------------------------------
def test_sequential_is_the_original_path_not_a_pool_of_one():
    """Nobody should pay pickling and spawn costs to run one thing at a time."""
    assert resolve_workers(None) == 1
    assert resolve_workers(0) == 1
    assert resolve_workers(1) == 1
    assert resolve_workers(2) >= 1


def test_workers_never_exceed_the_machine():
    import os

    assert resolve_workers(9999) <= (os.cpu_count() or 1)


def test_a_parallel_request_without_a_data_recipe_falls_back_to_sequential():
    """A worker builds its own client from a recipe; guessing at how to reconstruct
    someone's client is how a parallel run reads different bars than a serial one."""
    assert not _optimizer(workers=4)._parallel_available()
    assert _optimizer(workers=4, data_spec=SPEC)._parallel_available()


# --- candidate identity and seeding ----------------------------------------
def test_a_candidates_seed_comes_from_its_identity_not_its_position():
    """The same candidate must simulate identically whether it ran first, last, or
    in the sequential path."""
    first = _request({"fast_ema_period": 10, "slow_ema_period": 30})
    same = _request({"fast_ema_period": 10, "slow_ema_period": 30})
    other = _request({"fast_ema_period": 11, "slow_ema_period": 30})

    assert first.key == same.key and first.seed == same.seed
    assert other.key != first.key and other.seed != first.seed


def test_identity_matches_the_stores_own_definition():
    """A second definition of "the same candidate" would eventually disagree with
    the store, and the disagreement would show up as a miscounted campaign."""
    from tradeflow.store.trials import params_hash, universe_hash

    params = {"fast_ema_period": 10}
    key = candidate_key(STRATEGY, params, SYMBOLS, START, END)
    assert universe_hash(SYMBOLS) in key
    assert params_hash(params) in key
    # Symbol order and case do not change a universe, so they must not change a key.
    assert key == candidate_key(STRATEGY, params, ["bbb", "aaa"], START, END)


# --- the pool ---------------------------------------------------------------
def test_a_real_process_pool_runs_offline_and_deterministically():
    """A genuine pool, not a mock: spawn semantics are exactly what a mock hides.
    Every request argument must be importable and picklable, with no live client
    closed over."""
    requests = [_request({"fast_ema_period": fast, "slow_ema_period": 30}) for fast in (5, 8, 10, 12)]
    report = run_pool(requests, workers=2)

    assert report.completed == 4
    assert report.failures == 0 and not report.interrupted
    assert [r["key"] for r in report.results] == [r.key for r in requests]  # submission order
    assert all(r["metrics"] for r in report.results)


def test_results_come_back_in_submission_order_whatever_the_completion_order():
    """Aggregation must not depend on how the scheduler interleaved the work."""
    requests = [_request({"fast_ema_period": f, "slow_ema_period": 30}) for f in (5, 6, 7, 8, 9)]
    first = run_pool(requests, workers=3)
    second = run_pool(requests, workers=2)
    assert [r["key"] for r in first.results] == [r["key"] for r in second.results]
    assert [r["metrics"] for r in first.results] == [r["metrics"] for r in second.results]


def test_a_duplicate_candidate_is_dispatched_once_and_counted_once():
    """Two identical candidates are one trial. Running both would inflate the
    campaign's multiple-testing total with work that produced no information."""
    params = {"fast_ema_period": 10, "slow_ema_period": 30}
    report = run_pool([_request(params), _request(params), _request(params)], workers=2)

    assert report.completed == 1
    assert report.duplicates == 2
    assert "2 duplicate candidate(s) skipped" in parallel.summarize(report)


def test_a_worker_crash_fails_that_candidate_and_not_the_campaign():
    good = _request({"fast_ema_period": 10, "slow_ema_period": 30})
    # An unknown strategy raises inside the worker, which is a returned error.
    bad = EvalRequest(
        key="broken",
        strategy="no_such_strategy",
        params={},
        symbols=tuple(SYMBOLS),
        start=START,
        end=END,
        data=SPEC,
    )
    report = run_pool([good, bad], workers=2)

    assert report.completed == 2
    assert report.failures == 1
    errors = [r for r in report.results if r["error"]]
    assert len(errors) == 1 and errors[0]["key"] == "broken"
    # And the healthy candidate still produced its numbers.
    assert next(r for r in report.results if r["key"] == good.key)["metrics"]
    assert "1 failed" in parallel.summarize(report)


def test_an_empty_dispatch_is_not_an_error():
    report = run_pool([], workers=2)
    assert report.completed == 0 and not report.interrupted


def test_progress_reporting_happens_in_the_parent():
    """Workers never write to stdout, so N workers' chatter cannot interleave."""
    seen = []
    requests = [_request({"fast_ema_period": f, "slow_ema_period": 30}) for f in (5, 8)]
    run_pool(requests, workers=2, on_result=seen.append)
    assert len(seen) == 2


# --- the worker contract ----------------------------------------------------
def test_evaluate_returns_failures_rather_than_raising():
    result = parallel.evaluate(
        EvalRequest(key="k", strategy="nope", params={}, symbols=("AAA",), start=START, end=END, data=SPEC)
    )
    assert result["error"] and result["metrics"] == {}


def test_a_request_is_picklable_with_nothing_live_inside_it():
    import pickle

    request = _request({"fast_ema_period": 10, "slow_ema_period": 30})
    assert pickle.loads(pickle.dumps(request)) == request


def test_a_cost_model_round_trips_through_its_spec():
    """Two descriptions of one cost model would drift, and the drift would show up
    as a parallel run pricing trades differently from the sequential one."""
    from tradeflow.costs import ParametricCostModel

    original = ParametricCostModel(commission_bps=2.5, impact_eta=0.4, annual_borrow_bps=75.0)
    rebuilt = parallel._build_cost_model(parallel.cost_spec(original))

    assert rebuilt.commission_rate == pytest.approx(original.commission_rate)
    assert rebuilt.impact_eta == pytest.approx(original.impact_eta)
    assert rebuilt.annual_borrow_rate == pytest.approx(original.annual_borrow_rate)
    assert parallel.cost_spec(None) is None


# --- determinism vs the sequential path -------------------------------------
def _search(workers, data_spec=None):
    opt = _optimizer(workers=workers, data_spec=data_spec)
    return opt.grid_search(SYMBOLS, START, END, "sharpe_ratio", max_evals=6)


def test_the_same_seed_gives_the_same_trials_and_the_same_winner():
    """The headline test: parallelism changes throughput and nothing else."""
    sequential = _search(workers=1)
    concurrent = _search(workers=3, data_spec=SPEC)

    assert concurrent.best_params == sequential.best_params
    assert concurrent.best_score == pytest.approx(sequential.best_score)
    assert len(concurrent.results) == len(sequential.results)

    columns = sorted(set(sequential.results.columns) & set(concurrent.results.columns))
    assert concurrent.results[columns].round(8).to_dict("records") == (
        sequential.results[columns].round(8).to_dict("records")
    )


def test_ranking_is_a_total_order_so_ties_cannot_flip_the_winner():
    """Sorting on score alone leaves tied candidates in evaluation order, so a
    parallel run completing differently could pick a different best config."""
    import pandas as pd

    opt = _optimizer()
    rows = [
        {"fast_ema_period": 12, "slow_ema_period": 30, "sharpe_ratio": 1.0},
        {"fast_ema_period": 5, "slow_ema_period": 30, "sharpe_ratio": 1.0},
        {"fast_ema_period": 9, "slow_ema_period": 30, "sharpe_ratio": 1.0},
    ]
    winners = {
        opt._build_result(list(order), "sharpe_ratio").best_params["fast_ema_period"]
        for order in (rows, rows[::-1], [rows[1], rows[2], rows[0]])
    }
    assert winners == {5}  # the same winner from every input ordering
    assert isinstance(opt._build_result(rows, "sharpe_ratio").results, pd.DataFrame)


# --- the single-writer invariant -------------------------------------------
def test_a_parallel_search_journals_exactly_what_a_sequential_one_does(_isolated_state, monkeypatch):
    """Workers execute and return; only this process writes. The campaign count must
    not depend on how the work was scheduled."""

    def _trials():
        path = _isolated_state / "journal.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    monkeypatch.setattr(analysis, "_worker_data_spec", lambda *a, **k: SPEC)
    analysis.run_optimization(_client(), STRATEGY, SYMBOLS, START, END, method="grid", max_evals=4, workers=3)
    parallel_rows = _trials()

    (_isolated_state / "journal.jsonl").unlink()
    (_isolated_state / "trials.db").unlink(missing_ok=True)
    analysis.run_optimization(_client(), STRATEGY, SYMBOLS, START, END, method="grid", max_evals=4)
    sequential_rows = _trials()

    assert len(parallel_rows) == len(sequential_rows)
    assert {r["tool"] for r in parallel_rows} == {"trial:optimize"}
    # Same configs journaled, same metrics — only the timestamps and ids differ.
    assert sorted(json.dumps(r["resolved_config"], sort_keys=True) for r in parallel_rows) == sorted(
        json.dumps(r["resolved_config"], sort_keys=True) for r in sequential_rows
    )


def test_no_worker_ever_opens_the_trial_store():
    """The whole design rests on this: nothing in the worker's import path or its
    request touches the journal or the index."""
    import inspect

    source = inspect.getsource(parallel.evaluate)
    for forbidden in ("journal_trial", "TrialStore", "audit_log", "record("):
        assert forbidden not in source


def test_workers_default_conservatively_rather_than_to_every_core():
    """Each worker holds its own bar frames, so memory scales with the count."""
    assert parallel.DEFAULT_MAX_WORKERS <= 4
