"""A search stopped part-way names no winner, and both execution paths agree on that.

The defect: ``run_pool`` reports ``interrupted=True`` with the candidates that finished,
and ``ParameterOptimizer._evaluate_parallel`` never read it. The partial rows were ranked
like a whole search, a "best of N" was named from the first few, and the callers journaled
those rows as trials and printed the winner - exit status 0. The sequential path raises
``KeyboardInterrupt`` out of the search, so its callers journal nothing and name nothing.
These tests drive both paths to the same interruption and compare what they leave behind.
"""

import concurrent.futures as futures
import json
from datetime import datetime

import pytest

from quantred.marketdata.client import MarketDataClient
from quantred.marketdata.synthetic import SyntheticMarketData
from quantred.optimization import parallel
from quantred.optimization.optimizer import ParameterOptimizer, SearchInterrupted
from quantred.optimization.parallel import DataSpec
from quantred.services import analysis, audit
from quantred.services.registry import STRATEGIES

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2024, 9, 1)
STRATEGY = "demo_trend"
SPEC = DataSpec(kind="synthetic", seed=7)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(analysis, "_worker_data_spec", lambda *a, **k: SPEC)
    return tmp_path


def _client():
    return MarketDataClient(SyntheticMarketData(seed=7))


def _journal(tmp_path):
    path = tmp_path / "journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _interrupt_pool_after_first_result(monkeypatch):
    """Ctrl-C arriving in the parent once one candidate has finished, through the real
    ``run_pool`` and a real process pool."""
    real = futures.as_completed

    def interrupted(pending, *a, **k):
        for future in real(pending, *a, **k):
            yield future
            raise KeyboardInterrupt

    monkeypatch.setattr(parallel.futures, "as_completed", interrupted)


def _interrupt_sequential_after_first_result(monkeypatch):
    real = ParameterOptimizer._backtest
    calls = [0]

    def backtest(self, *a, **k):
        calls[0] += 1
        if calls[0] > 1:
            raise KeyboardInterrupt
        return real(self, *a, **k)

    monkeypatch.setattr(ParameterOptimizer, "_backtest", backtest)


def _leftovers(tmp_path):
    artifacts = tmp_path / "artifacts"
    return _journal(tmp_path), sorted(p.name for p in artifacts.glob("*")) if artifacts.exists() else []


def test_an_interrupted_parallel_search_names_no_winner_and_journals_nothing(_isolated_state, monkeypatch):
    """Before the fix this returned a result with a best config chosen from the one
    candidate that finished, and journaled that candidate as a trial of a search that
    never completed."""
    _interrupt_pool_after_first_result(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        analysis.run_optimization(
            _client(), STRATEGY, SYMBOLS, START, END, method="grid", max_evals=4, workers=2
        )
    assert _leftovers(_isolated_state) == ([], [])


def test_both_execution_paths_leave_the_same_trace_when_interrupted(_isolated_state, monkeypatch):
    """The parity check: one interruption, driven through each path, compared."""
    traces = {}
    with monkeypatch.context() as patched:
        _interrupt_pool_after_first_result(patched)
        with pytest.raises(KeyboardInterrupt):
            analysis.run_optimization(
                _client(), STRATEGY, SYMBOLS, START, END, method="grid", max_evals=4, workers=2
            )
        traces["parallel"] = _leftovers(_isolated_state)
    with monkeypatch.context() as patched:
        _interrupt_sequential_after_first_result(patched)
        with pytest.raises(KeyboardInterrupt):
            analysis.run_optimization(_client(), STRATEGY, SYMBOLS, START, END, method="grid", max_evals=4)
        traces["sequential"] = _leftovers(_isolated_state)
    assert traces["parallel"] == traces["sequential"] == ([], [])


def test_the_interruption_says_how_much_was_discarded(monkeypatch):
    opt = ParameterOptimizer(
        STRATEGIES[STRATEGY], _client(), initial_capital=100_000.0, seed=42, strategy_name=STRATEGY
    )
    opt.workers, opt.data_spec = 2, SPEC
    _interrupt_pool_after_first_result(monkeypatch)
    with pytest.raises(SearchInterrupted) as caught:
        opt.grid_search(SYMBOLS, START, END, "sharpe_ratio", max_evals=4)
    assert isinstance(caught.value, KeyboardInterrupt)
    assert caught.value.completed >= 1
    assert "no winner" in str(caught.value)


def test_a_bayesian_search_stops_at_the_interrupted_round(monkeypatch):
    """The loop used to carry on into the next round after an interrupted batch, running
    more pools on a run the user had already stopped."""
    pytest.importorskip("sklearn")
    rounds = []

    def fake_run_pool(requests, workers, *, on_result=None):
        rounds.append(len(requests))
        return parallel.PoolReport(interrupted=True)

    monkeypatch.setattr(parallel, "run_pool", fake_run_pool)
    opt = ParameterOptimizer(
        STRATEGIES[STRATEGY],
        _client(),
        initial_capital=100_000.0,
        seed=42,
        strategy_name=STRATEGY,
        workers=2,
        data_spec=SPEC,
    )
    # The initial points run sequentially; make them cheap by trusting the real engine.
    with pytest.raises(KeyboardInterrupt):
        opt.optimize_bayesian(SYMBOLS, START, END, "sharpe_ratio", n_initial=3, n_iterations=6)
    assert len(rounds) == 1


def test_the_cli_names_the_missing_winner_when_a_search_is_interrupted():
    from unittest import mock

    from quantred import cli

    def interrupted(args):
        raise SearchInterrupted(completed=3)

    namespace = mock.Mock()
    namespace.func = interrupted
    with mock.patch.object(cli, "parse_cli", return_value=namespace), mock.patch.object(cli, "setup_logging"):
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
    message = str(exit_info.value)
    assert "Interrupted" in message and "no winner" in message
    assert "no trial recorded" in message


def test_an_interrupted_walk_forward_inner_search_ends_the_validation(monkeypatch):
    """The walk-forward's in-sample search is a ``ParameterOptimizer``. Interrupted, it used
    to hand back a partial winner, the fold was scored out of sample with it, and the
    loop went on to the next fold - a validation built on searches that never finished."""
    from quantred.optimization.walk_forward import WalkForwardValidator

    pools = []
    real = futures.as_completed

    def interrupted(pending, *a, **k):
        pools.append(len(pending))
        for future in real(pending, *a, **k):
            yield future
            raise KeyboardInterrupt

    monkeypatch.setattr(parallel.futures, "as_completed", interrupted)
    validator = WalkForwardValidator(
        STRATEGIES[STRATEGY], _client(), seed=42, workers=2, data_spec=SPEC, strategy_name=STRATEGY
    )
    with pytest.raises(KeyboardInterrupt):
        validator.run(
            symbols=SYMBOLS, start=START, end=END, n_folds=2, holdout_days=20, method="grid", max_evals=4
        )
    assert len(pools) == 1, "the validation carried on to another search after the interrupt"


def test_an_interrupted_parallel_screen_leaves_the_same_trace_as_a_sequential_one(
    _isolated_state, monkeypatch
):
    """``screen --workers`` used to run sequentially whatever was asked, so it never met
    the parallel interrupt. Now that it dispatches, a Ctrl-C must not hand back a
    distribution over the few points that finished - a screen's whole report is a
    distribution, and one over a subset reads as the sweep that was asked for. Both paths
    stop with nothing written, compared here rather than each asserted alone."""
    traces = {}
    dispatched = []
    real_pool = parallel.run_pool

    def recording_pool(requests, workers, **kwargs):
        dispatched.append(workers)
        return real_pool(requests, workers, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(parallel, "run_pool", recording_pool)
        _interrupt_pool_after_first_result(patched)
        with pytest.raises(SearchInterrupted):
            analysis.run_screen(
                _client(), STRATEGY, SYMBOLS, START, END, method="grid", max_evals=4, workers=2
            )
        traces["parallel"] = _leftovers(_isolated_state)
    with monkeypatch.context() as patched:
        _interrupt_sequential_after_first_result(patched)
        with pytest.raises(KeyboardInterrupt):
            analysis.run_screen(_client(), STRATEGY, SYMBOLS, START, END, method="grid", max_evals=4)
        traces["sequential"] = _leftovers(_isolated_state)
    assert dispatched == [2], "the interrupted screen never took the parallel path"
    assert traces["parallel"] == traces["sequential"] == ([], [])
