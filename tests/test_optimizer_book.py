"""The book and the capital a searched candidate is evaluated at, and keyed under.

`position_limits` is not a tunable param, so a candidate built from params alone is the
class-default book (one position) wearing the config's name. Two defects came from that:
a walk-forward's in-sample search ran at the default book while its folds ran at the
configured one, and a stored candidate result was keyed on params and costs only, so a
search at one capital or book answered a request at another.

Each test captures the strategy the engine is *handed* (the effect), not whether a field
was forwarded.
"""

import types
from datetime import datetime

import pytest

from tests.fakes import FakeMarketData
from tests.test_walk_forward import PeriodicStrategy
from tradeflow.engine import backtest as backtest_module
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.marketdata.synthetic import SyntheticMarketData
from tradeflow.optimization import optimizer as optimizer_module
from tradeflow.optimization import parallel
from tradeflow.optimization import walk_forward as walk_forward_module
from tradeflow.optimization.optimizer import ParameterOptimizer
from tradeflow.optimization.parallel import DataSpec, EvalRequest, candidate_key
from tradeflow.optimization.walk_forward import WalkForwardValidator
from tradeflow.services import analysis, audit
from tradeflow.services.registry import STRATEGIES

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2024, 9, 1)
STRATEGY = "demo_trend"
SPEC = DataSpec(kind="synthetic", seed=7)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")


def _client():
    return MarketDataClient(SyntheticMarketData(seed=7))


def _optimizer(**kwargs):
    return ParameterOptimizer(
        STRATEGIES[STRATEGY], _client(), initial_capital=100_000.0, seed=42, strategy_name=STRATEGY, **kwargs
    )


def _record_books(monkeypatch, *modules):
    """max_positions of every strategy each module's engine is handed."""
    seen = {}
    for module in modules:
        real = module.BacktestEngine
        books = seen.setdefault(module.__name__, [])

        def engine(strategy, *args, _real=real, _books=books, **kwargs):
            _books.append(strategy.position_limits()["max_positions"])
            return _real(strategy, *args, **kwargs)

        monkeypatch.setattr(module, "BacktestEngine", engine)
    return seen


# --- defect: the in-sample search ran at the default book --------------------
def test_the_in_sample_search_runs_at_the_book_the_out_of_sample_folds_do(monkeypatch):
    seen = _record_books(monkeypatch, optimizer_module, walk_forward_module)
    client = MarketDataClient(FakeMarketData(SYMBOLS, n=600, freq="1D"))
    result = WalkForwardValidator(
        PeriodicStrategy, client, seed=42, position_limits={"max_positions": 8}
    ).run(
        symbols=SYMBOLS,
        start=datetime(2024, 1, 2),
        end=datetime(2025, 6, 1),
        n_folds=2,
        holdout_days=30,
        method="grid",
        max_evals=3,
    )

    assert result.folds
    in_sample = seen[optimizer_module.__name__]
    assert in_sample, "the in-sample search never reached its engine"
    assert set(in_sample) == {8}
    assert set(seen[walk_forward_module.__name__]) == {8}


def _request(params, **kwargs):
    complete = {**STRATEGIES[STRATEGY].create_with_defaults().config, **params}
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


def test_a_worker_evaluates_a_candidate_at_the_book_it_was_sent(monkeypatch):
    seen = _record_books(monkeypatch, backtest_module)
    result = parallel.evaluate(_request({"fast_ema_period": 8}, position_limits={"max_positions": 6}))
    assert result["error"] is None
    assert seen[backtest_module.__name__] == [6]


def test_the_parallel_dispatch_carries_the_book_to_every_request(monkeypatch):
    seen = _record_books(monkeypatch, backtest_module)
    monkeypatch.setattr(
        parallel,
        "run_pool",
        lambda requests, workers, on_result=None: types.SimpleNamespace(
            results=[parallel.evaluate(r) for r in requests]
        ),
    )
    monkeypatch.setattr(parallel, "summarize", lambda report: "")
    optimizer = _optimizer(workers=2, data_spec=SPEC, position_limits={"max_positions": 5})
    optimizer.workers = 2  # resolve_workers caps at the machine's cores
    optimizer.grid_search(SYMBOLS, START, END, "sharpe_ratio", max_evals=3)
    books = seen[backtest_module.__name__]
    assert books and set(books) == {5}


# --- defect: candidate memos ignored capital and book ------------------------
def _optimize(capital):
    return analysis.run_optimization(
        _client(), STRATEGY, SYMBOLS, START, END, method="grid", max_evals=3, capital=capital
    )


def test_a_candidate_scored_at_one_capital_is_not_served_to_another():
    assert _optimize(100_000.0)["n_memoized"] == 0
    other = _optimize(4_000.0)
    assert other["n_memoized"] == 0
    # The same request again is what the memo exists for.
    assert _optimize(4_000.0)["n_memoized"] == other["n_trials"]


def test_a_candidate_identity_changes_with_the_book_it_is_scored_at():
    one = _optimizer(position_limits={"max_positions": 1}).candidate_identity()
    eight = _optimizer(position_limits={"max_positions": 8}).candidate_identity()
    assert one != eight
    # No override resolves to the class book, so it keys as that book spelled out.
    assert _optimizer().candidate_identity() == _optimizer(position_limits={}).candidate_identity()
