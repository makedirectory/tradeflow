"""`screen --workers N` evaluates its points across worker processes.

The defect: the screen built its ``ParameterOptimizer`` with ``workers`` but no worker
data spec, and the optimizer runs sequentially without one (a worker cannot rebuild a
data client it was never told how to build). So ``--workers 4`` was accepted, documented
as "evaluate points in parallel", and changed nothing: an inert flag. The screen now
takes the worker data spec from the same rule ``optimize`` and ``walkforward`` use.

These tests drive the real path - a warmed bar cache read offline by spawned workers,
the environment ``--workers`` actually runs in - rather than a synthetic feed handed to
the workers, because a worker reading something other than the cache the parent read is
exactly the difference a sequential/parallel comparison exists to catch.
"""

import pytest

pytest.importorskip("pyarrow")

from quantred.marketdata.timeframe import Timeframe  # noqa: E402
from quantred.optimization import parallel  # noqa: E402
from quantred.services import analysis  # noqa: E402
from quantred.store.bars import CachedMarketData  # noqa: E402
from quantred.utils.timeutils import parse_market_date  # noqa: E402
from tests.fakes import FakeMarketData  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC"]
START_TEXT, END_TEXT = "2024-02-01", "2024-09-30"
START, END = parse_market_date(START_TEXT), parse_market_date(END_TEXT)


@pytest.fixture(autouse=True)
def _isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")


@pytest.fixture
def warmed_cache(tmp_path):
    """A bar cache holding the screen's window, written from the offline fake feed."""
    cache_dir = tmp_path / "bars"
    with CachedMarketData(FakeMarketData(SYMBOLS, n=300, freq="1D"), cache_dir=cache_dir) as cache:
        cache.warm(SYMBOLS, Timeframe.parse("1Day"), START, END)
    return cache_dir


@pytest.fixture
def pool_calls(monkeypatch):
    """Every dispatch to the real process pool, recorded and then run for real."""
    calls = []
    real = parallel.run_pool

    def recording(requests, workers, **kwargs):
        calls.append({"requests": list(requests), "workers": workers})
        return real(requests, workers, **kwargs)

    monkeypatch.setattr(parallel, "run_pool", recording)
    return calls


def _offline_client(cache_dir):
    from quantred.services.data import build_data_client

    return build_data_client(offline=True, cache_dir=cache_dir)


def test_screen_workers_dispatches_its_points_to_a_process_pool(warmed_cache, pool_calls, capsys):
    """Through the CLI a user types. Before the fix this ran every point in the parent
    and never reached the pool, with ``--workers 2`` accepted and silently ignored."""
    from quantred import cli

    args = cli.build_parser().parse_args(
        [
            "screen",
            "--strategy",
            "demo_trend",
            "--scanner",
            "none",
            "--symbols",
            ",".join(SYMBOLS),
            "--start",
            START_TEXT,
            "--end",
            END_TEXT,
            "--method",
            "random",
            "--max-evals",
            "4",
            "--offline",
            "--cache-dir",
            str(warmed_cache),
            "--workers",
            "2",
            "--json",
        ]
    )
    cli.cmd_screen(args)

    assert len(pool_calls) == 1, "the screen never reached the worker pool"
    dispatched = pool_calls[0]
    assert dispatched["workers"] == 2
    assert len(dispatched["requests"]) == 4
    # Workers read the cache this run was pointed at, not the default one.
    assert {r.data.cache_dir for r in dispatched["requests"]} == {str(warmed_cache)}
    assert {r.data.offline for r in dispatched["requests"]} == {True}
    assert '"evaluated": 4' in capsys.readouterr().out


def test_a_screen_reports_the_same_thing_with_one_worker_or_two(warmed_cache, pool_calls):
    """Parallelism changes wall-clock and nothing else: one comparison, both paths.

    The parallel run is required to have actually dispatched, so this cannot pass by
    both runs quietly taking the sequential path - which is precisely what the inert
    flag did."""
    import csv

    def screen(workers):
        report = analysis.run_screen(
            _offline_client(warmed_cache),
            "demo_trend",
            SYMBOLS,
            START,
            END,
            method="random",
            max_evals=6,
            seed=11,
            workers=workers,
            offline=True,
            cache_dir=warmed_cache,
        )
        with open(report["results_csv"], newline="") as handle:
            rows = list(csv.DictReader(handle))
        return report, rows

    sequential, sequential_rows = screen(1)
    assert pool_calls == []
    concurrent, concurrent_rows = screen(2)
    assert len(pool_calls) == 1 and pool_calls[0]["workers"] == 2

    assert concurrent["searched"]["evaluated"] == sequential["searched"]["evaluated"] == 6
    ignored = {"run_id", "results_csv"}
    assert {k: v for k, v in concurrent.items() if k not in ignored} == {
        k: v for k, v in sequential.items() if k not in ignored
    }
    assert concurrent_rows == sequential_rows


def test_a_sequential_screen_needs_no_worker_data_spec(warmed_cache, monkeypatch):
    """Both directions: one worker is the original in-process path, so nothing is
    dispatched and the cache is not touched on the workers' behalf."""

    def no_pool(*a, **k):
        raise AssertionError("a one-worker screen dispatched to a pool")

    monkeypatch.setattr(parallel, "run_pool", no_pool)
    report = analysis.run_screen(
        _offline_client(warmed_cache), "demo_trend", SYMBOLS, START, END, method="random", max_evals=3
    )
    assert report["searched"]["evaluated"] == 3


def test_an_offline_run_does_not_try_to_warm_a_cache_it_may_not_fetch_into(monkeypatch, caplog):
    """``warm_for`` built a *keyed* client whatever the spec said, so every
    ``--offline --workers`` run - screen, optimize, walkforward - logged a pre-warm
    failure and a traceback on a machine without keys, while the run itself was fine.
    An offline cache cannot be filled, so there is nothing to warm."""
    import logging

    from quantred.services import data

    def keyed_client(**kwargs):
        raise AssertionError(f"an offline warm built a client: {kwargs}")

    monkeypatch.setattr(data, "build_data_client", keyed_client)
    with caplog.at_level(logging.WARNING, logger="quantred.optimization.parallel"):
        parallel.warm_for(parallel.DataSpec(kind="cache", offline=True), SYMBOLS, "1Day", START, END)
    assert "pre-warm failed" not in caplog.text


def test_an_online_cache_spec_is_still_warmed(monkeypatch):
    """The other direction: the skip is for offline only."""
    from quantred.services import data

    built = []
    monkeypatch.setattr(data, "build_data_client", lambda **kwargs: built.append(kwargs) or _NoProvider())
    parallel.warm_for(parallel.DataSpec(kind="cache", offline=False), SYMBOLS, "1Day", START, END)
    assert built and built[0]["cache"] is True


class _NoProvider:
    provider = None
