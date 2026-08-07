"""Concurrent trial execution: workers evaluate, the parent records.

Candidate evaluation is embarrassingly parallel — each candidate is an independent
backtest over the same read-only bars — and the machines this runs on have cores to
spare. What made parallelism dangerous was never the compute; it was that the trial
store is built on a **single-writer** assumption: one process appends the journal,
one process updates the index, and memoization assumes there are no in-flight
duplicates. Parallelism done naively breaks the store's core promise (an accurate,
deduplicated campaign trial count) in exactly the silent way this project fights.

The resolution is that the framework already separates *execution* from *recording*.
Workers never write anything: they receive a fully-described candidate, run it, and
return numbers. The parent alone journals, updates the index, and answers
memoization lookups. **Concurrency of execution does not require concurrency of
journaling**, so nothing about storage changes — no schema, no locking, no
interleaved journal.

Three properties this module exists to preserve, in order of importance:

1. **The campaign trial count stays exact.** In-flight dedup means a candidate
   already dispatched is never dispatched twice, and a crashed candidate is still a
   candidate that was tried. A search that runs faster but miscounts trials is a net
   loss — it corrupts the deflated Sharpe silently, at scale.
2. **Results do not depend on scheduling.** Each candidate's seed derives from its
   own identity, not from dispatch order, and results are returned in submission
   order regardless of completion order. A parallel run and a sequential run of the
   same search produce the same set of trials and the same winner.
3. **Nothing is shared mutably.** Workers build their own strategy, engine, and data
   client from a picklable description. There are no live clients crossing the
   process boundary, which is also what makes this work under spawn (macOS,
   Windows) rather than only under fork.
"""

import concurrent.futures as futures
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Conservative by default. Each worker holds its own copy of the bar frames, so
#: memory scales with workers x per-worker footprint — a wide universe times many
#: workers can dwarf the sequential run. Raising this is a deliberate act.
DEFAULT_MAX_WORKERS = 4


def resolve_workers(requested: Optional[int]) -> int:
    """How many workers to actually use.

    ``None`` or anything below 2 means sequential — and sequential is not "a pool of
    one", it is the original code path, byte for byte. Nobody should pay pickling and
    process-spawn costs to run one thing at a time.
    """
    if not requested or requested < 2:
        return 1
    cores = os.cpu_count() or 1
    return max(1, min(int(requested), cores))


# --------------------------------------------------------------------------- #
# Picklable descriptions (nothing live crosses the process boundary)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataSpec:
    """A recipe for building a data client, rather than a client itself.

    A live market-data client holds sockets and vendor SDK objects and cannot be
    pickled to a spawned process. Passing the *recipe* keeps workers independent and
    keeps this module free of any vendor import.

    ``kind="synthetic"`` builds the deterministic keyless feed — the same one the
    offline demo uses — which makes a genuinely parallel run reproducible with no
    network and no credentials.
    """

    kind: str = "cache"
    seed: int = 42
    cache_dir: Optional[str] = None
    offline: bool = False

    def build(self):
        """Construct this worker's own data client."""
        from tradeflow.marketdata.client import MarketDataClient

        if self.kind == "synthetic":
            from tradeflow.marketdata.synthetic import SyntheticMarketData

            return MarketDataClient(SyntheticMarketData(seed=self.seed))

        from tradeflow.services.data import build_data_client

        # Cache-first: the parent warmed the ranges before dispatch, so workers read
        # local Parquet instead of N processes stampeding the vendor API.
        return build_data_client(cache=True, offline=self.offline, cache_dir=self.cache_dir)


@dataclass(frozen=True)
class EvalRequest:
    """One candidate, described completely enough to run anywhere.

    ``key`` is the candidate's dedup identity. It does triple duty: it is what
    in-flight dedup compares, what the per-candidate seed derives from, and the
    tie-breaker that makes ranking a total order. All three have to agree, so there
    is one field rather than three.
    """

    key: str
    strategy: str
    params: Dict[str, Any]
    symbols: Tuple[str, ...]
    start: datetime
    end: datetime
    capital: float = 100_000.0
    cost: Optional[Dict[str, Any]] = None
    data: DataSpec = field(default_factory=DataSpec)
    base_seed: int = 42

    @property
    def seed(self) -> int:
        """This candidate's seed, derived from its identity and never from its
        position in the queue — so the same candidate simulates identically whether
        it ran first, last, or in the sequential path."""
        digest = hashlib.sha256(f"{self.base_seed}:{self.key}".encode()).hexdigest()
        return int(digest[:8], 16)


def candidate_key(strategy: str, params: Dict[str, Any], symbols: Sequence[str], start, end) -> str:
    """The stable identity of one candidate.

    Deliberately reuses the trial store's own hashing, so "the same candidate" means
    exactly what it means everywhere else — dedup, memoization, and the campaign
    count. A second definition here would eventually disagree with the store, and
    the disagreement would show up as a miscounted campaign rather than an error.
    """
    from tradeflow.store.trials import params_hash, universe_hash

    window = f"{_iso(start)}|{_iso(end)}"
    return f"{strategy}|{universe_hash(symbols)}|{params_hash(params)}|{window}"


def _iso(value: Any) -> str:
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


# --------------------------------------------------------------------------- #
# The worker side
# --------------------------------------------------------------------------- #
def evaluate(request: EvalRequest) -> Dict[str, Any]:
    """Run one candidate and return its numbers. Top-level and picklable by design.

    This is the entire worker contract. It constructs its own strategy, cost model,
    data client, and engine; it never touches the journal, the trial-store index, or
    stdout. A failure is *returned*, not raised: one candidate that cannot run must
    not take the campaign with it, and a crashed candidate is still a configuration
    that was tried.
    """
    try:
        from tradeflow.engine.backtest import BacktestEngine
        from tradeflow.services.registry import resolve_strategy_class

        strategy = resolve_strategy_class(request.strategy)(dict(request.params))
        result = BacktestEngine(
            strategy,
            request.data.build(),
            cost_model=_build_cost_model(request.cost),
        ).run(list(request.symbols), request.start, request.end, request.capital)
        return {
            "key": request.key,
            "params": dict(request.params),
            "metrics": dict(result.metrics),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - a worker failure is data, not a crash
        return {
            "key": request.key,
            "params": dict(request.params),
            "metrics": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _build_cost_model(spec: Optional[Dict[str, Any]]):
    """A cost model from its picklable kwargs; ``None`` means a gross run."""
    if spec is None:
        return None
    from tradeflow.costs import ParametricCostModel

    return ParametricCostModel(**spec)


def cost_spec(cost_model) -> Optional[Dict[str, Any]]:
    """A live cost model as the kwargs a worker rebuilds it from.

    Reads the model's own attributes back out rather than asking callers to pass the
    construction arguments twice — two descriptions of one cost model would drift,
    and the drift would show up as a parallel run pricing trades differently from
    the sequential one.
    """
    if cost_model is None:
        return None
    return {
        "commission_bps": getattr(cost_model, "commission_rate", 0.0) * 1e4,
        "default_spread_bps": getattr(cost_model, "default_spread", 0.0) * 1e4,
        "impact_eta": getattr(cost_model, "impact_eta", 0.3),
        "participation_cap": getattr(cost_model, "participation_cap", 0.10),
        "annual_borrow_bps": getattr(cost_model, "annual_borrow_rate", 0.0) * 1e4,
        "linear_impact": bool(getattr(cost_model, "linear_impact", False)),
    }


# --------------------------------------------------------------------------- #
# The parent side
# --------------------------------------------------------------------------- #
@dataclass
class PoolReport:
    """What a dispatch produced, including what it deliberately did not run."""

    results: List[Dict[str, Any]] = field(default_factory=list)
    duplicates: int = 0
    failures: int = 0
    interrupted: bool = False

    @property
    def completed(self) -> int:
        return len(self.results)


def run_pool(
    requests: Sequence[EvalRequest],
    workers: int,
    *,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> PoolReport:
    """Evaluate ``requests`` across a process pool; return results in **submission
    order**.

    Submission order, not completion order, is the point: it makes downstream
    aggregation independent of how the scheduler happened to interleave the work.
    Combined with per-candidate seeds derived from identity, that is what lets a
    parallel run and a sequential run produce the same answer.

    Duplicates are dropped before dispatch rather than deduplicated afterwards. Two
    identical candidates are one trial — running both would inflate the campaign's
    multiple-testing total with work that produced no new information, which is the
    exact accounting the trial store exists to keep honest.

    Ctrl-C cancels what has not started, keeps what has already finished (that
    compute was really spent), and reports the run as partial rather than pretending
    it completed.
    """
    unique: List[EvalRequest] = []
    seen: Dict[str, int] = {}
    duplicates = 0
    for request in requests:
        if request.key in seen:
            duplicates += 1
            continue
        seen[request.key] = len(unique)
        unique.append(request)

    report = PoolReport(duplicates=duplicates)
    if not unique:
        return report

    ordered: List[Optional[Dict[str, Any]]] = [None] * len(unique)
    try:
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(evaluate, request): i for i, request in enumerate(unique)}
            try:
                for future in futures.as_completed(pending):
                    index = pending[future]
                    ordered[index] = future.result()
                    if on_result is not None:
                        # Progress printing happens HERE, in the parent. Workers never
                        # write to stdout, so N workers' chatter cannot interleave.
                        on_result(ordered[index])
            except KeyboardInterrupt:
                report.interrupted = True
                for future in pending:
                    future.cancel()
                logger.warning("Interrupted; keeping %d completed evaluation(s)", _count(ordered))
    except KeyboardInterrupt:  # raised while the pool is shutting down
        report.interrupted = True

    report.results = [row for row in ordered if row is not None]
    report.failures = sum(1 for row in report.results if row.get("error"))
    return report


def _count(rows: Sequence[Optional[Dict[str, Any]]]) -> int:
    return sum(1 for row in rows if row is not None)


def warm_for(data_spec: DataSpec, symbols: Sequence[str], timeframe: str, start, end) -> None:
    """Pre-warm the bar cache over the union of what the workers will read.

    Done once, in the parent, before any dispatch. Without it, N workers hitting a
    cold cache stampede the vendor API for the same ranges simultaneously — N times
    the requests, N times the rate-limit exposure, for one range of bars. A failure
    here is logged and swallowed: warming is an optimization, and a worker that has
    to fetch its own bars still produces the right answer.
    """
    if data_spec.kind != "cache":
        return
    try:
        from tradeflow.services.data import build_data_client
        from tradeflow.store.bars import CachedMarketData

        provider = build_data_client(cache=True, cache_dir=data_spec.cache_dir).provider
        if isinstance(provider, CachedMarketData):
            provider.warm(list(symbols), timeframe, start, end)
    except Exception:  # noqa: BLE001 - warming never blocks the run
        logger.warning("Cache pre-warm failed; workers will fetch their own bars", exc_info=True)


def summarize(report: PoolReport) -> str:
    """A one-line account of a dispatch, including what it dropped.

    Silence about skipped or failed work reads as "everything ran", so the counts
    are always stated even when they are zero-worthy.
    """
    parts = [f"{report.completed} evaluated"]
    if report.duplicates:
        parts.append(f"{report.duplicates} duplicate candidate(s) skipped")
    if report.failures:
        parts.append(f"{report.failures} failed")
    if report.interrupted:
        parts.append("INTERRUPTED — partial run")
    return ", ".join(parts)


def as_json(value: Any) -> str:
    """Stable JSON for logging a candidate's params in a deterministic order."""
    return json.dumps(value, sort_keys=True, default=str)
