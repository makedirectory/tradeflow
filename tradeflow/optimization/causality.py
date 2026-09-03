"""Causality probes: could this decision have been made when it was made?

**These are a different class of probe from the leakage probe, and conflating them is
how three days were lost.** The leakage probe shifts the whole feed forward and checks
the result changes; it tests for *future data* — a strategy reading bars it should not
have. It ran against a candidate and passed while the engine was executing every signal
one bar before it could have known, because a feed shift moves signal and price
together: the relationship "signal from bar i's close, filled at bar i's open" survives
the shift completely intact.

So a feed-shift probe cannot test intra-bar causality, and a passing one says nothing
about it. What follows tests the other question: for each decision, was every input to
it available *strictly before* the price it transacted at.

The method is perturbation, not inspection. Change something that only becomes knowable
after the fill, re-run, and require the decision at that instant to be unchanged. A
decision that moves consumed information from its own future — whatever the code looks
like, and whichever layer did it.

Each probe reports ``passed=None`` when the run gave it nothing to test. That is not a
pass: a probe that could not run and a probe that ran and found nothing are different
facts, and only one of them is evidence.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from tradeflow.engine.backtest import BacktestEngine
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.optimization.walk_forward import PrefetchedProvider

logger = logging.getLogger(__name__)

#: How many decision instants each perturbation probe examines. Every one costs a
#: re-run, so this is the knob between "cheap enough to run routinely" and "exhaustive".
#: Reported in the result, because a probe that examined 3 of 40 entries and said
#: "passed" without saying so would be claiming far more than it checked.
DEFAULT_SAMPLE = 3

#: Probe classes. Named, and carried on every result, so a report cannot present a
#: future-data verdict as an intra-bar one.
FUTURE_DATA = "future-data"
INTRA_BAR = "intra-bar causality"
AS_OF = "as-of clock"


@dataclass
class ProbeResult:
    """One probe's verdict, and what it actually examined.

    ``passed`` is deliberately three-valued. ``None`` means the probe could not be
    exercised — no trades, no competition for slots, no benchmark — and it must never
    be rendered as a pass. A run that gave a probe nothing to look at has not been
    cleared by it.
    """

    name: str
    probe_class: str
    passed: Optional[bool]
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "probe_class": self.probe_class,
            "passed": self.passed,
            "reason": self.reason,
            **self.detail,
        }


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #
def _replace(frame: pd.DataFrame, when, column: str, value: float) -> pd.DataFrame:
    """A copy of ``frame`` with one cell replaced, widening the column to hold it.

    The cast is not incidental. A user's bars can arrive with integer price or volume
    columns, and pandas refuses a float into an int column rather than silently
    truncating - so a probe that skipped this would raise on real data while passing
    every test written against float fixtures.
    """
    out = frame.copy()
    out[column] = out[column].astype("float64")
    out.loc[when, column] = float(value)
    return out


def _perturbed_close(frame: pd.DataFrame, when) -> Optional[pd.DataFrame]:
    """A copy of ``frame`` whose bar at ``when`` closes at the other end of its range.

    Deliberately *inside* the bar's existing high/low. Moving the close outside would
    force the high or low to move with it, and a stop or take-profit legitimately reads
    the high and low of the bar it fills on — so a wider bar would change exits for an
    honest reason and the probe would report a leak that is not there.

    ``None`` when the bar has no range to move within: a flat bar carries no
    post-open information to withhold, so there is nothing to test at it.
    """
    if when not in frame.index:
        return None
    row = frame.loc[when]
    low, high, close = float(row["low"]), float(row["high"]), float(row["close"])
    if high <= low:
        return None
    # The far end from where it actually closed, so the change is as large as the bar
    # allows. A small nudge can leave a threshold-crossing score unchanged, and a probe
    # that perturbs too gently passes everything.
    replacement = low if close - low > high - close else high
    if replacement == close:
        return None
    return _replace(frame, when, "close", replacement)


def _perturbed_volume(frame: pd.DataFrame, when) -> Optional[pd.DataFrame]:
    """A copy whose bar at ``when`` carries a very different volume.

    Volume is the other thing about a bar that is unknown at its open, and it is not
    inert: it feeds the trailing average-daily-volume the cost model charges impact
    against, so it can reach the affordability test that admits or declines an entry.
    """
    if when not in frame.index or "volume" not in frame:
        return None
    current = float(frame.loc[when, "volume"])
    return _replace(frame, when, "volume", current * 25.0 + 1_000.0)


def _entries_at(trades: pd.DataFrame, when) -> List[Dict[str, Any]]:
    """Every entry filled at ``when``, as comparable records."""
    if trades is None or trades.empty or "entry_time" not in trades:
        return []
    rows = trades[trades["entry_time"] == when]
    return sorted(
        (
            {
                "symbol": str(r["symbol"]),
                "side": str(r["side"]),
                "entry_price": round(float(r["entry_price"]), 8),
                "size": round(float(r["size"]), 8),
            }
            for _, r in rows.iterrows()
        ),
        key=lambda r: r["symbol"],
    )


def _signal_exits_at(trades: pd.DataFrame, when) -> List[Dict[str, Any]]:
    """Signal-driven exits at ``when``.

    Stop and take-profit exits are excluded on purpose: those legitimately read the
    high and low of the bar they fill on, so they are not evidence about the decision
    clock. A signal exit is, because it must come from a prior bar's score.
    """
    if trades is None or trades.empty or "exit_time" not in trades:
        return []
    rows = trades[(trades["exit_time"] == when) & (trades.get("exit_reason") == "signal")]
    return sorted(
        (
            {"symbol": str(r["symbol"]), "exit_price": round(float(r["exit_price"]), 8)}
            for _, r in rows.iterrows()
        ),
        key=lambda r: r["symbol"],
    )


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
class CausalityProbes:
    """Re-runs one backtest under controlled perturbations of its own data.

    Constructed with the frames the run used, so every probe is served from memory and
    the whole suite costs one fetch however many re-runs it makes.
    """

    def __init__(
        self,
        strategy_factory: Callable[[], Any],
        frames: Dict[str, pd.DataFrame],
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        capital: float = 100_000.0,
        cost_model: Optional[Any] = None,
        benchmark: Optional[str] = None,
        sample: int = DEFAULT_SAMPLE,
    ):
        self.strategy_factory = strategy_factory
        self.frames = frames
        self.symbols = list(symbols)
        self.start = start
        self.end = end
        self.capital = capital
        self.cost_model = cost_model
        self.benchmark = benchmark
        self.sample = max(int(sample), 1)

    def _run(self, frames: Dict[str, pd.DataFrame]):
        client = MarketDataClient(PrefetchedProvider(frames))
        engine = BacktestEngine(self.strategy_factory(), client, cost_model=self.cost_model)
        kwargs = {"benchmark": self.benchmark} if self.benchmark else {}
        return engine.run(self.symbols, self.start, self.end, self.capital, **kwargs)

    # --- 1. the execution clock ------------------------------------------------ #
    def execution_clock(self) -> ProbeResult:
        """For each fill, was every input to the decision available before its price?

        A fill happens at a bar's open. Everything else about that bar — where it
        closed, how much traded — becomes knowable only afterwards, so withholding it
        must not change what was decided at that bar. If it does, the decision read its
        own future, and the direction of the error is always flattering: the engine
        transacted at a price that existed before the information that justified it.

        This is the shape of the defect a feed shift cannot see. Both the close and the
        volume are withheld, separately, because they reach the decision by different
        routes — the close through the score, the volume through the trailing liquidity
        the cost model charges impact against, and so through affordability.
        """
        base = self._run(self.frames)
        instants = self._decision_instants(base)
        if not instants:
            return ProbeResult(
                "execution_clock",
                INTRA_BAR,
                None,
                "no fills to probe: the run transacted nothing, so its decision clock was never exercised",
            )

        findings = []
        for when in instants:
            for label, perturb in (("close", _perturbed_close), ("volume", _perturbed_volume)):
                # `or` is not available here: a DataFrame has no truth value. The
                # explicit None check is the point anyway - a frame with nothing to
                # withhold at this instant is left exactly as it was.
                altered = {}
                touched = False
                for symbol, frame in self.frames.items():
                    changed = perturb(frame, when)
                    altered[symbol] = frame if changed is None else changed
                    touched = touched or changed is not None
                if not touched:
                    continue
                run = self._run(altered)
                base_entries, run_entries = _entries_at(base.trades, when), _entries_at(run.trades, when)
                base_exits, run_exits = (
                    _signal_exits_at(base.trades, when),
                    _signal_exits_at(run.trades, when),
                )
                if base_entries != run_entries or base_exits != run_exits:
                    findings.append(
                        {
                            "when": str(when),
                            "withheld": label,
                            "entries_with_it": base_entries,
                            "entries_without_it": run_entries,
                            "signal_exits_with_it": base_exits,
                            "signal_exits_without_it": run_exits,
                        }
                    )

        detail = {
            "instants_probed": [str(i) for i in instants],
            "sampled": len(instants),
            "findings": findings,
        }
        if findings:
            first = findings[0]
            return ProbeResult(
                "execution_clock",
                INTRA_BAR,
                False,
                f"the decision at {first['when']} changed when that bar's {first['withheld']} was withheld, "
                f"so it was made using information that did not exist when the fill priced",
                detail,
            )
        return ProbeResult(
            "execution_clock",
            INTRA_BAR,
            True,
            f"withholding each probed bar's close and volume left that bar's decisions unchanged "
            f"({len(instants)} instant(s) probed)",
            detail,
        )

    def _decision_instants(self, base) -> List[Any]:
        """The instants worth probing: where this run actually decided something."""
        trades = base.trades
        if trades is None or trades.empty:
            return []
        stamps = list(dict.fromkeys(sorted(trades["entry_time"].tolist())))
        return stamps[: self.sample]

    # --- 2. same-bar ranking --------------------------------------------------- #
    def same_bar_ranking(self) -> ProbeResult:
        """Entry ordering must not consult the bar it transacts on.

        Distinct from the clock probe even though it perturbs the same thing. When more
        candidates signal than the book has slots, *which* of them is admitted is a
        second decision, made by ranking — and a ranking computed from the transacting
        bar's own score reintroduces the look-ahead the signal itself just lost, while
        every individual signal remains perfectly causal.

        It needs contention to be visible at all: with slots to spare, every candidate
        is admitted and the ordering never decides anything. A run that never filled
        the book reports ``None``, because nothing here has been cleared.
        """
        base = self._run(self.frames)
        contended = self._contended_instants(base)
        if not contended:
            return ProbeResult(
                "same_bar_ranking",
                INTRA_BAR,
                None,
                "no instant had more entry candidates than free slots, so ordering never chose between them",
            )

        for when in contended:
            altered = {}
            for symbol, frame in self.frames.items():
                changed = _perturbed_close(frame, when)
                altered[symbol] = frame if changed is None else changed
            run = self._run(altered)
            chose_with = [e["symbol"] for e in _entries_at(base.trades, when)]
            chose_without = [e["symbol"] for e in _entries_at(run.trades, when)]
            if chose_with != chose_without:
                return ProbeResult(
                    "same_bar_ranking",
                    INTRA_BAR,
                    False,
                    f"at {when} the book admitted {chose_with} with that bar's close and "
                    f"{chose_without} without it, so the ordering ranked on the bar it transacted on",
                    {"when": str(when), "admitted_with": chose_with, "admitted_without": chose_without},
                )
        return ProbeResult(
            "same_bar_ranking",
            INTRA_BAR,
            True,
            f"the admitted set was unchanged at {len(contended)} contended instant(s)",
            {"contended": [str(w) for w in contended]},
        )

    def _contended_instants(self, base) -> List[Any]:
        """Instants where the book was full immediately after entering.

        A proxy for contention, and an imperfect one: a bar can fill the last slot with
        no rival for it. It errs toward probing more instants rather than fewer, which
        is the right direction for something whose failure mode is claiming coverage.
        """
        trades = base.trades
        if trades is None or trades.empty:
            return []
        limits = self.strategy_factory().position_limits()
        max_positions = limits.get("max_positions") or 1
        out = []
        for when in dict.fromkeys(sorted(trades["entry_time"].tolist())):
            open_now = int(((trades["entry_time"] <= when) & (trades["exit_time"] > when)).sum())
            if open_now >= max_positions and _entries_at(trades, when):
                out.append(when)
        return out[: self.sample]

    # --- 3. benchmark alignment ------------------------------------------------ #
    def benchmark_alignment(self) -> ProbeResult:
        """The benchmark series must not be shifted relative to the strategy's.

        Stated as a causality claim rather than a correlation one, because correlation
        cannot distinguish a misaligned benchmark from an uncorrelated strategy:
        changing the benchmark's close at an instant must not move any benchmark return
        belonging to an equity step that had already finished. If it does, the pairing
        reaches backwards, and every alpha, beta and information ratio in the report is
        measured against a series the strategy could not have been running beside.
        """
        if not self.benchmark:
            return ProbeResult(
                "benchmark_alignment", INTRA_BAR, None, "no benchmark was supplied, so nothing was aligned"
            )
        base = self._run(self.frames)
        if base.benchmark_returns is None or base.equity_times is None or len(base.equity_times) < 4:
            return ProbeResult(
                "benchmark_alignment",
                INTRA_BAR,
                None,
                "the run produced no aligned benchmark series to probe",
            )
        frame = self.frames.get(self.benchmark)
        if frame is None or frame.empty:
            return ProbeResult(
                "benchmark_alignment", INTRA_BAR, None, f"no bars for the benchmark {self.benchmark!r}"
            )

        # Perturb late, so there is a long stretch of "already finished" steps in front
        # of it for a backwards-reaching pairing to disturb.
        stamps = list(base.equity_times[1:])
        pivot = stamps[max(len(stamps) * 3 // 4, 1)]
        candidates = [t for t in frame.index if t >= pivot]
        if not candidates:
            return ProbeResult(
                "benchmark_alignment", INTRA_BAR, None, "the benchmark has no bar late enough to perturb"
            )
        altered = dict(self.frames)
        altered[self.benchmark] = _replace(
            frame, candidates[0], "close", float(frame.loc[candidates[0], "close"]) * 1.5 + 1.0
        )

        run = self._run(altered)
        before, after = base.benchmark_returns, run.benchmark_returns
        if after is None or len(after) != len(before):
            return ProbeResult(
                "benchmark_alignment",
                INTRA_BAR,
                None,
                "the perturbed run produced a different-length benchmark series; nothing comparable to check",
            )

        changed = [k for k in range(len(before)) if not _close_enough(before.iloc[k], after.iloc[k])]
        early = [k for k in changed if stamps[k] < candidates[0]]
        detail = {
            "perturbed_at": str(candidates[0]),
            "steps_changed": len(changed),
            "steps_changed_before_the_perturbation": len(early),
        }
        if early:
            return ProbeResult(
                "benchmark_alignment",
                INTRA_BAR,
                False,
                f"changing the benchmark at {candidates[0]} moved {len(early)} benchmark return(s) belonging "
                f"to equity steps that had already finished — the two series are paired out of step",
                detail,
            )
        if not changed:
            return ProbeResult(
                "benchmark_alignment",
                INTRA_BAR,
                None,
                "changing the benchmark moved nothing at all, so this run's metrics do not depend on it "
                "and its alignment was not exercised",
                detail,
            )
        return ProbeResult(
            "benchmark_alignment",
            INTRA_BAR,
            True,
            f"a change at {candidates[0]} moved only benchmark returns at or after it",
            detail,
        )


def _close_enough(left: Any, right: Any) -> bool:
    """Equal, treating two absent values as equal — absent is not zero, but it is absent."""
    left_missing, right_missing = pd.isna(left), pd.isna(right)
    if left_missing or right_missing:
        return bool(left_missing and right_missing)
    return abs(float(left) - float(right)) <= 1e-12


# --------------------------------------------------------------------------- #
# 4. the as-of scanner clock
# --------------------------------------------------------------------------- #
def probe_as_of_scanner(
    scan: Callable[[Dict[str, pd.DataFrame], datetime], Sequence[str]],
    frames: Dict[str, pd.DataFrame],
    as_of: datetime,
) -> ProbeResult:
    """Universe selection must not see beyond the clock it claims to be reading.

    Selected once against everything on disk and once against a feed truncated at
    ``as_of``. A scanner honouring its own clock returns the same names either way; one
    that reads past it selects the winners of a period it was not supposed to know
    about, and a backtest over that universe is measuring hindsight — for every symbol,
    for the whole window, with nothing in the result that looks wrong.

    Takes the scan as a callable rather than reaching for a scanner registry, because
    this module sits below the layer that owns scanning and must not import upwards.
    """
    truncated = {}
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        stamp = _coerce_stamp(as_of, frame.index)
        truncated[symbol] = frame.loc[frame.index <= stamp]

    if not any(len(f) for f in truncated.values()):
        return ProbeResult(
            "as_of_scanner", AS_OF, None, f"no bars at or before {as_of}, so there was nothing to select from"
        )

    full = sorted(scan(frames, as_of))
    limited = sorted(scan(truncated, as_of))
    if not full and not limited:
        # Two empty selections agree, and agreeing about nothing is not evidence that a
        # clock was honoured. Reporting this as a pass would clear a scanner that was
        # never asked to choose.
        return ProbeResult(
            "as_of_scanner",
            AS_OF,
            None,
            "the scanner selected no names either way, so its clock was never exercised",
            {"as_of": str(as_of)},
        )
    if full != limited:
        return ProbeResult(
            "as_of_scanner",
            AS_OF,
            False,
            f"the universe changed when bars after {as_of} were withheld, so selection read past its own clock",
            {"with_future_bars": full, "without_them": limited, "as_of": str(as_of)},
        )
    return ProbeResult(
        "as_of_scanner",
        AS_OF,
        True,
        f"the same {len(full)} name(s) were selected with and without bars after {as_of}",
        {"universe": full, "as_of": str(as_of)},
    )


def _coerce_stamp(when: datetime, index) -> Any:
    """``when`` made comparable with a possibly tz-aware bar index."""
    stamp = pd.Timestamp(when)
    if getattr(index, "tz", None) is not None:
        stamp = stamp.tz_localize(index.tz) if stamp.tzinfo is None else stamp.tz_convert(index.tz)
    elif stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #
def run_causality_probes(
    strategy_factory: Callable[[], Any],
    frames: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    capital: float = 100_000.0,
    cost_model: Optional[Any] = None,
    benchmark: Optional[str] = None,
    sample: int = DEFAULT_SAMPLE,
    scan: Optional[Callable[[Dict[str, pd.DataFrame], datetime], Sequence[str]]] = None,
    scan_as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Every causality probe over one run, as one report.

    The summary refuses to collapse into a single verdict when any probe could not be
    exercised. "Three passed and one never ran" is not "passed", and a report that
    rounds it to one is the reassurance this whole subsystem exists to withhold.
    """
    probes = CausalityProbes(
        strategy_factory,
        frames,
        symbols,
        start,
        end,
        capital=capital,
        cost_model=cost_model,
        benchmark=benchmark,
        sample=sample,
    )
    results = [probes.execution_clock(), probes.same_bar_ranking(), probes.benchmark_alignment()]
    if scan is not None and scan_as_of is not None:
        results.append(probe_as_of_scanner(scan, frames, scan_as_of))
    else:
        results.append(
            ProbeResult(
                "as_of_scanner", AS_OF, None, "no scanner was used, so no selection clock was exercised"
            )
        )

    failed = [r for r in results if r.passed is False]
    unexercised = [r for r in results if r.passed is None]
    if failed:
        verdict = "non-causal"
    elif unexercised:
        verdict = "incomplete"
    else:
        verdict = "causal"
    return {
        "verdict": verdict,
        "probes": [r.as_dict() for r in results],
        "failed": [r.name for r in failed],
        "not_exercised": [r.name for r in unexercised],
        "note": (
            "These test intra-bar causality and the as-of clock. They are a different "
            "class from the feed-shift leakage probe, which tests for future data and "
            "cannot see a one-bar look-ahead at all — a shift moves signal and price "
            "together, so the relationship survives it. Neither substitutes for the other."
        ),
    }
