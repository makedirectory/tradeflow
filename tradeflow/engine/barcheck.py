"""Bar-quality guards for the live loop.

The live path hands whatever the vendor sends straight to the strategy. A stale
repeat, a bad tick, a bar for a halted symbol, or one arriving out of order all
become signals with nothing to veto them — and unlike a backtest, there is no
opportunity to notice afterwards.

Two rules govern everything here:

**Reject, never repair.** No guard modifies a bar, fills a gap, or interpolates a
missing value. The moment the live path "fixes" its inputs it stops being the thing
the backtest validated, and every historical result quietly stops describing what
will happen. A rejected bar is skipped and logged; the strategy simply does not see
it.

**A guard that rejects real data is worse than no guard.** Every veto discards
information the strategy was entitled to act on, and a threshold tight enough to
catch every bad tick will also catch the genuine moves that make money. Defaults are
deliberately loose: these exist to catch a corrupt *tick*, not a violent *day*.

Pure functions over ``(previous accepted bar, candidate) -> BarVerdict``, with no I/O
and no state beyond the last accepted bar per symbol — so the whole thing is
testable without a broker, a network, or a clock.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarVerdict:
    """Whether a bar may reach the strategy, and why not when it may not."""

    accepted: bool
    reason: Optional[str] = None
    detail: Optional[str] = None

    def __bool__(self) -> bool:
        return self.accepted


ACCEPTED = BarVerdict(True)


@dataclass
class BarChecks:
    """Thresholds for the guards. Every check defaults on; each can be disabled.

    ``max_return`` is the only real judgment call. At 0.35 it will not fire on any
    ordinary session — a stock genuinely moving 35% between consecutive bars is
    news, and the strategy should see it. It fires on the decimal-point errors and
    crossed quotes that a feed occasionally emits.
    """

    #: Reject a single-bar return larger than this in magnitude (fraction, not %).
    max_return: float = 0.35
    #: Reject a bar whose timestamp is older than this many intervals behind now.
    max_staleness_intervals: float = 3.0
    #: Reject a zero-volume bar once a symbol has been seen trading.
    check_zero_volume: bool = True
    #: Reject a bar at or before the last accepted timestamp for that symbol.
    check_monotonic: bool = True
    #: Reject internally inconsistent OHLC (high < low, close outside the range, ...).
    check_ohlc: bool = True
    #: Reject bars that arrive too late to be actionable.
    check_staleness: bool = True
    #: Reject implausible single-bar moves.
    check_spike: bool = True


@dataclass
class _SymbolState:
    """What the guards remember about one symbol: only what they need."""

    last_timestamp: Optional[datetime] = None
    last_close: Optional[float] = None
    has_traded: bool = False


@dataclass
class BarQualityFilter:
    """Applies the guards to a live bar stream, and counts what it rejected.

    The counts matter as much as the filtering. A guard quietly discarding a third
    of the feed looks identical, from the strategy's side, to a quiet market — so
    the rejection rate is tracked and reported rather than left to be inferred.
    """

    checks: BarChecks = field(default_factory=BarChecks)
    interval: timedelta = timedelta(minutes=5)
    #: Rejections per reason, and the total seen — the visibility half of the job.
    rejected: Dict[str, int] = field(default_factory=dict)
    seen: int = 0
    _state: Dict[str, _SymbolState] = field(default_factory=dict)

    def check(self, symbol: str, bar: Dict[str, float], timestamp: datetime, *, now=None) -> BarVerdict:
        """Whether this bar may reach the strategy.

        ``now`` is injectable so the staleness check is testable without freezing the
        clock. Accepting a bar updates this symbol's state; rejecting one does not —
        a bad bar must not become the baseline the next bar is judged against.
        """
        self.seen += 1
        state = self._state.setdefault(symbol, _SymbolState())
        verdict = self._evaluate(state, bar, timestamp, now=now)

        if not verdict.accepted:
            self.rejected[verdict.reason] = self.rejected.get(verdict.reason, 0) + 1
            logger.warning(
                "Rejected %s bar at %s: %s (%s)", symbol, timestamp, verdict.reason, verdict.detail
            )
            return verdict

        state.last_timestamp = timestamp
        state.last_close = float(bar["close"])
        if float(bar.get("volume") or 0.0) > 0:
            state.has_traded = True
        return verdict

    def _evaluate(self, state: _SymbolState, bar, timestamp, *, now) -> BarVerdict:
        checks = self.checks

        if checks.check_ohlc:
            verdict = _ohlc_verdict(bar)
            if not verdict.accepted:
                return verdict

        if checks.check_monotonic and state.last_timestamp is not None:
            if _as_utc(timestamp) <= _as_utc(state.last_timestamp):
                return BarVerdict(
                    False,
                    "out_of_order",
                    f"timestamp {timestamp} is not after the last accepted {state.last_timestamp}",
                )

        if checks.check_staleness:
            # Compared like-for-like: the reference is built in the bar's own
            # awareness rather than normalized to UTC. Assuming a naive timestamp is
            # UTC would make every bar from a vendor that serializes local time look
            # hours stale, and reject a perfectly good feed outright.
            reference = now if now is not None else _now_like(timestamp)
            age = _as_utc(reference) - _as_utc(timestamp)
            limit = self.interval * checks.max_staleness_intervals
            if age > limit:
                return BarVerdict(False, "stale", f"bar is {age} old, limit {limit}")

        if checks.check_spike and state.last_close:
            move = abs(float(bar["close"]) / state.last_close - 1.0)
            if move > checks.max_return:
                return BarVerdict(
                    False,
                    "spike",
                    f"single-bar move {move:.1%} exceeds {checks.max_return:.0%} "
                    f"({state.last_close} -> {bar['close']})",
                )

        if checks.check_zero_volume and state.has_traded:
            if float(bar.get("volume") or 0.0) <= 0:
                return BarVerdict(False, "zero_volume", "no volume on a symbol that has traded")

        return ACCEPTED

    def report(self) -> Dict[str, object]:
        """Rejection counts and rate — for logging at shutdown, or on a schedule.

        ``elevated`` is the flag worth acting on: a feed that is mostly being
        discarded is a data problem wearing a quiet market's clothes.
        """
        total = sum(self.rejected.values())
        rate = (total / self.seen) if self.seen else 0.0
        return {
            "seen": self.seen,
            "rejected": total,
            "rate": rate,
            "by_reason": dict(self.rejected),
            "elevated": rate > 0.05 and total > 5,
        }


def _ohlc_verdict(bar: Dict[str, float]) -> BarVerdict:
    """Internal consistency: the cheapest check and the one that catches corruption.

    A bar failing this is not a market event; it is a broken message.
    """
    try:
        o, h, low, c = (float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]))
    except (KeyError, TypeError, ValueError):
        return BarVerdict(False, "malformed", "bar is missing an OHLC field or it is not a number")

    if any(price <= 0 for price in (o, h, low, c)):
        return BarVerdict(False, "non_positive", f"non-positive price in OHLC ({o}, {h}, {low}, {c})")
    if h < low:
        return BarVerdict(False, "inverted_range", f"high {h} below low {low}")
    if not (low <= c <= h) or not (low <= o <= h):
        return BarVerdict(False, "outside_range", f"open/close outside [{low}, {h}]")
    volume = bar.get("volume")
    if volume is not None and float(volume) < 0:
        return BarVerdict(False, "negative_volume", f"volume {volume}")
    return ACCEPTED


def _now_like(timestamp: datetime) -> datetime:
    """``now`` in the same awareness as the bar, so the two are comparable.

    Staleness is a *duration*, so both sides must mean the same thing. A vendor
    serializing local naive timestamps is compared against local now; one sending
    aware timestamps against UTC now. Normalizing a naive timestamp to UTC instead
    would make an entire well-behaved feed look hours stale.
    """
    return datetime.now() if timestamp.tzinfo is None else datetime.now(timezone.utc)


def _as_utc(timestamp: datetime) -> datetime:
    """A timestamp as tz-aware UTC, for *relative* comparison only.

    Ordering and duration checks go through this so a feed that switches awareness
    mid-stream — naive on one bar, aware on the next — produces a verdict rather
    than a ``TypeError`` from inside the order path. The assumption is only ever
    used to compare two values, never to decide what time it is.

    The bar and the timestamp handed to the strategy are untouched: this converts a
    comparison key, never an input.
    """
    return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
