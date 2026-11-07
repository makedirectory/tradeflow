"""Window boundaries stay at New York midnight across daylight saving.

``aware + timedelta`` keeps a pytz value's January offset, so from March to November
every walk-forward fold, holdout, warmup and lookback boundary sat at 01:00 New York.
Daily bars are stamped at New York midnight, so a window starting on a summer day
skipped that day's bar: each out-of-sample fold began one bar late.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tests.fakes import FakeMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.optimization.walk_forward import PrefetchedProvider, WalkForwardValidator
from tradeflow.services.registry import resolve_strategy_class
from tradeflow.utils.timeutils import NEW_YORK, parse_market_date, shift_days


def _ny_wall(moment):
    return pd.Timestamp(moment).tz_convert("America/New_York")


@pytest.mark.parametrize(
    "start, days",
    [("2024-01-02", 150), ("2024-06-01", -150), ("2024-03-01", 15), ("2024-11-01", 5)],
)
def test_shifting_a_market_date_keeps_it_at_new_york_midnight(start, days):
    moment = parse_market_date(start)
    for shifted in (shift_days(moment, days), shift_days(pd.Timestamp(moment), days)):
        wall = _ny_wall(shifted)
        assert (wall.hour, wall.minute) == (0, 0), shifted
        assert wall.date() == (moment + timedelta(days=days)).date()


def test_the_other_kinds_of_datetime_shift_as_before():
    naive = datetime(2024, 1, 2)
    assert shift_days(naive, 3) == datetime(2024, 1, 5)
    utc = datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert shift_days(utc, 150) == utc + timedelta(days=150)
    zoned = datetime(2024, 1, 2, tzinfo=ZoneInfo("America/New_York"))
    assert shift_days(zoned, 150).hour == 0


def test_every_walk_forward_boundary_is_new_york_midnight_across_dst():
    validator = WalkForwardValidator(
        resolve_strategy_class("demo_trend"), MarketDataClient(FakeMarketData(["AAA"], n=50, freq="1D"))
    )
    folds, holdout = validator.build_folds(
        parse_market_date("2024-01-02"), parse_market_date("2025-01-02"), n_folds=4, holdout_days=60
    )
    # An out-of-sample window is half-open: it ends just before the next boundary.
    bounds = [
        b for f in folds for b in (f.is_start, f.is_end, f.oos_start, f.oos_end + timedelta(microseconds=1))
    ]
    bounds += list(holdout)
    summer = [b for b in bounds if _ny_wall(b).dst()]
    assert summer, "the window must cross daylight saving for this test to mean anything"
    for bound in bounds:
        assert (_ny_wall(bound).hour, _ny_wall(bound).minute) == (0, 0), bound


def test_a_summer_out_of_sample_fold_includes_its_first_days_bar():
    """The failure itself: a fold starting on a summer day returned bars from the next
    day, because the boundary was 01:00 and the bar was stamped at midnight."""
    validator = WalkForwardValidator(
        resolve_strategy_class("demo_trend"), MarketDataClient(FakeMarketData(["AAA"], n=50, freq="1D"))
    )
    folds, _ = validator.build_folds(
        parse_market_date("2024-01-02"), parse_market_date("2025-01-02"), n_folds=4
    )
    fold = next(f for f in folds if _ny_wall(f.oos_start).dst())
    stamps = pd.date_range("2024-01-01", "2025-01-31", freq="D", tz="America/New_York").tz_convert("UTC")
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=stamps)
    bars = PrefetchedProvider({"AAA": frame}).get_bars(["AAA"], "1Day", fold.oos_start, fold.oos_end)["AAA"]
    assert _ny_wall(bars.index[0]).date() == _ny_wall(fold.oos_start).date()


def test_a_transition_hour_wall_time_shifts_the_same_way_for_both_kinds():
    """Found in review: the pandas branch raised on a wall time that does not exist (or
    occurs twice) after the shift, where the datetime branch resolved it silently."""
    for wall in (datetime(2024, 3, 11, 2, 30), datetime(2024, 11, 4, 1, 30)):
        as_datetime = NEW_YORK.localize(wall)
        as_timestamp = pd.Timestamp(wall).tz_localize("America/New_York")
        a, b = shift_days(as_datetime, -1), shift_days(as_timestamp, -1)
        assert pd.Timestamp(a) == b, (a, b)


def test_consecutive_out_of_sample_folds_share_no_bar():
    """Found in review: each fold's window ended exactly where the next began, and windows
    are sliced inclusive at both ends, so the boundary bar's return counted in two folds."""
    validator = WalkForwardValidator(
        resolve_strategy_class("demo_trend"), MarketDataClient(FakeMarketData(["AAA"], n=50, freq="1D"))
    )
    folds, _ = validator.build_folds(
        parse_market_date("2024-01-02"), parse_market_date("2025-01-02"), n_folds=4
    )
    stamps = pd.date_range("2023-12-01", "2025-02-01", freq="D", tz="America/New_York").tz_convert("UTC")
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=stamps)
    provider = PrefetchedProvider({"AAA": frame})
    windows = [set(provider.get_bars(["AAA"], "1Day", f.oos_start, f.oos_end)["AAA"].index) for f in folds]
    for earlier, later in zip(windows, windows[1:]):
        assert not earlier & later


#: Day arithmetic left on purpose, each with why it cannot move a measured window.
_ALLOWED_DAY_ARITHMETIC = {
    (
        "tradeflow/services/analysis.py",
        "start = end - timedelta(days=lookback_days)",
    ): "naive now(): bar summary",
    (
        "tradeflow/services/analysis.py",
        "as_of_naive - timedelta(days=decay_lookback_days)",
    ): "explicitly naive",
    ("tradeflow/services/setup.py", "start = end - timedelta(days=7)"): "naive now(): credential probe",
    (
        "tradeflow/analytics/performance.py",
        "first_day = equity.index[0] - pd.Timedelta(days=1)",
    ): "curve anchor",
    ("tradeflow/store/bars.py", "finish.date() + timedelta(days=1)"): "date arithmetic, no zone",
    ("tradeflow/store/trials.py", "local.date() + timedelta(days=1)"): "date arithmetic, then localized",
    ("tradeflow/cli.py", "start = end - timedelta(days=365)"): "naive now(): first cache warm",
    ("tradeflow/cli.py", "datetime.now() - timedelta(days=2)"): "naive now(): demo default",
}


def test_no_new_research_clock_window_is_built_with_aware_day_arithmetic():
    """A static guard for the call sites the tests above do not reach: a new `+/-
    timedelta(days=...)` in research-clock code is either `shift_days` or listed here
    with why it cannot move a measured window."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"[+-]\s*(?:pd\.)?[tT]imedelta\(days=")
    found = []
    for folder in ("optimization", "research", "scanners", "data", "services", "analytics", "store"):
        found += sorted((root / "tradeflow" / folder).rglob("*.py"))
    found.append(root / "tradeflow" / "cli.py")
    unexpected = []
    for path in found:
        rel = path.relative_to(root).as_posix()
        for line in path.read_text().splitlines():
            if pattern.search(line) and not any(
                rel == file and snippet in line for (file, snippet) in _ALLOWED_DAY_ARITHMETIC
            ):
                unexpected.append(f"{rel}: {line.strip()}")
    assert unexpected == [], "use utils.timeutils.shift_days:\n" + "\n".join(unexpected)
