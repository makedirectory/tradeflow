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
from tradeflow.utils.timeutils import parse_market_date, shift_days


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
    bounds = [b for f in folds for b in (f.is_start, f.is_end, f.oos_start, f.oos_end)] + list(holdout)
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
