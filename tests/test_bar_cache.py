"""Tests for the persistent bar cache: `BarCoverage` gap computation
and `CachedMarketData`'s gap-fill/offline/refresh behavior.

Skipped automatically if the optional `store` extra (pyarrow) isn't installed,
same as tests/test_store.py which this builds on.
"""

import pytest

pytest.importorskip("pyarrow")

from datetime import datetime, timedelta, timezone  # noqa: E402

import pandas as pd  # noqa: E402

from tests.fakes import make_ohlcv  # noqa: E402
from tradeflow.data.store import ParquetBarStore  # noqa: E402
from tradeflow.marketdata.base import MarketDataProvider  # noqa: E402
from tradeflow.marketdata.timeframe import Timeframe  # noqa: E402
from tradeflow.store.bars import BarCoverage, CachedMarketData, CacheMiss  # noqa: E402

SYMBOLS = ["AAA", "BBB"]


class WindowedFakeProvider(MarketDataProvider):
    """Unlike tests/fakes.py::FakeMarketData, this respects the requested
    [start, end] window and counts calls per symbol - what gap-computation tests
    need to assert exactly which sub-ranges were fetched."""

    def __init__(self, symbols, n=400, freq="1D"):
        self._data = {s: make_ohlcv(n=n, seed=i, freq=freq) for i, s in enumerate(symbols)}
        self.calls = []  # list of (symbol, start, end)

    def get_bars(self, symbols, timeframe, start, end):
        assert isinstance(timeframe, Timeframe)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        out = {}
        for symbol in symbols:
            self.calls.append((symbol, start, end))
            frame = self._data.get(symbol)
            if frame is None:
                continue
            idx = frame.index
            lo = start_ts.tz_localize(idx.tz) if start_ts.tzinfo is None else start_ts.tz_convert(idx.tz)
            hi = end_ts.tz_localize(idx.tz) if end_ts.tzinfo is None else end_ts.tz_convert(idx.tz)
            sliced = frame.loc[(frame.index >= lo) & (frame.index <= hi)]
            if not sliced.empty:
                out[symbol] = sliced
        return out

    async def stream_bars(self, symbols, handler):  # pragma: no cover - unused
        raise NotImplementedError

    def supports_streaming(self) -> bool:
        return False


def _cache(tmp_path, upstream=None, offline=False):
    upstream = upstream or WindowedFakeProvider(SYMBOLS)
    cached = CachedMarketData(upstream, cache_dir=tmp_path / "bars", offline=offline)
    return cached, upstream


def _full_window(upstream, symbol="AAA"):
    idx = upstream._data[symbol].index
    return idx[0].to_pydatetime(), idx[-1].to_pydatetime()


# --- MarketDataProvider contract --------------------------------------------
def test_is_a_market_data_provider(tmp_path):
    cached, _ = _cache(tmp_path)
    assert isinstance(cached, MarketDataProvider)


# --- cold/warm fetch ----------------------------------------------------------
def test_cold_fetch_hits_upstream_and_warm_fetch_does_not(tmp_path):
    cached, upstream = _cache(tmp_path)
    start, end = _full_window(upstream)

    first = cached.get_bars(["AAA"], "1Day", start, end)
    assert len(upstream.calls) == 1
    assert list(first["AAA"].columns) == ["open", "high", "low", "close", "volume"]

    upstream.calls.clear()
    second = cached.get_bars(["AAA"], "1Day", start, end)
    assert upstream.calls == []  # fully warm - zero upstream calls
    assert len(second["AAA"]) == len(first["AAA"])
    assert second["AAA"]["close"].to_numpy() == pytest.approx(first["AAA"]["close"].to_numpy())


def test_round_trip_fidelity(tmp_path):
    cached, upstream = _cache(tmp_path)
    start, end = _full_window(upstream)
    got = cached.get_bars(["AAA"], "1Day", start, end)["AAA"]
    original = upstream._data["AAA"]
    original = original.loc[(original.index >= pd.Timestamp(start)) & (original.index <= pd.Timestamp(end))]
    assert len(got) == len(original)
    assert got["close"].to_numpy() == pytest.approx(original["close"].to_numpy())
    assert got.index.tz is not None


# --- gap computation (hidden factor 2, the crux) ------------------------------
def test_partial_range_gap_fetches_only_the_missing_subranges(tmp_path):
    cached, upstream = _cache(tmp_path)
    idx = upstream._data["AAA"].index
    a, b, c, d = idx[0], idx[100], idx[200], idx[-1]

    cached.get_bars(["AAA"], "1Day", b.to_pydatetime(), c.to_pydatetime())
    assert len(upstream.calls) == 1

    upstream.calls.clear()
    merged = cached.get_bars(["AAA"], "1Day", a.to_pydatetime(), d.to_pydatetime())["AAA"]

    # Exactly two upstream calls: [a, b) and (c, d] - never a full [a, d] refetch.
    assert len(upstream.calls) == 2
    ranges = sorted((pd.Timestamp(s), pd.Timestamp(e)) for _, s, e in upstream.calls)
    assert ranges[0][0] == a
    assert ranges[0][1] <= b
    assert ranges[1][0] >= c
    assert ranges[1][1] == d

    uncached_full = WindowedFakeProvider(SYMBOLS)
    expected = uncached_full._data["AAA"]
    expected = expected.loc[(expected.index >= a) & (expected.index <= d)]
    assert len(merged) == len(expected)
    assert merged["close"].to_numpy() == pytest.approx(expected["close"].to_numpy())


# --- offline mode --------------------------------------------------------------
def test_offline_raises_cache_miss_on_empty_cache(tmp_path):
    cached, upstream = _cache(tmp_path, offline=True)
    start, end = _full_window(upstream)
    with pytest.raises(CacheMiss):
        cached.get_bars(["AAA"], "1Day", start, end)
    assert upstream.calls == []


@pytest.mark.parametrize(
    "checkout, expected, absent",
    [
        (False, "`quantred cache warm --symbols AAA", "main.py"),
        (True, "`python main.py cache warm --symbols AAA", "quantred cache"),
    ],
)
def test_the_miss_names_the_warm_command_of_the_copy_that_is_running(
    tmp_path, monkeypatch, checkout, expected, absent
):
    """An installed copy has no main.py, so telling its user to run it is a dead end."""
    from tradeflow import settings

    monkeypatch.setattr(settings, "running_from_checkout", lambda: checkout)
    cached, upstream = _cache(tmp_path, offline=True)
    start, end = _full_window(upstream)
    with pytest.raises(CacheMiss) as miss:
        cached.get_bars(["AAA"], "1Day", start, end)
    assert expected in str(miss.value)
    assert absent not in str(miss.value)


def test_one_miss_names_every_missing_symbol_and_a_warm_command_that_fixes_it(tmp_path):
    """The miss used to stop at the first symbol and print `cache warm --symbols AAA` with
    no window: running it fixed one name, the rerun failed on the next, and the default
    warm window need not cover the request at all. Running the printed command has to be
    enough for the same offline request to succeed."""
    import re

    from tradeflow.utils.timeutils import parse_market_date

    cached, upstream = _cache(tmp_path, offline=True)
    start, end = _full_window(upstream)
    with pytest.raises(CacheMiss) as miss:
        cached.get_bars(SYMBOLS, "1Day", start, end)
    message = str(miss.value)
    assert message.count("cache warm") == 1
    command = re.search(r"`[^`]*cache warm ([^`]*)`", message).group(1)
    flags = dict(re.findall(r"--(\w+) (\S+)", command))
    assert flags["symbols"].split(",") == SYMBOLS

    online = CachedMarketData(WindowedFakeProvider(SYMBOLS), cache_dir=tmp_path / "bars")
    online.warm(
        flags["symbols"].split(","),
        flags["timeframe"],
        parse_market_date(flags["start"]),
        parse_market_date(flags["end"]),
    )
    online.close()
    again = CachedMarketData(WindowedFakeProvider(SYMBOLS), cache_dir=tmp_path / "bars", offline=True)
    assert set(again.get_bars(SYMBOLS, "1Day", start, end)) == set(SYMBOLS)


def test_the_vintage_stamp_miss_also_names_every_symbol(tmp_path):
    """Found in review: only get_bars checked every symbol first. Backtest, optimize and
    walkforward stamp the vintage before reading bars, and that path still stopped at the
    first symbol — the defect the consolidated miss was meant to remove."""
    cached, upstream = _cache(tmp_path, offline=True)
    start, end = _full_window(upstream)
    with pytest.raises(CacheMiss) as miss:
        cached.vintage_stamp(SYMBOLS, "1Day", start, end)
    assert f"--symbols {','.join(SYMBOLS)} " in str(miss.value)


def test_a_miss_in_a_custom_cache_directory_warms_that_directory(tmp_path):
    """Found in review: the warm command dropped --cache-dir, so a run against its own
    cache directory was told to fill the default one and missed again on the rerun."""
    custom = tmp_path / "elsewhere bars"
    cached = CachedMarketData(WindowedFakeProvider(SYMBOLS), cache_dir=custom, offline=True)
    start, end = _full_window(cached.upstream)
    with pytest.raises(CacheMiss) as miss:
        cached.get_bars(["AAA"], "1Day", start, end)
    assert f"--cache-dir '{custom}'" in str(miss.value)


def test_a_miss_in_the_default_cache_directory_does_not_name_it(tmp_path, monkeypatch):
    """The other direction: the default needs no flag, and naming the path would put a
    machine-specific directory into a command meant to be copied."""
    from tradeflow.store import bars

    monkeypatch.setattr(bars, "DEFAULT_CACHE_ROOT", tmp_path / "bars")
    cached, upstream = _cache(tmp_path, offline=True)
    start, end = _full_window(upstream)
    with pytest.raises(CacheMiss) as miss:
        cached.get_bars(["AAA"], "1Day", start, end)
    assert "--cache-dir" not in str(miss.value)


def test_the_date_note_appears_only_where_a_cache_was_warmed_before(tmp_path):
    """The note explains coverage recorded under the old date reading. A cold cache has
    none, yet the note was printed beside every miss — six times in one verdict."""
    from tradeflow.store.bars import DATE_CONTRACT_NOTICE

    cached, upstream = _cache(tmp_path, offline=True)
    start, end = _full_window(upstream)
    with pytest.raises(CacheMiss) as cold:
        cached.get_bars(["AAA"], "1Day", start, end)
    assert DATE_CONTRACT_NOTICE not in str(cold.value)

    online, _ = _cache(tmp_path)
    online.get_bars(["AAA"], "1Day", start, start + (end - start) / 2)
    online.close()
    partly, _ = _cache(tmp_path, offline=True)
    with pytest.raises(CacheMiss) as short:
        partly.get_bars(["AAA"], "1Day", start, end)
    assert DATE_CONTRACT_NOTICE in str(short.value)


def test_offline_serves_warm_cache_with_zero_upstream_calls(tmp_path):
    warm_upstream = WindowedFakeProvider(SYMBOLS)
    warm_cache = CachedMarketData(warm_upstream, cache_dir=tmp_path / "bars")
    start, end = _full_window(warm_upstream)
    online_result = warm_cache.get_bars(["AAA"], "1Day", start, end)["AAA"]

    offline_upstream = WindowedFakeProvider(SYMBOLS)  # would answer differently if ever called
    offline_cache = CachedMarketData(offline_upstream, cache_dir=tmp_path / "bars", offline=True)
    offline_result = offline_cache.get_bars(["AAA"], "1Day", start, end)["AAA"]

    assert offline_upstream.calls == []
    assert len(offline_result) == len(online_result)
    assert offline_result["close"].to_numpy() == pytest.approx(online_result["close"].to_numpy())


# --- read-merge-write regression (ParquetBarStore.write() full-replace hazard) -
def test_warming_a_second_disjoint_range_preserves_the_first(tmp_path):
    cached, upstream = _cache(tmp_path, upstream=WindowedFakeProvider(SYMBOLS, n=600, freq="1D"))
    idx = upstream._data["AAA"].index
    early = (idx[0].to_pydatetime(), idx[100].to_pydatetime())
    late = (idx[400].to_pydatetime(), idx[599].to_pydatetime())

    cached.get_bars(["AAA"], "1Day", *early)
    cached.get_bars(["AAA"], "1Day", *late)

    still_early = cached.get_bars(["AAA"], "1Day", *early)["AAA"]
    still_late = cached.get_bars(["AAA"], "1Day", *late)["AAA"]
    assert len(still_early) > 50
    assert len(still_late) > 50


# --- vintage stamp --------------------------------------------------------------
def test_vintage_stamp_stable_when_warm_changes_after_refresh(tmp_path):
    cached, upstream = _cache(tmp_path)
    start, end = _full_window(upstream)

    v1 = cached.vintage_stamp(["AAA"], "1Day", start, end)
    v2 = cached.vintage_stamp(["AAA"], "1Day", start, end)
    assert v1 is not None
    assert v1 == v2  # no new fetch happened - same stamp

    cached.refresh(["AAA"], "1Day", start, end)
    v3 = cached.vintage_stamp(["AAA"], "1Day", start, end)
    assert v3 != v1  # refresh forced a real re-fetch - the stamp must move


def test_vintage_stamp_none_before_any_fetch(tmp_path):
    coverage = BarCoverage(tmp_path / "coverage.db")
    assert coverage.vintage(["AAA"], "1day", datetime(2024, 1, 1), datetime(2024, 6, 1)) is None


# --- BarCoverage.gaps unit behavior ---------------------------------------------
def _day(n: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=n)


def test_gaps_merges_touching_intervals(tmp_path):
    coverage = BarCoverage(tmp_path / "coverage.db")
    coverage.record_fetch("AAA", "1day", _day(0), _day(10), "split")
    coverage.record_fetch("AAA", "1day", _day(10), _day(20), "split")
    assert coverage.gaps("AAA", "1day", _day(0), _day(20)) == []
    assert coverage.covered_intervals("AAA", "1day") == [(pd.Timestamp(_day(0)), pd.Timestamp(_day(20)))]


def test_gaps_empty_range_and_no_coverage(tmp_path):
    coverage = BarCoverage(tmp_path / "coverage.db")
    assert coverage.gaps("AAA", "1day", _day(0), _day(10)) == [
        (pd.Timestamp(_day(0)), pd.Timestamp(_day(10)))
    ]
    assert coverage.gaps("AAA", "1day", _day(5), _day(5)) == []


# --- refresh (corporate-action lever) -------------------------------------------
def test_refresh_clears_and_refetches(tmp_path):
    cached, upstream = _cache(tmp_path)
    start, end = _full_window(upstream)
    cached.get_bars(["AAA"], "1Day", start, end)
    assert len(upstream.calls) == 1

    upstream.calls.clear()
    summary = cached.refresh(["AAA"], "1Day", start, end)
    assert summary["AAA"]["refreshed"] is True
    assert len(upstream.calls) == 1  # re-fetched, not served from the now-cleared cache

    result = cached.get_bars(["AAA"], "1Day", start, end)["AAA"]
    assert not result.empty


def test_refresh_without_prior_coverage_or_window_is_a_documented_no_op(tmp_path):
    cached, upstream = _cache(tmp_path)
    summary = cached.refresh(["ZZZ"], "1Day")
    assert summary["ZZZ"]["refreshed"] is False
    assert upstream.calls == []


# --- derived-store precedent: rebuild + drift ------------------------------------
def test_coverage_rebuild_is_idempotent(tmp_path):
    store = ParquetBarStore(tmp_path / "bars")
    bars = {"AAA": make_ohlcv(n=300, seed=0, freq="1D")}
    store.write(bars, timeframe="1Day")

    coverage = BarCoverage(tmp_path / "coverage.db")
    stats1 = coverage.rebuild(store, ["AAA"], "1day", "split")
    assert stats1 == {"symbols": 1}
    gaps_after_first = coverage.gaps("AAA", "1day", bars["AAA"].index[0], bars["AAA"].index[-1])

    stats2 = coverage.rebuild(store, ["AAA"], "1day", "split")
    assert stats2 == stats1
    gaps_after_second = coverage.gaps("AAA", "1day", bars["AAA"].index[0], bars["AAA"].index[-1])
    assert gaps_after_first == gaps_after_second == []


def test_status_flags_drift_when_parquet_deleted_but_coverage_remains(tmp_path):
    cached, upstream = _cache(tmp_path)
    start, end = _full_window(upstream)
    cached.get_bars(["AAA"], "1Day", start, end)

    status_before = cached.status()
    assert status_before["drift"] == []

    cached.store.delete_symbol("AAA", "1Day")
    status_after = cached.status()
    assert any("AAA" in d for d in status_after["drift"])


# --- warm() summary ---------------------------------------------------------------
def test_warm_reports_already_cached_on_second_call(tmp_path):
    cached, upstream = _cache(tmp_path)
    start, end = _full_window(upstream)
    first = cached.warm(["AAA"], "1Day", start, end)
    assert first["AAA"]["already_cached"] is False

    second = cached.warm(["AAA"], "1Day", start, end)
    assert second["AAA"]["already_cached"] is True
