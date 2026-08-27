"""External strategy/scanner discovery.

The public package carries examples; proprietary signal IP can live in a separate
distribution that exposes entry points.
"""

from importlib import metadata

from tradeflow.scanners.base import ScannerStrategy
from tradeflow.services import registry
from tradeflow.strategies.ma_crossover import MovingAverageCrossoverStrategy


class PrivateTrendStrategy(MovingAverageCrossoverStrategy):
    """Private trend strategy used by a separate package."""


class PrivateScanner(ScannerStrategy):
    """Private scanner used by a separate package."""

    PARAM_RANGES = {
        "lookback": {"type": "int", "min": 2, "max": 20, "step": 1, "default": 5},
    }

    def initialize(self):
        pass

    def process_data(self, data):
        return data.copy()

    def generate_signals_df(self, data):
        out = data[["close"]].copy()
        out["signal"] = "SCANNER_HOLD"
        out["signal_strength"] = 0.0
        return out[["signal", "signal_strength"]]


class _FakeEntryPoint:
    def __init__(self, name, group, value):
        self.name = name
        self.group = group
        self._value = value

    def load(self):
        return self._value


class _FakeEntryPoints(list):
    def select(self, *, group):
        return _FakeEntryPoints([ep for ep in self if ep.group == group])


def test_refresh_registries_loads_private_entry_points(monkeypatch):
    real_entry_points = metadata.entry_points
    entry_points = _FakeEntryPoints(
        [
            _FakeEntryPoint("private_trend", registry.STRATEGY_ENTRY_POINT_GROUP, PrivateTrendStrategy),
            _FakeEntryPoint("private_scan", registry.SCANNER_ENTRY_POINT_GROUP, PrivateScanner),
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: entry_points)

    registry.refresh_registries()
    try:
        assert registry.STRATEGIES["private_trend"] is PrivateTrendStrategy
        assert registry.SCANNERS["private_scan"] is PrivateScanner
        assert "private_scan" in registry.SymbolScanner.available()
    finally:
        monkeypatch.setattr(metadata, "entry_points", real_entry_points)
        registry.refresh_registries()


def test_entry_point_factories_can_return_multiple_strategies(monkeypatch):
    real_entry_points = metadata.entry_points

    def contribute():
        return {"private_a": PrivateTrendStrategy, "private_b": PrivateTrendStrategy}

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda: _FakeEntryPoints(
            [_FakeEntryPoint("private_pack", registry.STRATEGY_ENTRY_POINT_GROUP, contribute)]
        ),
    )

    registry.refresh_registries()
    try:
        assert {"private_a", "private_b"} <= set(registry.STRATEGIES)
    finally:
        monkeypatch.setattr(metadata, "entry_points", real_entry_points)
        registry.refresh_registries()


def test_private_entry_points_cannot_override_builtins(monkeypatch):
    real_entry_points = metadata.entry_points
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda: _FakeEntryPoints(
            [_FakeEntryPoint("ma_crossover", registry.STRATEGY_ENTRY_POINT_GROUP, PrivateTrendStrategy)]
        ),
    )

    registry.refresh_registries()
    try:
        assert registry.STRATEGIES["ma_crossover"] is MovingAverageCrossoverStrategy
    finally:
        monkeypatch.setattr(metadata, "entry_points", real_entry_points)
        registry.refresh_registries()
