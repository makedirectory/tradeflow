"""External strategy/scanner discovery.

The public package carries examples; proprietary signal IP can live in a separate
distribution that exposes entry points.
"""

from importlib import metadata

from tradeflow.scanners.base import ScannerStrategy
from tradeflow.scanners.symbol_scanner import BUILTIN_SCANNERS as package_builtin_scanners
from tradeflow.scanners.symbol_scanner import SymbolScanner
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


# --- discovery degrades; it never brings the engine down ---------------------
class _NotAStrategy:
    """A contribution that will not validate."""


def test_one_unusable_class_does_not_take_its_siblings_with_it():
    """A pack contributing three strategies, one of them broken.

    The try wrapped the whole contribution loop, so an invalid class aborted it:
    siblings already added were kept, siblings not yet reached were silently dropped,
    and which of them survived came down to dict ordering. The warning named the
    entry point, so nothing said that `priv_c` had gone missing at all.
    """
    real = metadata.entry_points
    contributions = {
        "priv_a": PrivateTrendStrategy,
        "priv_bad": _NotAStrategy,
        "priv_c": PrivateTrendStrategy,
    }
    metadata.entry_points = lambda: _FakeEntryPoints(
        [_FakeEntryPoint("pack", registry.STRATEGY_ENTRY_POINT_GROUP, lambda: contributions)]
    )
    try:
        registry.refresh_registries()
        assert registry.STRATEGIES.get("priv_a") is PrivateTrendStrategy
        assert registry.STRATEGIES.get("priv_c") is PrivateTrendStrategy  # used to vanish
        assert "priv_bad" not in registry.STRATEGIES
    finally:
        metadata.entry_points = real
        registry.refresh_registries()


def test_unreadable_installed_metadata_leaves_the_builtins_standing():
    """Enumeration was the one step of discovery outside a try.

    Malformed metadata from any installed distribution propagated out of a call that
    had already cleared the registries — so the engine was not merely undiscovered,
    it was empty, and the exception took down every command that imports this module.
    """
    real = metadata.entry_points

    def _corrupt():
        raise RuntimeError("corrupt dist-info")

    metadata.entry_points = _corrupt
    try:
        registry.refresh_registries()  # must not raise
        assert set(registry.BUILTIN_STRATEGIES) <= set(registry.STRATEGIES)
        assert set(registry.BUILTIN_SCANNERS) <= set(registry.SCANNERS)
    finally:
        metadata.entry_points = real
        registry.refresh_registries()


def test_importing_the_registry_survives_unreadable_metadata():
    """The failure that mattered: discovery runs at import, so an unguarded raise
    there is an ImportError on every command rather than a missing extension."""
    import importlib

    real = metadata.entry_points

    def _corrupt():
        raise RuntimeError("corrupt dist-info")

    metadata.entry_points = _corrupt
    try:
        reloaded = importlib.reload(registry)  # must not raise
        assert set(reloaded.BUILTIN_STRATEGIES) <= set(reloaded.STRATEGIES)
    finally:
        metadata.entry_points = real
        importlib.reload(registry)


def test_a_pack_that_reads_the_registry_while_importing_sees_the_builtins():
    """Discovery imports third-party modules, and clearing first published an empty
    registry for exactly that window. A pack calling `resolve_strategy_class` at
    import time raised, and was then dropped as a broken contribution — so the
    failure looked like a bad pack rather than a bad moment to ask."""
    real = metadata.entry_points
    seen = {}

    def contribute():
        seen["strategies"] = dict(registry.STRATEGIES)
        seen["scanners"] = dict(registry.SCANNERS)
        return {"priv_a": PrivateTrendStrategy}

    metadata.entry_points = lambda: _FakeEntryPoints(
        [_FakeEntryPoint("pack", registry.STRATEGY_ENTRY_POINT_GROUP, contribute)]
    )
    try:
        registry.refresh_registries()
        assert set(registry.BUILTIN_STRATEGIES) <= set(seen["strategies"])
        assert set(registry.BUILTIN_SCANNERS) <= set(seen["scanners"])
        assert registry.STRATEGIES["priv_a"] is PrivateTrendStrategy
    finally:
        metadata.entry_points = real
        registry.refresh_registries()


def test_a_reload_does_not_promote_an_installed_scanner_into_the_builtins():
    """The reserved names have to come from this package's own literal.

    `BUILTIN_SCANNERS` was derived from `SymbolScanner.SCANNERS`, which discovery
    overwrites with built-ins *plus* installed contributions. Re-executing the module
    then captured a third-party scanner as a built-in — and the next refresh rejected
    that pack's own contribution as one that "cannot override a built-in name". The
    reservation the extension design rests on was poisoning itself.
    """
    import importlib

    real = metadata.entry_points
    metadata.entry_points = lambda: _FakeEntryPoints(
        [_FakeEntryPoint("priv_scan", registry.SCANNER_ENTRY_POINT_GROUP, PrivateScanner)]
    )
    try:
        registry.refresh_registries()
        assert "priv_scan" in registry.SCANNERS  # discovered, and now on SymbolScanner too
        assert "priv_scan" in SymbolScanner.SCANNERS

        reloaded = importlib.reload(registry)
        assert "priv_scan" not in reloaded.BUILTIN_SCANNERS
        assert set(reloaded.BUILTIN_SCANNERS) == set(package_builtin_scanners)
    finally:
        metadata.entry_points = real
        importlib.reload(registry)
