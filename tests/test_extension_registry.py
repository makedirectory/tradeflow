"""External strategy/scanner discovery.

The public package carries examples; proprietary signal IP can live in a separate
distribution that exposes entry points.
"""

from importlib import metadata

import pytest

from quantred.demo.strategies import DemoTrendStrategy
from quantred.scanners.base import ScannerStrategy
from quantred.scanners.symbol_scanner import BUILTIN_SCANNERS as package_builtin_scanners
from quantred.scanners.symbol_scanner import SymbolScanner
from quantred.services import registry
from quantred.strategies.base import Strategy


class PrivateTrendStrategy(DemoTrendStrategy):
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


@pytest.mark.parametrize(
    "legacy_group, registry_name, expected",
    [
        ("tradeflow.strategies", "STRATEGIES", PrivateTrendStrategy),
        ("tradeflow.scanners", "SCANNERS", PrivateScanner),
    ],
)
def test_a_pack_still_registered_under_the_old_group_name_is_discovered(
    monkeypatch, legacy_group, registry_name, expected
):
    """A pack built before the rename declares `tradeflow.*` groups and must keep working.

    The literal group strings are spelled out on purpose: deriving them from the
    constant under test would agree with a lookup that had dropped the legacy group.
    """
    real_entry_points = metadata.entry_points
    entry_points = _FakeEntryPoints([_FakeEntryPoint("legacy_pack_entry", legacy_group, expected)])
    monkeypatch.setattr(metadata, "entry_points", lambda: entry_points)

    registry.refresh_registries()
    try:
        assert getattr(registry, registry_name)["legacy_pack_entry"] is expected
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
            [_FakeEntryPoint("demo_trend", registry.STRATEGY_ENTRY_POINT_GROUP, PrivateTrendStrategy)]
        ),
    )

    registry.refresh_registries()
    try:
        assert registry.STRATEGIES["demo_trend"] is DemoTrendStrategy
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


def _reload_registry_preserving_identity():
    """Reload the registry module, then hand its consumers back the dicts they hold.

    Reloading rebinds ``STRATEGIES``/``SCANNERS`` to fresh objects, while every module
    that did ``from ... import STRATEGIES`` - the CLI's argparse ``choices=`` among
    them - keeps a live reference to the originals. Left that way a reload silently
    freezes the CLI's menu for the rest of the session, and the test that trips over
    it is some unrelated one much later.
    """
    import importlib

    live_strategies, live_scanners = registry.STRATEGIES, registry.SCANNERS
    reloaded = importlib.reload(registry)
    reloaded.STRATEGIES = live_strategies
    reloaded.SCANNERS = live_scanners
    reloaded.refresh_registries()
    return reloaded


def test_importing_the_registry_survives_unreadable_metadata():
    """The failure that mattered: discovery runs at import, so an unguarded raise
    there is an ImportError on every command rather than a missing extension."""
    real = metadata.entry_points

    def _corrupt():
        raise RuntimeError("corrupt dist-info")

    metadata.entry_points = _corrupt
    try:
        reloaded = _reload_registry_preserving_identity()  # must not raise
        assert set(reloaded.BUILTIN_STRATEGIES) <= set(reloaded.STRATEGIES)
    finally:
        metadata.entry_points = real
        _reload_registry_preserving_identity()


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
    real = metadata.entry_points
    metadata.entry_points = lambda: _FakeEntryPoints(
        [_FakeEntryPoint("priv_scan", registry.SCANNER_ENTRY_POINT_GROUP, PrivateScanner)]
    )
    try:
        registry.refresh_registries()
        assert "priv_scan" in registry.SCANNERS  # discovered, and now on SymbolScanner too
        assert "priv_scan" in SymbolScanner.SCANNERS

        reloaded = _reload_registry_preserving_identity()
        assert "priv_scan" not in reloaded.BUILTIN_SCANNERS
        # The scanner package's own literal plus the demo the engine itself owns,
        # and nothing that arrived by discovery.
        assert set(reloaded.BUILTIN_SCANNERS) == set(package_builtin_scanners) | {"demo_volume"}
    finally:
        metadata.entry_points = real
        _reload_registry_preserving_identity()


# --- the class attribute and the registry are one answer --------------------
def test_symbol_scanner_resolves_names_without_importing_the_registry_first():
    """A bare `import symbol_scanner` has to be usable on its own.

    Nothing is defined in that module any more, so its literal is empty and the class
    attribute it seeds is too. Only `refresh_registries()` ever filled the attribute,
    which made `SymbolScanner` work or not depending on whether some *other* module
    had been imported first - with no signal either way, and
    `docs/content/engineering/scanners.md` naming it as the driver to use.

    Run in a subprocess, because every other test in this suite has already imported
    the registry and would mask exactly the ordering this is about.
    """
    import subprocess
    import sys

    probe = (
        "from quantred.scanners.symbol_scanner import SymbolScanner;"
        "names = SymbolScanner.available();"
        "assert 'demo_volume' in names, names;"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]


def test_the_scanner_class_attribute_agrees_with_the_service_registry():
    """Two dicts holding one answer: compare them, rather than testing each alone."""
    registry.refresh_registries()

    assert SymbolScanner._registry() == registry.SCANNERS


def test_the_scanner_registry_survives_discovery_failing_late(monkeypatch):
    """The fallback the import-time guard exists for.

    If `refresh_registries()` raises before its last line, the service registry keeps
    its seeded reserved names - but the class attribute was left holding whatever it
    had, which after this package's built-ins moved out is nothing. Every scan then
    failed on a name `list_scanners()` still advertised.
    """
    monkeypatch.setattr(SymbolScanner, "SCANNERS", {})

    def explode(*_args, **_kwargs):
        raise RuntimeError("a broken distribution")

    monkeypatch.setattr(registry, "_merged_registry", explode)
    try:
        registry.refresh_registries()
    except RuntimeError:
        pass

    assert "demo_volume" in SymbolScanner.available()


# --- the demo label describes the class, not its ancestry -------------------
def test_a_pack_subclassing_the_demo_is_not_itself_reported_as_demo():
    """`DEMO` read through the MRO labelled private work as shipped scaffolding.

    The private-strategies doc invites a pack to start from `DemoTrendStrategy`, so
    this is the documented path, not a corner. Reporting that pack's real strategy as
    `demo: True` to the CLI, MCP and the agent is the label's own failure mode in
    reverse.
    """
    assert registry.is_demo(DemoTrendStrategy)
    assert not registry.is_demo(PrivateTrendStrategy)


class _FakeDist:
    def __init__(self, name):
        self.name = name


class _FakeDistEntryPoint(_FakeEntryPoint):
    def __init__(self, name, group, value, dist_name):
        super().__init__(name, group, value)
        self.dist = _FakeDist(dist_name)


def test_the_previous_distribution_beside_this_one_is_not_warned_about(monkeypatch, caplog):
    """`tradeflow-engine` 2.3.0 may stay installed next to `quantred` and declares the demo
    pair under the old groups, as a *different* class object (its own copy of the package).
    That must not trip the 'built-in names cannot be overridden' warning on every command."""
    import logging

    class OldCopyOfDemo(DemoTrendStrategy):
        """Stands in for the old package's own copy of the demo class."""

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda: _FakeEntryPoints(
            [_FakeDistEntryPoint("demo_trend", "tradeflow.strategies", OldCopyOfDemo, "tradeflow-engine")]
        ),
    )
    try:
        with caplog.at_level(logging.DEBUG, logger=registry.logger.name):
            registry.refresh_registries()
        assert registry.STRATEGIES["demo_trend"] is DemoTrendStrategy
        # Checked before the restoring refresh below, which enumerates the real
        # environment: a broken pack installed there would warn, and that is not this.
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    finally:
        monkeypatch.undo()
        registry.refresh_registries()


def test_a_pack_with_a_distribution_of_its_own_is_still_discovered(monkeypatch):
    """The skip is for this project's distributions only, not for anything with a `dist`."""
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda: _FakeEntryPoints(
            [
                _FakeDistEntryPoint(
                    "pack_trend", registry.STRATEGY_ENTRY_POINT_GROUP, PrivateTrendStrategy, "some-pack"
                )
            ]
        ),
    )
    try:
        registry.refresh_registries()
        assert registry.STRATEGIES["pack_trend"] is PrivateTrendStrategy
    finally:
        monkeypatch.undo()
        registry.refresh_registries()


class _CountingEntryPoint(_FakeDistEntryPoint):
    def __init__(self, *args):
        super().__init__(*args)
        self.loads = 0

    def load(self):
        self.loads += 1
        return super().load()


def test_this_distributions_own_entry_points_go_through_the_load_path(monkeypatch):
    """The root manifest declares the demo pair so every install exercises discovery.

    Skipping every entry point of the engine's own distribution made that declaration
    decorative: the pair arrived only through the seeded built-ins. Only the previous
    distribution is skipped.
    """
    ours = _CountingEntryPoint(
        "demo_trend", registry.STRATEGY_ENTRY_POINT_GROUP, DemoTrendStrategy, "quantred"
    )
    old = _CountingEntryPoint("demo_trend", "tradeflow.strategies", DemoTrendStrategy, "tradeflow-engine")
    monkeypatch.setattr(metadata, "entry_points", lambda: _FakeEntryPoints([ours, old]))
    try:
        registry.refresh_registries()
    finally:
        monkeypatch.undo()
        registry.refresh_registries()
    assert ours.loads == 1
    assert old.loads == 0


def test_the_installed_demo_declarations_are_found_by_discovery():
    """Against the real installed metadata: the demo pair is reachable as an entry point of
    this distribution, and is not filtered out before it loads."""
    found = {ep.name for ep in registry._select_entry_points(registry.STRATEGY_ENTRY_POINT_GROUP)}
    assert "demo_trend" in found
    ours = [
        ep
        for ep in registry._select_entry_points(registry.STRATEGY_ENTRY_POINT_GROUP)
        if ep.name == "demo_trend"
    ]
    assert not any(registry._from_legacy_distribution(ep) for ep in ours)
    assert (
        registry._entry_point_classes(registry.STRATEGY_ENTRY_POINT_GROUP, registry.Strategy)["demo_trend"]
        is DemoTrendStrategy
    )


def test_a_failure_names_the_group_the_pack_declared_not_the_canonical_one(monkeypatch, caplog):
    import logging

    class Broken(_FakeDistEntryPoint):
        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda: _FakeEntryPoints([Broken("old_pack", "tradeflow.strategies", None, "some-pack")]),
    )
    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        try:
            registry.refresh_registries()
        finally:
            monkeypatch.undo()
            registry.refresh_registries()
    messages = [r.getMessage() for r in caplog.records]
    assert any("tradeflow.strategies entry point 'old_pack'" in m for m in messages)


def test_a_pack_using_the_old_import_name_is_said_once(monkeypatch, caplog):
    import logging
    import sys

    class _AliasFinder:  # the shim's finder, recognised by name
        pass

    monkeypatch.setitem(sys.modules, "tradeflow", sys)
    monkeypatch.setattr(sys, "meta_path", [_AliasFinder(), *sys.meta_path])
    monkeypatch.setattr(registry, "_WARNED_OLD_IMPORT", False)
    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        registry._warn_if_a_pack_imports_the_old_name(already_loaded=False)
        registry._warn_if_a_pack_imports_the_old_name(already_loaded=False)
    said = [r for r in caplog.records if "old `tradeflow` name" in r.getMessage()]
    assert len(said) == 1


def test_no_old_name_message_without_the_alias(monkeypatch, caplog):
    import logging
    import sys

    monkeypatch.delitem(sys.modules, "tradeflow", raising=False)
    monkeypatch.setattr(registry, "_WARNED_OLD_IMPORT", False)
    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        registry._warn_if_a_pack_imports_the_old_name(already_loaded=False)
    assert not caplog.records


class _AliasFinder:  # stands in for the shim's finder, which is recognised by name
    def find_spec(self, *args, **kwargs):
        return None


class _PackImportingTheOldName:
    """An entry point whose loading imports `tradeflow`, as a pack written against it does."""

    name = "old_import_pack"
    group = "quantred.strategies"
    dist = None

    def load(self):
        import sys

        sys.modules.setdefault("tradeflow", sys)
        return {}


def _discover_with(monkeypatch, caplog, old_name_loaded_first):
    import logging
    import sys

    if old_name_loaded_first:
        monkeypatch.setitem(sys.modules, "tradeflow", sys)
    else:
        monkeypatch.delitem(sys.modules, "tradeflow", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_AliasFinder(), *sys.meta_path])
    monkeypatch.setattr(registry, "_WARNED_OLD_IMPORT", False)
    monkeypatch.setattr(registry, "_select_entry_points", lambda group: [_PackImportingTheOldName()])
    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        registry._entry_point_classes("quantred.strategies", Strategy)
    return [r for r in caplog.records if "old `tradeflow` name" in r.getMessage()]


def test_a_pack_that_imports_the_old_name_while_loading_is_said(monkeypatch, caplog):
    assert len(_discover_with(monkeypatch, caplog, old_name_loaded_first=False)) == 1


def test_launching_through_the_old_name_does_not_blame_a_pack(monkeypatch, caplog):
    """`python -m tradeflow.cli` imports the old name before discovery runs. The check
    read that as a pack using it and printed a false line on every such launch - the form
    MCP client configs written before the rename use."""
    assert _discover_with(monkeypatch, caplog, old_name_loaded_first=True) == []
