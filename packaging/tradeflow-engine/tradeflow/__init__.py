"""Compatibility shim: ``tradeflow`` is now ``quantred``.

Importing this package warns, then makes ``tradeflow`` and every ``tradeflow.<x>`` resolve
to the *same module objects* as ``quantred`` / ``quantred.<x>``. Identity matters: a pack
that still subclasses ``tradeflow.strategies.base.Strategy`` must pass the registry's
``issubclass`` check against ``quantred``'s ``Strategy``, so nothing is re-executed or
copied.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import quantred

warnings.warn(
    "The `tradeflow` package is now `quantred`; import from `quantred` instead. "
    "This compatibility alias comes from the final `tradeflow-engine` release and will be removed.",
    DeprecationWarning,
    stacklevel=2,
)

_OLD = "tradeflow"
_NEW = "quantred"
#: Real files in this package; the finder leaves them alone (see ``tradeflow/cli.py``).
_REAL_MODULES = frozenset({f"{_OLD}.cli"})


class _AliasLoader(importlib.abc.Loader):
    """Hands back an already-imported ``quantred`` module under its old name."""

    def __init__(self, module):
        self._module = module
        self._original_spec = module.__spec__

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        # ``module_from_spec`` overwrites ``__spec__`` with the alias spec; restore the
        # module's own so ``quantred.x`` keeps describing itself as ``quantred.x``.
        module.__spec__ = self._original_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + ".") or fullname in _REAL_MODULES:
            return None
        real = _NEW + fullname[len(_OLD) :]
        try:
            module = importlib.import_module(real)
        except ModuleNotFoundError as exc:
            # Only "there is no such quantred module" means "not ours"; a dependency
            # missing inside a real module must surface as itself.
            if exc.name and (real == exc.name or real.startswith(exc.name + ".")):
                return None
            raise
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module))


def _install() -> None:
    # By name: a reload redefines the class, and an isinstance check would then miss the
    # finder the first import installed.
    if not any(type(f).__name__ == "_AliasFinder" for f in sys.meta_path):
        # First, not appended: for ``tradeflow.strategies.base`` the parent's ``__path__`` is
        # quantred's directory, so the ordinary path finder would otherwise find
        # ``base.py`` there and execute a second copy under the old name.
        sys.meta_path.insert(0, _AliasFinder())


_install()


def __getattr__(name):
    try:
        return getattr(quantred, name)
    except AttributeError:
        pass
    try:
        return importlib.import_module(f"{_OLD}.{name}")
    except ModuleNotFoundError:
        raise AttributeError(f"module {_OLD!r} has no attribute {name!r}") from None
