"""Validation sandbox for proposals - the rules that keep an eager AI from
manufacturing overfit garbage at scale (its words would be more enthusiastic).

Two jobs:

1. **Research-hygiene gates** on every proposal: a rationale must be present, and
   no more than :data:`MAX_TUNABLE_PARAMS` parameters may be varied. These are the
   cheap, load-bearing rules that stop the loop manufacturing overfit garbage at
   scale.
2. **Contract validation** of agent-authored strategy *code*: it must define a
   concrete :class:`~tradeflow.strategies.base.Strategy` subclass that implements the
   abstract hooks, declares a valid ``PARAM_RANGES`` (<= 5 searchable), carries a
   docstring (the hypothesis), and actually constructs.

Isolation note: generated code is validated and executed with a restricted global
namespace (no ``os``/``sys``/network imports, limited builtins). This is a
*proposal artifact*, never auto-merged. True OS-level isolation (separate process,
no network, resource limits) is the production hardening called for in  and is
left as a deliberate follow-up; :func:`load_strategy_from_code` is the single
choke point where that would be enforced.
"""

import builtins as _builtins
from typing import Optional, Tuple, Type

from tradeflow.optimization.param_space import ParameterSpace
from tradeflow.strategies.base import Strategy

#: Hard cap on tunable parameters per strategy (more knobs = more overfit surface).
MAX_TUNABLE_PARAMS = 5

#: Config keys the execution path reads off every strategy, regardless of mechanism:
#: ``calculate_position_size`` needs the first two, and the backtest engine and live
#: trader both read ``take_profit`` when opening a position. A generated strategy that
#: omits them passes every other check and then raises on the first bar - producing a
#: zero-trade run that is indistinguishable from "no edge" unless we catch it here.
REQUIRED_CONFIG_KEYS = ("risk_per_trade", "stop_loss", "take_profit")

#: Import names a generated strategy is allowed to use.
_ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "math",
    "tradeflow.indicators.indicators",
    "tradeflow.indicators",
    "tradeflow.strategies.base",
    "tradeflow.strategies.signals",
    "tradeflow.strategies",
}

#: Builtins denied to generated code (no filesystem / process / eval surface).
_DENIED_BUILTINS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "memoryview",
}


class HygieneError(ValueError):
    """A proposal violated a research-hygiene rule."""


def validate_hygiene(proposal, strategy_cls: Optional[Type[Strategy]] = None) -> Tuple[bool, str]:
    """Check a proposal against the hygiene rules; return ``(ok, reason)``.

    Never raises - the caller journals the reason and skips the proposal.
    """
    if not (proposal.hypothesis or "").strip():
        return False, "missing hypothesis (rationale required before evaluation)"

    if proposal.kind == "tune":
        if strategy_cls is None:
            return False, "tune proposal without a known strategy class"
        tuned = proposal.tuned_params or list(proposal.params)
        if len(tuned) > MAX_TUNABLE_PARAMS:
            return False, f"{len(tuned)} tuned params exceeds the cap of {MAX_TUNABLE_PARAMS}"
        ranges = strategy_cls.PARAM_RANGES
        for name, value in proposal.params.items():
            if name not in ranges:
                return False, f"unknown parameter {name!r}"
            spec = ranges[name]
            if "min" in spec and "max" in spec and not (spec["min"] <= value <= spec["max"]):
                return False, f"{name}={value} outside [{spec['min']}, {spec['max']}]"
        return True, "ok"

    if proposal.kind == "code":
        if not (proposal.code or "").strip():
            return False, "code proposal without source"
        return True, "ok"

    return False, f"unknown proposal kind {proposal.kind!r}"


def load_strategy_from_code(code: str, *, class_name: Optional[str] = None) -> Type[Strategy]:
    """Compile generated source and return its validated ``Strategy`` subclass.

    Raises :class:`HygieneError` if the code imports disallowed modules, fails the
    abstract-method contract, declares > 5 searchable params, lacks a docstring,
    or does not construct from its declared defaults.
    """
    namespace: dict = {"__builtins__": _restricted_builtins()}
    namespace["__import__"] = _guarded_import
    try:
        compiled = compile(code, "<generated-strategy>", "exec")
        exec(compiled, namespace)  # noqa: S102 - restricted namespace, validated below
    except Exception as exc:  # noqa: BLE001 - any failure is a rejection, not a crash
        raise HygieneError(f"generated code failed to import: {exc}") from exc

    candidates = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
    ]
    if class_name:
        candidates = [c for c in candidates if c.__name__ == class_name]
    if not candidates:
        raise HygieneError("no concrete Strategy subclass defined")
    cls = candidates[-1]

    _validate_contract(cls)
    return cls


def _validate_contract(cls: Type[Strategy]) -> None:
    if getattr(cls, "__abstractmethods__", None):
        raise HygieneError(
            f"{cls.__name__} leaves abstract methods unimplemented: {sorted(cls.__abstractmethods__)}"
        )
    if not (cls.__doc__ or "").strip():
        raise HygieneError(f"{cls.__name__} has no docstring (the hypothesis is required)")
    if not isinstance(getattr(cls, "PARAM_RANGES", None), dict):
        raise HygieneError(f"{cls.__name__} must declare a PARAM_RANGES dict")
    space = ParameterSpace(cls.PARAM_RANGES)
    if len(space.searchable) > MAX_TUNABLE_PARAMS:
        raise HygieneError(
            f"{cls.__name__} has {len(space.searchable)} searchable params (cap {MAX_TUNABLE_PARAMS})"
        )
    # Must construct from its own defaults.
    defaults = {name: spec["default"] for name, spec in cls.PARAM_RANGES.items() if "default" in spec}
    defaults.setdefault("timeframe", getattr(cls, "TIMEFRAME", "1Day"))
    try:
        instance = cls(dict(defaults))
    except Exception as exc:  # noqa: BLE001
        raise HygieneError(f"{cls.__name__} does not construct from defaults: {exc}") from exc

    # Sizing and exit handling read these off the config on every bar. Missing keys
    # would surface as a silent zero-trade run, not an error, so reject here instead.
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in instance.config]
    if missing:
        raise HygieneError(
            f"{cls.__name__} does not declare required config {missing} "
            f"(the execution path reads {list(REQUIRED_CONFIG_KEYS)} on every bar)"
        )


def _guarded_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if name in _ALLOWED_IMPORTS or root in {"pandas", "numpy", "math"} or name.startswith("tradeflow."):
        if name.startswith("tradeflow.") and name not in _ALLOWED_IMPORTS and root == "tradeflow":
            # Only allow the explicitly whitelisted tradeflow.* modules.
            if not any(name == allowed or name.startswith(allowed + ".") for allowed in _ALLOWED_IMPORTS):
                raise HygieneError(f"generated code may not import {name!r}")
        return _builtins.__import__(name, *args, **kwargs)
    raise HygieneError(f"generated code may not import {name!r}")


def _restricted_builtins() -> dict:
    allowed = {k: getattr(_builtins, k) for k in dir(_builtins) if k not in _DENIED_BUILTINS}
    allowed["__import__"] = _guarded_import
    return allowed
