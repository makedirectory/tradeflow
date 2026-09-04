"""Validation sandbox for proposals - the rules that keep an eager AI from
manufacturing overfit garbage at scale (its words would be more enthusiastic).

Two jobs:

1. **Research-hygiene gates** on every proposal: a rationale must be present, and
   no more than :data:`MAX_TUNABLE_PARAMS` parameters may be varied. These are the
   cheap, load-bearing rules that stop the loop manufacturing overfit garbage at
   scale.
2. **Contract validation** of agent-authored strategy/scanner *code*: a strategy
   must define a concrete :class:`~tradeflow.strategies.base.Strategy` subclass that
   implements the abstract hooks, declares a valid ``PARAM_RANGES`` (<= 5
   searchable), carries a docstring (the hypothesis), and actually constructs. A
   scanner must define a concrete :class:`~tradeflow.scanners.base.ScannerStrategy`
   subclass that emits the scanner signal vocabulary and a numeric
   ``signal_strength``.

Isolation note: generated code is validated and executed with a restricted global
namespace (no ``os``/``sys``/network imports, limited builtins). This is a
*proposal artifact*, never auto-merged. True OS-level isolation (separate process,
no network, resource limits) is the production hardening called for in  and is
left as a deliberate follow-up; :func:`load_strategy_from_code` and
:func:`load_scanner_from_code` are the choke points where that would be enforced.
"""

import builtins as _builtins
from collections.abc import Mapping
from typing import Optional, Tuple, Type

from tradeflow.optimization.param_space import ParameterSpace
from tradeflow.scanners.base import SCANNER_BUY, SCANNER_HOLD, SCANNER_SELL, ScannerStrategy
from tradeflow.strategies.base import Strategy

#: Hard cap on tunable parameters per strategy (more knobs = more overfit surface).
MAX_TUNABLE_PARAMS = 5

#: Config keys the execution path reads off every strategy, regardless of mechanism:
#: ``calculate_position_size`` needs the first two, and the backtest engine and live
#: trader both read ``take_profit`` when opening a position. A generated strategy that
#: omits them passes every other check and then raises on the first bar - producing a
#: zero-trade run that is indistinguishable from "no edge" unless we catch it here.
REQUIRED_CONFIG_KEYS = ("risk_per_trade", "stop_loss", "take_profit")

#: Import names generated strategy/scanner code is allowed to use.
#:
#: The list exists to keep generated code away from the filesystem, the process, the
#: network and the broker. ``typing`` and ``collections.abc`` reach none of those -
#: they are annotation vocabulary and grant no capability - and leaving them off
#: rejected every strategy and scanner this package ships, since all of them annotate
#: their ``PARAM_RANGES``. A validator that refuses the repository's own examples is
#: measuring its allowlist, not the draft.
_ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "math",
    "typing",
    "collections.abc",
    "tradeflow.indicators.indicators",
    "tradeflow.indicators",
    "tradeflow.scanners.base",
    "tradeflow.scanners",
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
    cls = _load_class_from_code(code, Strategy, "strategy", class_name)
    _validate_strategy_contract(cls)
    return cls


def load_scanner_from_code(code: str, *, class_name: Optional[str] = None) -> Type[ScannerStrategy]:
    """Compile generated source and return its validated ``ScannerStrategy`` subclass.

    Raises :class:`HygieneError` if the code imports disallowed modules, fails the
    abstract-method contract, declares too many searchable params, lacks a docstring,
    or does not emit a scanner signal frame on sample OHLCV data.
    """
    cls = _load_class_from_code(code, ScannerStrategy, "scanner", class_name)
    _validate_scanner_contract(cls)
    return cls


def _load_class_from_code(code: str, base_cls: Type, label: str, class_name: Optional[str] = None) -> Type:
    namespace: dict = {"__builtins__": _restricted_builtins()}
    namespace["__import__"] = _guarded_import
    try:
        compiled = compile(code, f"<generated-{label}>", "exec")
        exec(compiled, namespace)  # noqa: S102 - restricted namespace, validated below
    except Exception as exc:  # noqa: BLE001 - any failure is a rejection, not a crash
        raise HygieneError(f"generated code failed to import: {exc}") from exc

    candidates = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, base_cls) and obj is not base_cls
    ]
    if class_name:
        candidates = [c for c in candidates if c.__name__ == class_name]
    if not candidates:
        raise HygieneError(f"no concrete {base_cls.__name__} subclass defined")
    return candidates[-1]


def _validate_common_contract(cls: Type, base_name: str) -> None:
    if getattr(cls, "__abstractmethods__", None):
        raise HygieneError(
            f"{cls.__name__} leaves abstract methods unimplemented: {sorted(cls.__abstractmethods__)}"
        )
    if not (cls.__doc__ or "").strip():
        raise HygieneError(f"{cls.__name__} has no docstring (the hypothesis is required)")
    if not isinstance(getattr(cls, "PARAM_RANGES", None), dict):
        raise HygieneError(f"{cls.__name__} must declare a PARAM_RANGES dict")
    # Every rejection leaves here as a HygieneError, including a malformed spec.
    # `PARAM_RANGES = {"lookback": 20}` is what generated code actually writes, and
    # reading it raised a bare TypeError out of ParameterSpace - through validators
    # whose entire contract is to answer "is this valid?" with a verdict.
    for param, spec in cls.PARAM_RANGES.items():
        if not isinstance(spec, Mapping):
            raise HygieneError(
                f"{cls.__name__} parameter {param!r} must be a spec mapping with "
                f"min/max/step/default, not a bare {type(spec).__name__}"
            )
    try:
        space = ParameterSpace.for_class(cls)
    except ValueError as exc:
        # Same contract as the malformed-spec rejection above: a validator whose whole
        # job is to answer "is this valid?" must answer it, not raise something else.
        # A draft can declare constraints, and a constraint naming a parameter that
        # does not exist is exactly the kind of thing generated code writes.
        raise HygieneError(f"{cls.__name__} declares an unusable PARAM_CONSTRAINTS: {exc}") from exc
    if len(space.searchable) > MAX_TUNABLE_PARAMS:
        # Deliberately stricter than the curated strategies this package ships: a
        # draft is an unreviewed proposal, and knobs are overfit surface. Two shipped
        # strategies exceed it, which is why the shipped-artifact test below admits
        # this one rejection and no other.
        raise HygieneError(
            f"{cls.__name__} has {len(space.searchable)} searchable params "
            f"(draft {base_name} cap is {MAX_TUNABLE_PARAMS})"
        )


def _defaults(cls: Type) -> dict:
    return {name: spec["default"] for name, spec in cls.PARAM_RANGES.items() if "default" in spec}


def _validate_strategy_contract(cls: Type[Strategy]) -> None:
    _validate_common_contract(cls, "strategy")
    # Must construct from its own defaults.
    defaults = _defaults(cls)
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


def _validate_scanner_contract(cls: Type[ScannerStrategy]) -> None:
    _validate_common_contract(cls, "scanner")
    try:
        instance = cls(_defaults(cls))
        instance.initialize()
    except Exception as exc:  # noqa: BLE001
        raise HygieneError(f"{cls.__name__} does not construct from defaults: {exc}") from exc

    import pandas as pd

    sample = _scanner_sample(cls, instance)
    try:
        signals = instance.generate_signals_df(instance.process_data(sample))
    except Exception as exc:  # noqa: BLE001
        raise HygieneError(f"{cls.__name__} cannot produce scanner signals on sample data: {exc}") from exc
    required = {"signal", "signal_strength"}
    if not required <= set(signals.columns):
        raise HygieneError(f"{cls.__name__} scanner signals must include {sorted(required)}")
    valid_signals = {SCANNER_BUY, SCANNER_SELL, SCANNER_HOLD}
    emitted = set(signals["signal"].dropna())
    if not emitted <= valid_signals:
        raise HygieneError(f"{cls.__name__} emits unknown scanner signals: {sorted(emitted - valid_signals)}")
    try:
        pd.to_numeric(signals["signal_strength"])
    except Exception as exc:  # noqa: BLE001
        raise HygieneError(f"{cls.__name__} signal_strength must be numeric") from exc


def _scanner_sample(cls: Type, instance):
    """OHLCV long enough for a scanner's own warm-up, and eventful enough to move it.

    A fixed five-bar frame was neither, and every scanner needs more than five bars.
    One that guards on ``required_data_points()`` was rejected for a sample too short
    to scan; one that does not returned an all-NaN, all-HOLD frame - which passes the
    signal-vocabulary check trivially, because ``{HOLD}`` is a subset of the
    vocabulary. So the check could not fail, for any scanner, in either direction.

    The tail carries a volume spike on a decisive up-bar because a scanner that never
    emits an actionable label is a scanner whose actionable labels went unchecked.
    """
    import pandas as pd

    try:
        required = int(instance.required_data_points())
    except Exception as exc:  # noqa: BLE001 - a scanner that cannot say is not valid
        raise HygieneError(f"{cls.__name__} cannot report required_data_points: {exc}") from exc
    if required < 1:
        raise HygieneError(f"{cls.__name__} reports required_data_points={required}")

    n = max(required + 5, 30)
    opens = [100.0 + i * 0.25 for i in range(n)]
    closes = [price + 0.1 for price in opens]
    volumes = [300_000.0] * n
    # The event: a decisive up-bar on heavy volume, so thresholds on ratio, move and
    # liquidity are all cleared at once and an actionable label is actually reachable.
    opens[-1], closes[-1] = 100.0, 103.0
    volumes[-1] = 2_000_000.0
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 1.0 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 1.0 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": volumes,
        },
        index=pd.date_range("2024-01-02", periods=n, freq="D"),
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
