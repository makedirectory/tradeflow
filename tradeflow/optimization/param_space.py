"""Parameter search space derived from a ``PARAM_RANGES`` declaration.

Turns the ``{name: {min, max, step, default, type}}`` spec that strategies and
scanners already declare into the concrete things an optimizer needs:

* full **grid** of step-aligned combinations,
* **random** samples,
* **normalize/denormalize** to a ``[0, 1]`` vector (for surrogate models).

Only parameters that declare ``min``/``max``/``step`` are searched; everything
else is held at its default, so a search is always a complete, valid config.

A space may also declare **constraints** between parameters - ``exit_period``
below ``entry_period``, say. They are enforced by construction rather than by
rejection: the grid never contains an invalid point and the random sampler never
draws one, because an evaluated invalid combination is a journaled trial, and a
journaled trial permanently raises the deflation bar for every future candidate in
its family. Wasted trials are not just wasted compute.
"""

import math
import operator
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tradeflow.utils.numeric import step_decimals

#: The comparisons a declarative constraint may use. Deliberately a fixed table
#: rather than an expression to evaluate: a space is data a sampler introspects, and
#: something that has to be executed to be understood cannot be introspected at all.
_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

#: One declared constraint: ``(left, op, right)``, where each side is either the name
#: of a parameter in the same space or a literal number.
Constraint = Tuple[str, str, Any]


class ParameterConstraints:
    """The declared relationships between parameters in one space.

    Two questions, and they are not the same one: ``holds`` asks whether a complete
    assignment is valid, and ``feasible`` asks whether a *partial* one can still be
    completed. The sampler needs the second - that is what lets it narrow each
    parameter's candidate values as it goes, instead of drawing a whole point and
    throwing it away.
    """

    def __init__(self, declared: Sequence[Constraint], names: Sequence[str]):
        self.declared: Tuple[Constraint, ...] = tuple(declared)
        for left, op, right in self.declared:
            if op not in _OPERATORS:
                raise ValueError(f"Unknown constraint operator {op!r}; expected one of {sorted(_OPERATORS)}")
            for side in (left, right):
                if isinstance(side, str) and side not in names:
                    raise ValueError(f"Constraint references unknown parameter {side!r}")

    def __bool__(self) -> bool:
        return bool(self.declared)

    def _resolve(self, side: Any, params: Dict[str, Any]) -> Any:
        """A side's value, or ``None`` when it names a parameter not yet assigned."""
        if isinstance(side, str):
            return params.get(side)
        return side

    def holds(self, params: Dict[str, Any]) -> bool:
        """Whether a complete assignment satisfies every declared constraint."""
        return self.violations(params) == ()

    def violations(self, params: Dict[str, Any]) -> Tuple[str, ...]:
        """Every constraint this assignment breaks, rendered for a human.

        Returned rather than raised: a caller screening a user-supplied config wants
        to report all of them at once, and a caller sampling wants a boolean.
        """
        broken = []
        for left, op, right in self.declared:
            lhs, rhs = self._resolve(left, params), self._resolve(right, params)
            if lhs is None or rhs is None:
                continue  # not fully assigned; `feasible` is the question for that
            if not _OPERATORS[op](lhs, rhs):
                broken.append(f"{left} {op} {right} (got {lhs} {op} {rhs})")
        return tuple(broken)

    def feasible(self, params: Dict[str, Any]) -> bool:
        """Whether a partial assignment still violates nothing it has decided.

        A constraint with an unassigned side is not yet judged: it may still be
        satisfiable, and refusing it here would prune valid regions of the space.
        """
        return self.violations(params) == ()

    def describe(self) -> Tuple[str, ...]:
        return tuple(f"{left} {op} {right}" for left, op, right in self.declared)

    def can_bind(self, extremes: Dict[str, Tuple[Any, Any]]) -> bool:
        """Whether any declared constraint could exclude a point in these ranges.

        ``extremes`` maps a parameter to its ``(lowest, highest)`` declared value. A
        constraint that every combination of the extremes already satisfies cannot
        remove anything, and saying so is worth the few lines: it lets a space whose
        constraints are decorative keep drawing exactly the points it drew before one
        was declared. A declaration that changes nothing should change nothing —
        including which configs a seeded random search happens to visit.

        Conservative by construction: equality operators, and anything it cannot
        bound, answer "yes, it could".
        """
        for left, op, right in self.declared:
            lo_l, hi_l = self._extreme(left, extremes)
            lo_r, hi_r = self._extreme(right, extremes)
            if None in (lo_l, hi_l, lo_r, hi_r):
                return True
            # The worst case for each ordering: the combination most likely to fail.
            if op == "<" and hi_l < lo_r:
                continue
            if op == "<=" and hi_l <= lo_r:
                continue
            if op == ">" and lo_l > hi_r:
                continue
            if op == ">=" and lo_l >= hi_r:
                continue
            return True
        return False

    @staticmethod
    def _extreme(side: Any, extremes: Dict[str, Tuple[Any, Any]]) -> Tuple[Any, Any]:
        if isinstance(side, str):
            return extremes.get(side, (None, None))
        return (side, side)


class ParameterSpace:
    """A searchable view over a ``PARAM_RANGES`` mapping."""

    def __init__(self, param_ranges: Dict[str, Dict[str, Any]], constraints: Sequence[Constraint] = ()):
        self.param_ranges = param_ranges
        self.searchable: List[str] = [
            name for name, spec in param_ranges.items() if all(k in spec for k in ("min", "max", "step"))
        ]
        self.defaults: Dict[str, Any] = {
            name: spec["default"] for name, spec in param_ranges.items() if "default" in spec
        }
        self.constraints = ParameterConstraints(constraints, list(param_ranges))

    @classmethod
    def for_class(cls, strategy_class) -> "ParameterSpace":
        """The space a strategy (or scanner) class declares, constraints included.

        One construction path, so a class that declares constraints gets them applied
        wherever its space is built. Building ``ParameterSpace(cls.PARAM_RANGES)``
        directly is what silently drops them, which is the failure this exists to
        prevent - an invalid combination that gets evaluated is a journaled trial.
        """
        return cls(
            getattr(strategy_class, "PARAM_RANGES", {}) or {},
            getattr(strategy_class, "PARAM_CONSTRAINTS", ()) or (),
        )

    def _values_for(self, name: str) -> List[Any]:
        spec = self.param_ranges[name]
        decimals = step_decimals(spec["step"])
        n_steps = int(round((spec["max"] - spec["min"]) / spec["step"])) + 1
        values = []
        for i in range(n_steps):
            value = round(spec["min"] + i * spec["step"], decimals)
            value = min(spec["max"], max(spec["min"], value))
            value = int(value) if spec["type"] == "int" else value
            if value not in values:
                values.append(value)
        return values

    def unconstrained_grid_size(self) -> int:
        """Points in the full Cartesian product, before constraints remove any."""
        return math.prod(len(self._values_for(name)) for name in self.searchable) if self.searchable else 0

    def grid_size(self) -> int:
        """Number of *valid* points in the full grid.

        Computed without materializing the product when nothing is constrained. With
        constraints it enumerates, because the count a caller budgets against has to
        be the count it will actually get - reporting the unconstrained size and then
        yielding fewer points is how a cap ends up meaning something different from
        what it said.
        """
        if not self._constraints_can_bind():
            return self.unconstrained_grid_size()
        return len(self.grid())

    def grid(self) -> List[Dict[str, Any]]:
        """Every step-aligned, constraint-satisfying combination of searchable params.

        Note: this is the full Cartesian product and can be enormous for
        many-parameter spaces. Callers that cap the number of evaluations should
        check :meth:`grid_size` first and fall back to :meth:`random_samples`.
        """
        axes = [self._values_for(name) for name in self.searchable]
        points = [{**self.defaults, **dict(zip(self.searchable, combo))} for combo in product(*axes)]
        if not self._constraints_can_bind():
            return points
        return [point for point in points if self.constraints.holds(point)]

    def random_samples(self, n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
        """``n`` random step-aligned configs, none of which violate a constraint.

        Drawn parameter by parameter, each one restricted to the values still
        consistent with what has already been decided - so an invalid region is
        unreachable rather than reached and discarded. A draw that paints itself into
        a corner (no value left for a later parameter) is restarted rather than
        returned half-built; the restart budget is bounded so a space whose
        constraints admit nothing fails loudly instead of hanging.
        """
        if not self._constraints_can_bind():
            samples = []
            for _ in range(n):
                params = dict(self.defaults)
                for name in self.searchable:
                    params[name] = rng.choice(self._values_for(name)).item()
                samples.append(params)
            return samples

        samples = []
        attempts = 0
        budget = max(20 * n, 100)
        while len(samples) < n and attempts < budget:
            attempts += 1
            drawn = self._draw_constrained(rng)
            if drawn is not None:
                samples.append(drawn)
        if len(samples) < n:
            raise ValueError(
                f"Could only draw {len(samples)} of {n} configs satisfying "
                f"{', '.join(self.constraints.describe())} — the constraints may admit "
                f"little or none of the declared ranges"
            )
        return samples

    def _draw_constrained(self, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
        """One constraint-satisfying draw, or ``None`` if this attempt dead-ended."""
        params = dict(self.defaults)
        for name in self.searchable:
            allowed = [v for v in self._values_for(name) if self.constraints.feasible({**params, name: v})]
            if not allowed:
                return None
            params[name] = allowed[int(rng.integers(len(allowed)))]
            if isinstance(params[name], np.generic):
                params[name] = params[name].item()
        return params if self.constraints.holds(params) else None

    # --- normalization, for surrogate-model optimizers -------------------- #
    def to_unit_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Map a config to a ``[0, 1]`` vector over the searchable parameters."""
        vec = []
        for name in self.searchable:
            spec = self.param_ranges[name]
            vec.append((params[name] - spec["min"]) / (spec["max"] - spec["min"]))
        return np.array(vec, dtype="float64")

    def from_unit_vector(self, vec: np.ndarray) -> Dict[str, Any]:
        """Inverse of :meth:`to_unit_vector`, snapped to the step grid.

        The one path constraints cannot be enforced *by construction*: a surrogate
        proposes a point in a continuous box and snapping it can land outside the
        feasible region. The caller checks :meth:`is_valid` and drops such a proposal
        before evaluating it, so it still costs no trial — but this is why
        reject-after-draw is not the design anywhere it can be avoided.
        """
        params = dict(self.defaults)
        for i, name in enumerate(self.searchable):
            spec = self.param_ranges[name]
            raw = vec[i] * (spec["max"] - spec["min"]) + spec["min"]
            snapped = round(raw / spec["step"]) * spec["step"]
            snapped = min(spec["max"], max(spec["min"], round(snapped, step_decimals(spec["step"]))))
            params[name] = int(snapped) if spec["type"] == "int" else snapped
        return params

    def _constraints_can_bind(self) -> bool:
        """Whether the declared constraints can exclude anything in this space."""
        if not self.constraints:
            return False
        extremes = {}
        for name in self.searchable:
            values = self._values_for(name)
            if values:
                extremes[name] = (min(values), max(values))
        for name, value in self.defaults.items():
            extremes.setdefault(name, (value, value))
        return self.constraints.can_bind(extremes)

    def is_valid(self, params: Dict[str, Any]) -> bool:
        """Whether a complete assignment satisfies every declared constraint."""
        return self.constraints.holds(params)

    def violations(self, params: Dict[str, Any]) -> Tuple[str, ...]:
        """Every constraint ``params`` breaks, for a caller that has to explain why."""
        return self.constraints.violations(params)

    @property
    def dimensions(self) -> int:
        return len(self.searchable)
