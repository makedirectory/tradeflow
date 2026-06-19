"""Parameter search space derived from a ``PARAM_RANGES`` declaration.

Turns the ``{name: {min, max, step, default, type}}`` spec that strategies and
scanners already declare into the concrete things an optimizer needs:

* full **grid** of step-aligned combinations,
* **random** samples,
* **normalize/denormalize** to a ``[0, 1]`` vector (for surrogate models).

Only parameters that declare ``min``/``max``/``step`` are searched; everything
else is held at its default, so a search is always a complete, valid config.
"""

import math
from itertools import product
from typing import Any, Dict, List

import numpy as np

from src.utils.numeric import step_decimals


class ParameterSpace:
    """A searchable view over a ``PARAM_RANGES`` mapping."""

    def __init__(self, param_ranges: Dict[str, Dict[str, Any]]):
        self.param_ranges = param_ranges
        self.searchable: List[str] = [
            name for name, spec in param_ranges.items() if all(k in spec for k in ("min", "max", "step"))
        ]
        self.defaults: Dict[str, Any] = {
            name: spec["default"] for name, spec in param_ranges.items() if "default" in spec
        }

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

    def grid_size(self) -> int:
        """Number of points in the full grid, computed without materialising it."""
        return math.prod(len(self._values_for(name)) for name in self.searchable) if self.searchable else 0

    def grid(self) -> List[Dict[str, Any]]:
        """Every step-aligned combination of searchable parameters.

        Note: this is the full Cartesian product and can be enormous for
        many-parameter spaces. Callers that cap the number of evaluations should
        check :meth:`grid_size` first and fall back to :meth:`random_samples`.
        """
        axes = [self._values_for(name) for name in self.searchable]
        return [{**self.defaults, **dict(zip(self.searchable, combo))} for combo in product(*axes)]

    def random_samples(self, n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
        """``n`` random step-aligned configs."""
        samples = []
        for _ in range(n):
            params = dict(self.defaults)
            for name in self.searchable:
                params[name] = rng.choice(self._values_for(name)).item()
            samples.append(params)
        return samples

    # --- normalization, for surrogate-model optimizers -------------------- #
    def to_unit_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Map a config to a ``[0, 1]`` vector over the searchable parameters."""
        vec = []
        for name in self.searchable:
            spec = self.param_ranges[name]
            vec.append((params[name] - spec["min"]) / (spec["max"] - spec["min"]))
        return np.array(vec, dtype="float64")

    def from_unit_vector(self, vec: np.ndarray) -> Dict[str, Any]:
        """Inverse of :meth:`to_unit_vector`, snapped to the step grid."""
        params = dict(self.defaults)
        for i, name in enumerate(self.searchable):
            spec = self.param_ranges[name]
            raw = vec[i] * (spec["max"] - spec["min"]) + spec["min"]
            snapped = round(raw / spec["step"]) * spec["step"]
            snapped = min(spec["max"], max(spec["min"], round(snapped, step_decimals(spec["step"]))))
            params[name] = int(snapped) if spec["type"] == "int" else snapped
        return params

    @property
    def dimensions(self) -> int:
        return len(self.searchable)
