"""Constraint-solver portfolio allocation - because "put it all on the one with the
highest score" is a strategy, just not a good one.

Given a set of candidate symbols, each with a *score* (a higher score = more
attractive, however the caller chooses to define it - expected return, signal
strength, momentum, inverse volatility, ...), decide how to weight a portfolio
across them subject to hard constraints:

* invest at most 100% of capital,
* hold at most ``max_positions`` names (cardinality),
* cap any single name at ``max_weight`` and floor a *held* name at ``min_weight``.

This is a small mixed-integer program solved with Google OR-Tools. Binary
"is this name held?" variables enforce the cardinality and min-weight-if-held
constraints; continuous weight variables carry the allocation.

OR-Tools is an optional dependency (``portfolio`` extra) and is imported lazily,
so the base install stays lean.
"""

import logging
from dataclasses import dataclass
from typing import List

from src.utils.numeric import round_quantity

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A symbol eligible for allocation."""

    symbol: str
    score: float  # higher = more attractive (caller-defined factor)
    price: float


@dataclass
class Allocation:
    """A solved allocation for one symbol."""

    symbol: str
    weight: float  # fraction of capital [0, 1]
    dollars: float
    shares: float


class PortfolioAllocator:
    """Weights positions across candidates via a MIP constraint solver."""

    def __init__(
        self,
        max_positions: int = 5,
        max_weight: float = 0.25,
        min_weight: float = 0.0,
        allow_fractional_shares: bool = False,
    ):
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 <= min_weight <= max_weight:
            raise ValueError("min_weight must be in [0, max_weight]")
        self.max_positions = max_positions
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.allow_fractional_shares = allow_fractional_shares

    def allocate(self, candidates: List[Candidate], capital: float) -> List[Allocation]:
        """Return the optimal allocation across ``candidates`` for ``capital``.

        Only candidates with a positive score are considered (a non-positive
        score never improves the objective and would only consume a slot).
        """
        eligible = [c for c in candidates if c.score > 0 and c.price > 0]
        if not eligible or capital <= 0:
            return []

        solver = self._make_solver()
        weights, selected = self._build_model(solver, eligible)

        status = solver.Solve()
        if status not in (solver.OPTIMAL, solver.FEASIBLE):
            logger.warning("Portfolio solver returned no usable solution (status=%s)", status)
            return []

        allocations = []
        for candidate, weight_var, select_var in zip(eligible, weights, selected):
            weight = weight_var.solution_value()
            if select_var.solution_value() < 0.5 or weight <= 1e-9:
                continue
            dollars = weight * capital
            allocations.append(
                Allocation(
                    symbol=candidate.symbol,
                    weight=weight,
                    dollars=dollars,
                    shares=round_quantity(dollars / candidate.price, self.allow_fractional_shares),
                )
            )
        return sorted(allocations, key=lambda a: a.weight, reverse=True)

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #
    def _build_model(self, solver, candidates: List[Candidate]):
        weights, selected = [], []
        for candidate in candidates:
            weight = solver.NumVar(0.0, self.max_weight, f"w_{candidate.symbol}")
            select = solver.BoolVar(f"x_{candidate.symbol}")
            # A name's weight is bounded by its selection flag (0 weight if unheld),
            # and floored at min_weight when held.
            solver.Add(weight <= self.max_weight * select)
            solver.Add(weight >= self.min_weight * select)
            weights.append(weight)
            selected.append(select)

        solver.Add(solver.Sum(weights) <= 1.0)  # invest <= 100%
        solver.Add(solver.Sum(selected) <= self.max_positions)  # cardinality cap

        solver.Maximize(solver.Sum(c.score * w for c, w in zip(candidates, weights)))
        return weights, selected

    @staticmethod
    def _make_solver():
        try:
            from ortools.linear_solver import pywraplp
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "The portfolio manager requires Google OR-Tools. Install the optional extra:\n"
                "    uv sync --extra portfolio"
            ) from exc

        solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:  # pragma: no cover
            raise RuntimeError("Could not create an OR-Tools CBC solver")
        return solver
