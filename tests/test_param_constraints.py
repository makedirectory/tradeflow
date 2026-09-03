"""Invalid parameter combinations must be unreachable, not rejected after the fact.

An invalid combination that gets evaluated is a journaled trial, and a journaled trial
permanently raises the deflated-Sharpe bar for its whole family. So a wasted trial is
not just wasted compute: it makes every future candidate in that family harder to
promote, forever. That is why these tests assert that nothing invalid is ever *drawn*,
rather than that something invalid is caught.
"""

import numpy as np
import pytest

from tradeflow.optimization.param_space import ParameterConstraints, ParameterSpace

_RANGES = {
    "entry_period": {"type": "int", "min": 5, "max": 20, "step": 5, "default": 20},
    "exit_period": {"type": "int", "min": 5, "max": 20, "step": 5, "default": 5},
    "threshold": {"type": "float", "min": 0.1, "max": 0.3, "step": 0.1, "default": 0.2},
}
_ORDERED = (("exit_period", "<", "entry_period"),)


def _space(constraints=_ORDERED, ranges=None):
    return ParameterSpace(ranges or _RANGES, constraints)


# --- the grid ---------------------------------------------------------------------
def test_the_grid_contains_no_point_the_constraints_reject():
    space = _space()

    assert all(p["exit_period"] < p["entry_period"] for p in space.grid())


def test_the_unconstrained_product_would_have_contained_them():
    """Both directions. A filter over a product that never had a violation in it
    passes without doing anything, and is indistinguishable from one that works."""
    unconstrained = _space(constraints=()).grid()

    assert any(p["exit_period"] >= p["entry_period"] for p in unconstrained)


def test_the_reported_grid_size_is_the_number_of_points_a_caller_will_get():
    """`grid_size` is what a caller budgets `max_evals` against. Reporting the
    unconstrained product and then yielding fewer points makes a cap mean something
    different from what it said."""
    space = _space()

    assert space.grid_size() == len(space.grid())
    assert space.grid_size() < space.unconstrained_grid_size()


# --- random sampling --------------------------------------------------------------
def test_no_random_draw_is_ever_invalid():
    """Drawn parameter by parameter, each restricted to what is still consistent with
    what has been decided — so the invalid region is unreachable, not reached and
    discarded. Enough draws that a reject-after-draw implementation with an off-by-one
    would show up."""
    draws = _space().random_samples(500, np.random.default_rng(3))

    assert len(draws) == 500
    assert all(d["exit_period"] < d["entry_period"] for d in draws)


def test_a_draw_still_covers_the_whole_feasible_region():
    """A sampler that satisfies constraints by collapsing onto one corner satisfies
    them and searches nothing."""
    draws = _space().random_samples(400, np.random.default_rng(5))

    assert len({d["entry_period"] for d in draws}) > 1
    assert len({d["exit_period"] for d in draws}) > 1
    assert len({d["threshold"] for d in draws}) > 1


def test_constraints_that_admit_nothing_fail_loudly_rather_than_hanging():
    """The other end of "keep drawing until it is valid": a space whose constraints
    exclude everything must say so, not spin."""
    impossible = _space(
        constraints=(("exit_period", ">", "entry_period"), ("exit_period", "<", "entry_period"))
    )

    with pytest.raises(ValueError, match="constraints"):
        impossible.random_samples(5, np.random.default_rng(0))


# --- a declaration that changes nothing must change nothing ------------------------
def test_a_constraint_that_cannot_bind_leaves_the_sampler_exactly_as_it_was():
    """The shipped demo strategy declares fast < slow over ranges (5-20, 21-60) that
    cannot violate it. Routing such a space through the constraint-aware sampler would
    change which configs a seeded random search visits — and therefore which trials a
    campaign records — for a declaration that excludes nothing."""
    ranges = {
        "fast": {"type": "int", "min": 5, "max": 20, "step": 1, "default": 10},
        "slow": {"type": "int", "min": 21, "max": 60, "step": 1, "default": 30},
    }
    declared = ParameterSpace(ranges, (("fast", "<", "slow"),))
    plain = ParameterSpace(ranges)

    assert declared.random_samples(20, np.random.default_rng(11)) == plain.random_samples(
        20, np.random.default_rng(11)
    )


def test_a_constraint_that_can_bind_is_not_skipped():
    """Both directions: the shortcut above must not swallow a real constraint."""
    overlapping = {
        "fast": {"type": "int", "min": 5, "max": 30, "step": 1, "default": 10},
        "slow": {"type": "int", "min": 10, "max": 60, "step": 1, "default": 30},
    }
    space = ParameterSpace(overlapping, (("fast", "<", "slow"),))

    assert space._constraints_can_bind()
    assert all(d["fast"] < d["slow"] for d in space.random_samples(200, np.random.default_rng(2)))


# --- malformed declarations are rejected where they are declared -------------------
def test_a_constraint_naming_a_parameter_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="unknown parameter"):
        _space(constraints=(("exit_period", "<", "no_such_param"),))


def test_an_unknown_operator_is_refused():
    with pytest.raises(ValueError, match="operator"):
        _space(constraints=(("exit_period", "=<", "entry_period"),))


def test_a_literal_is_a_legal_side():
    """Not every constraint relates two parameters; a floor or ceiling is one too."""
    space = _space(constraints=(("entry_period", ">=", 10),))

    assert all(p["entry_period"] >= 10 for p in space.grid())


# --- a partial assignment is not judged as if it were complete ---------------------
def test_an_undecided_parameter_does_not_make_a_partial_draw_infeasible():
    """The sampler asks about partial assignments. Treating an unassigned side as a
    violation would prune regions that are perfectly reachable, and the pruning would
    look exactly like a constraint doing its job."""
    constraints = ParameterConstraints(_ORDERED, list(_RANGES))

    assert constraints.feasible({"exit_period": 5})  # entry_period not chosen yet
    assert not constraints.feasible({"exit_period": 20, "entry_period": 5})


# --- one definition, reached two ways ---------------------------------------------
def test_a_strategy_rejects_a_config_its_declared_constraints_forbid():
    """The declaration is what a sampler reads; construction has to enforce the same
    one. A strategy that restated the comparison in `initialize` gave the sampler
    nothing to introspect, which is the half that matters — a combination a sampler
    draws gets evaluated, and an evaluated combination is a journaled trial."""
    from tradeflow.strategies.base import Strategy

    class _Constrained(Strategy):
        PARAM_RANGES = dict(_RANGES)
        PARAM_CONSTRAINTS = _ORDERED

        def calculate_required_lookback(self) -> int:
            return 1

        def initialize(self) -> None:
            pass

        def process_data(self, data):
            return data

        def generate_signals(self, data, symbol=None):
            return data

        def calculate_scores(self, data, symbol=None):
            return data

    with pytest.raises(ValueError, match="exit_period < entry_period"):
        _Constrained({"entry_period": 5, "exit_period": 20, "threshold": 0.2})

    # Both directions: the boundary case is accepted, or the guard is indistinguishable
    # from one that rejects everything.
    assert _Constrained({"entry_period": 20, "exit_period": 15, "threshold": 0.2}) is not None


def test_the_shipped_strategy_declares_its_ordering_rather_than_restating_it():
    """The demo strategy's fast/slow rule used to live only in `initialize`, where a
    sampler could not see it."""
    from tradeflow.demo.strategies import DemoTrendStrategy

    assert ("fast_ema_period", "<", "slow_ema_period") in DemoTrendStrategy.PARAM_CONSTRAINTS
    assert ParameterSpace.for_class(DemoTrendStrategy).constraints.describe() == (
        "fast_ema_period < slow_ema_period",
    )


def test_draft_code_declaring_an_unusable_constraint_gets_a_verdict_not_a_traceback():
    """A validator whose whole job is to answer "is this valid?" must answer it. The
    same contract the malformed-PARAM_RANGES rejection already has."""
    from tradeflow.research.sandbox import HygieneError, load_strategy_from_code

    code = '''
from tradeflow.strategies.base import Strategy

class Draft(Strategy):
    """A draft whose constraint names a parameter it never declared."""

    PARAM_RANGES = {"a": {"type": "int", "min": 1, "max": 5, "step": 1, "default": 2}}
    PARAM_CONSTRAINTS = (("a", "<", "b"),)

    def calculate_required_lookback(self):
        return 1

    def initialize(self):
        pass

    def process_data(self, data):
        return data

    def generate_signals(self, data, symbol=None):
        return data

    def calculate_scores(self, data, symbol=None):
        return data
'''
    with pytest.raises(HygieneError, match="PARAM_CONSTRAINTS"):
        load_strategy_from_code(code)
