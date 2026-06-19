"""Portfolio allocator tests (OR-Tools constraint solver).

Skipped automatically if the optional ``portfolio`` extra (ortools) isn't installed.
"""

import pytest

pytest.importorskip("ortools")

from src.portfolio.allocator import Candidate, PortfolioAllocator  # noqa: E402


def _candidates():
    return [
        Candidate("AAA", score=0.10, price=100.0),
        Candidate("BBB", score=0.08, price=50.0),
        Candidate("CCC", score=0.05, price=25.0),
        Candidate("DDD", score=0.01, price=10.0),
    ]


def test_respects_cardinality_and_weight_caps():
    allocator = PortfolioAllocator(max_positions=2, max_weight=0.5)
    allocations = allocator.allocate(_candidates(), capital=100_000)

    assert len(allocations) <= 2
    assert all(a.weight <= 0.5 + 1e-9 for a in allocations)
    assert sum(a.weight for a in allocations) <= 1.0 + 1e-9


def test_prefers_higher_scores():
    allocator = PortfolioAllocator(max_positions=1, max_weight=1.0)
    allocations = allocator.allocate(_candidates(), capital=100_000)
    assert len(allocations) == 1
    assert allocations[0].symbol == "AAA"  # highest score


def test_dollars_and_shares_consistent():
    allocator = PortfolioAllocator(max_positions=2, max_weight=0.5)
    allocations = allocator.allocate(_candidates(), capital=100_000)
    for a in allocations:
        assert a.dollars == pytest.approx(a.weight * 100_000)
        assert a.shares == int(a.dollars / next(c.price for c in _candidates() if c.symbol == a.symbol))


def test_non_positive_scores_excluded():
    allocator = PortfolioAllocator(max_positions=5, max_weight=1.0)
    candidates = [Candidate("AAA", score=0.0, price=100.0), Candidate("BBB", score=-0.1, price=50.0)]
    assert allocator.allocate(candidates, capital=100_000) == []


def test_empty_inputs():
    allocator = PortfolioAllocator()
    assert allocator.allocate([], capital=100_000) == []
    assert allocator.allocate(_candidates(), capital=0) == []
