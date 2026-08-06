"""Tests for the benchmark-as-a-portfolio helpers: loading w_B,
restricting/renormalizing to Σ's covered universe, and reverse optimization.

Offline and deterministic.
"""

import json

import numpy as np
import pytest

from tradeflow.portfolio.benchmark import (
    implied_returns,
    load_benchmark_weights,
    restrict_and_renormalize,
)
from tradeflow.risk.base import RiskMatrix

SIGMA = np.array([[0.04, 0.006, 0.002], [0.006, 0.09, 0.003], [0.002, 0.003, 0.05]])
SYMS = ["A", "B", "C"]


def _risk() -> RiskMatrix:
    return RiskMatrix(SYMS, SIGMA)


# --- load_benchmark_weights ---------------------------------------------------
def test_equal_source_is_uniform_over_symbols():
    w = load_benchmark_weights("equal", SYMS)
    assert w == {"A": pytest.approx(1 / 3), "B": pytest.approx(1 / 3), "C": pytest.approx(1 / 3)}


def test_equal_source_empty_universe_is_empty():
    assert load_benchmark_weights("equal", []) == {}


def test_json_file_is_parsed_and_renormalized(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"A": 2.0, "B": 1.0, "C": 1.0}))  # sums to 4, not 1
    w = load_benchmark_weights(str(path), SYMS)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert w["A"] == pytest.approx(0.5)
    assert w["B"] == pytest.approx(0.25)


def test_csv_file_is_parsed(tmp_path):
    path = tmp_path / "bench.csv"
    path.write_text("symbol,weight\nA,0.5\nB,0.3\nC,0.2\n")
    w = load_benchmark_weights(str(path), SYMS)
    assert w == {"A": pytest.approx(0.5), "B": pytest.approx(0.3), "C": pytest.approx(0.2)}


def test_unknown_extension_raises(tmp_path):
    path = tmp_path / "bench.txt"
    path.write_text("A,0.5")
    with pytest.raises(ValueError):
        load_benchmark_weights(str(path), SYMS)


def test_missing_file_raises():
    with pytest.raises(ValueError):
        load_benchmark_weights("/no/such/file.csv", SYMS)


def test_all_zero_weight_file_raises(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"A": 0.0}))
    with pytest.raises(ValueError):
        load_benchmark_weights(str(path), SYMS)


# --- restrict_and_renormalize --------------------------------------------------
def test_restrict_drops_uncovered_names_and_reports_coverage():
    raw = {"A": 0.5, "B": 0.3, "ZZZ": 0.2}
    restricted, coverage = restrict_and_renormalize(raw, SYMS)
    assert "ZZZ" not in restricted
    assert coverage == pytest.approx(0.8)
    assert abs(sum(restricted.values()) - 1.0) < 1e-12
    assert restricted["A"] == pytest.approx(0.5 / 0.8)


def test_restrict_full_coverage_is_a_no_op_up_to_renormalization():
    raw = {"A": 0.5, "B": 0.3, "C": 0.2}
    restricted, coverage = restrict_and_renormalize(raw, SYMS)
    assert coverage == pytest.approx(1.0)
    assert restricted == pytest.approx(raw)


def test_restrict_zero_overlap_is_empty_with_zero_coverage():
    restricted, coverage = restrict_and_renormalize({"ZZZ": 1.0}, SYMS)
    assert restricted == {}
    assert coverage == 0.0


# --- implied_returns (reverse optimization) -----------------------------------
def test_implied_returns_matches_beta_times_mu_b():
    wb = {"A": 0.5, "B": 0.3, "C": 0.2}
    mu = implied_returns(wb, _risk(), mu_b=0.05)
    beta = _risk().implied_beta(wb)
    for s in SYMS:
        assert mu[s] == pytest.approx(float(beta[s]) * 0.05)


def test_implied_returns_scales_linearly_with_mu_b():
    wb = {"A": 0.5, "B": 0.3, "C": 0.2}
    mu_low = implied_returns(wb, _risk(), mu_b=0.04)
    mu_high = implied_returns(wb, _risk(), mu_b=0.08)
    for s in SYMS:
        assert mu_high[s] == pytest.approx(2 * mu_low[s])
