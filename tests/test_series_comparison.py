"""Correlating two recorded trials' return series, and refusing to.

Most of what matters here is the refusals. A correlation is a claim about a
relationship and there is no partial version of one, so a pair that cannot support the
claim has to produce nothing rather than a number with a caveat beside it.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from tradeflow.analytics.reporting import format_series_comparison
from tradeflow.analytics.series_comparison import (
    COMPARED,
    MIN_OVERLAP,
    REFUSED,
    compare_series,
)

DATES = [(datetime(2024, 1, 2) + timedelta(days=i)).date().isoformat() for i in range(300)]


def _entry(trial_id, values, *, dates=None, accounting=5, strategy="demo_trend"):
    return {
        "trial_id": trial_id,
        "dates": DATES[: len(values)] if dates is None else dates,
        "values": list(values),
        "accounting": accounting,
        "strategy": strategy,
        "kind": "backtest",
    }


def _twins(n=300, noise=0.0005, seed=11):
    """Two series that are the same bet expressed twice, and one that is not."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0004, 0.011, n)
    return (
        base + rng.normal(0, noise, n),
        base + rng.normal(0, noise, n),
        rng.normal(0.0004, 0.011, n),
    )


# --- the thing it is for ------------------------------------------------------------
def test_two_results_that_are_one_result_show_up_as_one_result():
    a, b, _ = _twins()

    report = compare_series([_entry("a", a), _entry("b", b)])

    pair = report["pairs"][0]
    assert pair["status"] == COMPARED
    assert pair["correlation"] > 0.9
    assert pair["overlap"] == 300
    assert report["n_compared"] == 1 and report["n_refused"] == 0


def test_independent_results_are_not_reported_as_related():
    """Both directions: a check that only ever finds duplicates is not a check."""
    a, _, c = _twins()

    pair = compare_series([_entry("a", a), _entry("c", c)])["pairs"][0]

    assert pair["status"] == COMPARED
    assert abs(pair["correlation"]) < 0.3


# --- the refusals -------------------------------------------------------------------
def test_a_pair_below_the_minimum_overlap_is_refused_not_caveated():
    a, b, _ = _twins()
    short = _entry("b", b[:20], dates=DATES[:20])

    pair = compare_series([_entry("a", a), short])["pairs"][0]

    assert pair["status"] == REFUSED
    assert pair["correlation"] is None
    assert "20 shared dates" in pair["reason"] and str(MIN_OVERLAP) in pair["reason"]
    # The overlap it does have is still reported — the refusal says how far short.
    assert pair["overlap"] == 20


def test_a_pair_at_exactly_the_minimum_overlap_is_compared():
    """The boundary. A guard that rejects the case it was drawn to permit is
    indistinguishable from one that rejects everything."""
    a, b, _ = _twins()
    n = MIN_OVERLAP

    pair = compare_series([_entry("a", a[:n], dates=DATES[:n]), _entry("b", b[:n], dates=DATES[:n])])[
        "pairs"
    ][0]

    assert pair["status"] == COMPARED
    assert pair["overlap"] == MIN_OVERLAP


def test_series_from_different_accounting_versions_are_refused_by_default():
    """The two series were produced by engines that compute different things, so their
    correlation is partly a fact about the instruments."""
    a, b, _ = _twins()

    pair = compare_series([_entry("a", a, accounting=4), _entry("b", b, accounting=5)])["pairs"][0]

    assert pair["status"] == REFUSED
    assert "v4" in pair["reason"] and "v5" in pair["reason"]


def test_a_forced_cross_accounting_pair_is_computed_and_labelled_incomparable():
    a, b, _ = _twins()

    pair = compare_series(
        [_entry("a", a, accounting=4), _entry("b", b, accounting=5)], across_accounting=True
    )["pairs"][0]

    assert pair["status"] == COMPARED
    assert pair["comparable"] is False
    assert "INCOMPARABLE" in pair["reason"]


def test_a_trial_with_no_recorded_series_is_refused_by_name():
    a, _, _ = _twins()
    missing = {
        "trial_id": "b",
        "dates": None,
        "values": None,
        "accounting": 5,
        "strategy": "demo_trend",
        "kind": "walkforward",
    }

    report = compare_series([_entry("a", a), missing])

    assert report["series"][1]["available"] is False
    assert "no return series was recorded" in report["series"][1]["reason"]
    assert report["pairs"][0]["status"] == REFUSED


def test_an_empty_stored_series_is_distinct_from_none_stored():
    """ "Nothing was stored" and "a series was stored and it is empty" are different
    facts about the record, and only one of them is about the strategy."""
    empty = _entry("b", [], dates=[])

    summary = compare_series([empty])["series"][0]

    assert summary["available"] is False
    assert summary["periods"] == 0
    assert "empty" in summary["reason"]


def test_a_flat_series_cannot_correlate_and_says_so():
    a, _, _ = _twins()

    pair = compare_series([_entry("a", a), _entry("flat", [0.0] * 300)])["pairs"][0]

    assert pair["status"] == REFUSED
    assert "does not vary" in pair["reason"]


# --- precision travels with the number ----------------------------------------------
def test_a_correlation_carries_an_interval_that_widens_with_a_short_overlap():
    """The guard against a coefficient over a thin overlap wearing four decimals. It is
    not the decimals that lie, it is the missing error bar."""
    a, b, _ = _twins()
    n = MIN_OVERLAP

    thin = compare_series([_entry("a", a[:n], dates=DATES[:n]), _entry("b", b[:n], dates=DATES[:n])])[
        "pairs"
    ][0]
    thick = compare_series([_entry("a", a), _entry("b", b)])["pairs"][0]

    thin_width = thin["interval"]["high"] - thin["interval"]["low"]
    thick_width = thick["interval"]["high"] - thick["interval"]["low"]
    assert thin_width > thick_width


# --- the matrix does not hide the refusals ------------------------------------------
def test_the_matrix_holds_none_where_nothing_was_computed_never_zero():
    """In a correlation matrix a zero is the strong claim that two results move
    independently — precisely the claim a refusal is unable to make."""
    a, b, _ = _twins()
    report = compare_series([_entry("a", a), _entry("b", b[:20], dates=DATES[:20])])

    values = report["matrix"]["values"]
    assert values[0][0] == 1.0
    assert values[0][1] is None and values[1][0] is None


def test_every_pair_appears_in_the_diagnostics_including_the_refused_ones():
    a, b, c = _twins()
    report = compare_series([_entry("a", a), _entry("b", b[:20], dates=DATES[:20]), _entry("c", c)])

    assert report["n_pairs"] == 3
    assert len(report["pairs"]) == 3
    assert report["n_compared"] + report["n_refused"] == 3


# --- the renderer -------------------------------------------------------------------
def test_the_headline_carries_the_overlap_count_and_its_span():
    a, b, _ = _twins()

    printed = format_series_comparison(compare_series([_entry("a", a), _entry("b", b)]))

    assert "overlap 300 periods" in printed
    assert f"{DATES[0]}..{DATES[299]}" in printed


def test_a_refused_pair_is_printed_beside_the_compared_ones():
    """A listing that shows the correlations and drops the refusals reads as a complete
    comparison of everything that was asked about."""
    a, b, c = _twins()

    printed = format_series_comparison(
        compare_series([_entry("a", a), _entry("b", b[:20], dates=DATES[:20]), _entry("c", c)])
    )

    assert "REFUSED" in printed
    assert "2 refused" in printed


def test_no_duplicate_verdict_is_drawn_from_an_uncorrelated_highest_pair():
    """The highest correlation is always reported; the sentence about what it means is
    not. Printing "one bet held twice" beside +0.10 would be a verdict its own number
    contradicts."""
    a, _, c = _twins()

    printed = format_series_comparison(compare_series([_entry("a", a), _entry("c", c)]))

    assert "Highest:" in printed
    assert "one bet held twice" not in printed


def test_a_near_duplicate_pair_does_get_the_verdict():
    a, b, _ = _twins()

    printed = format_series_comparison(compare_series([_entry("a", a), _entry("b", b)]))

    assert "one bet held twice" in printed


# --- the store accessor -------------------------------------------------------------
def test_a_trial_with_no_stored_series_reads_as_none_not_as_empty(tmp_path):
    from tradeflow.store.trials import TrialStore

    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "j.jsonl") as store:
        assert store.returns_for("nothing-here") is None

        store.record_returns("t1", DATES[:5], [0.01, -0.02, 0.0, 0.03, 0.01])
        series = store.returns_for("t1")

    assert series["dates"] == DATES[:5]
    assert series["values"] == pytest.approx([0.01, -0.02, 0.0, 0.03, 0.01])


def test_a_series_whose_dates_and_values_disagree_is_refused_rather_than_truncated(tmp_path):
    """A shape the writer refuses to produce, so its presence means something else
    wrote the row. Truncating to the shorter of the two and calling the result a return
    series is the guess this must not make."""
    import json

    from tradeflow.store.trials import TrialStore

    with TrialStore(tmp_path / "trials.db", journal_path=tmp_path / "j.jsonl") as store:
        store._conn.execute(
            "INSERT INTO trial_returns (trial_id, dates_json, returns_json) VALUES (?, ?, ?)",
            ("bad", json.dumps(DATES[:5]), json.dumps([0.01, 0.02])),
        )
        store._conn.commit()

        assert store.returns_for("bad") is None


def test_a_perfect_correlation_says_why_it_has_no_interval():
    """Found by running the installed wheel against two series that happened to be
    identical in shape. The Fisher-z transform is undefined at |r| = 1, and the renderer
    blamed the overlap for the missing interval — reporting the strongest possible
    result as though it rested on thin evidence."""
    values = [0.01 * ((i % 11) - 5) for i in range(120)]
    identical = compare_series([_entry("a", values), _entry("b", [v + 0.002 for v in values])])
    pair = identical["pairs"][0]

    assert pair["status"] == COMPARED
    assert pair["correlation"] == pytest.approx(1.0)
    assert pair["interval"] is None
    assert "perfectly correlated" in pair["interval_note"]

    printed = format_series_comparison(identical)
    assert "perfectly correlated" in printed
    assert "overlap" in printed  # the overlap is still reported, just not blamed


def test_an_ordinary_correlation_carries_an_interval_and_no_note():
    """Both directions: the note must not appear where an interval exists."""
    a, b, _ = _twins()

    pair = compare_series([_entry("a", a), _entry("b", b)])["pairs"][0]

    assert pair["interval"] is not None
    assert pair["interval_note"] == ""
