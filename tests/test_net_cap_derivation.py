"""Choosing a directional cap from what a strategy actually ran at.

`max_net_exposure` is a fraction, and every number this project has used for it so far
was picked rather than derived. The point of this is not to produce a recommendation —
it is to make the trade-off visible: any cap below the observed maximum would have
changed the book that was validated.
"""

import pandas as pd

from tradeflow.analytics.exposure import MIN_SAMPLES, derive_net_cap, format_net_cap


def _exposure(values, gross_max=0.8):
    """Built the way the engine builds it, from the same helper the engine calls.

    A fixture that assembled this dict by hand could drift from what a backtest
    actually produces, and then every assertion here would be about a shape nothing
    emits.
    """
    from tradeflow.analytics.exposure import candidate_caps

    series = pd.Series(values, dtype="float64")
    stats = {
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }
    return {
        "samples": len(values),
        "skipped": 0,
        "gross_max": gross_max,
        "net_signed_mean": float(series.mean()),
        "net_abs": stats,
        "candidates": [
            {
                "cap": cap,
                "binding_rate": float((series > cap).mean()),
                "above_observed_max": cap >= stats["max"],
            }
            for cap in candidate_caps(stats)
        ],
    }


def _tilted(n=400, tilt=0.18):
    """A book with a persistent lean, shaped like a long/short one that is not neutral."""
    step = tilt / n
    return [tilt + i * step for i in range(n)]


# --- the recommendation -----------------------------------------------------------
def test_the_recommended_cap_would_never_have_bound():
    """The only choice that leaves the validated book intact. A cap that would have
    fired is a decision to trade something other than what was tested."""
    exposure = _exposure(_tilted())

    derivation = derive_net_cap(exposure)

    assert derivation["recommended"] >= exposure["net_abs"]["max"]


def test_there_is_always_a_candidate_above_the_observed_maximum():
    """A percentile ladder alone can sit entirely under a fat tail, and then reports
    that no cap leaves the book intact when one obviously does."""
    heavy_tail = [0.05] * 99 + [0.9]

    candidates = derive_net_cap(_exposure(heavy_tail))["candidates"]

    assert any(c["binding_rate"] == 0.0 for c in candidates)


def test_every_candidate_reports_what_it_would_have_done():
    """The number alone is the least useful half. A cap is not free for being loose:
    it either never binds, or the thing running is not the thing that was tested."""
    for candidate in derive_net_cap(_exposure(_tilted()))["candidates"]:
        assert 0.0 <= candidate["binding_rate"] <= 1.0


def test_a_tighter_cap_binds_more_often():
    """Monotonicity, because a ladder that did not order this way would be noise."""
    candidates = derive_net_cap(_exposure(_tilted()))["candidates"]
    rates = [c["binding_rate"] for c in candidates]

    assert rates == sorted(rates, reverse=True)


# --- a book with no tilt to cap ---------------------------------------------------
def test_a_neutral_book_is_told_it_has_nothing_to_cap(capsys):
    """The bug this found when it was first rendered: a perfectly neutral book has a
    p95 of zero, every candidate collapsed to zero, and the report concluded that *no*
    cap leaves the book intact — the exact opposite of the truth."""
    derivation = derive_net_cap(_exposure([0.0] * 60))

    assert derivation["no_tilt_carried"] is True
    assert derivation["candidates"] == []
    printed = "\n".join(format_net_cap(derivation))
    assert "never carried a measurable tilt" in printed
    assert "every positive cap is unbinding" in printed


def test_a_neutral_book_is_not_reported_as_undecidable(capsys):
    """Both directions on the same bug: the alarming message must not appear."""
    printed = "\n".join(format_net_cap(derive_net_cap(_exposure([0.0] * 60))))

    assert "No candidate leaves the validated book intact" not in printed


# --- honesty about the sample -----------------------------------------------------
def test_a_thin_history_says_so_before_it_says_anything_else():
    """A percentile from a handful of steps is exactly the number that gets quoted as
    evidence once somebody writes it down, so the caveat leads."""
    lines = format_net_cap(derive_net_cap(_exposure([0.1, 0.3, 0.2])))

    assert "NOT ENOUGH HISTORY" in lines[1]  # immediately under the heading


def test_a_long_history_carries_no_caveat():
    """Both directions: a warning that always fires is one nobody reads."""
    lines = format_net_cap(derive_net_cap(_exposure(_tilted(n=MIN_SAMPLES + 10))))

    assert not any("NOT ENOUGH HISTORY" in line for line in lines)


def test_a_systematic_lean_is_named_by_direction():
    """|net| says how big a tilt to allow; the signed mean says whether the strategy
    leans one way by construction, which is a different question."""
    exposure = _exposure(_tilted())
    exposure["net_signed_mean"] = -0.22

    assert "leans short" in "\n".join(format_net_cap(derive_net_cap(exposure)))


def test_no_exposure_is_reported_as_not_derivable_rather_than_zero():
    """Absent is not zero: no samples means no answer, not a cap of nothing."""
    derivation = derive_net_cap({})

    assert derivation["available"] is False
    assert "not derivable" in format_net_cap(derivation)[0]


def test_the_output_never_presents_itself_as_a_verdict():
    printed = "\n".join(format_net_cap(derive_net_cap(_exposure(_tilted()))))

    assert "starting point, not a verdict" in printed
