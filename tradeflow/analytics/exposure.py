"""Choosing a directional cap from what a strategy actually ran at.

`max_net_exposure` is a fraction, and picking one from anything but the strategy's own
history is a guess dressed as a limit. This turns the exposure the backtest carried into
the only thing that can honestly inform that choice: a distribution, and — for any cap
you are considering — how often it would have bound.

The framing matters more than the arithmetic. **Any cap below the observed maximum would
have changed the book that was validated.** A cap is not free just because it is loose:
it either never binds, in which case it documents an intent rather than enforcing one, or
it binds, in which case the thing running is no longer the thing that was tested. This
reports both halves and picks nothing.

Research clock: reads a completed backtest, decides nothing about a live run.
"""

from typing import Any, Dict, List

#: Caps worth showing, as multiples of the observed p95. Below 1.0 bites into normal
#: operation; above ~1.5 is loose enough that it only catches an excursion.
_HEADROOMS = (1.0, 1.25, 1.5)

#: A little above the observed maximum, so there is always one candidate that provably
#: never binds. Without it a fat-tailed book offers only caps that would have changed it.
_MAX_HEADROOM = 1.1

#: Below this many samples a percentile is an anecdote. The derivation still runs — the
#: numbers are real — but the caveat leads, because a p95 from a handful of steps is the
#: kind of number that gets quoted as though it were evidence.
MIN_SAMPLES = 30


def candidate_caps(stats: Dict[str, float]) -> List[float]:
    """The caps worth evaluating, derived from the observed distribution alone.

    The max-based anchor guarantees at least one cap that provably never bound. A
    percentile-only ladder can sit entirely below a fat tail, and then reports that no
    cap leaves the book intact when one obviously does.
    """
    if not stats or stats.get("max", 0.0) <= 0:
        # A book that never carried a measurable tilt. Every positive cap is unbinding,
        # so offering candidates would invent a choice that does not exist.
        return []
    caps = {round(stats["p95"] * headroom, 4) for headroom in _HEADROOMS}
    caps.add(round(stats["max"] * _MAX_HEADROOM, 4))
    return sorted(cap for cap in caps if cap > 0)


def net_cap_candidates(exposure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Caps worth considering, each with what it would have done historically.

    Binding rates are computed where the samples live — during the backtest — and
    carried here as scalars. Keeping the full series on the result would put one row per
    step into anything that serializes it, for a number only ever read at a handful of
    caps.
    """
    return list((exposure or {}).get("candidates") or [])


def derive_net_cap(exposure: Dict[str, Any]) -> Dict[str, Any]:
    """What the history says about a directional cap, and what it does not.

    ``recommended`` is the smallest candidate that would never have bound — the only
    choice that leaves the validated book intact. It is a starting point, not a verdict:
    a cap that never binds is documentation, and tightening it is a deliberate decision
    to trade something different from what was tested.
    """
    stats = (exposure or {}).get("net_abs")
    if not stats:
        return {"available": False, "reason": "no exposure was sampled"}

    candidates = net_cap_candidates(exposure)
    never_binds = [c for c in candidates if c["binding_rate"] == 0.0]
    recommended = min(never_binds, key=lambda c: c["cap"]) if never_binds else None
    samples = exposure.get("samples", 0)
    return {
        "available": True,
        "samples": samples,
        "thin": samples < MIN_SAMPLES,
        "no_tilt_carried": stats["max"] <= 0,
        "observed": stats,
        "signed_mean": exposure.get("net_signed_mean"),
        "gross_max": exposure.get("gross_max"),
        "candidates": candidates,
        # None when every candidate would have bound: the honest answer is then "no cap
        # derived from this history leaves it unchanged", not the least-bad number.
        "recommended": recommended["cap"] if recommended else None,
    }


def format_net_cap(derivation: Dict[str, Any]) -> List[str]:
    """Render the derivation, leading with what it cannot tell you."""
    if not derivation.get("available"):
        return [f"  Net exposure: not derivable — {derivation.get('reason', 'no data')}."]

    observed, lines = derivation["observed"], []
    lines.append(f"\n=== Directional tilt actually carried ({derivation['samples']} steps) ===")
    if derivation.get("thin"):
        # Leads, because a percentile from a handful of steps is exactly the kind of
        # number that gets quoted as evidence once it is written down.
        lines.append(
            f"  NOT ENOUGH HISTORY — {derivation['samples']} steps is too few for a "
            f"percentile to mean anything. Read the max; ignore the rest."
        )
    lines.append(
        f"  |net| / equity        median {observed['median']:.1%}  p90 {observed['p90']:.1%}  "
        f"p95 {observed['p95']:.1%}  p99 {observed['p99']:.1%}  max {observed['max']:.1%}"
    )
    signed = derivation.get("signed_mean")
    if signed is not None:
        lean = "long" if signed > 0 else "short"
        lines.append(
            f"  signed mean           {signed:+.1%} — the book leans {lean} by construction"
            if abs(signed) > 0.01
            else f"  signed mean           {signed:+.1%} — no systematic lean"
        )
    if derivation.get("gross_max") is not None:
        lines.append(f"  gross max             {derivation['gross_max']:.1%} of equity")

    if derivation.get("no_tilt_carried"):
        lines.append(
            "\n  This book never carried a measurable tilt, so every positive cap is "
            "unbinding.\n  A cap here documents an intent; it does not constrain "
            "anything this history shows."
        )
        return lines

    lines.append("\n  A cap and what it would have done:")
    for candidate in derivation["candidates"]:
        rate = candidate["binding_rate"]
        effect = (
            "never binds — documents the intent, enforces nothing new"
            if rate == 0.0
            else f"would have bound on {rate:.1%} of steps — a different book from the validated one"
        )
        lines.append(f"    --max-net-exposure {candidate['cap']:.2f}   {effect}")

    if derivation["recommended"] is not None:
        lines.append(
            f"\n  Smallest cap that leaves the validated book intact: {derivation['recommended']:.2f}"
        )
    else:
        lines.append(
            "\n  No candidate leaves the validated book intact. Any cap from this "
            "history would have changed what was tested — decide deliberately."
        )
    lines.append(
        "  This is a starting point, not a verdict. A cap that never binds is "
        "documentation;\n  tightening it is a decision to trade something other than "
        "what was validated."
    )
    return lines
