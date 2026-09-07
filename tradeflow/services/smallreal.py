"""The validated contract at reduced capital, with its proportions intact.

Observing execution — real fills, real slippage, real fees — means placing real orders,
and placing them at full size is not something anybody wants to do to find out whether
the plumbing works. So the book gets smaller. The question this module exists to answer
is *which numbers move when it does*, because the obvious answer is wrong and the wrong
answer is what caused the problem this mode was built to remove: a position ceiling
shrunk by hand until fills happened, which biases the sample toward low-priced names and
turns every high-price signal into an invisible non-trade. The strategy under observation
stops being the strategy that was validated, and the telemetry describes a book nobody
would trade.

**The rule, in one line:**

    Fractions scale. Counts stay counts. Dollar strategy limits scale.
    Venue floors stay absolute.

Each clause is a different unit, and applying one clause to another clause's limit is a
distortion in one direction or the other:

* ``max_gross_exposure``, ``max_net_exposure`` and ``max_total_risk`` are fractions *of
  capital*. They already scale, because capital did. Multiplying them by the shrink
  factor as well would shrink them twice — a 0.80 gross cap becoming 0.04 — and bind far
  tighter than anything that was validated.
* ``max_positions`` is the book's *shape*: how many names compete for the same budget,
  and therefore how diversified the thing being measured is. Shrinking it is precisely
  the distortion, arriving through the front door.
* ``max_position_size`` is in dollars. A $12,500 ceiling on a $250,000 book is a 5%
  ceiling; left at $12,500 over $12,500 of capital it is above the whole book and binds
  nothing, so the validated contract had a ceiling and the run would have none.
* ``min_notional`` is a venue fact, not a share of the book. Scaling it down would claim
  the broker accepts smaller orders than it does, which makes the run easier to fill and
  less real.

**The floor not scaling is a cost, taken deliberately and then measured.** It will refuse
more orders at small capital, and refuse them unevenly — expensive names first. So will
share granularity, which no amount of care avoids: a book with a few hundred dollars per
position cannot buy one share of a four-figure stock. Rather than pretend otherwise, the
preflight reports the price above which a name cannot be traded at all at this scale, and
the refusals are coded so they can be counted. The bias becomes a number in the report
instead of a silence in the sample.
"""

from typing import Any, Dict, Optional, Tuple

#: The rule, kept as a string because it is quoted in the preflight and in the docs and
#: must not be paraphrased differently in three places.
SCALING_RULE = (
    "Fractions scale. Counts stay counts. Dollar strategy limits scale. Venue floors stay absolute."
)

#: Fractions *of capital*. Already proportional, so the declared value is carried across
#: untouched — scaling it would apply the shrink a second time.
FRACTIONS = ("max_gross_exposure", "max_net_exposure", "max_total_risk")

#: The book's shape. How many names share the budget is what is being measured, not how
#: much money is in it.
COUNTS = ("max_positions",)

#: Dollar ceilings that express a share of the book, and therefore have to be restated
#: when the book changes size.
DOLLAR_CEILINGS = ("max_position_size",)

#: Dollar floors that express a fact about the venue. The account's minimum does not get
#: smaller because this run chose to.
VENUE_FLOORS = ("min_notional",)


class ContractError(ValueError):
    """A contract that cannot be scaled, with the reason and the way out.

    Every one of these is a refusal to start rather than a warning to read, because the
    thing on the other side places real orders. The message says what was missing and
    what to do instead: a refusal an operator cannot act on gets worked around.
    """


def validated_contract(
    *,
    config_capital: Optional[float],
    config_limits: Optional[Dict[str, Any]],
    strategy_class,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """What the config says was validated: ``(capital, book)``.

    The book is resolved through the one resolver that also writes saved configs, so a
    contract read here and a config written elsewhere cannot disagree about it. A
    partial ``position_limits`` is merged over the class defaults, because that is what
    the run would actually trade.

    A config that recorded no book at all is refused. Falling back to the class default
    would scale proportions nobody validated — which is the defect that made a config
    recording an eight-position validation trade one position, one step earlier in the
    chain. The capital may legitimately be absent, and the caller decides whether the
    scale it was given can survive that.
    """
    from tradeflow.strategies.base import resolve_book

    if config_capital is not None and float(config_capital) <= 0:
        raise ContractError(
            f"this config records a validated capital of {config_capital}, which is not "
            "an amount anything can have been validated at. A fraction of it is not a "
            "smaller book, it is a meaningless one — and the run would size every "
            "position to nothing while looking like it was trading.\n"
            "  Fix the capital in the config, or pass --capital to state this run's own."
        )
    if not config_limits:
        raise ContractError(
            "this config records no position_limits, so there is no validated book to "
            "take proportions from. Small-real trades the validated contract at reduced "
            "capital; without the contract it would be scaling the strategy class's "
            "defaults, which nobody validated.\n"
            "  Promote a config from a recorded trial (`trials promote <id> "
            "--save-config PATH`), which writes the book it was validated at."
        )
    return config_capital, resolve_book(strategy_class, config_limits)


def resolve_scale(
    *,
    validated_capital: Optional[float],
    scale: Optional[float] = None,
    capital: Optional[float] = None,
) -> Tuple[float, Optional[float], str]:
    """``(capital to deploy, shrink factor, how it was stated)``.

    Exactly one of ``scale`` and ``capital``. There is deliberately no default: a run
    that can place real orders must not deploy an amount nobody chose, which is the same
    rule the dry run applies to its stated capital and the same class of defect as a
    position book that quietly resolved to one.

    Both together are refused rather than reconciled even when they agree. Two sources
    for one number is a thing to keep in step, and the arithmetic that checks them
    agree is the arithmetic that would be wrong.

    ``scale`` needs a recorded validated capital to be a fraction *of* something. Where
    the config has none, the caller is told to state the capital outright.
    """
    if scale is None and capital is None:
        raise ContractError(
            "small-real needs the size of the run stated: pass --scale (a fraction of "
            "the validated capital) or --capital (an amount). There is no default, "
            "because this mode places real orders and the amount it deploys must be one "
            "somebody chose."
        )
    if scale is not None and capital is not None:
        raise ContractError(
            "--scale and --capital both state the size of this run, and two sources for "
            "one number is one too many. Pass whichever you meant."
        )

    if scale is not None:
        if validated_capital is None:
            raise ContractError(
                "--scale is a fraction of the capital this config was validated at, and "
                "this config records none, so there is nothing to take a fraction of.\n"
                "  Pass --capital with the amount to deploy instead."
            )
        if scale <= 0:
            raise ContractError(f"--scale must be positive, not {scale}")
        if scale > 1:
            raise ContractError(
                f"--scale {scale} would deploy more than the capital this config was "
                f"validated at (${validated_capital:,.2f}). Small-real runs the validated "
                "contract smaller, never larger; there is no evidence for the book above."
            )
        return float(validated_capital) * float(scale), float(scale), f"--scale {scale:g}"

    if capital <= 0:
        raise ContractError(f"--capital must be positive, not {capital}")
    if validated_capital is not None and capital > validated_capital:
        raise ContractError(
            f"--capital ${capital:,.2f} is more than the ${validated_capital:,.2f} this "
            "config was validated at. Small-real runs the validated contract smaller, "
            "never larger; there is no evidence for the book above."
        )
    # Absent stays absent. Without a validated capital there is no ratio, and reporting
    # one derived from the deployed amount alone would be inventing the denominator.
    derived = float(capital) / float(validated_capital) if validated_capital else None
    return float(capital), derived, "--capital"


def scale_book(validated_book: Dict[str, Any], scale: Optional[float]) -> Dict[str, Any]:
    """The validated book restated for a smaller amount of capital.

    Every key of the input appears in the output, so a limit this rule has no opinion
    about is carried rather than dropped — an omitted limit reads as unbounded, which is
    the one direction a book must never move by accident.

    ``scale`` of ``None`` means no ratio is known. The fractions and counts are still
    correct (they do not depend on one), so the book is returned with its dollar
    ceilings untouched; :func:`unscalable_ceilings` is what refuses that case before it
    can be run.
    """
    scaled = dict(validated_book)
    if scale is None:
        return scaled
    for key in DOLLAR_CEILINGS:
        value = scaled.get(key)
        if value is not None:
            scaled[key] = float(value) * scale
    return scaled


def unscalable_ceilings(validated_book: Dict[str, Any], scale: Optional[float]) -> Tuple[str, ...]:
    """Dollar ceilings that cannot be restated because no ratio is known.

    Reached only when a capital was named against a config that recorded none of its
    own. A ceiling left at its validated value over a fraction of the capital is above
    the whole book and binds nothing, so the run would be missing a limit the validated
    contract had — silently, and in the direction that lets positions get too big.
    """
    if scale is not None:
        return ()
    return tuple(key for key in DOLLAR_CEILINGS if validated_book.get(key) is not None)


def per_position_budget(capital: float, book: Dict[str, Any]) -> Optional[float]:
    """Roughly what one position gets at this size, or ``None`` if nothing bounds it.

    The smaller of the explicit dollar ceiling and an even split of the deployable book
    across the names it may hold. Deliberately approximate and labelled as such wherever
    it is shown: the sizer decides the real number from the strategy's risk budget and
    the symbol's stop distance, and it varies per name. What this is for is the order of
    magnitude — whether a position at this scale is worth hundreds or thousands — which
    is what decides whether the run can trade its universe at all.
    """
    candidates = []
    ceiling = book.get("max_position_size")
    if ceiling:
        candidates.append(float(ceiling))
    positions = book.get("max_positions")
    if positions:
        deployable = capital * float(book.get("max_gross_exposure") or 1.0)
        candidates.append(deployable / float(positions))
    return min(candidates) if candidates else None


def max_loss_envelope(capital: float, book: Dict[str, Any]) -> Optional[float]:
    """What the book gives up if every open position stops out, or ``None``.

    ``max_total_risk`` is a stop-weighted budget, so this is exactly what it means:
    the fraction of deployable capital lost when every stop fills at its price. That
    condition is the whole caveat and it is stated wherever this is printed — a gap
    through a stop fills below it, so the envelope is a floor on the loss and not a
    ceiling on it.

    ``None`` where no risk budget is declared, which is the honest answer. An undeclared
    budget is unbounded, and rendering that as zero would print the most reassuring
    possible number for the least bounded possible book.
    """
    risk = book.get("max_total_risk")
    return capital * float(risk) if risk else None


def contract(
    *,
    validated_capital: Optional[float],
    validated_book: Dict[str, Any],
    scale: Optional[float],
    capital: float,
    capital_source: str,
) -> Dict[str, Any]:
    """The scaled contract as data, for a renderer to print and a ledger to record.

    One object, so the preflight a human reads and the session header the telemetry
    keeps cannot describe the run differently. Every limit is reported beside the
    validated value it came from and the reason it did or did not move, because "this
    number is unchanged" and "this number was overlooked" are indistinguishable in a
    list of numbers.
    """
    scaled_book = scale_book(validated_book, scale)
    treatments = {
        **{key: "fraction of capital — unchanged" for key in FRACTIONS},
        **{key: "count — unchanged" for key in COUNTS},
        **{key: "dollar ceiling — scaled" for key in DOLLAR_CEILINGS},
        **{key: "venue floor — not scaled" for key in VENUE_FLOORS},
    }
    return {
        "mode": "small_real",
        "rule": SCALING_RULE,
        "validated_capital": validated_capital,
        "capital": capital,
        "capital_source": capital_source,
        "scale": scale,
        "validated_book": dict(validated_book),
        "book": scaled_book,
        "limit_treatment": {
            key: treatments.get(key, "carried unchanged — no rule for this limit") for key in scaled_book
        },
        "per_position_budget": per_position_budget(capital, scaled_book),
        "max_loss_envelope": max_loss_envelope(capital, scaled_book),
        # Said rather than left to be inferred from a clean-looking report. This is the
        # bias the mode cannot design away, only measure.
        "not_covered": (
            "a name priced above roughly one position's budget cannot be traded at this "
            "scale at all, and the venue floor refuses more orders here than it did at "
            "full size. Both refusals are counted in the telemetry; neither is a "
            "property of the strategy"
        ),
        "journaled": False,
    }


def adopted_book_note(held: int, book: Dict[str, Any]) -> Optional[str]:
    """What positions already on the account mean for this run, or ``None``.

    The engine adopts whatever the broker holds at start-up, which is right — a process
    that believes it is flat cannot exit a position it owns. But a book carried over
    from a full-size session is not this contract's book, and it lands in this run's
    telemetry: exits of those positions are fills at the *old* size, recorded in a
    ledger that exists precisely so two sizes are never averaged together.

    Reported rather than refused. On a restart the adopted positions *are* this run's
    own, and nothing here can tell the two cases apart; refusing would block the
    legitimate one. What must not happen is the operator not being told — most sharply
    when the adopted count already fills the book, because then no entry can be admitted
    and the session measures nothing while looking like it is running.
    """
    if not held:
        return None
    limit = book.get("max_positions")
    note = (
        f"this account already holds {held} position(s) that this run did not open. The "
        "engine will adopt them, so their exits will be recorded in this session's "
        "telemetry at whatever size they were opened at — not at this run's."
    )
    if limit and held >= int(limit):
        note += (
            f"\n  They already fill the scaled book ({held} of {limit}), so no new entry "
            "can be admitted and this session would measure no entry execution at all."
        )
    return note


def account_shortfall(account, capital: float) -> Optional[str]:
    """Why this account cannot fund the scaled contract, or ``None`` if it can.

    Sizing caps the account at the configured capital rather than replacing it, so an
    account holding less than the scaled contract asks for does not fail — it quietly
    trades something smaller, with all the proportions this module exists to preserve
    silently wrong. That is the one outcome worse than refusing to start, because the
    telemetry it produces looks exactly like telemetry from the contract that was asked
    for.

    **Every field the cap applies to is checked, not only equity.** The first version
    checked equity alone, and the sizer sizes off ``buying_power`` — so an account with
    ample equity and restricted buying power passed the guard and then traded a book
    smaller than the contract, which is the exact failure this exists to prevent
    arriving one field over. Whichever field binds is the one that decides the book, so
    the guard has to look at all of them.

    An unreadable account returns ``None``: the preflight already reports that it could
    not be read, and refusing on an absent number would turn a broker hiccup into a
    claim about the balance.
    """
    if account is None:
        return None
    short = {
        name: float(value)
        for name, value in (
            ("equity", getattr(account, "equity", None)),
            ("cash", getattr(account, "cash", None)),
            ("buying power", getattr(account, "buying_power", None)),
        )
        if value is not None and float(value) < capital
    }
    if not short:
        return None
    detail = ", ".join(f"{name} ${value:,.2f}" for name, value in short.items())
    return (
        f"this account cannot fund the ${capital:,.2f} this run is sized for: {detail}. "
        "Sizing caps at whatever the account has — and the sizer sizes off buying power "
        "— so the run would trade a smaller book than the one whose proportions it "
        "claims to preserve, and the telemetry would not say so.\n"
        "  Fund the account, or lower the scale to one it can carry."
    )
