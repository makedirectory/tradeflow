"""Scaling a validated contract without distorting it.

The mode exists because measuring execution used to mean shrinking the book's caps by
hand until fills happened, which biases the sample toward low-priced names. So the one
thing these tests have to pin is *which numbers move* when capital shrinks:

    Fractions scale. Counts stay counts. Dollar strategy limits scale.
    Venue floors stay absolute.

Getting any clause wrong is a distortion in one direction or the other, and each is
asserted against a book where the wrong answer is visibly different from the right one.
"""

import pytest

from tradeflow.services import smallreal
from tradeflow.services.registry import STRATEGIES

# A contract whose every limit is set, so a rule that silently drops one is visible.
# Illustrative figures: round numbers chosen to make the arithmetic checkable by eye.
VALIDATED_CAPITAL = 200_000.0
VALIDATED_BOOK = {
    "max_positions": 8,
    "max_position_size": 10_000.0,
    "max_total_risk": 0.05,
    "max_gross_exposure": 0.80,
    "max_net_exposure": 0.30,
    "min_notional": 50.0,
}
SCALE = 0.05  # a twentieth


# --- the rule, clause by clause ----------------------------------------------------
def test_fractions_of_capital_are_carried_across_untouched():
    """They already scaled, because capital did. Multiplying them again would shrink
    them twice — a 0.80 gross cap becoming 0.04 — and bind far tighter than anything
    that was validated.

    The keys are named rather than read from `smallreal.FRACTIONS`: a test that iterates
    the list it is checking passes vacuously the moment somebody empties that list,
    which is exactly the edit this exists to catch.
    """
    scaled = smallreal.scale_book(VALIDATED_BOOK, SCALE)

    assert scaled["max_gross_exposure"] == 0.80
    assert scaled["max_net_exposure"] == 0.30
    assert scaled["max_total_risk"] == 0.05


def test_every_limit_belongs_to_exactly_one_clause_of_the_rule():
    """The rule is four clauses over one book, so a limit in two of them is scaled
    twice and a limit in none is carried by accident rather than by decision. Asserted
    against the shipped default book, so a limit added there without a clause fails
    here instead of silently taking the fallback."""
    from tradeflow.strategies.base import DEFAULT_POSITION_LIMITS

    clauses = (
        smallreal.FRACTIONS,
        smallreal.COUNTS,
        smallreal.DOLLAR_CEILINGS,
        smallreal.VENUE_FLOORS,
    )
    classified = [key for clause in clauses for key in clause]

    assert len(classified) == len(set(classified)), "a limit appears in two clauses"
    assert set(classified) == set(DEFAULT_POSITION_LIMITS), (
        "every declared limit needs a clause: "
        f"{set(DEFAULT_POSITION_LIMITS) ^ set(classified)} has none or does not exist"
    )


def test_the_position_count_is_the_books_shape_and_does_not_shrink():
    """How many names compete for the same budget is what is being measured. Shrinking
    it is the distortion this mode exists to remove, arriving through the front door."""
    scaled = smallreal.scale_book(VALIDATED_BOOK, SCALE)

    assert scaled["max_positions"] == 8


def test_a_dollar_ceiling_is_restated_so_it_still_binds():
    """$10,000 on a $200,000 book is a 5% ceiling. Left at $10,000 over $10,000 of
    capital it is above the whole book and binds nothing, so the validated contract
    had a ceiling and the run would have none."""
    scaled = smallreal.scale_book(VALIDATED_BOOK, SCALE)

    assert scaled["max_position_size"] == pytest.approx(500.0)
    # The proportion is what survives, which is the property rather than the number.
    assert scaled["max_position_size"] / (VALIDATED_CAPITAL * SCALE) == pytest.approx(
        VALIDATED_BOOK["max_position_size"] / VALIDATED_CAPITAL
    )


def test_the_venue_floor_is_not_scaled():
    """A broker's minimum does not get smaller because this run chose to. Scaling it
    down would claim the venue accepts orders it would refuse, which makes the run
    easier to fill and less real."""
    scaled = smallreal.scale_book(VALIDATED_BOOK, SCALE)

    assert scaled["min_notional"] == 50.0


def test_every_declared_limit_survives_the_scaling():
    """A limit this rule has no opinion about must be carried, not dropped: an omitted
    limit reads as unbounded, which is the one direction a book must never move by
    accident."""
    book = {**VALIDATED_BOOK, "some_future_limit": 3}

    scaled = smallreal.scale_book(book, SCALE)

    assert set(scaled) == set(book)
    assert scaled["some_future_limit"] == 3


def test_each_limit_reports_why_it_did_or_did_not_move():
    """ "This number is unchanged" and "this number was overlooked" are
    indistinguishable in a list of numbers."""
    payload = smallreal.contract(
        validated_capital=VALIDATED_CAPITAL,
        validated_book=VALIDATED_BOOK,
        scale=SCALE,
        capital=VALIDATED_CAPITAL * SCALE,
        capital_source="--scale 0.05",
    )

    treatment = payload["limit_treatment"]
    assert treatment["max_gross_exposure"] == "fraction of capital — unchanged"
    assert treatment["max_positions"] == "count — unchanged"
    assert treatment["max_position_size"] == "dollar ceiling — scaled"
    assert treatment["min_notional"] == "venue floor — not scaled"
    assert set(treatment) == set(payload["book"])


# --- the size of the run is stated, never chosen for you ----------------------------
def test_neither_scale_nor_capital_is_refused():
    """No default. This mode places real orders, so the amount it deploys must be one
    somebody chose — the same rule the dry run applies to its stated capital."""
    with pytest.raises(smallreal.ContractError, match="needs the size of the run stated"):
        smallreal.resolve_scale(validated_capital=VALIDATED_CAPITAL)


def test_both_scale_and_capital_are_refused_even_when_they_agree():
    """Two sources for one number is a thing to keep in step, and the arithmetic that
    checks they agree is the arithmetic that would be wrong."""
    with pytest.raises(smallreal.ContractError, match="one too many"):
        smallreal.resolve_scale(
            validated_capital=VALIDATED_CAPITAL, scale=SCALE, capital=VALIDATED_CAPITAL * SCALE
        )


def test_a_scale_derives_the_capital_and_a_capital_derives_the_scale():
    """Both directions of one relation, so the preflight can show the operator the
    number they did not type — which is where an order-of-magnitude slip shows up."""
    capital, scale, source = smallreal.resolve_scale(validated_capital=VALIDATED_CAPITAL, scale=SCALE)
    assert (capital, scale) == (10_000.0, 0.05) and source == "--scale 0.05"

    capital, scale, source = smallreal.resolve_scale(validated_capital=VALIDATED_CAPITAL, capital=10_000.0)
    assert (capital, scale) == (10_000.0, 0.05) and source == "--capital"


def test_a_scale_needs_a_validated_capital_to_be_a_fraction_of():
    """And says what to do instead, because a refusal an operator cannot act on gets
    worked around."""
    with pytest.raises(smallreal.ContractError, match="--capital with the amount"):
        smallreal.resolve_scale(validated_capital=None, scale=SCALE)


def test_a_capital_without_a_validated_one_reports_an_unknown_ratio():
    """Absent stays absent: with no denominator there is no ratio, and deriving one
    from the deployed amount alone would be inventing it."""
    capital, scale, _ = smallreal.resolve_scale(validated_capital=None, capital=10_000.0)

    assert capital == 10_000.0 and scale is None


@pytest.mark.parametrize("scale", [1.01, 2.0])
def test_a_scale_above_one_is_refused(scale):
    """Small-real runs the validated contract smaller, never larger. There is no
    evidence for the book above what was validated."""
    with pytest.raises(smallreal.ContractError, match="never larger"):
        smallreal.resolve_scale(validated_capital=VALIDATED_CAPITAL, scale=scale)


def test_the_whole_validated_contract_is_the_accepted_boundary():
    """Both directions: the guard rejects more than was validated and must accept
    exactly what was."""
    capital, scale, _ = smallreal.resolve_scale(validated_capital=VALIDATED_CAPITAL, scale=1.0)

    assert capital == VALIDATED_CAPITAL and scale == 1.0


def test_a_capital_above_the_validated_one_is_refused_too():
    """The same rule reached by the other spelling. A guard on one of two synonyms is
    a guard somebody routes around by typing the other."""
    with pytest.raises(smallreal.ContractError, match="never larger"):
        smallreal.resolve_scale(validated_capital=VALIDATED_CAPITAL, capital=VALIDATED_CAPITAL + 1)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_size_is_refused_in_either_spelling(bad):
    for kwargs in ({"scale": bad}, {"capital": bad}):
        with pytest.raises(smallreal.ContractError, match="must be positive"):
            smallreal.resolve_scale(validated_capital=VALIDATED_CAPITAL, **kwargs)


# --- a contract that cannot be scaled is refused, not guessed at --------------------
def test_a_config_with_no_recorded_book_is_refused():
    """Falling back to the strategy class's defaults would scale proportions nobody
    validated, which is the defect that made a config recording an eight-position
    validation trade one position, one step earlier in the chain."""
    with pytest.raises(smallreal.ContractError, match="no position_limits"):
        smallreal.validated_contract(
            config_capital=VALIDATED_CAPITAL,
            config_limits=None,
            strategy_class=STRATEGIES["demo_trend"],
        )


def test_a_partial_book_is_completed_from_the_class_the_way_a_run_would_be():
    """A config records only what was set, and the run trades the merge. Resolved
    through the same resolver that writes saved configs, so a contract read here and a
    config written elsewhere cannot disagree about the book."""
    capital, book = smallreal.validated_contract(
        config_capital=VALIDATED_CAPITAL,
        config_limits={"max_positions": 8},
        strategy_class=STRATEGIES["demo_trend"],
    )

    assert capital == VALIDATED_CAPITAL
    assert book["max_positions"] == 8
    # Filled from the class, not invented here.
    assert "max_total_risk" in book and "min_notional" in book


def test_a_dollar_ceiling_with_no_known_ratio_is_named_as_unscalable():
    """A ceiling left at its validated value over a fraction of the capital is above
    the whole book and binds nothing, so the run would be missing a limit the validated
    contract had — silently, in the direction that lets positions get too big."""
    assert smallreal.unscalable_ceilings(VALIDATED_BOOK, scale=None) == ("max_position_size",)
    # Both directions: with a ratio there is nothing to refuse, and a book with no
    # dollar ceiling is scalable without one.
    assert smallreal.unscalable_ceilings(VALIDATED_BOOK, scale=SCALE) == ()
    assert smallreal.unscalable_ceilings({**VALIDATED_BOOK, "max_position_size": None}, None) == ()


# --- the numbers a preflight has to show -------------------------------------------
def test_the_max_loss_envelope_is_the_risk_budget_at_this_capital():
    assert smallreal.max_loss_envelope(10_000.0, VALIDATED_BOOK) == pytest.approx(500.0)


def test_an_undeclared_risk_budget_has_no_envelope_rather_than_a_zero_one():
    """An undeclared budget is unbounded, and rendering that as zero would print the
    most reassuring possible number for the least bounded possible book."""
    assert smallreal.max_loss_envelope(10_000.0, {**VALIDATED_BOOK, "max_total_risk": None}) is None


def test_the_per_position_budget_takes_the_binding_one_of_the_two_bounds():
    """The ceiling and an even split across the names the book may hold. Whichever is
    smaller is what a position actually gets, and it is what decides whether a name is
    tradable at this scale at all."""
    scaled = smallreal.scale_book(VALIDATED_BOOK, SCALE)  # $500 ceiling, 8 positions

    # 10_000 * 0.80 gross / 8 positions = $1,000 a name; the $500 ceiling binds first.
    assert smallreal.per_position_budget(10_000.0, scaled) == pytest.approx(500.0)
    # And with the ceiling lifted, the split binds instead.
    assert smallreal.per_position_budget(10_000.0, {**scaled, "max_position_size": None}) == pytest.approx(
        1_000.0
    )


def test_a_book_that_bounds_nothing_has_no_per_position_budget():
    """Rather than a number derived from limits nobody set."""
    assert smallreal.per_position_budget(10_000.0, {"max_positions": None}) is None


def test_the_contract_states_the_bias_it_cannot_design_away():
    """A name priced above one position's budget cannot be traded at this scale, and the
    venue floor refuses more here than at full size. A reader must find that out from
    the report, not conclude from a clean-looking sample that it did not happen."""
    payload = smallreal.contract(
        validated_capital=VALIDATED_CAPITAL,
        validated_book=VALIDATED_BOOK,
        scale=SCALE,
        capital=10_000.0,
        capital_source="--scale 0.05",
    )

    assert "cannot be traded at this scale" in payload["not_covered"]
    assert "counted in the telemetry" in payload["not_covered"]
    assert payload["journaled"] is False


# --- the account has to be able to fund what was asked for -------------------------
def _account(equity=10_000.0, cash=None, buying_power=None):
    from tradeflow.brokers.base import AccountSnapshot

    return AccountSnapshot(
        cash=equity if cash is None else cash,
        equity=equity,
        buying_power=equity if buying_power is None else buying_power,
        portfolio_value=equity,
    )


def test_an_account_too_small_for_the_scaled_contract_is_named():
    """Sizing caps at whatever the account has rather than failing, so the run would
    trade a smaller book than the one whose proportions it claims to preserve — and the
    telemetry would look exactly like telemetry from the contract that was asked for."""
    shortfall = smallreal.account_shortfall(_account(equity=5_000.0), capital=10_000.0)

    assert shortfall is not None and "$5,000.00" in shortfall and "$10,000.00" in shortfall


def test_restricted_buying_power_is_a_shortfall_even_when_equity_is_ample():
    """Found by an independent review. The guard checked equity alone, and the sizer
    sizes off *buying power* — so an account with plenty of equity and restricted buying
    power passed, then traded a book smaller than the contract. The exact failure this
    guard exists to prevent, arriving one field over."""
    shortfall = smallreal.account_shortfall(_account(equity=50_000.0, buying_power=3_000.0), capital=10_000.0)

    assert shortfall is not None
    assert "buying power $3,000.00" in shortfall
    # And it does not accuse the fields that were fine.
    assert "equity" not in shortfall.split("Sizing caps")[0]


def test_restricted_cash_is_a_shortfall_too():
    shortfall = smallreal.account_shortfall(_account(equity=50_000.0, cash=1_000.0), capital=10_000.0)

    assert shortfall is not None and "cash $1,000.00" in shortfall


def test_an_account_that_can_fund_it_is_not_refused():
    """Both directions, including the exact boundary, on every field the cap applies
    to — a guard that rejects the account it exists to permit is indistinguishable from
    one that rejects everything."""
    assert smallreal.account_shortfall(_account(equity=10_000.0), capital=10_000.0) is None
    assert smallreal.account_shortfall(_account(equity=10_000.01), capital=10_000.0) is None
    assert (
        smallreal.account_shortfall(
            _account(equity=10_000.0, cash=10_000.0, buying_power=40_000.0), capital=10_000.0
        )
        is None
    )


def test_an_unreadable_account_is_not_treated_as_an_empty_one():
    """Refusing on an absent number would turn a broker hiccup into a claim about the
    balance. The preflight already reports that it could not be read."""
    assert smallreal.account_shortfall(None, capital=10_000.0) is None


# --- a book the run did not open --------------------------------------------------
def test_an_empty_account_gets_no_adoption_note():
    """Both directions: the note must not fire on the ordinary case."""
    assert smallreal.adopted_book_note(0, VALIDATED_BOOK) is None


def test_adopted_positions_are_reported_as_telemetry_from_another_size():
    """The engine adopts whatever is open, which is right — a process that believes it
    is flat cannot exit a position it owns. But their exits are fills at the size they
    were opened at, landing in a ledger that exists so two sizes are never averaged."""
    note = smallreal.adopted_book_note(3, VALIDATED_BOOK)

    assert note is not None
    assert "3 position(s) that this run did not open" in note
    assert "not at this run's" in note
    # Three of eight leaves room, so the sharper warning must not fire.
    assert "no new entry can be admitted" not in note


def test_an_adopted_book_that_already_fills_the_contract_says_so_sharply():
    """The case found by running it against a real paper account holding twelve
    positions under a scaled book of eight: every entry is refused as "book is full", so
    the session measures no entry execution while looking like it is running."""
    note = smallreal.adopted_book_note(12, VALIDATED_BOOK)

    assert "12 of 8" in note
    assert "no new entry can be admitted" in note
    assert "measure no entry execution at all" in note


def test_an_unbounded_book_cannot_be_filled_by_an_adoption():
    """Absent is not zero: a book with no position cap has no count for an adoption to
    reach, so the sharper warning has nothing to assert."""
    note = smallreal.adopted_book_note(12, {**VALIDATED_BOOK, "max_positions": None})

    assert note is not None and "no new entry can be admitted" not in note


@pytest.mark.parametrize("bad", [0.0, -100.0])
def test_a_config_recording_a_non_positive_validated_capital_is_refused(bad):
    """A fraction of zero is not a smaller book; a fraction of a negative one is not a
    book at all. Both were accepted, producing a contract that sized every position to
    nothing while the preflight rendered it as a normal run."""
    with pytest.raises(smallreal.ContractError, match="not an amount anything can have"):
        smallreal.validated_contract(
            config_capital=bad,
            config_limits=VALIDATED_BOOK,
            strategy_class=STRATEGIES["demo_trend"],
        )


def test_a_config_recording_no_capital_is_still_allowed():
    """Both directions: absent is not invalid. A config with no capital is refused only
    a `--scale`, and `--capital` states this run's own."""
    capital, book = smallreal.validated_contract(
        config_capital=None,
        config_limits=VALIDATED_BOOK,
        strategy_class=STRATEGIES["demo_trend"],
    )

    assert capital is None and book["max_positions"] == 8


# --- a recorded zero is not an absence ---------------------------------------------
# The project's rule is "absent is not zero". These are its inverse, which truthiness
# checks get wrong in the direction that matters: a limit explicitly set to zero is the
# most restrictive book there is, and reading it as unset renders it as the least.
def test_a_zero_risk_budget_is_an_envelope_of_zero_not_an_absent_one():
    """`max_total_risk: 0` permits no loss at all. It read as "not declared", which the
    preflight renders as *unbounded* — the most reassuring line printed for the most
    restrictive setting."""
    assert smallreal.max_loss_envelope(10_000.0, {**VALIDATED_BOOK, "max_total_risk": 0.0}) == 0.0
    # Both directions: an undeclared budget is still genuinely absent.
    assert smallreal.max_loss_envelope(10_000.0, {**VALIDATED_BOOK, "max_total_risk": None}) is None


def test_a_zero_gross_cap_deploys_nothing_rather_than_everything():
    """`or 1.0` read a declared cap of zero as an undeclared one, so a book permitted no
    exposure at all reported a positive per-position budget."""
    book = {**VALIDATED_BOOK, "max_gross_exposure": 0.0, "max_position_size": None}

    assert smallreal.per_position_budget(10_000.0, book) == 0.0
    # An undeclared cap still means the whole capital is deployable.
    undeclared = {**VALIDATED_BOOK, "max_gross_exposure": None, "max_position_size": None}
    assert smallreal.per_position_budget(10_000.0, undeclared) == pytest.approx(1_250.0)


def test_a_zero_dollar_ceiling_is_a_real_ceiling():
    assert smallreal.per_position_budget(10_000.0, {**VALIDATED_BOOK, "max_position_size": 0.0}) == 0.0


def test_a_book_that_may_hold_no_positions_is_already_full():
    """Definitionally full, and the warning was silenced for exactly that case."""
    note = smallreal.adopted_book_note(3, {**VALIDATED_BOOK, "max_positions": 0})

    assert note is not None and "no new entry can be admitted" in note
