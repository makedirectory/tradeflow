"""The CLI and the MCP server must identify a run the same way.

`services.analysis._find_cached_trial` promises that "a trial run over MCP and one run
over the CLI dedup against each other identically". Nothing asserted it, and it broke
the moment book limits were folded into one surface's dedup key and not the other's —
a trial recorded from the CLI would silently stop being found from MCP, and every
memoization and multiple-testing count would quietly split in two.

The two surfaces build the key separately on purpose: the CLI has argparse namespaces
and a bar-cache vintage, the service has plain arguments. That is exactly why the
equivalence needs a test rather than an assumption.
"""

import pytest

from tradeflow.cli import _dedup_params, parse_cli
from tradeflow.services.analysis import _cost_key as service_cost_key
from tradeflow.services.analysis import limits_key
from tradeflow.services.registry import STRATEGIES

_PARAMS = {"fast_ema_period": 10, "slow_ema_period": 30}


def _cli_key(argv, limits):
    return _dedup_params(_PARAMS, parse_cli(argv), None, limits)


def _service_key(limits, *, gross=False, commission_bps=1.0, impact_eta=0.3, borrow_bps=50.0):
    """Built exactly as `run_backtest` builds it."""
    return {
        **_PARAMS,
        "_cost": service_cost_key(gross, commission_bps, impact_eta, borrow_bps),
        **limits_key(limits),
    }


def test_the_two_surfaces_agree_on_a_default_run():
    assert _cli_key(["backtest"], None) == _service_key(None)


def test_the_two_surfaces_agree_when_limits_are_declared():
    """The case that broke: limits reached the CLI's key and not the service's."""
    limits = {"max_positions": 8, "max_position_size": 1200.0, "max_gross_exposure": 0.9}

    assert _cli_key(["backtest"], limits) == _service_key(limits)


def test_the_two_surfaces_agree_on_cost_assumptions():
    argv = ["backtest", "--commission-bps", "7.5", "--impact-eta", "0.9", "--borrow-bps", "120"]

    assert _cli_key(argv, None) == _service_key(None, commission_bps=7.5, impact_eta=0.9, borrow_bps=120.0)


def test_the_two_surfaces_agree_on_a_gross_run():
    assert _cli_key(["backtest", "--gross"], None) == _service_key(None, gross=True)


def test_limits_actually_change_the_key():
    """Both directions: an equivalence test passes trivially if neither side folds
    anything, so the fold has to be shown to do something first."""
    tight = _service_key({"max_gross_exposure": 0.5})
    loose = _service_key({"max_gross_exposure": 0.9})

    assert tight != loose


def test_unset_limits_key_exactly_as_before_they_existed():
    """Every trial recorded before this fold must still match itself, or the whole
    store re-keys and stops finding its own history."""
    assert limits_key(None) == {}
    assert limits_key({"max_gross_exposure": None, "min_notional": None}) == {}


def test_a_real_strategy_s_limits_round_trip_through_both_surfaces():
    """Against an actual strategy's declared limits rather than a hand-built dict, so
    a change to the limit set cannot pass this while breaking the surfaces."""
    limits = STRATEGIES["demo_trend"].create_with_defaults().position_limits()

    assert _cli_key(["backtest"], limits) == _service_key(limits)


@pytest.mark.parametrize("flag,value", [("--max-positions", "8"), ("--max-gross-exposure", "0.9")])
def test_a_typed_cli_limit_reaches_the_shared_key(flag, value):
    """The flags only matter if what they set arrives in the identity."""
    from tradeflow.cli import _apply_limit_overrides

    args = parse_cli(["backtest", flag, value])
    strategy = STRATEGIES["demo_trend"].create_with_defaults()
    _apply_limit_overrides(args, strategy)

    assert _dedup_params(_PARAMS, args, None, strategy.position_limits()) != _cli_key(["backtest"], None)


# --- the validated book must be the configured book -------------------------------
def test_a_walk_forward_fold_trades_the_book_its_config_asked_for():
    """`position_limits` is not a tunable param, so building a candidate from its
    params alone silently drops it — a config asking for eight positions was validated
    at whatever the strategy class declares, which was one. Validation and deployment
    then describe different books, which is the one thing a walk-forward exists to
    rule out."""
    from tradeflow.optimization.walk_forward import WalkForwardValidator

    cls = STRATEGIES["demo_trend"]
    tuned = {k: cls.create_with_defaults().config[k] for k in cls.PARAM_RANGES}
    validator = WalkForwardValidator(
        cls, None, position_limits={"max_positions": 8, "max_gross_exposure": 0.9}
    )

    limits = validator._make(tuned).position_limits()

    assert limits["max_positions"] == 8
    assert limits["max_gross_exposure"] == 0.9


def test_a_walk_forward_without_config_limits_is_unchanged():
    """Both directions: no config limits must leave the strategy exactly as it was, or
    every existing walk-forward silently re-bases."""
    from tradeflow.optimization.walk_forward import WalkForwardValidator

    cls = STRATEGIES["demo_trend"]
    tuned = {k: cls.create_with_defaults().config[k] for k in cls.PARAM_RANGES}

    assert (
        WalkForwardValidator(cls, None)._make(tuned).position_limits()
        == cls.create_with_defaults().position_limits()
    )


def test_config_limits_do_not_leak_between_candidates():
    """Each fold builds its own candidate; a shared mutable limits dict would let one
    candidate's overrides follow the next."""
    from tradeflow.optimization.walk_forward import WalkForwardValidator

    cls = STRATEGIES["demo_trend"]
    tuned = {k: cls.create_with_defaults().config[k] for k in cls.PARAM_RANGES}
    validator = WalkForwardValidator(cls, None, position_limits={"max_positions": 8})

    first = validator._make(tuned)
    first.config["position_limits"]["max_positions"] = 99

    assert validator._make(tuned).position_limits()["max_positions"] == 8


# --- MCP is a transport over the same service -------------------------------------
def test_every_mcp_backtest_argument_is_one_the_service_accepts():
    """MCP is a transport: parse, call one service function, render. An argument it
    accepts that the service does not is a call that fails at runtime, and an agent
    cannot notice a stale signature — it acts on one."""
    import inspect

    from tradeflow.mcp import server as mcp_server
    from tradeflow.services import analysis

    source = inspect.getsource(mcp_server)
    start = source.index("def run_backtest(")
    signature = source[start : source.index(") -> Dict[str, Any]:", start)]
    names = {
        line.split(":")[0].strip()
        for line in signature.splitlines()[1:]
        if ":" in line and not line.strip().startswith("#")
    }
    service = set(inspect.signature(analysis.run_backtest).parameters)

    assert names - service - {"strategy", "symbols", "start", "end"} == set()


def test_mcp_exposes_the_fill_assumption_knob():
    """It is a research diagnostic and MCP is the research surface; a strategy whose
    gain concentrates in target exits is exactly what an agent should be able to
    stress without a human running the CLI for it."""
    import inspect

    from tradeflow.mcp import server as mcp_server

    assert "take_profit_margin_bps" in inspect.getsource(mcp_server)


def test_no_mcp_description_sends_an_installed_reader_to_a_checkout():
    """Descriptions are an interface, not documentation. An installed copy has no
    `main.py` and no Makefile, so a bare checkout-only command is a dead end — and the
    reader here is an agent, which cannot go looking for the file and give up."""
    import inspect
    import re

    from tradeflow.mcp import server as mcp_server

    # Backticked commands only — prose like "make validated code available" is not an
    # instruction, and a matcher that cannot tell them apart teaches people to ignore it.
    invocations = re.findall(r"`+\s*(python main\.py|make)\b[^`]*`+", inspect.getsource(mcp_server))
    source = inspect.getsource(mcp_server)
    for match in re.finditer(r"`+\s*(?:python main\.py|make)\b[^`]*`+", source):
        line_start = source.rfind("\n", 0, match.start()) + 1
        line_end = source.find("\n", match.end())
        line = source[line_start:line_end]
        assert "tradeflow " in line, f"checkout-only instruction with no installed form: {line.strip()}"
    assert invocations or True  # the assertion above is the check; this documents intent


def test_the_journal_has_one_location():
    """`services.audit` writes it and `store.trials` indexes it, from two constants kept
    in step by a comment. If they diverged the store would index a different file from
    the one being written, and the multiple-testing correction would deflate against
    half its evidence — with nothing erroring, because both paths are valid."""
    from tradeflow.services.audit import DEFAULT_TRIAL_JOURNAL
    from tradeflow.store.trials import DEFAULT_JOURNAL_PATH

    assert DEFAULT_TRIAL_JOURNAL == DEFAULT_JOURNAL_PATH


# --- a walk-forward is memoized by its recipe, and both surfaces build one ---------
def _wf_service_key(limits=None, **overrides):
    """Built exactly as `run_walk_forward` builds it."""
    from tradeflow.services.analysis import walk_forward_recipe

    kwargs = dict(
        mode="anchored",
        n_folds=4,
        train_days=None,
        test_days=None,
        embargo_days=None,
        holdout_days=0,
        method="grid",
        objective="sharpe_ratio",
        max_evals=50,
        seed=42,
        cost_key=service_cost_key(False, 1.0, 0.3, 50.0),
        limits=limits,
    )
    return walk_forward_recipe(**{**kwargs, **overrides})


def _wf_cli_key(argv, limits=None):
    from tradeflow.cli import _walkforward_recipe

    args = parse_cli(argv)
    args.config_position_limits = limits
    return _walkforward_recipe(args, None)


def test_the_two_surfaces_agree_on_the_same_walk_forward():
    """Like for like: `--folds` defaults to None on the CLI and 4 in the service, so
    the two surfaces' *defaults* differ. That is a defaults question, not a
    key-construction one — this asserts the construction."""
    assert _wf_cli_key(["walkforward", "--folds", "4"]) == _wf_service_key()


def test_the_two_surfaces_agree_on_the_book_a_walk_forward_validated():
    """The case that broke: `run_backtest` folded the book limits into its key and the
    walk-forward recipe did not, so two configs differing only in `max_positions`
    hashed alike — and the second was served the first's result, reporting a
    one-position validation as an eight-position book."""
    limits = {"max_positions": 8, "max_gross_exposure": 0.9}

    assert _wf_cli_key(["walkforward", "--folds", "4"], limits) == _wf_service_key(limits)


def test_the_book_actually_changes_a_walk_forward_s_identity():
    """Both directions: an equivalence that folds nothing on either side passes
    trivially, so the fold has to be shown to do something first."""
    from tradeflow.store.trials import params_hash

    one = _wf_service_key({"max_positions": 1})
    eight = _wf_service_key({"max_positions": 8})

    assert params_hash(one) != params_hash(eight)


def test_a_walk_forward_without_limits_keys_exactly_as_before_they_existed():
    """Every validation already in the store must still find itself."""
    assert "_limits" not in _wf_service_key(None)


# --- MCP is a transport over the same service -------------------------------------
@pytest.mark.parametrize("tool", ["run_backtest", "run_walk_forward"])
def test_every_mcp_argument_is_one_the_service_accepts(tool):
    """MCP is a transport: parse, call one service function, render. An argument it
    accepts that the service does not is a call that fails at runtime, and an agent
    cannot notice a stale signature — it acts on one."""
    import inspect

    from tradeflow.mcp import server as mcp_server
    from tradeflow.services import analysis

    source = inspect.getsource(mcp_server)
    start = source.index(f"def {tool}(")
    signature = source[start : source.index(") -> Dict[str, Any]:", start)]
    names = {
        line.split(":")[0].strip()
        for line in signature.splitlines()[1:]
        if ":" in line and not line.strip().startswith("#")
    }
    service = set(inspect.signature(getattr(analysis, tool)).parameters)

    assert names - service - {"strategy", "symbols", "start", "end"} == set()


@pytest.mark.parametrize("flag", ["benchmark", "position_limits"])
def test_a_walk_forward_knob_the_cli_has_is_reachable_over_mcp(flag):
    """A gate that cannot be configured over a surface is not a stricter gate on that
    surface — it is one that never runs. Without `benchmark` every fold reports
    `benchmark_available: False`, so the benchmark-relative prerequisites come back
    unevaluated rather than failed, and nothing says so."""
    import inspect

    from tradeflow.mcp import server as mcp_server
    from tradeflow.services import analysis

    assert flag in inspect.signature(analysis.run_walk_forward).parameters

    source = inspect.getsource(mcp_server)
    start = source.index("def run_walk_forward(")
    body = source[start : source.index("return _logged(", start)]
    assert f"{flag}={flag}" in body, f"MCP accepts no {flag}, or accepts it and drops it"
