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


def _service_default(name):
    """A service function's own default, read from its signature rather than restated.

    A test that hardcodes the value it expects agrees with itself: it kept passing
    while the two surfaces disagreed, because both halves of the comparison were
    written by hand.
    """
    import inspect

    from tradeflow.services import analysis

    return inspect.signature(analysis.run_walk_forward).parameters[name].default


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


#: Knobs whose *spelling* differs by transport but whose meaning is identical. argparse
#: cannot take a mapping, so the CLI offers the two narrowings a sweep actually needs as
#: flat flags; the tool takes the structures they parse into. A rename belongs here only
#: when both surfaces reach the same service argument — it is not a place to excuse a
#: knob one surface simply lacks.
_SCREEN_RENAMED = {"range": "param_ranges", "max_positions": "position_limits"}


@pytest.mark.parametrize("flag", ["range", "max_positions", "objective", "method", "max_evals", "seed"])
def test_a_screen_knob_the_cli_has_is_reachable_over_mcp(flag):
    """A screen configured differently on each surface is two screens. The direction
    that has bitten this project is a CLI flag with no MCP equivalent: an agent cannot
    notice the omission, it just screens the full range at the class's own book and
    reports a distribution for a search nobody asked for."""
    import inspect

    from tradeflow.mcp import server as mcp_server

    source = inspect.getsource(mcp_server)
    start = source.index("def run_screen(")
    signature = source[start : source.index(") -> Dict[str, Any]:", start)]

    assert f"{_SCREEN_RENAMED.get(flag, flag)}:" in signature


def test_every_screen_flag_the_cli_takes_has_an_mcp_counterpart():
    """Enumerated from the parser rather than listed by hand, so a flag added to one
    surface and not the other fails here instead of being written down."""
    import inspect

    from tradeflow.cli import build_parser
    from tradeflow.mcp import server as mcp_server

    screen = next(
        parser
        for action in build_parser()._subparsers._group_actions
        for name, parser in action.choices.items()
        if name == "screen"
    )
    cli_dests = {a.dest for a in screen._actions} - {
        "help",
        "func",
        # Transport-shaped, not run-shaped: how the CLI got its bars, where it prints,
        # and which recorded point it re-runs are not parameters of the screen itself.
        "json",
        "cache",
        "offline",
        "cache_dir",
        "config",
        "re_resolve_universe",
        "scanner",
        "scan_as_of",
        "confirm",
        "force",
        # The service resolves a universe from symbols; the scanner runs before it.
        "symbols",
    }
    source = inspect.getsource(mcp_server)
    begin = source.index("def run_screen(")
    signature = source[begin : source.index(") -> Dict[str, Any]:", begin)]
    mcp_names = {line.split(":")[0].strip() for line in signature.splitlines()[1:] if ":" in line}

    missing = {_SCREEN_RENAMED.get(dest, dest) for dest in cli_dests} - mcp_names
    assert missing == set(), f"MCP run_screen is missing {sorted(missing)}"


def test_a_renamed_screen_knob_actually_reaches_the_argument_it_is_renamed_to():
    """The rename table is the loophole in the test above: an entry could excuse a flag
    that reaches nothing. So follow the two that are renamed all the way to the service
    argument and check they arrive."""
    from tradeflow.cli import _screen_limits

    args = parse_cli(["screen", "--max-positions", "8"])
    args.config_position_limits = None

    assert _screen_limits(args) == {"max_positions": 8}
    assert dict(parse_cli(["screen", "--range", "fast_ema_period=5:9:1"]).range) == {
        "fast_ema_period": {"min": 5.0, "max": 9.0, "step": 1.0}
    }


def test_a_configs_book_and_an_explicit_one_do_not_silently_disagree():
    """Both sources exist, so the precedence has to be stated: the flag you typed wins
    over the file, and the file's other limits survive."""
    from tradeflow.cli import _screen_limits

    args = parse_cli(["screen", "--max-positions", "8"])
    args.config_position_limits = {"max_positions": 1, "max_gross_exposure": 0.9}

    assert _screen_limits(args) == {"max_positions": 8, "max_gross_exposure": 0.9}


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
        n_folds=_service_default("n_folds"),
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
    """Like for like: the construction, with both surfaces given the same folds."""
    assert _wf_cli_key(["walkforward", "--folds", "4"]) == _wf_service_key(n_folds=4)


def test_the_two_surfaces_agree_on_a_default_walk_forward():
    """The stronger claim, which used not to hold: `--folds` defaulted to None on the
    CLI and 4 in the service. Both build four folds — `build_folds` falls back to
    `n_folds or 4` — so a default run over each surface validated *identically* and
    keyed *differently*, and a walk-forward recorded over one was never found again
    over the other. A miss is not free: it journals a fresh trial and permanently
    raises the deflation bar for that family."""
    assert _wf_cli_key(["walkforward"]) == _wf_service_key()


def test_the_default_fold_count_is_one_value_across_every_surface():
    """Read from the signatures rather than restated here, so a surface that changes
    its default fails this instead of quietly disagreeing again."""
    import inspect

    from tradeflow.mcp import server as mcp_server
    from tradeflow.services import analysis

    cli_default = parse_cli(["walkforward"]).folds
    assert cli_default == _service_default("n_folds")
    assert cli_default == inspect.signature(analysis.run_draft_walk_forward).parameters["n_folds"].default

    source = inspect.getsource(mcp_server)
    for tool in ("run_walk_forward", "run_draft_walk_forward"):
        start = source.index(f"def {tool}(")
        signature = source[start : source.index(") -> Dict[str, Any]:", start)]
        line = next(ln for ln in signature.splitlines() if ln.strip().startswith("n_folds"))
        assert line.strip().rstrip(",").endswith("= None"), f"{tool} disagrees: {line.strip()}"


def test_the_two_surfaces_agree_on_the_book_a_walk_forward_validated():
    """The case that broke: `run_backtest` folded the book limits into its key and the
    walk-forward recipe did not, so two configs differing only in `max_positions`
    hashed alike — and the second was served the first's result, reporting a
    one-position validation as an eight-position book."""
    limits = {"max_positions": 8, "max_gross_exposure": 0.9}

    assert _wf_cli_key(["walkforward", "--folds", "4"], limits) == _wf_service_key(limits, n_folds=4)


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
@pytest.mark.parametrize("tool", ["run_backtest", "run_screen", "run_walk_forward", "run_causality_probes"])
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


# --- trial analytics reach both surfaces the same way -------------------------------
def _mcp_signature(name: str) -> str:
    """One MCP tool's parameter list, read out of the server's own source.

    The `def name(` prefix is stripped rather than skipped by line, because a tool
    whose parameters fit on one line has no second line and would read as taking no
    parameters at all — a parity check that passes by seeing nothing.
    """
    import inspect

    from tradeflow.mcp import server as mcp_server

    source = inspect.getsource(mcp_server)
    start = source.index(f"def {name}(") + len(f"def {name}(")
    return source[start : source.index(") -> Dict[str, Any]:", start)]


def _parameter_names(signature: str) -> set:
    """Parameter names out of a signature's source text.

    Split on commas at bracket depth zero rather than by line: a tool whose parameters
    fit on one line is one line, and `List[str]` and `Optional[Dict[str, Any]]` carry
    commas of their own.
    """
    names, depth, current = set(), 0, ""
    for char in signature:
        if char in "[({":
            depth += 1
        elif char in "])}":
            depth -= 1
        if char == "," and depth == 0:
            if ":" in current:
                names.add(current.split(":")[0].strip())
            current = ""
        else:
            current += char
    if ":" in current:
        names.add(current.split(":")[0].strip())
    return names


def _cli_subcommand(name: str):
    from tradeflow.cli import build_parser

    trials = next(
        parser
        for action in build_parser()._subparsers._group_actions
        for cmd, parser in action.choices.items()
        if cmd == "trials"
    )
    return next(
        parser
        for action in trials._subparsers._group_actions
        for cmd, parser in action.choices.items()
        if cmd == name
    )


@pytest.mark.parametrize(("command", "tool"), [("analyze", "analyze_trial"), ("compare", "compare_trials")])
def test_every_trial_analytics_flag_the_cli_takes_has_an_mcp_counterpart(command, tool):
    """Enumerated from the parser rather than listed by hand. A knob added to one
    surface and not the other fails here instead of being noticed by an agent that
    cannot notice anything — it just runs the default and reports the result."""
    cli_dests = {a.dest for a in _cli_subcommand(command)._actions} - {
        "help",
        "func",
        # Transport-shaped, not run-shaped: where the CLI prints and which database
        # file it opens are not parameters of the analysis.
        "json",
        "db",
    }
    signature = _mcp_signature(tool)
    mcp_names = _parameter_names(signature)

    assert cli_dests - mcp_names == set(), f"MCP {tool} is missing {sorted(cli_dests - mcp_names)}"


def test_the_minimum_overlap_default_is_one_value_across_every_surface():
    """Identical construction is not parity if the two callers reach it with different
    arguments. A comparison refusing below 60 on one surface and below some other
    number on the other is two different commands wearing one name — and the last time
    a default differed between surfaces, a walk-forward recorded over one was never
    found again over the other."""
    import inspect

    from tradeflow.analytics.series_comparison import MIN_OVERLAP
    from tradeflow.services.analysis import compare_trials

    cli_default = {a.dest: a.default for a in _cli_subcommand("compare")._actions}["min_overlap"]
    service_default = inspect.signature(compare_trials).parameters["min_overlap"].default

    assert cli_default == service_default == MIN_OVERLAP
    # Read from the MCP source rather than the imported symbol, so a literal typed in
    # place of the constant is caught rather than silently agreeing today.
    assert "min_overlap: int = SERIES_MIN_OVERLAP" in _mcp_signature("compare_trials")


def test_both_surfaces_refuse_a_partial_trade_table_by_default():
    """`allow_partial` has to default the same way on both. A capped table summed
    silently on one surface and refused on the other means the same trial answers the
    same question two different ways depending on who asked."""
    import inspect

    from tradeflow.services.analysis import analyze_trial

    cli_default = {a.dest: a.default for a in _cli_subcommand("analyze")._actions}["allow_partial"]
    service_default = inspect.signature(analyze_trial).parameters["allow_partial"].default

    assert cli_default is False and service_default is False
    assert "allow_partial: bool = False" in _mcp_signature("analyze_trial")


def test_the_trial_store_is_opened_one_way():
    """It was opened three ways: the CLI's, the analysis service's and the MCP
    server's, each free to differ on *which* journal it indexed — the one decision
    they must never disagree about, because the multiple-testing correction rests on
    there being one journal."""
    import inspect

    from tradeflow.cli import _open_trial_store as cli_opener
    from tradeflow.mcp.server import _trial_store as mcp_opener
    from tradeflow.services.analysis import _open_trial_store as service_opener

    for opener in (cli_opener, service_opener, mcp_opener):
        assert "open_trial_store" in inspect.getsource(opener), (
            f"{opener.__qualname__} does not delegate to the shared opener"
        )


def test_all_three_openers_reach_the_same_journal(tmp_path, monkeypatch):
    """And the test that actually compares them, rather than three that each pass.

    The redirect is the case that matters: `store.trials.default_journal_path()` and
    `services.audit.DEFAULT_TRIAL_JOURNAL` resolve alike in production and differ the
    moment a journal is redirected, which is every test in this suite.
    """
    from tradeflow.cli import _open_trial_store as cli_opener
    from tradeflow.mcp.server import _trial_store as mcp_opener
    from tradeflow.services import audit
    from tradeflow.services.analysis import _open_trial_store as service_opener

    journal = tmp_path / "redirected.jsonl"
    journal.write_text("")
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal, raising=False)

    paths = []
    for opener in (cli_opener, service_opener, mcp_opener):
        with opener() as store:
            paths.append((store.journal_path, store.db_path))

    assert len(set(paths)) == 1, f"the three openers reached different journals: {paths}"
    assert paths[0][0] == journal


# --- review findings ----------------------------------------------------------------
def test_the_audit_record_names_the_knob_that_changed_the_answer():
    """Review finding. `compare_trials` logged `trial_ids` and `min_overlap` but not
    `across_accounting` — the one parameter deciding whether cross-era pairs were
    computed or refused. An audit trail that cannot tell a default comparison from one
    that deliberately crossed an accounting boundary is not an audit trail of that
    decision. `analyze_trial` logs its `allow_partial`; this now matches."""
    import inspect

    from tradeflow.mcp import server as mcp_server

    source = inspect.getsource(mcp_server)
    body = source[source.index("def compare_trials(") : source.index("def best_trials(")]
    inputs = body[body.index("inputs = {") : body.index("with _trial_store()")]

    for knob in ("trial_ids", "min_overlap", "across_accounting"):
        assert knob in inputs, f"compare_trials does not journal {knob}"


def test_promote_and_campaign_read_the_same_journal():
    """Review finding. `trials campaign` took `--journal` and `trials promote`, which
    embeds the very same block, did not — so `_promote_trial` always read the default
    journal however `--db` was pointed. Aimed at an archived store, `campaign` showed
    the recipe while `promote` wrote `recipe.available: false` into the config: one
    block, produced two ways, disagreeing."""
    for command in ("promote", "campaign"):
        dests = {a.dest for a in _cli_subcommand(command)._actions}
        assert "journal" in dests, f"trials {command} cannot be pointed at a journal"
        assert "db" in dests


def test_a_promoted_config_finds_the_recipe_in_a_redirected_journal(tmp_path, monkeypatch):
    """And the test that proves the flag is wired rather than merely accepted: promote
    against a journal that is not the default, and the recipe has to arrive."""
    import json
    from datetime import datetime

    import pandas as pd

    from tradeflow.cli import build_parser
    from tradeflow.services.analysis import walk_forward_recipe
    from tradeflow.services.audit import journal_trial
    from tradeflow.store.trials import db_path_for_journal

    journal = tmp_path / "elsewhere.jsonl"
    trial_id = journal_trial(
        "walkforward",
        strategy="demo_trend",
        symbols=["AAA"],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 4, 30),
        params={"fast_ema_period": 9},
        metrics={"sharpe_ratio": 1.0},
        extra={"promotable": True},
        returns=pd.Series([0.001] * 90, index=pd.date_range("2024-01-02", periods=90)),
        dedup_params=walk_forward_recipe(
            mode="anchored",
            n_folds=None,
            train_days=252,
            test_days=63,
            embargo_days=5,
            holdout_days=60,
            method="grid",
            objective="sharpe_ratio",
            max_evals=50,
            seed=42,
            cost_key={},
            limits=None,
        ),
        path=journal,
    )
    # The default journal is somewhere else entirely, which is the whole point.
    monkeypatch.setattr(
        "tradeflow.services.audit.DEFAULT_TRIAL_JOURNAL", tmp_path / "default.jsonl", raising=False
    )
    out = tmp_path / "promoted.json"
    args = build_parser().parse_args(
        [
            "trials",
            "promote",
            trial_id,
            "--save-config",
            str(out),
            "--db",
            str(db_path_for_journal(journal)),
            "--journal",
            str(journal),
        ]
    )
    args.func(args)

    recipe = json.loads(out.read_text())["provenance"]["campaign"]["recipe"]
    assert recipe["available"] is True
    assert recipe["validation"]["train_days"] == 252
