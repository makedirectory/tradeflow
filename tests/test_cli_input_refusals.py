"""A newcomer's likeliest mistakes are refused with a message that says what to type.

Each of these used to reach the user as a stack trace, a credentials wall hiding the
real problem, or no refusal at all - a run over a nonsense input that then printed a
recovery command which could never work.
"""

from datetime import datetime, timedelta

import pytest

from tradeflow import cli
from tradeflow.settings import CLI_NAME, SettingsError
from tradeflow.store.bars import CacheMiss
from tradeflow.utils.timeutils import NEW_YORK


def _refusal(argv, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.parse_cli(argv)
    assert exited.value.code == 2
    return capsys.readouterr().err


def _no_keys(monkeypatch):
    def _refuse():
        raise SettingsError("Missing Alpaca credentials")

    monkeypatch.setattr("tradeflow.settings.load_settings", _refuse)


# --- --offline needs a cache, not keys ----------------------------------------------


def test_an_offline_research_command_without_keys_reports_the_missing_bars(monkeypatch, tmp_path):
    """Every research command built the trading client and discarded it, so a cache-only
    run demanded broker keys and said nothing about the bars it could not find."""
    _no_keys(monkeypatch)
    args = cli.parse_cli(["scan", "--offline", "--symbols", "AAA", "--cache-dir", str(tmp_path / "bars")])
    with pytest.raises(CacheMiss) as miss:
        args.func(args)
    assert "AAA" in str(miss.value) and "cache warm" in str(miss.value)


def test_an_online_research_command_still_asks_for_keys(monkeypatch):
    """The other direction: only a cache-only run may skip them."""
    _no_keys(monkeypatch)
    args = cli.parse_cli(["scan", "--symbols", "AAA"])
    with pytest.raises(SettingsError):
        args.func(args)


def test_research_commands_never_build_a_trading_client(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "build_data_and_broker", lambda **kw: pytest.fail("a broker was built"))
    args = cli.parse_cli(["scan", "--offline", "--symbols", "AAA", "--cache-dir", str(tmp_path / "bars")])
    with pytest.raises(CacheMiss):
        args.func(args)


# --- windows ------------------------------------------------------------------------


def test_a_reversed_window_is_refused_before_anything_runs(capsys):
    """It surfaced as a cache miss with a silently re-ordered warm window, and `cache warm`
    over it reported an empty cache as already cached."""
    for command in (["backtest"], ["verdict"], ["cache", "warm", "--symbols", "AAA"]):
        err = _refusal([*command, "--start", "2024-06-01", "--end", "2024-01-01"], capsys)
        assert "ends (2024-01-01) before it starts (2024-06-01)" in err
        assert err.startswith(f"{CLI_NAME} {command[0]}: error:")


def test_a_window_starting_in_the_future_is_refused(capsys):
    """Its warm command could never succeed."""
    start = (datetime.now(NEW_YORK) + timedelta(days=400)).strftime("%Y-%m-%d")
    end = (datetime.now(NEW_YORK) + timedelta(days=500)).strftime("%Y-%m-%d")
    assert "in the future" in _refusal(["backtest", "--start", start, "--end", end], capsys)


def test_an_ordinary_window_and_an_end_after_today_are_accepted():
    """The boundary: a window may end after now (it is clipped by the data), and the
    default window must pass, or every command without dates would refuse."""
    from tradeflow.utils.timeutils import window_refusal

    cli.parse_cli(["backtest", "--start", "2024-01-02", "--end", "2024-01-03"])
    cli.parse_cli(["backtest"])
    now = datetime(2024, 6, 1, tzinfo=NEW_YORK)
    assert (
        window_refusal(datetime(2024, 5, 1, tzinfo=NEW_YORK), datetime(2024, 7, 1, tzinfo=NEW_YORK), now)
        is None
    )
    assert window_refusal(datetime(2024, 6, 1, tzinfo=NEW_YORK), datetime(2024, 7, 1, tzinfo=NEW_YORK), now)


def test_a_service_caller_gets_the_same_refusal_and_nothing_is_journaled(tmp_path, monkeypatch):
    """Over MCP a verdict on a reversed window was recorded as a trial, raising the
    multiple-testing count for a run that measured nothing."""
    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import analysis, audit

    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal)
    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=300, freq="1D"))
    start, end = cli._date("2024-06-01"), cli._date("2024-01-01")
    for run in (analysis.run_verdict, analysis.run_backtest, analysis.run_walk_forward):
        with pytest.raises(ValueError, match="before it starts"):
            run(client, "demo_trend", ["AAA", "BBB"], start, end)
    assert not journal.exists() or not journal.read_text().strip()


# --- names, dates and amounts -------------------------------------------------------


def test_a_mistyped_scanner_is_refused_with_a_suggestion(capsys):
    """It was a fourteen-line traceback from inside the scan; a mistyped strategy has
    always been an argparse refusal listing the choices."""
    err = _refusal(["scan", "--scanner", "demo_volum"], capsys)
    assert "unknown scanner 'demo_volum'" in err
    assert "Did you mean 'demo_volume'?" in err
    assert "Traceback" not in err


def test_none_and_an_installed_scanner_are_accepted():
    assert cli.parse_cli(["backtest", "--scanner", "none"]).scanner == "none"
    assert cli.parse_cli(["backtest", "--scanner", "demo_volume"]).scanner == "demo_volume"


@pytest.mark.parametrize("value", ["2024/01/02", "01-02-2024", "2025-13-45"])
def test_a_malformed_date_names_the_format_not_a_private_function(value, capsys):
    err = _refusal(["backtest", "--start", value], capsys)
    assert "_date" not in err
    assert "YYYY-MM-DD" in err and repr(value) in err


@pytest.mark.parametrize(
    "argv",
    [
        ["backtest", "--capital", "0"],
        ["verdict", "--capital", "-5"],
        ["backtest", "--capital", "nan"],
        ["backtest", "--max-positions", "0"],
        ["allocate", "--max-positions", "-1"],
    ],
)
def test_an_amount_that_cannot_fund_a_book_is_refused(argv, capsys):
    """Nothing refused these; a run sized every position from them."""
    assert "error: argument" in _refusal(argv, capsys)


def test_the_smallest_meaningful_amounts_are_accepted():
    args = cli.parse_cli(["backtest", "--capital", "0.01", "--max-positions", "1"])
    assert (args.capital, args.max_positions) == (0.01, 1)


# --- rejected keys ------------------------------------------------------------------


def test_rejected_keys_end_the_command_with_their_message_not_a_stack(monkeypatch):
    """The fix text was good and sat under sixty lines of the vendor SDK's stack."""
    from argparse import Namespace

    from tradeflow.brokers.errors import AuthenticationError

    def _rejected(args):
        raise AuthenticationError("Alpaca rejected these credentials (HTTP 401).")

    monkeypatch.setattr(cli, "parse_cli", lambda: Namespace(func=_rejected))
    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == "Alpaca rejected these credentials (HTTP 401)."


# --- a hand-edited config -----------------------------------------------------------


def test_a_typo_d_config_key_is_named_with_the_parameter_it_left_unset(tmp_path):
    """`"parms"` for `"params"` reached the reader as `KeyError: 'slow_ema_period'` under
    causes - renamed strategies, missing packages - that pointed away from the typo."""
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"strategy": "demo_trend", "parms": {"fast_ema_period": 3}}))
    args = cli.parse_cli(["backtest", "--config", str(path)])
    with pytest.raises(SystemExit) as exited:
        cli.apply_run_config(args)
    message = str(exited.value.code)
    assert "'parms' is not a config key - did you mean 'params'?" in message
    assert "which the config's params do not set" in message


def test_every_key_the_writer_produces_is_one_the_hint_recognises(tmp_path):
    """The writer and the hint's list are two statements of one format; a field added to
    `save_config` and not to the list would be reported to its reader as a typo."""
    import inspect

    from tradeflow.optimization.config_store import CONFIG_KEYS, Provenance, save_config, unrecognised_keys

    every = {
        name: ({"a": 1} if name in ("params", "position_limits", "cost") else "x")
        for name in inspect.signature(save_config).parameters
        if name not in ("path", "provenance")
    }
    written = json_load(save_config(tmp_path / "c.json", provenance=Provenance(), **every))
    assert unrecognised_keys(written) == []
    assert set(written) == set(CONFIG_KEYS)


def json_load(path):
    import json

    return json.loads(path.read_text())


def test_the_missing_keys_message_says_what_needs_them_and_the_keyless_route():
    """The reader has usually just run the keyless demo; the message listed the missing
    variables and a setup command without saying why this command was different, or
    that a warmed cache runs without keys."""
    from tradeflow import settings

    message = settings._missing_credentials_message(["APCA_API_KEY_ID"])
    assert "market data" in message and "--offline" in message


def _parse_refusal(capsys, *argv):
    from tradeflow.cli import parse_cli

    with pytest.raises(SystemExit) as stop:
        parse_cli(list(argv))
    assert stop.value.code == 2
    return capsys.readouterr().err


def test_a_stray_flag_is_refused_with_the_subcommands_own_usage(capsys):
    """`optimize --range` printed the root usage line and `quantred: error:`, a map of the
    whole tool for a mistake about one command's flags."""
    err = _parse_refusal(capsys, "optimize", "--strategy", "demo_trend", "--range", "fast_ema_period=5:10:1")
    assert err.startswith(f"usage: {CLI_NAME} optimize ")
    assert f"{CLI_NAME} optimize: error: unrecognized arguments: --range" in err


def test_a_flag_that_belongs_to_a_sibling_command_names_it(capsys):
    """`--range` exists on `screen`; saying so beats leaving the reader to guess a typo."""
    err = _parse_refusal(capsys, "optimize", "--range", "fast_ema_period=5:10:1")
    assert "--range is an option of `screen`, not of `optimize`" in err


def test_a_flag_nobody_has_gets_no_sibling_hint(capsys):
    """The hint must not fire for a genuine typo, or it names a command at random."""
    err = _parse_refusal(capsys, "optimize", "--no-such-flag")
    assert "unrecognized arguments: --no-such-flag" in err
    assert "is an option of" not in err


def test_a_nested_subcommand_reports_its_own_usage(capsys):
    err = _parse_refusal(capsys, "trials", "show", "abcd1234", "--bogus")
    assert err.startswith(f"usage: {CLI_NAME} trials show ")


# --- names a strategy or a book does not have --------------------------------------


def test_a_misspelt_parameter_is_refused_not_run_at_the_default(tmp_path, monkeypatch):
    """`config={"fast_emaperiod": 9}` ran at the defaults and was served the defaults'
    memoized trial, with nothing saying the override reached nothing."""
    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import analysis, audit

    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    client = MarketDataClient(FakeMarketData(["AAA"], n=300, freq="1D"))
    start, end = cli._date("2024-01-02"), cli._date("2024-06-01")
    with pytest.raises(ValueError, match=r"'fast_emaperiod' \(did you mean 'fast_ema_period'\?\)"):
        analysis.run_backtest(client, "demo_trend", ["AAA"], start, end, config={"fast_emaperiod": 9})
    with pytest.raises(ValueError, match=r"'max_positoins' \(did you mean 'max_positions'\?\)"):
        analysis.run_walk_forward(
            client, "demo_trend", ["AAA"], start, end, position_limits={"max_positoins": 3}
        )
    assert not (tmp_path / "journal.jsonl").exists()


def test_a_declared_parameter_and_a_real_limit_are_accepted():
    """The other direction, including the keys that are not parameters."""
    from tradeflow.services.registry import resolve_strategy_class
    from tradeflow.strategies.base import refuse_unknown_limits, refuse_unknown_params

    cls = resolve_strategy_class("demo_trend")
    refuse_unknown_params(cls, {"fast_ema_period": 9, "position_limits": {}, "_cost": {}})
    refuse_unknown_limits({"max_positions": 3, "min_notional": 5.0})


def test_a_saved_config_with_a_stale_parameter_is_refused(tmp_path):
    """The loader's docstring said unrecognised params raise; the constructor let them
    through, so a config naming a since-renamed parameter traded the default."""
    import json

    path = tmp_path / "c.json"
    path.write_text(json.dumps({"strategy": "demo_trend", "params": {"fast_emaperiod": 9}}))
    args = cli.parse_cli(["backtest", "--config", str(path)])
    with pytest.raises(SystemExit) as exited:
        cli.apply_run_config(args)
    assert "has no parameter 'fast_emaperiod'" in str(exited.value.code)


# --- arguments no run can mean ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"mode": "bogus"}, "mode must be one of"),
        ({"method": "bogus"}, "method must be one of"),
        ({"objective": "sharpe"}, "did you mean 'sharpe_ratio'"),
        ({"holdout_days": -5}, "holdout_days cannot be negative"),
        ({"capital": -5.0}, "capital must be greater than zero"),
        ({"max_evals": 0}, "max_evals must be at least 1"),
    ],
)
def test_a_walk_forward_argument_no_run_can_mean_is_refused_and_not_journaled(
    kwargs, fragment, tmp_path, monkeypatch
):
    """Over MCP each of these ran, echoed the value back and journaled a trial: an
    unknown objective scored every candidate as -inf, so the "best" was arbitrary."""
    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import analysis, audit

    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal)
    client = MarketDataClient(FakeMarketData(["AAA"], n=300, freq="1D"))
    with pytest.raises(analysis.RunRefused, match=fragment):
        analysis.run_walk_forward(
            client, "demo_trend", ["AAA"], cli._date("2024-01-02"), cli._date("2024-06-01"), **kwargs
        )
    assert not journal.exists()


def test_a_backtest_that_read_no_bars_is_refused_not_scored_as_zero(tmp_path, monkeypatch):
    """`symbols=["ZZZZ"]` returned final capital equal to initial and every ratio 0.0,
    journaled as a trial; only a log line said no bars were fetched."""
    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import analysis, audit

    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal)
    client = MarketDataClient(FakeMarketData(["AAA"], n=300, freq="1D"))
    start, end = cli._date("2024-01-02"), cli._date("2024-06-01")
    with pytest.raises(analysis.RunRefused, match="No bars for any of ZZZZ"):
        analysis.run_backtest(client, "demo_trend", ["ZZZZ"], start, end)
    with pytest.raises(analysis.RunRefused, match="symbols is empty"):
        analysis.run_backtest(client, "demo_trend", [], start, end)
    assert not journal.exists()
    # The boundary: a universe with bars still runs.
    assert analysis.run_backtest(client, "demo_trend", ["AAA"], start, end, journal=False)["metrics"]


def test_the_cli_refuses_an_unknown_objective_and_ends_a_refused_run_without_a_stack(capsys, monkeypatch):
    from argparse import Namespace

    from tradeflow.services.analysis import RunRefused

    assert "did you mean 'sortino_ratio'" in _refusal(["optimize", "--objective", "sortino"], capsys)
    assert "cannot be negative" in _refusal(["walkforward", "--holdout-days", "-1"], capsys)
    assert "expected one or more" in _refusal(["backtest", "--symbols", " , "], capsys)

    def _refused(args):
        raise RunRefused("No bars for any of ZZZZ.")

    monkeypatch.setattr(cli, "parse_cli", lambda: Namespace(func=_refused))
    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == "No bars for any of ZZZZ."


def test_a_malformed_date_says_so_on_every_surface():
    """The MCP server returned strptime's own text - "does not match format '%Y-%m-%d'" -
    which never said which formats work; the message now lives in the one parser."""
    from tradeflow.utils.timeutils import parse_market_date

    with pytest.raises(ValueError, match=r"YYYY-MM-DD .* got '01/02/2024'"):
        parse_market_date("01/02/2024")


def test_walk_forward_gates_say_what_they_are_computed_over(tmp_path, monkeypatch):
    """The fold-median gates sat beside a stitched aggregate with the same metric names;
    a gate Sharpe of 0.21 next to an aggregate 0.20 read as a bug."""
    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import analysis, audit

    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(analysis, "ARTIFACT_DIR", tmp_path / "artifacts")
    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=500, freq="1D"))
    result = analysis.run_walk_forward(
        client,
        "demo_trend",
        ["AAA", "BBB"],
        cli._date("2023-01-02"),
        cli._date("2024-06-01"),
        n_folds=2,
        max_evals=2,
    )
    checks = result["gate_report"]["checks"]
    assert checks["oos_sharpe"]["basis"] == "median across folds"
    assert checks["min_oos_trades"]["basis"] == "total across folds"
    assert checks["deflated_sharpe"]["basis"] == "stitched out-of-sample aggregate"
