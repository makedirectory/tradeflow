"""A saved config as run configuration, not just a strategy.

One file sets what to run - strategy, params, scanner, universe, capital, cost - so a
tuned config can live in a private repository beside the strategies it belongs to and
then drive a backtest, an allocation or a verdict without restating any of it.

The window is deliberately not in the file: a config that carried its own tuning dates
would make every later run re-evaluate that period by default.
"""

import json

import pytest

from tradeflow.cli import apply_run_config, parse_cli
from tradeflow.optimization import config_store

_PARAMS = {
    "fast_ema_period": 18,
    "slow_ema_period": 24,
    "risk_per_trade": 0.04,
    "stop_loss": 0.02,
    "take_profit": 0.03,
}


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "alpha.json"
    path.write_text(
        json.dumps(
            {
                "strategy": "ma_crossover",
                "scanner": "volume",
                "symbols": ["AAA", "BBB", "CCC"],
                "capital": 250_000.0,
                "cost": {"gross": False, "commission_bps": 2.5, "impact_eta": 0.4, "borrow_bps": 30.0},
                "params": _PARAMS,
                "provenance": {},
            }
        )
    )
    return str(path)


def test_the_file_supplies_every_input_the_command_line_left_unsaid(saved):
    args = parse_cli(["backtest", "--config", saved, "--start", "2024-01-02", "--end", "2024-06-01"])

    tuned = apply_run_config(args)

    assert args.strategy == "ma_crossover"
    assert args.scanner == "volume"
    assert args.symbols == ["AAA", "BBB", "CCC"]
    assert args.capital == 250_000.0
    assert args.commission_bps == 2.5
    assert tuned == _PARAMS


def test_a_flag_the_user_typed_beats_the_file(saved):
    """Including a flag typed with its own default value.

    Which is why this reads the argv tokens rather than comparing against defaults: a
    flag passed explicitly and one omitted produce the same parsed value, and a config
    silently overriding something the user typed is the worst outcome available.
    """
    args = parse_cli(
        [
            "backtest",
            "--config",
            saved,
            "--symbols",
            "ZZZ",
            "--capital",
            "1000",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-01",
        ]
    )

    apply_run_config(args)

    assert args.symbols == ["ZZZ"]
    assert args.capital == 1000.0
    assert args.commission_bps == 2.5  # untyped, so still the file's


def test_a_command_takes_only_the_fields_it_has(saved, capsys):
    """`risk` summarizes a universe's covariance and has no strategy at all.

    So the same file is usable there without being meaningful there, and the report
    names only what was applied — claiming a strategy the command never uses would be
    the exact misreport this surface exists to prevent.
    """
    args = parse_cli(["risk", "--config", saved])

    apply_run_config(args)

    assert args.symbols == ["AAA", "BBB", "CCC"]
    printed = capsys.readouterr().out
    assert "symbols=<config>" in printed
    assert "strategy=" not in printed


def test_a_strategy_that_contradicts_the_config_is_refused(saved):
    """The params in the file belong to the strategy in the file. Handing one
    strategy's tuned params to another is not an outcome worth guessing at."""
    args = parse_cli(["backtest", "--config", saved, "--strategy", "volume_spike"])

    with pytest.raises(SystemExit) as exit_info:
        apply_run_config(args)

    message = str(exit_info.value)
    assert "ma_crossover" in message and "volume_spike" in message


def test_the_window_comes_from_the_run_and_says_so(saved, capsys):
    """A config pinned to its tuning window would quietly re-evaluate that period on
    every later run, which is the one thing a reusable config must not do."""
    assert "windows" not in json.loads(open(saved).read())
    args = parse_cli(["backtest", "--config", saved, "--start", "2024-01-02", "--end", "2024-06-01"])

    apply_run_config(args)

    assert "window 2024-01-02..2024-06-01 (from this run, not the config)" in capsys.readouterr().out


def test_no_config_is_a_no_op(saved):
    args = parse_cli(["backtest", "--strategy", "volume_spike", "--symbols", "AAA"])

    assert apply_run_config(args) == {}
    assert args.strategy == "volume_spike"


def test_the_tuned_params_reach_the_service_not_just_the_namespace(monkeypatch):
    """The property that makes `--config` mean anything on the analysis commands.

    They pass a strategy *name* to their service, which builds it from defaults. If the
    params are not threaded through as well, a run reports that it loaded a tuned config
    and then scores the universe with the strategy's defaults - a wrong answer wearing
    the right label. Asserting the signature accepts `config` would not catch that; only
    watching what reaches the constructor does.
    """
    from datetime import datetime

    from tests.fakes import FakeMarketData
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.services import analysis

    seen = []
    real = analysis._strategy

    def _spy(name, config=None):
        seen.append(config)
        return real(name, config)

    monkeypatch.setattr(analysis, "_strategy", _spy)
    client = MarketDataClient(FakeMarketData(["AAA", "BBB", "SPY"], n=300, freq="1D"))

    analysis.compute_alphas(
        client, "ma_crossover", ["AAA", "BBB"], datetime(2024, 6, 1), config=_PARAMS, scanner="none"
    )

    assert _PARAMS in seen, f"the tuned params never reached the strategy: {seen}"


def test_walkforward_saves_the_run_inputs_not_only_the_params(tmp_path):
    """A file holding params alone cannot configure a run: the universe the scanner
    resolved, the capital and the cost model are all part of what was validated."""
    path = config_store.save_config(
        tmp_path / "c.json",
        strategy="ma_crossover",
        params=_PARAMS,
        scanner="volume",
        symbols=["AAA", "BBB"],
        capital=50_000.0,
        cost={"gross": False, "commission_bps": 1.0},
    )

    loaded = config_store.load_config(path)

    assert loaded["symbols"] == ["AAA", "BBB"]
    assert loaded["capital"] == 50_000.0
    assert loaded["cost"]["commission_bps"] == 1.0


def test_a_config_saved_without_run_inputs_still_loads(tmp_path):
    """Files already sitting in someone's private repo predate these keys, and absent
    is not empty: they simply supply less."""
    path = config_store.save_config(tmp_path / "old.json", strategy="ma_crossover", params=_PARAMS)

    loaded = config_store.load_config(path)

    assert "symbols" not in loaded
    assert loaded["params"] == _PARAMS


# --- backtest and live use the same layering as everything else ---------------
def test_backtest_lets_an_explicit_scanner_override_the_config(saved, monkeypatch):
    """Reported from real use, both directions.

    `backtest` kept its own pre-existing config branch when the shared layering was
    introduced, and that branch read `if cfg_scanner: scanner = cfg_scanner` - so the
    file always won and `--scanner` was inert. Passing a scanner got the config's, and
    passing `--scanner none` still ran the config's scanner.
    """
    args = parse_cli(
        ["backtest", "--config", saved, "--scanner", "none", "--start", "2024-01-02", "--end", "2024-06-01"]
    )

    apply_run_config(args)

    assert args.scanner == "none"


def test_backtest_takes_its_universe_from_the_config(saved):
    """The other half of the same defect, and the more dangerous half.

    `backtest` never applied the config's `symbols`, so `args.symbols` stayed at
    DEFAULT_UNIVERSE - which is why a run against a 61-symbol saved config went and
    fetched RIVN. `verdict` handled the same file correctly, which is what made it
    look scanner-specific rather than a whole command left on the old path.
    """
    from tradeflow.cli import DEFAULT_UNIVERSE

    args = parse_cli(["backtest", "--config", saved, "--start", "2024-01-02", "--end", "2024-06-01"])

    apply_run_config(args)

    assert args.symbols == ["AAA", "BBB", "CCC"]
    assert "RIVN" not in args.symbols
    assert args.symbols != DEFAULT_UNIVERSE


def test_live_takes_its_universe_and_scanner_from_the_config(saved):
    """`live` carried the identical branch, so it had the identical defect."""
    args = parse_cli(["live", "--config", saved, "--scanner", "none"])

    apply_run_config(args)

    assert args.scanner == "none"
    assert args.symbols == ["AAA", "BBB", "CCC"]


def test_every_command_with_config_shares_one_layering_path():
    """The defect was one command keeping its own copy of this logic. Asserting the
    behaviour per command would not have caught it; asserting there is one path does."""
    import inspect

    from tradeflow import cli

    for command in ("cmd_backtest", "cmd_live", "cmd_verdict", "cmd_alphas", "cmd_info", "cmd_horizon"):
        source = inspect.getsource(getattr(cli, command))
        assert "apply_run_config" in source, f"{command} does not use the shared layering"
        assert "cfg_scanner" not in source, f"{command} still has its own config branch"
