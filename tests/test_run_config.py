"""A saved config as run configuration, not just a strategy.

One file sets what to run - strategy, params, scanner, universe, capital, cost - so a
tuned config can live in a private repository beside the strategies it belongs to and
then drive a backtest, an allocation or a verdict without restating any of it.

The window is deliberately not in the file: a config that carried its own tuning dates
would make every later run re-evaluate that period by default.
"""

import json
from datetime import datetime

import pytest

from tradeflow.cli import apply_run_config, build_parser, parse_cli
from tradeflow.optimization import config_store

_MA_RANGES = __import__("tradeflow.services.registry", fromlist=["x"]).STRATEGIES["ma_crossover"].PARAM_RANGES

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
    assert args.symbols == ["AAA", "BBB", "CCC"]
    # The scanner is read from the config and then pinned off, because the config's
    # universe is replayed rather than re-scanned - see the universe tests below.
    assert args.scanner == "none"
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


# --- the saved universe is a decision, not a starting point --------------------
@pytest.fixture
def with_universe(tmp_path):
    """A config recording both universes: 61 resolved from 85 candidates."""
    from tradeflow.services.registry import STRATEGIES

    path = tmp_path / "universe.json"
    path.write_text(
        json.dumps(
            {
                "strategy": "ma_crossover",
                "scanner": "volume",
                "symbols": [f"R{i}" for i in range(61)],
                "candidate_symbols": [f"C{i}" for i in range(85)],
                "params": {n: s["default"] for n, s in STRATEGIES["ma_crossover"].PARAM_RANGES.items()},
                "provenance": {},
            }
        )
    )
    return str(path)


def test_a_saved_universe_is_replayed_not_rescanned(with_universe, capsys):
    """The defect this fixes.

    `--config` restored the 61 names the scanner *resolved*, and the command then
    passed them straight back through the scanner - filtering an already-filtered set,
    at a different clock. A config that recorded a 61-name book could silently replay
    as a subset of it, and nothing said which had happened.
    """
    args = parse_cli(["backtest", "--config", with_universe])

    apply_run_config(args)

    assert len(args.symbols) == 61
    assert args.scanner == "none"  # pinned off, which is how replay is expressed
    assert "replayed from config, 61 symbols" in capsys.readouterr().out


def test_re_resolving_uses_the_saved_candidates_not_the_resolved_book(with_universe, capsys):
    """Re-running a scanner over the 61 it already picked is a second filter, not the
    original decision repeated. Only the candidate list makes a genuine re-scan
    possible, which is why both are saved."""
    args = parse_cli(["backtest", "--config", with_universe, "--re-resolve-universe"])

    apply_run_config(args)

    assert len(args.symbols) == 85  # the candidates, not the resolved book
    assert args.scanner == "volume"
    assert "re-resolved from 85 saved candidates" in capsys.readouterr().out


def test_an_older_config_without_candidates_says_what_it_can_only_do(tmp_path, capsys):
    """Configs saved before candidates were recorded can only re-scan the resolved
    book. That is a second filter, so it has to be named rather than passed off as a
    re-resolution."""
    from tradeflow.services.registry import STRATEGIES

    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "strategy": "ma_crossover",
                "scanner": "volume",
                "symbols": ["R0", "R1"],
                "params": {n: s["default"] for n, s in STRATEGIES["ma_crossover"].PARAM_RANGES.items()},
                "provenance": {},
            }
        )
    )
    args = parse_cli(["backtest", "--config", str(path), "--re-resolve-universe"])

    apply_run_config(args)

    assert "no candidates recorded" in capsys.readouterr().out


def test_an_explicit_scanner_beats_the_replay_and_says_so(with_universe, capsys):
    """Flags win, including over the replay default - but a typed scanner and a saved
    book are two instructions that disagree, so the report names which was honoured."""
    args = parse_cli(["backtest", "--config", with_universe, "--scanner", "volume"])

    apply_run_config(args)

    assert args.scanner == "volume"  # not pinned off
    assert "saved book re-scanned" in capsys.readouterr().out


def test_symbols_and_re_resolve_together_report_the_collision(with_universe, capsys):
    """The edge where "flags win" and "a saved config is a decision" collide.

    `--symbols` has already replaced the saved book, so `--re-resolve-universe` has
    nothing left to re-resolve. Silently ignoring it would let a flag look honoured.
    """
    args = parse_cli(["backtest", "--config", with_universe, "--symbols", "X,Y", "--re-resolve-universe"])

    apply_run_config(args)

    assert args.symbols == ["X", "Y"]
    assert "has nothing to re-resolve" in capsys.readouterr().out


def test_without_a_config_nothing_about_scanning_changes(capsys):
    """Replay only applies to a universe that came from a config. An ordinary run must
    behave exactly as it did."""
    args = parse_cli(["backtest", "--strategy", "volume_spike", "--scanner", "volume"])

    apply_run_config(args)

    assert args.scanner == "volume"
    assert capsys.readouterr().out == ""


# --- promoting a validated trial into a portable config ------------------------
def _journaled_trial(tmp_path, monkeypatch, promotable=True, candidates=None):
    """One journaled trial plus a store rebuilt from it, as a real run leaves behind."""
    from tradeflow.services import audit
    from tradeflow.store.trials import TrialStore, db_path_for_journal

    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(audit, "DEFAULT_TRIAL_JOURNAL", journal)
    audit.journal_trial(
        "walkforward",
        strategy="ma_crossover",
        symbols=["R0", "R1", "R2"],
        candidate_symbols=candidates,
        start=datetime(2024, 1, 2),
        end=datetime(2025, 1, 2),
        params={**{n: s["default"] for n, s in _MA_RANGES.items()}, "_cost": {"gross": False}},
        metrics={"sharpe_ratio": 1.4},
        extra={"promotable": promotable, "n_trials": 12},
        path=journal,
    )
    store = TrialStore(db_path_for_journal(journal), journal_path=journal)
    store.rebuild(journal)
    trial_id = [
        json.loads(line)["run_id"]
        for line in journal.read_text().splitlines()
        if str(json.loads(line).get("tool", "")).startswith("trial:")
    ][-1]
    return store, trial_id


def _promote(store, trial_id, out, force=False):
    from tradeflow.cli import _promote_trial

    argv = ["trials", "promote", trial_id, "--save-config", str(out)] + (["--force"] if force else [])
    _promote_trial(store, build_parser().parse_args(argv))


def test_a_validated_trial_promotes_without_being_re_run(tmp_path, monkeypatch, capsys):
    """`--save-config` writes after a walk-forward, so saving a config you already
    validated meant validating it again - and the memo only serves an identical recipe,
    which it is not once a seed changed to ask a different question."""
    store, trial_id = _journaled_trial(tmp_path, monkeypatch)
    out = tmp_path / "promoted.json"

    _promote(store, trial_id, out)

    config = json.loads(out.read_text())
    assert config["strategy"] == "ma_crossover"
    assert config["symbols"] == ["R0", "R1", "R2"]
    assert trial_id in config["provenance"]["notes"]
    assert "Promoted trial" in capsys.readouterr().out


def test_the_dedup_keys_do_not_leak_into_the_promoted_params(tmp_path, monkeypatch):
    """Journaled params carry the dedup key's reserved entries. `_cost` is not a
    strategy parameter - the schema has its own field - and passing it through would
    leave a stray key in every strategy the config constructs."""
    store, trial_id = _journaled_trial(tmp_path, monkeypatch)
    out = tmp_path / "promoted.json"

    _promote(store, trial_id, out)

    config = json.loads(out.read_text())
    assert not [name for name in config["params"] if name.startswith("_")]
    assert config["cost"] == {"gross": False}  # lifted to where it belongs


def test_a_promoted_config_replays_the_book_it_validated(tmp_path, monkeypatch, capsys):
    """The alignment that makes promotion worth having: promote then replay has to
    reproduce the validated decision, not run a nearby experiment."""
    store, trial_id = _journaled_trial(tmp_path, monkeypatch)
    out = tmp_path / "promoted.json"
    _promote(store, trial_id, out)
    capsys.readouterr()

    args = parse_cli(["backtest", "--config", str(out)])
    apply_run_config(args)

    assert args.symbols == ["R0", "R1", "R2"]
    assert args.scanner == "none"
    assert "replayed from config, 3 symbols" in capsys.readouterr().out


def test_a_promoted_config_carries_candidates_when_the_trial_recorded_them(tmp_path, monkeypatch, capsys):
    """So `--re-resolve-universe` can offer a genuine re-scan rather than a second
    filter over the already-resolved book."""
    store, trial_id = _journaled_trial(tmp_path, monkeypatch, candidates=["C0", "C1", "C2", "C3"])
    out = tmp_path / "promoted.json"
    _promote(store, trial_id, out)
    capsys.readouterr()

    args = parse_cli(["backtest", "--config", str(out), "--re-resolve-universe"])
    apply_run_config(args)

    assert len(args.symbols) == 4
    assert "re-resolved from 4 saved candidates" in capsys.readouterr().out


def test_a_trial_without_candidates_promotes_as_incomplete_rather_than_inventing_them(
    tmp_path, monkeypatch, capsys
):
    """Trials journaled before candidates were recorded stay *less complete*. Treating
    the resolved book as its own candidate list would make a second filter look like a
    re-resolution."""
    store, trial_id = _journaled_trial(tmp_path, monkeypatch, candidates=None)
    out = tmp_path / "promoted.json"

    _promote(store, trial_id, out)

    assert "candidate_symbols" not in json.loads(out.read_text())
    assert "no candidate list recorded" in capsys.readouterr().out


def test_a_trial_that_did_not_clear_its_gates_is_refused_without_force(tmp_path, monkeypatch):
    """Promoting one silently would put a config on disk whose own provenance says it
    failed. Refuse, and say how to override."""
    store, trial_id = _journaled_trial(tmp_path, monkeypatch, promotable=False)
    out = tmp_path / "promoted.json"

    with pytest.raises(SystemExit) as exit_info:
        _promote(store, trial_id, out)

    assert "not promotable" in str(exit_info.value) and "--force" in str(exit_info.value)
    assert not out.exists()


def test_forcing_a_failed_trial_records_the_verdict_in_the_config(tmp_path, monkeypatch, capsys):
    store, trial_id = _journaled_trial(tmp_path, monkeypatch, promotable=False)
    out = tmp_path / "promoted.json"

    _promote(store, trial_id, out, force=True)

    assert "NOT promotable" in json.loads(out.read_text())["provenance"]["notes"]
    assert "WARNING" in capsys.readouterr().out


# --- universe provenance --------------------------------------------------------
def test_provenance_names_the_source_and_the_scan_clock():
    """A 61-name large-cap list is not "the market", and a report that leaves the
    universe in the background invites it to be read as one."""
    from tradeflow.analytics.reporting import format_universe_provenance

    rendered = "\n".join(
        format_universe_provenance(
            candidates=[str(i) for i in range(85)],
            resolved=[str(i) for i in range(61)],
            scanner="alpha_pack_trend_quality",
            scan_clock="2026-08-22T00:00:00-04:00",
            source="--symbols",
            replayed=False,
        )
    )

    assert "85 names from --symbols" in rendered
    assert "alpha_pack_trend_quality as of 2026-08-22" in rendered
    assert "61 of 85 names" in rendered
    assert "resolved this run" in rendered


def test_provenance_distinguishes_a_replay_from_a_fresh_resolution():
    """The distinction 042 §2.2 exists for, carried into the report so a reader never
    has to infer which happened."""
    from tradeflow.analytics.reporting import format_universe_provenance

    replayed = "\n".join(
        format_universe_provenance(
            candidates=["A", "B"],
            resolved=["A", "B"],
            scanner="none",
            scan_clock=None,
            source="the saved config",
            replayed=True,
        )
    )

    assert "replayed from config" in replayed
    assert "candidates traded as-is" in replayed


def test_provenance_states_survivorship_rather_than_measuring_it():
    """Measuring survivorship needs point-in-time membership this project does not
    ingest. Saying a hand-supplied list is survivorship-prone by construction is
    honest; computing a number from a list that has none would not be.
    """
    from tradeflow.analytics.reporting import format_universe_provenance

    rendered = "\n".join(
        format_universe_provenance(
            candidates=["A"],
            resolved=["A"],
            scanner="none",
            scan_clock=None,
            source="--symbols",
            replayed=False,
        )
    )

    assert "not point-in-time" in rendered
    assert "today's names applied to history" in rendered


# --- a frozen config states what it risks --------------------------------------
def _frozen(tmp_path, limits):
    from tradeflow.services.registry import STRATEGIES

    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps(
            {
                "strategy": "ma_crossover",
                "scanner": "none",
                "symbols": ["A", "B", "C"],
                "capital": 8000.0,
                "params": {n: s["default"] for n, s in STRATEGIES["ma_crossover"].PARAM_RANGES.items()},
                "provenance": {},
                **({"position_limits": limits} if limits else {}),
            }
        )
    )
    return str(path)


def test_a_config_s_risk_limits_reach_the_strategy(tmp_path, capsys):
    """Writing limits into a file changes nothing unless they arrive.

    The shipped default of `max_positions: 1` is exactly the inheritance a frozen
    config exists to prevent - a 61-name config quietly holding one position is a defect
    already found in backtests, and a live run would repeat it with money.
    """
    from tradeflow.cli import _strategy_from

    path = _frozen(tmp_path, {"max_positions": 8, "max_position_size": 1200.0, "min_notional": 25.0})
    args = parse_cli(["backtest", "--config", path])
    tuned = apply_run_config(args)
    capsys.readouterr()

    limits = _strategy_from(args, tuned).position_limits()

    assert limits["max_positions"] == 8  # not the shipped 1
    assert limits["max_position_size"] == 1200.0
    assert limits["min_notional"] == 25.0


def test_limits_absent_from_an_older_config_fall_back_to_the_defaults(tmp_path, capsys):
    """Merged, not replaced: a config written before limits were recorded still gets
    the strategy's own values for keys it never carried."""
    from tradeflow.cli import _strategy_from

    args = parse_cli(["backtest", "--config", _frozen(tmp_path, None)])
    tuned = apply_run_config(args)
    capsys.readouterr()

    limits = _strategy_from(args, tuned).position_limits()

    assert limits["max_positions"] == 1  # the strategy's own default, unchanged
    assert set(limits) >= {"max_positions", "max_total_risk", "max_gross_exposure", "min_notional"}


def test_a_partial_limits_block_keeps_the_rest(tmp_path, capsys):
    from tradeflow.cli import _strategy_from

    args = parse_cli(["backtest", "--config", _frozen(tmp_path, {"max_positions": 12})])
    tuned = apply_run_config(args)
    capsys.readouterr()

    limits = _strategy_from(args, tuned).position_limits()

    assert limits["max_positions"] == 12
    assert limits["max_total_risk"] == 0.05  # untouched by a partial override


def test_a_saved_config_records_its_limits_in_full(tmp_path):
    """So the file says what it risks rather than resolving it at start-up."""
    from tradeflow.optimization.config_store import load_config, save_config
    from tradeflow.services.registry import STRATEGIES

    path = save_config(
        tmp_path / "c.json",
        strategy="ma_crossover",
        params={"a": 1},
        symbols=["A"],
        capital=8000.0,
        position_limits=STRATEGIES["ma_crossover"].create_with_defaults().position_limits(),
    )

    limits = load_config(path)["position_limits"]
    assert set(limits) >= {"max_positions", "max_position_size", "max_total_risk", "max_gross_exposure"}
