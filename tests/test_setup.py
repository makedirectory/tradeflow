"""First-run setup: secrets hygiene, non-destructive writes, and the doctor.

All offline. The properties defended here are the ones whose absence would be
expensive rather than annoying: a secret reaching a log or an error message, a
rewrite eating a key this project does not own, a torn file that reads as
half-configured credentials, and a network blip reported as "your keys are wrong".
"""

from datetime import datetime

import pytest

from tradeflow.services import setup

REAL_KEY = "PKTESTKEYID0000001"
REAL_SECRET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text)
    return path


# --- masking ----------------------------------------------------------------
def test_masking_never_reveals_a_usable_secret():
    masked = setup.mask(REAL_SECRET)
    assert REAL_SECRET not in masked
    assert masked.startswith("abcd") and masked.endswith("6789")
    # A short value is masked entirely — revealing most of a short string reveals it.
    assert setup.mask("abc123") == "******"
    assert setup.mask(None) == "(not set)"
    assert setup.mask("<APCA_API_KEY_ID>") == "(placeholder)"


def test_placeholder_detection_defers_to_the_settings_loader():
    """Two definitions of "filled in" would drift; the one that matters is the
    one the loader enforces."""
    from tradeflow.settings import _PLACEHOLDERS

    for placeholder in _PLACEHOLDERS:
        assert setup.is_placeholder(placeholder)
    assert not setup.is_placeholder(REAL_KEY)


# --- inspection -------------------------------------------------------------
def test_inspect_reports_missing_placeholder_and_filled_states(tmp_path):
    assert setup.inspect_env(tmp_path / "absent.env", environ={}).missing == list(setup.CREDENTIAL_KEYS)

    path = _write(tmp_path, "APCA_API_KEY_ID=<APCA_API_KEY_ID>\nAPCA_API_SECRET_KEY=<APCA_API_SECRET_KEY>\n")
    assert setup.inspect_env(path, environ={}).missing == list(setup.CREDENTIAL_KEYS)

    path = _write(tmp_path, f"APCA_API_KEY_ID={REAL_KEY}\nAPCA_API_SECRET_KEY={REAL_SECRET}\n")
    state = setup.inspect_env(path, environ={})
    assert state.complete and state.missing == []


def test_an_exported_variable_beats_the_file(tmp_path):
    """Someone whose keys come from the environment must not be told to fill in a
    file that would have no effect."""
    path = _write(tmp_path, "APCA_API_KEY_ID=<APCA_API_KEY_ID>\n")
    state = setup.inspect_env(path, environ={"APCA_API_KEY_ID": REAL_KEY})
    assert state.resolved("APCA_API_KEY_ID") == REAL_KEY
    assert "APCA_API_KEY_ID" not in state.missing


def test_a_summary_is_fully_masked(tmp_path):
    path = _write(tmp_path, f"APCA_API_KEY_ID={REAL_KEY}\nAPCA_API_SECRET_KEY={REAL_SECRET}\n")
    summary = setup.inspect_env(path, environ={}).summary()
    assert REAL_KEY not in str(summary) and REAL_SECRET not in str(summary)


# --- writing ----------------------------------------------------------------
def test_writing_preserves_every_line_this_project_does_not_own(tmp_path):
    """A .env is shared territory: comments, ordering, and foreign keys survive."""
    original = (
        "# my notes\n"
        "SOME_OTHER_TOOL_KEY=keep-me\n"
        "\n"
        "APCA_API_KEY_ID=<APCA_API_KEY_ID>\n"
        "# trailing comment\n"
        "STRIPE_SECRET=also-keep-me\n"
    )
    path = _write(tmp_path, original)
    setup.write_env({"APCA_API_KEY_ID": REAL_KEY}, path)

    written = path.read_text()
    assert "# my notes" in written
    assert "SOME_OTHER_TOOL_KEY=keep-me" in written
    assert "STRIPE_SECRET=also-keep-me" in written
    assert "# trailing comment" in written
    assert f"APCA_API_KEY_ID={REAL_KEY}" in written
    assert "<APCA_API_KEY_ID>" not in written


def test_a_backup_exists_before_anything_is_modified(tmp_path):
    path = _write(tmp_path, "APCA_API_KEY_ID=old-value\n")
    result = setup.write_env({"APCA_API_KEY_ID": REAL_KEY}, path)
    assert result["backup"]
    from pathlib import Path

    assert "old-value" in Path(result["backup"]).read_text()


def test_a_new_key_is_appended_rather_than_replacing_the_file(tmp_path):
    path = _write(tmp_path, "# header\nOTHER=1\n")
    setup.write_env({"PAPER_TRADE": "true"}, path)
    written = path.read_text()
    assert "# header" in written and "OTHER=1" in written and "PAPER_TRADE=true" in written


def test_writing_creates_the_file_when_there_is_none(tmp_path):
    path = tmp_path / "fresh.env"
    result = setup.write_env({"APCA_API_KEY_ID": REAL_KEY}, path)
    assert path.exists() and result["backup"] is None
    assert path.read_text() == f"APCA_API_KEY_ID={REAL_KEY}\n"


def test_the_write_is_atomic(tmp_path, monkeypatch):
    """A crash mid-write must leave the old file or the new one — never a torn one
    the loader would read as half-configured credentials."""
    import os

    path = _write(tmp_path, "APCA_API_KEY_ID=old-value\n")

    def explode(src, dst):
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError):
        setup.write_env({"APCA_API_KEY_ID": REAL_KEY}, path)
    assert path.read_text() == "APCA_API_KEY_ID=old-value\n"
    # And no partial file is left holding a copy of the secret.
    assert not (tmp_path / ".env.tmp").exists()


def test_the_result_never_echoes_what_was_written(tmp_path):
    result = setup.write_env({"APCA_API_SECRET_KEY": REAL_SECRET}, tmp_path / ".env")
    assert REAL_SECRET not in str(result)


def test_build_updates_only_writes_what_was_answered():
    assert setup.build_updates() == {}
    assert setup.build_updates(key=REAL_KEY) == {"APCA_API_KEY_ID": REAL_KEY}
    assert setup.build_updates(paper_trade=False) == {"PAPER_TRADE": "false"}


# --- credential validation --------------------------------------------------
class _Rejecting:
    def get_bars(self, *args, **kwargs):
        raise RuntimeError(f"401 Unauthorized for key {REAL_KEY}")


class _Unreachable:
    def get_bars(self, *args, **kwargs):
        raise ConnectionError("temporary failure in name resolution")


class _Empty:
    def get_bars(self, *args, **kwargs):
        return {}


class _Working:
    def __init__(self):
        self.calls = []

    def get_bars(self, symbols, timeframe, start, end):
        self.calls.append((tuple(symbols), timeframe))
        import pandas as pd

        return {symbols[0]: pd.DataFrame({"close": [1.0]}, index=[datetime(2024, 1, 2)])}


def test_the_three_outcomes_are_distinguishable():
    """ "Could not reach Alpaca" is not "keys rejected": sending someone to
    regenerate working keys is a worse failure than saying nothing."""
    assert setup.check_credentials(_Working()).status == setup.CREDENTIALS_OK
    assert setup.check_credentials(_Rejecting()).status == setup.CREDENTIALS_REJECTED
    assert setup.check_credentials(_Unreachable()).status == setup.CREDENTIALS_UNREACHABLE

    messages = {
        setup.check_credentials(client).message for client in (_Working(), _Rejecting(), _Unreachable())
    }
    assert len(messages) == 3


def test_no_failure_message_ever_contains_the_secret():
    """A vendor SDK's exception text can echo request parameters, so the message
    is composed here rather than passed through."""
    for client in (_Rejecting(), _Unreachable(), _Empty()):
        message = setup.check_credentials(client).message
        assert REAL_KEY not in message
        assert REAL_SECRET not in message
        assert "401" not in message


def test_an_empty_response_is_not_reported_as_a_rejection():
    check = setup.check_credentials(_Empty())
    assert check.status == setup.CREDENTIALS_UNREACHABLE
    assert "accepted" in check.message


def test_validation_uses_one_cheap_request():
    client = _Working()
    setup.check_credentials(client)
    assert len(client.calls) == 1
    assert client.calls[0][1] == "1Day"


def test_an_ambiguous_error_is_treated_as_unreachable_not_rejected():
    class _Vague:
        def get_bars(self, *args, **kwargs):
            raise RuntimeError("something went wrong")

    assert setup.check_credentials(_Vague()).status == setup.CREDENTIALS_UNREACHABLE


# --- doctor mode ------------------------------------------------------------
def test_the_doctor_reports_every_check_not_just_the_first_miss(tmp_path):
    checks = setup.run_checks(tmp_path / "absent.env", environ={})
    names = [c.name for c in checks]
    assert "env file" in names
    assert set(setup.CREDENTIAL_KEYS) <= set(names)
    assert "PAPER_TRADE" in names and "state directory" in names
    assert any(n.startswith("extra:") for n in names)
    # Both credentials are reported, not only the first one that failed.
    assert sum(1 for c in checks if c.name in setup.CREDENTIAL_KEYS and not c.passed) == 2


def test_the_doctor_writes_nothing(tmp_path):
    path = tmp_path / "absent.env"
    setup.run_checks(path, environ={})
    assert not path.exists()


def test_live_trading_enabled_fails_its_check_loudly(tmp_path):
    path = _write(tmp_path, "PAPER_TRADE=false\n")
    paper = next(c for c in setup.run_checks(path, environ={}) if c.name == "PAPER_TRADE")
    assert not paper.passed
    assert "real money" in paper.detail


def test_a_missing_extra_is_collected_for_the_one_install_command(tmp_path):
    """Rows no longer carry their own command (two of them undo each other); they name
    the extra, and the doctor prints a single command over the union."""
    checks = setup.run_checks(tmp_path / ".env", environ={})
    rows = [c for c in checks if c.name.startswith("extra:")]
    assert rows
    for row in rows:
        assert row.passed or row.installs == (row.name.split(": ")[1],)
        assert "uv sync" not in (row.next_action or "") and "uv tool" not in (row.next_action or "")
    assert set(setup.missing_extras(checks)) >= {r.name.split(": ")[1] for r in rows if not r.passed}


def test_the_doctor_never_prints_a_raw_credential(tmp_path):
    path = _write(tmp_path, f"APCA_API_KEY_ID={REAL_KEY}\nAPCA_API_SECRET_KEY={REAL_SECRET}\n")
    rendered = "\n".join(f"{c.name} {c.detail}" for c in setup.run_checks(path, environ={}))
    assert REAL_KEY not in rendered and REAL_SECRET not in rendered


# --- the CLI surface --------------------------------------------------------
def test_cli_check_exits_non_zero_when_credentials_are_missing(tmp_path, monkeypatch, capsys):
    from tradeflow import cli as main

    monkeypatch.setattr(
        setup,
        "run_checks",
        lambda *a, **k: [
            setup.Check("APCA_API_KEY_ID", "blocked", "missing", needed_for=frozenset({"data"}))
        ],
    )
    args = main.build_parser().parse_args(["init", "--check"])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1
    assert "[blocked]" in capsys.readouterr().out


def test_cli_check_ignores_missing_optional_extras(tmp_path, monkeypatch, capsys):
    """A missing extra is not a broken setup — only the essentials fail the run."""
    from tradeflow import cli as main

    monkeypatch.setattr(
        setup,
        "run_checks",
        lambda *a, **k: [
            setup.Check("extra: viz", "optional_missing", "not installed — uv sync --extra viz")
        ],
    )
    args = main.build_parser().parse_args(["init", "--check"])
    args.func(args)
    assert "Ready for 'data'" in capsys.readouterr().out


def test_cli_non_interactive_builds_the_env_from_the_environment(tmp_path, monkeypatch, capsys):
    from tradeflow import cli as main

    monkeypatch.setenv("APCA_API_KEY_ID", REAL_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", REAL_SECRET)
    path = tmp_path / ".env"
    args = main.build_parser().parse_args(["init", "--non-interactive", "--env-path", str(path)])
    args.func(args)

    written = path.read_text()
    assert f"APCA_API_KEY_ID={REAL_KEY}" in written
    assert f"APCA_API_SECRET_KEY={REAL_SECRET}" in written
    # ...and the confirmation printed to the terminal is masked.
    assert REAL_SECRET not in capsys.readouterr().out


def test_cli_non_interactive_refuses_when_there_is_nothing_to_write(tmp_path, monkeypatch):
    from tradeflow import cli as main

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("PAPER_TRADE", raising=False)
    for name in MODEL_ENV:
        monkeypatch.delenv(name, raising=False)
    args = main.build_parser().parse_args(["init", "--non-interactive", "--env-path", str(tmp_path / ".env")])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert "APCA_API_KEY_ID" in str(exc.value)


def test_cli_interactive_skip_path_writes_nothing(tmp_path, monkeypatch, capsys):
    """Pressing Enter at the key prompt leaves a valid keyless demo setup."""
    import getpass

    from tradeflow import cli as main

    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")  # no model provider
    path = tmp_path / ".env"
    args = main.build_parser().parse_args(["init", "--env-path", str(path)])
    args.func(args)

    assert not path.exists()
    out = capsys.readouterr().out
    # Named the way *this* copy can run it — and that now turns on how the software was
    # reached, not on where its state happens to live. The suite runs from a checkout,
    # so the Makefile target is the correct instruction here.
    assert "Skipped" in out and "make demo" in out


def test_an_installed_copy_is_told_to_use_its_command(tmp_path, monkeypatch, capsys):
    """Both directions. The point is not that one phrasing wins — it is that the
    instruction matches how the software was reached."""
    import getpass

    from tradeflow import cli as main

    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    monkeypatch.setattr("tradeflow.settings.running_from_checkout", lambda: False)
    args = main.build_parser().parse_args(["init", "--env-path", str(tmp_path / ".env")])
    args.func(args)

    assert "tradeflow demo" in capsys.readouterr().out


def test_cli_interactive_writes_keys_and_confirms_paper_trading(tmp_path, monkeypatch, capsys):
    import getpass

    from tradeflow import cli as main

    answers = iter([REAL_KEY, REAL_SECRET])
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: next(answers))
    # Keep PAPER_TRADE, no model provider, then decline the cache warm.
    replies = iter(["y", "none", "n"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(replies))
    monkeypatch.setattr(
        setup, "check_credentials", lambda *a, **k: setup.CredentialCheck(setup.CREDENTIALS_OK, "fine")
    )
    monkeypatch.setattr("tradeflow.services.data.build_data_client", lambda *a, **k: object())

    path = tmp_path / ".env"
    args = main.build_parser().parse_args(["init", "--env-path", str(path)])
    args.func(args)

    written = path.read_text()
    assert f"APCA_API_KEY_ID={REAL_KEY}" in written
    assert "PAPER_TRADE=true" in written
    assert REAL_SECRET not in capsys.readouterr().out


def test_cli_interactive_requires_a_typed_phrase_to_disable_paper_trading(tmp_path, monkeypatch):
    """A yes/no prompt is too easy to answer wrongly for a choice that moves real
    money — and anything other than the exact phrase keeps paper trading."""
    import getpass

    from tradeflow import cli as main

    answers = iter([REAL_KEY, REAL_SECRET])
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: next(answers))
    # Decline to keep paper trading, fail the typed confirmation, no model, no warm.
    replies = iter(["n", "yes please", "none", "n"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(replies))
    monkeypatch.setattr(
        setup, "check_credentials", lambda *a, **k: setup.CredentialCheck(setup.CREDENTIALS_OK, "fine")
    )
    monkeypatch.setattr("tradeflow.services.data.build_data_client", lambda *a, **k: object())

    path = tmp_path / ".env"
    args = main.build_parser().parse_args(["init", "--env-path", str(path)])
    args.func(args)
    assert "PAPER_TRADE=true" in path.read_text()


# --- install hints and masking ----------------------------------------------
def test_extras_hints_are_phrased_for_the_copy_that_is_running(monkeypatch, tmp_path):
    """An installed copy has no checkout to `uv sync` in - the third place this same
    dead end turned up. The wording itself is covered in `test_extras_map.py`."""
    from tradeflow import settings
    from tradeflow.utils import extras

    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    monkeypatch.setattr(settings, "_looks_like_checkout", lambda _p: False)
    assert "uv sync" not in extras.install_hint("store")
    assert "uv tool install --force" in extras.install_hint("store")

    monkeypatch.setattr(settings, "_looks_like_checkout", lambda _p: True)
    assert "uv sync" in extras.install_hint("store")
    assert "--extra store" in extras.install_hint("store")


def test_the_doctor_reports_every_published_extra(tmp_path):
    """`ai` and `openai` were missing from the doctor, so a reader choosing a research
    provider was never told their module was absent."""
    from tradeflow.utils import extras

    rows = {c.name for c in setup.run_checks(tmp_path / ".env", environ={}) if c.name.startswith("extra:")}
    assert rows == {f"extra: {name}" for name in extras.EXTRAS}


def test_only_credentials_are_masked(tmp_path):
    """Masking PAPER_TRADE to **** hides nothing and costs the reader the one value
    they most want to confirm."""
    path = _write(tmp_path, f"APCA_API_KEY_ID={REAL_KEY}\nPAPER_TRADE=true\n")
    summary = setup.inspect_env(path, environ={}).summary()

    assert summary["PAPER_TRADE"] == "true"
    assert REAL_KEY not in summary["APCA_API_KEY_ID"]


def test_a_written_settings_value_is_not_masked_in_the_confirmation(tmp_path):
    result = setup.write_env({"PAPER_TRADE": "true", "APCA_API_SECRET_KEY": REAL_SECRET}, tmp_path / ".env")
    assert result["updated"]["PAPER_TRADE"] == "true"
    assert REAL_SECRET not in str(result)


# --- readiness per task -------------------------------------------------------
from tradeflow.utils import extras as _extras  # noqa: E402

GOOD_ENV = f"APCA_API_KEY_ID={REAL_KEY}\nAPCA_API_SECRET_KEY={REAL_SECRET}\n"
MODEL_KEY = "sk-test-model-key-0123456789"


def _present(monkeypatch, *, sdks=()):
    """Pretend exactly these extras are installed (all others absent)."""
    monkeypatch.setattr(_extras, "installed", lambda name: name in sdks)


def _status(checks, name):
    return next(c for c in checks if c.name == name)


@pytest.mark.parametrize(
    "keys, sdks, provider, ready, blocked",
    [
        # data-only: keys, no SDKs
        (True, (), None, {"demo", "data"}, {"research", "mcp"}),
        # selected provider without its SDK
        (True, ("mcp",), "anthropic", {"demo", "data", "mcp"}, {"research"}),
        # SDK present, no key
        (True, ("ai", "mcp"), "anthropic", {"demo", "data", "mcp"}, {"research"}),
        # SDK and key both there
        (True, ("ai", "mcp"), "ollama", {"demo", "data", "mcp", "research"}, set()),
        # no market keys: only the demo runs
        (False, ("ai", "mcp"), "ollama", {"demo"}, {"data", "mcp", "research"}),
    ],
)
def test_readiness_is_computed_per_task(tmp_path, monkeypatch, keys, sdks, provider, ready, blocked):
    _present(monkeypatch, sdks=sdks)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = _write(tmp_path, GOOD_ENV if keys else "")
    result = setup.readiness(setup.run_checks(path, environ={}, provider=provider))
    assert {t for t, v in result.items() if v["ready"]} == ready
    assert {t for t, v in result.items() if not v["ready"]} == blocked


def test_a_selected_provider_names_what_is_missing(tmp_path, monkeypatch):
    _present(monkeypatch, sdks=())
    checks = setup.run_checks(_write(tmp_path, GOOD_ENV), environ={}, provider="anthropic")
    row = _status(checks, "model: anthropic")
    assert row.status == "blocked"
    assert "'ai' extra" in row.detail and "ANTHROPIC_API_KEY" in row.detail

    _present(monkeypatch, sdks=("ai",))
    row = _status(
        setup.run_checks(_write(tmp_path, GOOD_ENV), environ={}, provider="anthropic"), "model: anthropic"
    )
    assert row.status == "blocked" and "ANTHROPIC_API_KEY" in row.detail and "extra" not in row.detail


def test_a_ready_provider_masks_its_key(tmp_path, monkeypatch):
    _present(monkeypatch, sdks=("ai",))
    checks = setup.run_checks(
        _write(tmp_path, GOOD_ENV), environ={"ANTHROPIC_API_KEY": MODEL_KEY}, provider="anthropic"
    )
    row = _status(checks, "model: anthropic")
    assert row.status == "ready" and MODEL_KEY not in row.detail


def test_with_nothing_selected_any_ready_provider_makes_research_ready(tmp_path, monkeypatch):
    _present(monkeypatch, sdks=("openai",))
    checks = setup.run_checks(
        _write(tmp_path, GOOD_ENV), environ={"OPENAI_API_KEY": MODEL_KEY}, provider=None
    )
    assert setup.readiness(checks)["research"]["ready"]
    # ...and an unselected, unconfigured provider is optional, not blocked.
    assert _status(checks, "model: anthropic").status == "optional_missing"


def test_ollama_is_unverified_until_probed(tmp_path, monkeypatch):
    _present(monkeypatch, sdks=())
    checks = setup.run_checks(_write(tmp_path, GOOD_ENV), environ={}, provider=None)
    assert _status(checks, "model: ollama").status == "not_checked"
    assert not setup.readiness(checks)["research"]["ready"]
    chosen = setup.readiness(setup.run_checks(_write(tmp_path, GOOD_ENV), environ={}, provider="ollama"))
    assert chosen["research"]["ready"] and chosen["research"]["unverified"] == ["model: ollama"]


def test_optional_extras_missing_are_optional_and_exit_zero(tmp_path, monkeypatch):
    _present(monkeypatch, sdks=("mcp",))
    _stub_state(monkeypatch)
    checks = setup.run_checks(_write(tmp_path, GOOD_ENV), environ={})
    for name in ("store", "viz", "optimize", "portfolio", "ai", "openai"):
        assert _status(checks, f"extra: {name}").status == "optional_missing"
    assert setup.doctor_ok(checks, "data")


def test_an_unwritable_state_root_is_a_permission_failure_not_a_credential_one(tmp_path, monkeypatch):
    from pathlib import Path

    def deny(self, *a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", deny)
    row = _status(setup.run_checks(tmp_path / "absent.env", environ={}), "state directory")
    assert row.status == "failed"
    assert "permission denied writing" in row.detail
    assert "credential" not in (row.detail + (row.next_action or "")).lower()
    assert "TRADEFLOW_HOME" in row.next_action


def _run_init(monkeypatch, capsys, *argv):
    from tradeflow import cli as main

    args = main.build_parser().parse_args(["init", *argv])
    code = 0
    try:
        args.func(args)
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out


@pytest.fixture
def keyed_env(tmp_path, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", REAL_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", REAL_SECRET)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_an_optional_extra_is_never_printed_as_fail(monkeypatch, capsys, keyed_env):
    _present(monkeypatch, sdks=())
    _, out = _run_init(monkeypatch, capsys, "--check")
    extra_lines = [line for line in out.splitlines() if "extra:" in line]
    assert extra_lines
    assert not any("[FAIL]" in line for line in extra_lines)
    assert any("[optional]" in line for line in extra_lines)


def test_a_missing_model_does_not_block_a_backtest(monkeypatch, capsys, keyed_env):
    _present(monkeypatch, sdks=())
    _stub_state(monkeypatch)
    code, out = _run_init(monkeypatch, capsys, "--check", "--for", "data")
    assert code == 0
    assert "ready for:  data" in out and "blocked for: research" in out


def test_research_is_blocked_without_any_provider(monkeypatch, capsys, keyed_env):
    _present(monkeypatch, sdks=())
    _stub_state(monkeypatch)
    code, out = _run_init(monkeypatch, capsys, "--check", "--for", "research")
    assert code == 1
    assert "model provider" in out


def _stub_state(monkeypatch):
    """Keep developer-machine state warnings (a checkout's earlier journal) out of the
    exit-code assertions: they are real rows, but not what these tests are about."""
    monkeypatch.setattr(setup, "_state_checks", lambda: [])


def test_a_plain_check_makes_no_network_call(monkeypatch, capsys, keyed_env):
    def boom(*a, **k):
        raise AssertionError("network used by a plain check")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("socket.create_connection", boom)
    import tradeflow.research.llm as llm_mod

    monkeypatch.setattr(llm_mod, "build_llm_client", boom)
    _stub_state(monkeypatch)
    code, _ = _run_init(monkeypatch, capsys, "--check", "--provider", "ollama")
    assert code == 0


def test_probe_model_needs_a_provider_and_exits_non_zero_on_a_failed_probe(monkeypatch, capsys, keyed_env):
    _stub_state(monkeypatch)
    code, _ = _run_init(monkeypatch, capsys, "--probe-model")
    assert code != 0

    monkeypatch.setattr(
        setup, "probe_model", lambda p, m=None, **k: setup.ProbeResult(p, "m", "auth", "rejected")
    )
    code, out = _run_init(monkeypatch, capsys, "--probe-model", "--provider", "ollama")
    assert code == 1 and "[FAIL] ollama / m  auth: rejected" in out

    monkeypatch.setattr(setup, "probe_model", lambda p, m=None, **k: setup.ProbeResult(p, "m", "ok", "fine"))
    code, out = _run_init(monkeypatch, capsys, "--probe-model", "--provider", "ollama")
    assert code == 0 and "[ok] ollama / m  ok" in out


# --- model provider setup ----------------------------------------------------
MODEL_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_TIMEOUT", "OLLAMA_NUM_CTX")
MODEL_SECRET = "sk-test-0123456789abcdefghijkl"


@pytest.fixture
def wizard(tmp_path, monkeypatch):
    """Drive the interactive wizard with scripted prompts and no network.

    ``run(getpass_answers, input_answers, existing=None)`` returns ``(env_path, output)``;
    any extra prompt beyond the script fails the test, so a prompt that appears
    unasked is caught rather than answered with a default.
    """
    import getpass

    from tradeflow import cli as main

    # The wizard asks for keys only when none resolve, and other tests' dotenv loading can
    # leave real-looking ones in the process environment.
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE", *MODEL_ENV):
        monkeypatch.delenv(name, raising=False)

    def probe(*a, **k):
        raise AssertionError("init must never probe a model on its own")

    monkeypatch.setattr(setup, "probe_model", probe)
    monkeypatch.setattr(
        setup, "check_credentials", lambda *a, **k: setup.CredentialCheck(setup.CREDENTIALS_OK, "fine")
    )
    monkeypatch.setattr("tradeflow.services.data.build_data_client", lambda *a, **k: object())

    def run(secrets, replies, existing=None, capsys=None):
        path = tmp_path / ".env"
        if existing is not None:
            path.write_text(existing)
        secret_iter, reply_iter = iter(secrets), iter(replies)
        monkeypatch.setattr(getpass, "getpass", lambda *a, **k: next(secret_iter))
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(reply_iter))
        args = main.build_parser().parse_args(["init", "--env-path", str(path)])
        args.func(args)
        assert next(secret_iter, None) is None and next(reply_iter, None) is None, "unused answers"
        return path

    return run


def test_init_writes_a_model_key_without_touching_the_alpaca_keys(wizard, tmp_path, capsys):
    existing = f"APCA_API_KEY_ID={REAL_KEY}\nAPCA_API_SECRET_KEY={REAL_SECRET}\nPAPER_TRADE=true\n"
    # Keep the existing credentials, choose anthropic, give a key.
    path = wizard([MODEL_SECRET], ["n", "anthropic"], existing=existing)

    written = path.read_text()
    assert written.startswith(existing.rstrip("\n"))
    assert f"ANTHROPIC_API_KEY={MODEL_SECRET}" in written
    out = capsys.readouterr().out
    assert MODEL_SECRET not in out and REAL_SECRET not in out
    assert "--provider anthropic --probe-model" in out


def test_init_writes_an_ollama_endpoint_timeout_and_context(wizard):
    path = wizard(
        [
            "",
        ],
        ["ollama", "http://gpu-box:11434/api/chat", "300", "8192", ""],
    )
    written = path.read_text()
    # The endpoint is stored as the server root, the way the settings reader sees it.
    assert "OLLAMA_BASE_URL=http://gpu-box:11434\n" in written
    assert "OLLAMA_TIMEOUT=300" in written and "OLLAMA_NUM_CTX=8192" in written


def test_init_reprompts_on_a_bad_ollama_value_instead_of_writing_it(wizard, capsys):
    path = wizard([""], ["ollama", "", "soon", "60", "", ""])
    written = path.read_text()
    assert "OLLAMA_TIMEOUT=60" in written and "soon" not in written
    assert "OLLAMA_TIMEOUT must be a positive number" in capsys.readouterr().out


def test_init_leaves_an_existing_model_key_alone_when_skipped(wizard):
    existing = f"ANTHROPIC_API_KEY={MODEL_SECRET}\nOLLAMA_TIMEOUT=45\n"
    path = wizard([""], ["none"], existing=existing)
    assert path.read_text() == existing


def test_init_does_not_blank_a_model_key_when_the_key_prompt_is_left_empty(wizard):
    existing = f"OPENAI_API_KEY={MODEL_SECRET}\n"
    path = wizard(["", ""], ["openai"], existing=existing)  # chose openai, then pressed Enter at the key
    assert path.read_text() == existing


def test_init_never_probes_the_model_on_its_own(wizard):
    # The fixture's probe_model raises if called; choosing each provider must not.
    wizard(["", MODEL_SECRET], ["openai"])
    wizard([""], ["ollama", "", "", "", ""])


def test_init_names_the_missing_sdk_for_a_hosted_provider(wizard, monkeypatch, capsys):
    monkeypatch.setattr("tradeflow.cli.extras_installed", lambda name: False)
    wizard(["", MODEL_SECRET], ["anthropic"])
    out = capsys.readouterr().out
    assert "'ai' extra" in out


def test_a_model_secret_is_masked_in_the_env_summary_and_the_write_result(tmp_path):
    path = tmp_path / ".env"
    result = setup.write_env(setup.build_updates(anthropic_key=MODEL_SECRET), path)
    assert MODEL_SECRET not in str(result)
    assert MODEL_SECRET not in str(setup.inspect_env(path, environ={}).summary())
    assert setup.inspect_env(path, environ={}).summary()["ANTHROPIC_API_KEY"] != "(not set)"


def test_build_updates_skips_blank_and_placeholder_model_answers():
    assert setup.build_updates(anthropic_key="", openai_key="  ", ollama_timeout=None) == {}
    assert setup.build_updates(anthropic_key=MODEL_SECRET) == {"ANTHROPIC_API_KEY": MODEL_SECRET}


@pytest.mark.parametrize(
    "kwargs,variable",
    [
        ({"ollama_base_url": "localhost:11434"}, "OLLAMA_BASE_URL"),
        ({"ollama_base_url": "ftp://host"}, "OLLAMA_BASE_URL"),
        ({"ollama_timeout": "0"}, "OLLAMA_TIMEOUT"),
        ({"ollama_timeout": "fast"}, "OLLAMA_TIMEOUT"),
        ({"ollama_num_ctx": "-3"}, "OLLAMA_NUM_CTX"),
        ({"ollama_num_ctx": "4k"}, "OLLAMA_NUM_CTX"),
    ],
)
def test_build_updates_rejects_a_bad_model_setting_by_name(kwargs, variable):
    with pytest.raises(ValueError, match=variable):
        setup.build_updates(**kwargs)


def test_the_wizard_accepts_exactly_what_the_settings_reader_accepts(tmp_path, monkeypatch):
    """One parser for both, so a value init writes never makes research refuse to start."""
    from tradeflow import settings

    for value in ("90", "2.5"):
        written = setup.build_updates(ollama_timeout=value)["OLLAMA_TIMEOUT"]
        monkeypatch.setenv("OLLAMA_TIMEOUT", written)
        assert settings.ollama_timeout() == float(value)


def test_non_interactive_init_accepts_only_a_model_key(tmp_path, monkeypatch, capsys):
    from tradeflow import cli as main

    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE", *MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", MODEL_SECRET)
    path = tmp_path / ".env"
    args = main.build_parser().parse_args(
        ["init", "--non-interactive", "--provider", "openai", "--env-path", str(path)]
    )
    args.func(args)

    assert path.read_text() == f"OPENAI_API_KEY={MODEL_SECRET}\n"
    out = capsys.readouterr().out
    assert MODEL_SECRET not in out
    assert "--provider openai --probe-model" in out


def test_non_interactive_init_reads_the_ollama_settings(tmp_path, monkeypatch):
    from tradeflow import cli as main

    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE", *MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://box:11434/")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "240")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    path = tmp_path / ".env"
    args = main.build_parser().parse_args(
        ["init", "--non-interactive", "--provider", "ollama", "--env-path", str(path)]
    )
    args.func(args)
    assert path.read_text() == "OLLAMA_BASE_URL=http://box:11434\nOLLAMA_TIMEOUT=240\nOLLAMA_NUM_CTX=4096\n"


def test_non_interactive_init_copies_model_keys_only_for_the_named_provider(tmp_path, monkeypatch, capsys):
    """Model keys are very often exported for unrelated tools. Copying whatever the shell
    holds put billable keys into a project file nobody asked for."""
    from tradeflow import cli as main

    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE", *MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "PKEXAMPLEKEYID123456")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "examplesecretexamplesecretexamplesecret1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", MODEL_SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", MODEL_SECRET + "2")
    path = tmp_path / ".env"
    args = main.build_parser().parse_args(["init", "--non-interactive", "--env-path", str(path)])
    args.func(args)
    written = path.read_text()
    assert "ANTHROPIC_API_KEY" not in written and "OPENAI_API_KEY" not in written
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out  # said, by name, that it was left out

    path2 = tmp_path / "second.env"
    args = main.build_parser().parse_args(
        ["init", "--non-interactive", "--provider", "anthropic", "--env-path", str(path2)]
    )
    args.func(args)
    written = path2.read_text()
    assert f"ANTHROPIC_API_KEY={MODEL_SECRET}" in written and "OPENAI_API_KEY" not in written


def test_non_interactive_with_only_model_variables_and_no_provider_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    from tradeflow import cli as main

    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE", *MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", MODEL_SECRET)
    path = tmp_path / ".env"
    args = main.build_parser().parse_args(["init", "--non-interactive", "--env-path", str(path)])
    with pytest.raises(SystemExit) as caught:
        args.func(args)
    assert "--provider" in str(caught.value) and "OPENAI_API_KEY" in str(caught.value)
    assert not path.exists()


def test_non_interactive_init_refuses_a_bad_model_setting_without_writing(tmp_path, monkeypatch):
    from tradeflow import cli as main

    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE", *MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_TIMEOUT", "never")
    path = tmp_path / ".env"
    args = main.build_parser().parse_args(
        ["init", "--non-interactive", "--provider", "ollama", "--env-path", str(path)]
    )
    with pytest.raises(SystemExit, match="OLLAMA_TIMEOUT"):
        args.func(args)
    assert not path.exists()


def test_next_steps_say_the_client_launches_the_mcp_server(capsys):
    from tradeflow import cli as main
    from tradeflow.settings import DOCS_URL

    main._print_next_steps()
    out = " ".join(capsys.readouterr().out.split())
    assert "client launches" in out and "over stdio" in out
    assert "does not connect any chat" in out
    assert f"{DOCS_URL}/usage/agents" in out


def test_the_private_strategies_link_has_no_docs_prefix(tmp_path, monkeypatch):
    from tradeflow import cli as main
    from tradeflow.settings import DOCS_URL

    monkeypatch.setattr("tradeflow.services.setup.example_pack_source", lambda: None)
    args = main.build_parser().parse_args(["init", "--example-pack", str(tmp_path / "pack")])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    message = str(exc.value)
    assert f"{DOCS_URL}/usage/private-strategies" in message
    assert "/docs/usage" not in message


# --- the MCP launch command ---------------------------------------------------
def test_the_launch_command_is_absolute_and_runnable_from_anywhere(monkeypatch, tmp_path):
    """A client spawns this with no shell, no cwd and a small PATH."""
    from pathlib import Path

    command = setup.mcp_launch_command()
    assert Path(command[0]).is_absolute()
    assert command[-1] == "mcp"
    monkeypatch.chdir(tmp_path)
    assert setup.mcp_launch_command() == command


def test_an_installed_launch_command_uses_the_script_beside_the_interpreter(monkeypatch, tmp_path):
    import sys

    from tradeflow import settings

    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").write_text("")
    (bindir / "tradeflow").write_text("")
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))
    monkeypatch.setattr(settings, "_looks_like_checkout", lambda _p: False)
    assert setup.mcp_launch_command() == [str(bindir / "tradeflow"), "mcp"]
    (bindir / "tradeflow").unlink()
    assert setup.mcp_launch_command() == [str(bindir / "python"), "-m", "tradeflow.cli", "mcp"]


def test_a_checkout_launch_command_names_main_py_by_absolute_path():
    from pathlib import Path

    from tradeflow import settings

    command = setup.mcp_launch_command()
    assert command[1] == str(settings.PROJECT_ROOT / "main.py") and Path(command[1]).is_file()
    assert setup.mcp_client_config() == {
        "mcpServers": {"tradeflow": {"command": command[0], "args": command[1:]}}
    }


# --- the .env holds billable keys ---------------------------------------------
@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX file modes")
def test_the_env_file_and_its_backup_are_owner_only(tmp_path):
    import os
    import stat

    path = tmp_path / ".env"
    path.write_text("APCA_API_KEY_ID=x\n")
    os.chmod(path, 0o644)  # how most editors and `cp` leave it
    result = setup.write_env({"ANTHROPIC_API_KEY": MODEL_SECRET}, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(os.stat(result["backup"]).st_mode) == 0o600
    assert not (tmp_path / ".env.tmp").exists()
    # A fresh file is private from the moment it exists, not after a chmod.
    fresh = tmp_path / "fresh.env"
    setup.write_env({"ANTHROPIC_API_KEY": MODEL_SECRET}, fresh)
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o600


# --- a configured Ollama is a set-up provider ----------------------------------
def _model_row(checks, name="model provider"):
    return next(c for c in checks if c.name == name)


def test_an_ollama_endpoint_the_wizard_wrote_is_unverified_not_absent(tmp_path, monkeypatch):
    """`init` writes OLLAMA_*; the next `--check --for research` called that "no provider
    is set up", contradicting the file it had just written."""
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tradeflow.utils.extras.installed", lambda extra: False)
    path = _write(tmp_path, "OLLAMA_BASE_URL=http://gpu-box:11434\n")
    checks = setup.run_checks(path, environ={})
    row = _model_row(checks)
    assert row.status == "not_checked"
    assert "ollama configured" in row.detail and "unverified" in row.detail
    assert "model provider" in setup.readiness(checks)["research"]["unverified"]
    assert "model provider" not in setup.readiness(checks)["research"]["blocking"]

    bare = setup.run_checks(tmp_path / "none.env", environ={})
    assert _model_row(bare).status == "blocked"


def test_a_probe_result_replaces_not_checked_in_the_provider_row(tmp_path):
    ok = setup.ProbeResult("ollama", "llama3.1", "ok", "the provider answered one request")
    checks = setup.run_checks(tmp_path / ".env", environ={}, provider="ollama", probe=ok)
    row = _model_row(checks, "model: ollama")
    assert row.status == "ready" and "answered" in row.detail
    assert "model: ollama" not in setup.readiness(checks)["research"]["unverified"]

    bad = setup.ProbeResult("ollama", "llama3.1", "unreachable", "nothing answered at http://x")
    checks = setup.run_checks(tmp_path / ".env", environ={}, provider="ollama", probe=bad)
    row = _model_row(checks, "model: ollama")
    assert row.status == "blocked" and "unreachable" in row.detail
    assert "model: ollama" in setup.readiness(checks)["research"]["blocking"]


# --- init's flags and its probe line --------------------------------------------
def _init(argv):
    from tradeflow import cli as main

    args = main.parse_cli(["init", *argv])
    args.func(args)


@pytest.mark.parametrize(
    "argv, fragment",
    [
        (["--provider", "anthropic"], "--provider only applies"),
        (["--for", "research"], "--for only applies"),
        (["--model", "m"], "--model only applies"),
    ],
)
def test_an_init_flag_the_mode_would_discard_is_refused_before_anything_runs(argv, fragment, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("the wizard must not start"))
    with pytest.raises(SystemExit) as caught:
        _init(argv)
    assert fragment in str(caught.value)


def test_probe_model_without_a_provider_fails_before_the_report_prints(capsys):
    with pytest.raises(SystemExit) as caught:
        _init(["--probe-model"])
    assert "--probe-model needs --provider" in str(caught.value)
    assert "setup check" not in capsys.readouterr().out


def test_the_probe_line_says_billed_only_for_a_provider_that_bills(monkeypatch, capsys):
    monkeypatch.setattr(
        setup, "probe_model", lambda provider, model=None: setup.ProbeResult(provider, "m", "ok", "fine")
    )
    for provider, billed in (("anthropic", True), ("ollama", False)):
        try:
            _init(["--probe-model", "--provider", provider])
        except SystemExit:  # not ready for `data` when no keys resolve; either way the report printed
            pass
        out = capsys.readouterr().out
        assert ("may be billed" in out) is billed, provider


def test_the_printed_ollama_probe_command_names_the_chosen_model(capsys):
    from tradeflow import cli as main

    main._print_provider_followup(setup, "ollama", model="qwen2.5:7b")
    out = capsys.readouterr().out
    assert "--provider ollama --probe-model --model qwen2.5:7b" in out
    main._print_provider_followup(setup, "ollama")
    assert "--model" not in capsys.readouterr().out


def test_the_wizard_passes_a_named_ollama_model_to_the_probe_command(wizard, capsys):
    wizard([""], ["ollama", "", "", "", "qwen2.5:7b"])
    assert "--provider ollama --probe-model --model qwen2.5:7b" in capsys.readouterr().out


def test_importing_the_setup_service_does_not_load_the_research_package():
    """The research package is the agent, its proposer and the provider SDK shims; the
    doctor needs none of it to say whether a key is set. (numpy still arrives through
    `tradeflow.services.__init__`, which is outside this module.)"""
    import subprocess
    import sys

    code = (
        "import sys, tradeflow.services.setup; "
        "bad = [m for m in ('tradeflow.research', 'tradeflow.research.agent') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


# --- what the doctor labels, and what exits non-zero ---------------------------
def test_earlier_checkout_state_is_a_notice_not_a_failure(tmp_path, monkeypatch):
    """It exited non-zero even for `--for demo`, which needs no state at all."""
    monkeypatch.setattr("tradeflow.settings.orphaned_checkout_state", lambda: tmp_path)
    checks = setup.run_checks(tmp_path / ".env", environ={})
    row = next(c for c in checks if c.name == "earlier state found")
    assert row.status == "not_checked" and row.next_action
    assert setup.doctor_ok(checks, "demo")


def test_a_row_that_gates_only_another_task_is_not_labelled_blocked(monkeypatch, capsys):
    from tradeflow import cli as main

    rows = [
        setup.Check("extra: mcp", "blocked", "not installed", needed_for=frozenset({"mcp"})),
        setup.Check("APCA_API_KEY_ID", "blocked", "missing", needed_for=frozenset({"data"})),
    ]
    monkeypatch.setattr(setup, "run_checks", lambda *a, **k: rows)
    with pytest.raises(SystemExit):
        main.parse_cli(["init", "--check"]).func(main.parse_cli(["init", "--check"]))
    out = capsys.readouterr().out.splitlines()
    mcp_line = next(line for line in out if "extra: mcp" in line)
    key_line = next(line for line in out if "APCA_API_KEY_ID" in line)
    assert "[optional]" in mcp_line and "[blocked]" in key_line
    # Names line up whatever the label's width.
    assert mcp_line.index("extra: mcp") == key_line.index("APCA_API_KEY_ID")
