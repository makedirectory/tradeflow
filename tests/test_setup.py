"""First-run setup: secrets hygiene, non-destructive writes, and the doctor.

All offline. The properties defended here are the ones whose absence would be
expensive rather than annoying: a secret reaching a log or an error message, a
rewrite eating a key this project does not own, a torn file that reads as
half-configured credentials, and a network blip reported as "your keys are wrong".
"""

from datetime import datetime

import pytest

from src.services import setup

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
    from src.settings import _PLACEHOLDERS

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


def test_a_missing_extra_reports_the_exact_install_command(tmp_path):
    rows = [c for c in setup.run_checks(tmp_path / ".env", environ={}) if c.name.startswith("extra:")]
    assert rows
    for row in rows:
        if not row.passed:
            assert "uv sync --extra" in row.detail


def test_the_doctor_never_prints_a_raw_credential(tmp_path):
    path = _write(tmp_path, f"APCA_API_KEY_ID={REAL_KEY}\nAPCA_API_SECRET_KEY={REAL_SECRET}\n")
    rendered = "\n".join(f"{c.name} {c.detail}" for c in setup.run_checks(path, environ={}))
    assert REAL_KEY not in rendered and REAL_SECRET not in rendered


# --- the CLI surface --------------------------------------------------------
def test_cli_check_exits_non_zero_when_credentials_are_missing(tmp_path, monkeypatch, capsys):
    import main

    monkeypatch.setattr(
        setup, "run_checks", lambda *a, **k: [setup.Check("APCA_API_KEY_ID", False, "missing")]
    )
    args = main.build_parser().parse_args(["init", "--check"])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_check_ignores_missing_optional_extras(tmp_path, monkeypatch, capsys):
    """A missing extra is not a broken setup — only the essentials fail the run."""
    import main

    monkeypatch.setattr(
        setup,
        "run_checks",
        lambda *a, **k: [setup.Check("extra: viz", False, "not installed — uv sync --extra viz")],
    )
    args = main.build_parser().parse_args(["init", "--check"])
    args.func(args)
    assert "Setup looks good" in capsys.readouterr().out


def test_cli_non_interactive_builds_the_env_from_the_environment(tmp_path, monkeypatch, capsys):
    import main

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
    import main

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("PAPER_TRADE", raising=False)
    args = main.build_parser().parse_args(["init", "--non-interactive", "--env-path", str(tmp_path / ".env")])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert "APCA_API_KEY_ID" in str(exc.value)


def test_cli_interactive_skip_path_writes_nothing(tmp_path, monkeypatch, capsys):
    """Pressing Enter at the key prompt leaves a valid keyless demo setup."""
    import getpass

    import main

    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "")
    path = tmp_path / ".env"
    args = main.build_parser().parse_args(["init", "--env-path", str(path)])
    args.func(args)

    assert not path.exists()
    out = capsys.readouterr().out
    assert "Skipped" in out and "make demo" in out


def test_cli_interactive_writes_keys_and_confirms_paper_trading(tmp_path, monkeypatch, capsys):
    import getpass

    import main

    answers = iter([REAL_KEY, REAL_SECRET])
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: next(answers))
    # Keep PAPER_TRADE, then decline the cache warm.
    replies = iter(["y", "n"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(replies))
    monkeypatch.setattr(
        setup, "check_credentials", lambda *a, **k: setup.CredentialCheck(setup.CREDENTIALS_OK, "fine")
    )
    monkeypatch.setattr("src.services.data.build_data_client", lambda *a, **k: object())

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

    import main

    answers = iter([REAL_KEY, REAL_SECRET])
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: next(answers))
    # Decline to keep paper trading, then fail the typed confirmation, then no warm.
    replies = iter(["n", "yes please", "n"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(replies))
    monkeypatch.setattr(
        setup, "check_credentials", lambda *a, **k: setup.CredentialCheck(setup.CREDENTIALS_OK, "fine")
    )
    monkeypatch.setattr("src.services.data.build_data_client", lambda *a, **k: object())

    path = tmp_path / ".env"
    args = main.build_parser().parse_args(["init", "--env-path", str(path)])
    args.func(args)
    assert "PAPER_TRADE=true" in path.read_text()
