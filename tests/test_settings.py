"""Tests for credential resolution (env / .env) + validation."""

from pathlib import Path

import pytest

from tradeflow import settings
from tradeflow.settings import SettingsError, load_settings


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Run each test with no .env file and a clean environment."""
    monkeypatch.setattr(settings, "_dotenv_loaded", False)
    monkeypatch.setattr(settings, "ENV_PATH", Path("/nonexistent/.env"))
    for key in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_env_vars_are_used(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "key123")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret456")
    monkeypatch.setenv("PAPER_TRADE", "false")

    s = load_settings()
    assert s.alpaca_key == "key123"
    assert s.alpaca_secret == "secret456"
    assert s.paper_trade is False


def test_paper_trade_defaults_true(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "key123")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret456")
    assert load_settings().paper_trade is True


def test_missing_credentials_raise():
    with pytest.raises(SettingsError) as exc:
        load_settings()
    assert "APCA_API_KEY_ID" in str(exc.value)


def test_placeholder_values_count_as_missing(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "<APCA_API_KEY_ID>")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "<APCA_API_SECRET_KEY>")
    with pytest.raises(SettingsError):
        load_settings()


def test_get_credential_falls_back_to_default():
    assert settings.get_credential("DOES_NOT_EXIST", default="fallback") == "fallback"


def test_dotenv_file_is_loaded(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('# a comment\nAPCA_API_KEY_ID="filek"\nAPCA_API_SECRET_KEY=files\nPAPER_TRADE=true\n')
    monkeypatch.setattr(settings, "ENV_PATH", env_file)
    monkeypatch.setattr(settings, "_dotenv_loaded", False)

    s = load_settings()
    assert s.alpaca_key == "filek"
    assert s.alpaca_secret == "files"


# --- the message a keyless user actually sees -------------------------------
def _missing_message(monkeypatch, tmp_path, *, checkout: bool):
    from tradeflow import settings

    monkeypatch.setenv("TRADEFLOW_HOME", str(tmp_path))
    monkeypatch.setattr(settings, "_looks_like_checkout", lambda _p: checkout)
    monkeypatch.setattr(settings, "get_credential", lambda *a, **k: None)
    try:
        settings.load_settings()
    except settings.SettingsError as exc:
        return str(exc)
    raise AssertionError("expected SettingsError")


def test_an_installed_copy_is_never_told_to_use_files_it_does_not_have(monkeypatch, tmp_path):
    """This is the first thing most people see. It used to tell a pip-installed user
    to copy a .env.example they do not have and run a make target that does not
    exist — two dead ends, and no mention of the command that actually fixes it."""
    message = _missing_message(monkeypatch, tmp_path, checkout=False)

    assert "tradeflow init" in message
    assert ".env.example" not in message
    assert "make " not in message
    assert str(tmp_path / ".env") in message  # says exactly where the file goes


def test_a_checkout_keeps_its_own_instructions(monkeypatch, tmp_path):
    message = _missing_message(monkeypatch, tmp_path, checkout=True)
    assert "make init" in message
    assert ".env.example" in message


def test_both_paths_name_the_keyless_demo_and_where_keys_come_from(monkeypatch, tmp_path):
    for checkout in (True, False):
        message = _missing_message(monkeypatch, tmp_path, checkout=checkout)
        assert "demo" in message, "someone with no keys must be told they can still try it"
        assert "alpaca.markets" in message
        assert "APCA_API_KEY_ID" in message  # the env-var path, for scripts
