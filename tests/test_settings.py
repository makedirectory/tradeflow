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
