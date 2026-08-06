"""One place credentials and runtime settings come from - resolved, validated,
and with a clear error when something's missing.

Every setting is resolved in the same predictable order:

1. **Process environment variables** - the standard, 12-factor way.
2. **A ``.env`` file** in the project root, loaded into the environment on first
   use. No dependency: a tiny built-in parser handles ``KEY=value`` lines, and
   real environment variables already set always win (standard dotenv behavior).

Alpaca credentials are required for any command that touches the market
(``scan``/``backtest``/``live``/``optimize``/...); the offline ``demo`` command
and the test suite need nothing. LLM provider keys for the research agent flow
through the very same resolver, so there is a single, obvious answer to "where
does this credential come from?"
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Repo root (this file is ``<root>/tradeflow/settings.py``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

#: Values that mean "not actually filled in" (the ``.env.example`` placeholders).
_PLACEHOLDERS = frozenset({"", "<APCA_API_KEY_ID>", "<APCA_API_SECRET_KEY>"})

_dotenv_loaded = False


class SettingsError(RuntimeError):
    """Raised when required configuration is missing or malformed.

    Carries a human-readable, actionable message; CLI entry points print it and
    exit cleanly rather than dumping a traceback.
    """


def _load_dotenv_once() -> None:
    """Parse ``.env`` into ``os.environ`` exactly once (no-op if absent).

    Minimal on purpose: ``KEY=value`` lines, ``#`` comments and blanks skipped,
    surrounding quotes stripped. Existing environment variables are never
    overwritten, so an explicit ``export`` still takes precedence over the file.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_credential(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a single named setting from the environment (or ``.env``).

    Used for LLM provider keys (``ANTHROPIC_API_KEY`` ...) and the Alpaca keys
    alike, so every credential follows one rule.
    """
    _load_dotenv_once()
    return os.environ.get(name) or default


def _parse_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_paper_trade() -> bool:
    """Resolve ``PAPER_TRADE`` (defaults to ``True`` - the safe choice)."""
    _load_dotenv_once()
    if "PAPER_TRADE" in os.environ:
        return _parse_bool(os.environ["PAPER_TRADE"])
    return True


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings - currently the Alpaca account credentials."""

    alpaca_key: str
    alpaca_secret: str
    paper_trade: bool = True


def load_settings() -> Settings:
    """Resolve and validate the Alpaca credentials, or raise :class:`SettingsError`.

    Raising (rather than returning partial settings) means no command can run
    against the market with half-configured keys.
    """
    key = get_credential("APCA_API_KEY_ID")
    secret = get_credential("APCA_API_SECRET_KEY")

    missing = [
        name
        for name, value in (("APCA_API_KEY_ID", key), ("APCA_API_SECRET_KEY", secret))
        if not value or value in _PLACEHOLDERS
    ]
    if missing:
        raise SettingsError(
            "Missing Alpaca credentials: "
            + ", ".join(missing)
            + ".\nCopy .env.example to .env and add your Alpaca paper-trading keys "
            "(free at https://app.alpaca.markets/ -> Paper Account -> API Keys).\n"
            "Environment variables are honored too. "
            "No keys needed to explore? Run `make demo`."
        )
    return Settings(alpaca_key=key, alpaca_secret=secret, paper_trade=_resolve_paper_trade())
