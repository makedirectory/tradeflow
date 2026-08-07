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

#: Repo root (this file is ``<root>/tradeflow/settings.py``). Meaningful only in a
#: checkout — an installed copy lives in site-packages, where this points at nothing
#: useful, which is exactly why the state root below is resolved separately.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


#: The distribution name in ``pyproject.toml``. A checkout is identified by this
#: rather than by the mere presence of a ``pyproject.toml`` — every Python project
#: has one of those, and mistaking someone else's for ours would put our journal in
#: their repository.
DISTRIBUTION_NAME = "tradeflow-engine"


def _looks_like_checkout(path: Path) -> bool:
    """Whether ``path`` is a TradeFlow working copy rather than any old directory.

    Matched against :data:`DISTRIBUTION_NAME` so a rename cannot silently break
    detection: getting this wrong sends a developer's journal to ``~/.tradeflow``
    while their repo still holds the old one, which is the split-campaign failure
    :func:`state_root` exists to prevent — and it fails silently, with no error.
    """
    manifest = path / "pyproject.toml"
    try:
        return f'name = "{DISTRIBUTION_NAME}"' in manifest.read_text()
    except OSError:
        return False


def state_root() -> Path:
    """Where the journal, trial store, bar cache, and configs live.

    Every path in this project used to be relative to the working directory, which
    is right in a checkout and wrong everywhere else: an installed ``tradeflow``
    would write its journal into whatever directory the user happened to be standing
    in, scattering a campaign across the filesystem.

    That is the failure worth preventing. The entire multiple-testing correction
    rests on **one** journal accumulating every trial — a campaign split across two
    roots deflates its Sharpe against half the evidence it should, and nothing
    errors. So the root is resolved in one place, most explicit first:

    1. ``TRADEFLOW_HOME``, when set — the escape hatch, and what a container mounts.
    2. The current directory, when it is a TradeFlow checkout — so a developer's
       ``logs/`` and ``configs/`` keep working exactly as they always have.
    3. ``~/.tradeflow`` otherwise — one predictable home for an installed copy.

    ``init --check`` prints the result, because "where did my trials go" must never
    be a question a user has to reverse-engineer.
    """
    override = os.environ.get("TRADEFLOW_HOME")
    if override:
        return Path(override).expanduser()
    cwd = Path.cwd()
    if _looks_like_checkout(cwd):
        return cwd
    if _looks_like_checkout(PROJECT_ROOT):
        return PROJECT_ROOT
    return Path.home() / ".tradeflow"


def state_path(*parts: str) -> Path:
    """A path under the state root, with its parent directory created.

    Callers ask for ``state_path("logs", "research_journal.jsonl")`` rather than
    building a relative path, so there is one answer to where state lives.
    """
    path = state_root().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


#: Where credentials are read from. In a checkout this is the repo's own ``.env``,
#: exactly as before; an installed copy reads ``~/.tradeflow/.env``, since there is
#: no repo to put one in.
ENV_PATH = state_root() / ".env"

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
