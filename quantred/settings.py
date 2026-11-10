"""One place credentials and runtime settings come from - resolved, validated,
and with a clear error when something's missing.

Every setting is resolved in the same predictable order:

1. **Process environment variables** - the standard, 12-factor way.
2. **A ``.env`` file** in the state root (``~/.quantred/.env``, or
   ``$QUANTRED_HOME/.env``), loaded into the environment on first use. No dependency: a tiny built-in parser handles ``KEY=value`` lines, and
   real environment variables already set always win (standard dotenv behavior).

Alpaca credentials are required for any command that touches the market
(``scan``/``backtest``/``live``/``optimize``/...); the offline ``demo`` command
and the test suite need nothing. LLM provider keys for the research agent flow
through the very same resolver, so there is a single, obvious answer to "where
does this credential come from?"
"""

import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: Repo root (this file is ``<root>/quantred/settings.py``). Meaningful only in a
#: checkout — an installed copy lives in site-packages, where this points at nothing
#: useful, which is exactly why the state root below is resolved separately.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


#: The distribution name in ``pyproject.toml``. A checkout is identified by this
#: rather than by the mere presence of a ``pyproject.toml`` — every Python project
#: has one of those, and mistaking someone else's for ours would put our journal in
#: their repository.
DISTRIBUTION_NAME = "quantred"

#: Names this project was previously distributed under. A checkout of an older
#: revision still declares one of these, and must still be recognised as ours.
LEGACY_DISTRIBUTION_NAMES: Tuple[str, ...] = ("tradeflow-engine",)

#: The executable a reader types. The one definition of the command's name: argparse's
#: ``prog``, the version banner, the MCP launch command and every printed instruction
#: read it, so renaming the executable is this line and not a hunt through the
#: package. Deliberately separate from the import package (``quantred``), the
#: entry-point groups and :data:`DISTRIBUTION_NAME`, none of which a rename of the
#: command changes.
CLI_NAME = "quantred"

#: The product's display name, as a reader sees it in banners, report titles and
#: footers, chart titles and server log lines. Distinct from :data:`CLI_NAME` (what is
#: typed) and from the import package, which a rebrand does not touch.
BRAND = "QuantRed"

#: Executables that still launch this copy after :data:`CLI_NAME` changes. Never
#: printed as an instruction; looked up only where something must *find* a script
#: that exists on disk (an MCP client config written under an older name). This
#: distribution installs none of them: the ``tradeflow`` script comes from the final
#: ``tradeflow-engine`` release, which depends on this one, so it exists beside the
#: interpreter only when that package is installed.
LEGACY_CLI_NAMES: Tuple[str, ...] = ("tradeflow",)

#: The name this copy's MCP server reports to a client, and the key its client
#: configuration is written under. One definition for the server, the printed JSON
#: and the ``claude mcp add`` line.
MCP_SERVER_NAME = "quantred"

#: The documentation root. Pages are served from the site root (there is no ``/docs/``
#: prefix), so a link built from this constant and a page path is a page that exists.
DOCS_URL = "https://quantred.mk-dir.com"
#: Where the source lives: the one place an installed copy can get the example pack.
REPOSITORY_URL = "https://github.com/makedirectory/quantred"

#: The entry-point groups an installed pack registers strategies and scanners under.
#: Here, at the bottom layer, because both the registry that loads them and the
#: install hint that names the packages providing them need the same two strings.
STRATEGY_ENTRY_POINT_GROUP = "quantred.strategies"
SCANNER_ENTRY_POINT_GROUP = "quantred.scanners"

#: The groups packs registered under before the import package was renamed. Still read,
#: never written: a pack declared against the old names keeps registering without an
#: edit, and the import shim the old distribution ships lets its imports resolve.
LEGACY_ENTRY_POINT_GROUPS: Dict[str, Tuple[str, ...]] = {
    STRATEGY_ENTRY_POINT_GROUP: ("tradeflow.strategies",),
    SCANNER_ENTRY_POINT_GROUP: ("tradeflow.scanners",),
}


def entry_point_groups(group: str) -> Tuple[str, ...]:
    """``group`` and every older name it was registered under, current name first."""
    return (group, *LEGACY_ENTRY_POINT_GROUPS.get(group, ()))


def is_own_distribution(name: str) -> bool:
    """Whether a distribution ``name`` is this project, under its current or an old name.

    A receipt or entry point written while the project was still ``tradeflow-engine``
    describes this copy, not a pack someone else installed; counting it as a pack would
    tell a reader to re-add the engine to its own environment.
    """
    return name.lower().replace("_", "-") in (DISTRIBUTION_NAME, *LEGACY_DISTRIBUTION_NAMES)


def _looks_like_checkout(path: Path) -> bool:
    """Whether ``path`` is a QuantRed working copy rather than any old directory.

    Matched against :data:`DISTRIBUTION_NAME` so a rename cannot silently break
    detection, and so someone else's ``pyproject.toml`` is not mistaken for ours.

    The names in :data:`LEGACY_DISTRIBUTION_NAMES` count too: a working copy from before
    the rename declares the old one, and refusing to recognise it would give a developer
    in it the installed copy's instructions (``quantred demo`` and a ``uv tool`` install
    line) in place of ``python main.py`` and ``uv sync``. That is all this decides; where
    state lives is :func:`state_root`'s question and does not read it.

    Only the ``[project]`` table's ``name`` counts. A ``name = "quantred"`` line under
    some other table - an index called ``quantred``, say - belongs to a project that
    merely mentions us. (``tomllib`` would be exact, but it is 3.11+ and this supports
    3.10.)
    """
    manifest = path / "pyproject.toml"
    try:
        text = manifest.read_text()
    except OSError:
        return False
    names = {DISTRIBUTION_NAME, *LEGACY_DISTRIBUTION_NAMES}
    table = ""
    for line in text.splitlines():
        header = re.match(r"^\s*(\[\[?)\s*([^\[\]]*?)\s*\]\]?\s*(#.*)?$", line)
        if header:
            table = header.group(2) if header.group(1) == "[" else ""
            continue
        if table == "project":
            # Valid TOML allows indentation and single quotes; both are still the name.
            found = re.match(r"""^\s*name\s*=\s*(["'])([^"']*)\1\s*(#.*)?$""", line)
            if found and found.group(2) in names:
                return True
    return False


def invocation(command: str, *, make_target: Optional[str] = None) -> str:
    """How to run ``command`` on the copy that is actually running.

    An installed copy has no ``main.py`` and no Makefile, so "python main.py verdict"
    and "make demo" send the reader looking for files that were never there.

    Lives in ``settings`` because it is not a CLI concern and is needed below the
    services: a service that tells a reader how to fetch something, and this module's
    own missing-credentials message, need the same answer. A second copy of this
    decision is how one surface starts printing instructions another knows are wrong.
    ``services.setup.invocation`` and the CLI's ``_invocation`` delegate here.
    """
    if not started_from_checkout_shim():
        return f"{invoked_legacy_name() or CLI_NAME} {command}"
    return f"make {make_target}" if make_target else f"python main.py {command}"


def invoked_legacy_name() -> Optional[str]:
    """The old command name this process was started as, when it is the only one on PATH.

    A ``uv tool`` install upgraded in place from the old distribution exposes only
    ``tradeflow``: the reader types it, and an instruction saying ``quantred init``
    names a command their shell does not have. So the instruction echoes the name that
    ran - but only while :data:`CLI_NAME` is genuinely absent, so a reader who has both
    is told the current name. ``None`` otherwise.
    """
    started = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ""
    if started in LEGACY_CLI_NAMES and shutil.which(CLI_NAME) is None:
        return started
    return None


def started_from_checkout_shim() -> bool:
    """Whether this process was started the contributor's way - ``python main.py`` or a
    ``make`` target - from a working copy, so ``make``/``main.py`` instructions fit.

    A checkout's virtual environment also carries the installed command, and someone
    who typed it was told to run ``python main.py cache warm``, which fails anywhere
    but the repository root. So the instruction echoes how the process was started:
    the installed command's name when that is what ran, the shim otherwise. Anything
    unrecognised (a test runner, ``python -c``) keeps the checkout's own spelling.
    """
    if not running_from_checkout() or in_container():
        return False
    started = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ""
    return started not in (CLI_NAME, *LEGACY_CLI_NAMES)


#: Set by the project's Dockerfile. The image carries `main.py` and `pyproject.toml`, so
#: it looks like a checkout, but it has no Makefile and no `.env.example`: a container
#: told to run `make init` was sent to files that were never copied in.
CONTAINER_ENV = "QUANTRED_CONTAINER"


def in_container() -> bool:
    """Whether this process runs in the project's own image."""
    return bool(os.environ.get(CONTAINER_ENV))


def running_from_checkout() -> bool:
    """Whether the running ``quantred`` package was imported from a working copy.

    Decided by where the package itself lives, never by the working directory: an
    installed copy run from inside some checkout of this repository is still an
    installed copy, and telling its user to ``uv sync`` there is wrong. It also means
    there is no ``Path.cwd()`` on the path of the flatten report, the halt message or
    the missing-credentials message, which raises when the directory was deleted -
    in code that runs after money has already moved.

    Deliberately separate from :func:`state_root`. *How the software was reached*
    decides which instructions make sense - ``make demo`` against ``quantred demo``.
    *Where state lives* is a safety question with a different answer. Conflating them
    is how a message meant for a contributor reached an installed user, and it would
    now also send a contributor the wrong install command.
    """
    return _looks_like_checkout(PROJECT_ROOT)


#: The variable that names the state root and the default directory under the home
#: directory, with the names each had before the project was renamed. One definition:
#: every message that names either reads these.
STATE_HOME_ENV = "QUANTRED_HOME"
LEGACY_STATE_HOME_ENV = "TRADEFLOW_HOME"
STATE_DIR_NAME = ".quantred"
LEGACY_STATE_DIR_NAME = ".tradeflow"


def state_root() -> Path:
    """Where the journal, trial store, bar cache, and configs live.

    Resolved most explicit first:

    1. ``QUANTRED_HOME``, when set - the escape hatch, what a container mounts, and
       how a contributor opts back in to checkout-local state.
    2. ``TRADEFLOW_HOME``, the variable's name before the project was renamed, when
       that is set and the new one is not.
    3. ``~/.quantred`` - unless it does not exist and ``~/.tradeflow`` does, in which
       case the old directory is still the state root. Always, including inside a
       checkout.

    **The old names are read, never migrated.** State recorded before the rename is a
    journal, a position ledger and possibly a standing halt, and a root that moved by
    itself would leave any process still running the earlier version writing to a fresh,
    empty one beside it - with no halts. So an existing root keeps being used under its
    old name, :func:`legacy_state_note` says so on every run, and moving it is one
    command the owner runs when nothing else is using it.

    **The default does not depend on how you got here, and it never writes into a
    working copy.** It used to: a checkout was its own state root, so a contributor's
    ``logs/`` kept working the way it always had. That was convenient and it was wrong
    in two directions at once. A private strategy's trial evidence, saved configs and
    live position ledger accumulated inside a git working tree, protected by nothing
    but ``.gitignore``; and ``git clean -xd`` would take the lot, live ledger included.

    A user arriving through PyPI or over MCP may never learn there is a working tree,
    a ``logs/`` directory or an ignore file, so a warning cannot carry this. The safe
    root has to be the one they get without knowing to ask - which means it cannot be
    something ``init`` arranges, because ``init`` may never be run.

    The multiple-testing correction still rests on **one** journal per root: a campaign
    split across two of them deflates its Sharpe against half the evidence it should,
    and nothing errors. That is why the escape hatch is a single explicit variable
    rather than a per-pack or per-project rule.

    ``init --check`` prints the result, because "where did my trials go" must never be
    a question a user has to reverse-engineer - and after this change it is a question
    every existing contributor will have.
    """
    override = os.environ.get(STATE_HOME_ENV) or os.environ.get(LEGACY_STATE_HOME_ENV)
    if override:
        return Path(override).expanduser()
    current, legacy = Path.home() / STATE_DIR_NAME, Path.home() / LEGACY_STATE_DIR_NAME
    if not current.exists() and legacy.is_dir():
        return legacy
    return current


def legacy_state_note() -> Optional[str]:
    """What to tell someone whose state is still found under a pre-rename name.

    ``None`` when nothing old is in play. Three cases, each with the one command that
    ends it: the old variable is what named the root; the old default directory is the
    root because the new one does not exist; or both directories exist, so the old one
    is no longer read and whatever it holds is missing from every answer.
    """
    if os.environ.get(STATE_HOME_ENV):
        return None
    if os.environ.get(LEGACY_STATE_HOME_ENV):
        return (
            f"The state root comes from {LEGACY_STATE_HOME_ENV}, the variable's old name. "
            f"It still works; rename it to {STATE_HOME_ENV} when convenient."
        )
    current, legacy = Path.home() / STATE_DIR_NAME, Path.home() / LEGACY_STATE_DIR_NAME
    if not legacy.is_dir():
        return None
    if not current.exists():
        return (
            f"State is still read from ~/{LEGACY_STATE_DIR_NAME}, the directory's old name. "
            f"To move it, with nothing else running:  mv ~/{LEGACY_STATE_DIR_NAME} ~/{STATE_DIR_NAME}"
        )
    return (
        f"~/{LEGACY_STATE_DIR_NAME} still exists and is NOT being read: the state root is "
        f"~/{STATE_DIR_NAME}. Trials, the position ledger and any halt recorded in the old "
        f"directory are missing from every answer until you merge it in or remove it."
    )


def git_worktree_containing(path: Path) -> Optional[Path]:
    """The git working tree ``path`` sits inside, or ``None``.

    State inside a working tree is exposed to everything that acts on a repository:
    an ignore file one edit away from disclosing it, an archive of the tree, and
    ``git clean -xd``, which deletes it outright.
    """
    try:
        candidate = path.expanduser().resolve()
    except OSError:
        return None
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    return None


def orphaned_checkout_state() -> Optional[Path]:
    """A checkout's own state directory that is no longer the active root.

    Returned so it can be *said out loud*. Contributors accumulated journals and trial
    stores inside their working copies for as long as a checkout was its own root, and
    silently resolving somewhere else would deflate their next campaign against none of
    that evidence without erroring - the exact failure :func:`state_root` exists to
    prevent, caused by the fix for a different one.
    """
    if not _looks_like_checkout(PROJECT_ROOT):
        return None
    logs = PROJECT_ROOT / "logs"
    try:
        if state_root().expanduser().resolve() == PROJECT_ROOT.resolve():
            return None
        has_evidence = any(
            (logs / name).exists()
            for name in ("research_journal.jsonl", "trials.db", "position_ledger.jsonl")
        )
    except OSError:
        return None
    return PROJECT_ROOT if has_evidence else None


def trial_journal_path() -> Path:
    """Where the research journal lives — one definition, for every reader and writer.

    It was defined twice, in ``services.audit`` and ``store.trials``, kept in step by a
    comment. Had they diverged the store would have indexed a different file from the
    one being written, and the multiple-testing correction would have deflated against
    half its evidence — with nothing erroring, because both paths are valid.

    Here rather than in either of them because ``settings`` is the layer both already
    depend on; the reverse would be a cycle.
    """
    return state_path("logs", "research_journal.jsonl")


def state_path(*parts: str) -> Path:
    """A path under the state root, with its parent directory created.

    Callers ask for ``state_path("logs", "research_journal.jsonl")`` rather than
    building a relative path, so there is one answer to where state lives.
    """
    path = state_root().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


#: Where credentials are read from: the state root's ``.env``, the same file for a
#: checkout, an installed copy and the container. A ``.env`` in a checkout's own root
#: is never read - one location, so a key is never in the file nobody consults.
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


#: Ollama's default local endpoint.
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120.0


def normalize_ollama_url(url: str) -> str:
    """The server root, whether the user gave it bare or as a full chat/generate URL.

    People copy the URL from the API reference, which names the endpoint, and the client
    appends the endpoint itself - so ``.../api/chat`` would otherwise become
    ``.../api/chat/api/chat`` and fail as a 404 that reads like a missing model.
    """
    cleaned = url.strip().rstrip("/")
    for suffix in ("/api/chat", "/api/generate"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.rstrip("/")


def ollama_base_url() -> str:
    """``OLLAMA_BASE_URL`` (environment / ``.env``), normalised; no network."""
    return normalize_ollama_url(get_credential("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL) or DEFAULT_OLLAMA_URL)


def parse_ollama_timeout(raw: Optional[str]) -> float:
    """``OLLAMA_TIMEOUT`` text as seconds; blank means the default.

    One parser for the settings reader and for ``init``, so a value the wizard accepts
    is exactly one a research run will accept.
    """
    if raw is None or not raw.strip():
        return DEFAULT_OLLAMA_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not value > 0 or value == float("inf"):
        raise SettingsError(f"OLLAMA_TIMEOUT must be a positive number of seconds, got {raw!r}.")
    return value


def parse_ollama_num_ctx(raw: Optional[str]) -> Optional[int]:
    """``OLLAMA_NUM_CTX`` text as a token count; blank means send nothing (``None``)."""
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise SettingsError(f"OLLAMA_NUM_CTX must be a positive whole number of tokens, got {raw!r}.")
    return value


def ollama_timeout() -> float:
    """``OLLAMA_TIMEOUT`` in seconds. A slow local GPU can need minutes per response."""
    return parse_ollama_timeout(get_credential("OLLAMA_TIMEOUT"))


def ollama_num_ctx() -> Optional[int]:
    """``OLLAMA_NUM_CTX``: the context length to request, or ``None`` to send nothing.

    Never inferred - a model's own default is the right behaviour until someone with a
    memory-constrained machine says otherwise.
    """
    return parse_ollama_num_ctx(get_credential("OLLAMA_NUM_CTX"))


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


def paper_trade_mode() -> bool:
    """Whether orders would go to the paper account, without needing credentials.

    Separate from :func:`load_settings` because reporting the broker mode must work in
    a preflight that has not yet proved it can reach a venue - refusing to say "this is
    paper" because the keys are missing would withhold the one fact most worth printing
    before an order path starts.
    """
    return _resolve_paper_trade()


#: Feeds a live run may be pinned to. ``sip`` is the full consolidated tape;
#: ``iex`` is a single venue, free but partial.
DATA_FEEDS = ("iex", "sip", "delayed_sip")


def data_feed() -> Optional[str]:
    """Which Alpaca market-data feed to pin, or ``None`` to leave the SDK's defaults.

    ``None`` is deliberately the default and must stay that way. Pinning a feed
    globally would mean an entitled account silently trading a partial venue, or a
    delayed tape, with nothing in the output to say so - the failure this exists to
    prevent, inverted. A run that needs a specific feed says so explicitly.
    """
    _load_dotenv_once()
    value = (os.environ.get("ALPACA_DATA_FEED") or "").strip().lower()
    if not value:
        return None
    if value not in DATA_FEEDS:
        raise SettingsError(
            f"ALPACA_DATA_FEED={value!r} is not a feed this supports ({', '.join(DATA_FEEDS)})."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings - currently the Alpaca account credentials."""

    alpaca_key: str
    alpaca_secret: str
    paper_trade: bool = True


#: Where free paper-trading keys come from, said the same way everywhere.
ALPACA_KEYS_URL = "https://app.alpaca.markets/ (Paper Account -> API Keys)"


def _missing_credentials_message(missing: list) -> str:
    """What to do about missing keys, phrased for the copy that is running.

    This is the first thing most people see, and it used to tell an installed user
    to copy a ``.env.example`` they do not have and run a ``make`` target that does
    not exist — two dead ends and no mention of the one command that actually fixes
    it. A pip-installed copy has no repository, so any instruction that assumes one
    is worse than no instruction: it sends someone looking for a file that was never
    there.
    """
    setup_path = state_root() / ".env"
    lines = [
        "Missing Alpaca credentials: " + ", ".join(missing) + ".",
        # Said outright because the reader has usually just run the keyless demo, and
        # nothing else told them why this command is different.
        "Fetching market data and reaching the broker both need them; most research commands "
        "(not `research` or `demo-agent`) can run --offline from bars already in the local cache without them.",
    ]

    if started_from_checkout_shim():
        lines += [
            "",
            f"  make init                 guided setup — writes {setup_path}",
            "                            (or copy .env.example there and fill it in)",
            "  make demo                 no keys needed — the full pipeline on synthetic data",
        ]
    elif in_container():
        # The compose file mounts the host's state-root .env read-only at the container's
        # state root, so `init` inside the container could not write it; the keys go in
        # on the host.
        lines += [
            "",
            "  Put them in the host's ~/.quantred/.env (or $QUANTRED_HOME/.env); compose "
            f"mounts it read-only at {setup_path}.",
            f"  {invocation('demo'):<25} no keys needed — the full pipeline on synthetic data",
        ]
    else:
        lines += [
            "",
            f"  {invocation('init'):<25} guided setup — writes " + str(setup_path),
            f"  {invocation('demo'):<25} no keys needed — the full pipeline on synthetic data",
        ]
    lines += [
        "",
        f"Free paper-trading keys: {ALPACA_KEYS_URL}",
        "Environment variables (APCA_API_KEY_ID / APCA_API_SECRET_KEY) are honored too.",
    ]
    return "\n".join(lines)


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
        raise SettingsError(_missing_credentials_message(missing))
    return Settings(alpaca_key=key, alpaca_secret=secret, paper_trade=_resolve_paper_trade())
