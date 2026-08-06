"""First-run setup: inspecting, validating, and writing the ``.env`` contract.

Pure functions with no prompts and no printing, so the interactive wizard in the
CLI is a thin sequence over them and every branch is testable offline. The
prompting lives in ``main.py``; the decisions live here.

**Secrets hygiene is the whole point.** A key that reaches this module must not
reach a log, a journal, an exception message, or a printed summary. Every value
that comes back out for display is masked, and the validation path is built to
report *why* it failed without quoting what was submitted.

**Never destroy an existing setup.** A ``.env`` is shared dotenv convention and may
hold keys this project does not own. Rewrites preserve every line this project
did not touch, byte for byte, and a backup is written before any modification.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.settings import _PLACEHOLDERS, ENV_PATH

logger = logging.getLogger(__name__)

#: The keys this project owns. Anything else in a ``.env`` belongs to someone else
#: and is preserved untouched.
OWNED_KEYS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE")

#: The two that must be real for any command that touches the market.
CREDENTIAL_KEYS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")

#: A small, liquid default universe for the optional first cache warm.
DEFAULT_WARM_UNIVERSE = ("SPY", "AAPL", "MSFT", "NVDA", "AMZN")


def is_placeholder(value: Optional[str]) -> bool:
    """Whether a value means "not actually filled in".

    Defers to ``settings.py``'s own placeholder set rather than re-deciding what
    counts as unset — two definitions of "filled in" will drift, and the one that
    matters is the one the loader enforces.
    """
    return value is None or value.strip() in _PLACEHOLDERS


def mask(value: Optional[str]) -> str:
    """A value in a form safe to print: ``AKFO…9J2X``, or a plain absence marker.

    The only representation of a secret this module ever emits. Short values are
    masked entirely rather than partially, since revealing most of a short string
    reveals the string.
    """
    if not value:
        return "(not set)"
    if is_placeholder(value):
        return "(placeholder)"
    stripped = value.strip()
    if len(stripped) < 12:
        return "*" * len(stripped)
    return f"{stripped[:4]}…{stripped[-4:]}"


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
@dataclass
class EnvState:
    """What the current ``.env`` (and process environment) actually contains."""

    path: Path
    exists: bool
    values: Dict[str, str] = field(default_factory=dict)
    from_environment: Dict[str, str] = field(default_factory=dict)

    @property
    def missing(self) -> List[str]:
        """Credential keys that are absent or still a placeholder."""
        return [k for k in CREDENTIAL_KEYS if is_placeholder(self.resolved(k))]

    @property
    def complete(self) -> bool:
        return not self.missing

    def resolved(self, key: str) -> Optional[str]:
        """The value a command would actually see: a real environment variable
        wins over the file, exactly as the settings loader resolves it."""
        return self.from_environment.get(key) or self.values.get(key)

    def summary(self) -> Dict[str, str]:
        """A printable, fully masked view. Never returns a raw secret."""
        return {key: mask(self.resolved(key)) for key in OWNED_KEYS}


def inspect_env(path: Optional[Any] = None, environ: Optional[Dict[str, str]] = None) -> EnvState:
    """Read the current setup without changing anything.

    Reports what the file holds *and* what the process environment overrides, so a
    user whose keys come from an exported variable is not told to fill in a file
    that would have no effect.
    """
    env_path = Path(path) if path else ENV_PATH
    environ = os.environ if environ is None else environ
    values: Dict[str, str] = {}
    if env_path.exists():
        for key, value, _ in _parse_env_lines(env_path.read_text().splitlines()):
            if key is not None:
                values[key] = value
    from_environment = {k: environ[k] for k in OWNED_KEYS if environ.get(k)}
    return EnvState(path=env_path, exists=env_path.exists(), values=values, from_environment=from_environment)


def _parse_env_lines(lines: List[str]) -> List[Tuple[Optional[str], str, str]]:
    """Each line as ``(key or None, value, raw)``.

    ``key is None`` marks a comment or blank — a line to preserve verbatim rather
    than interpret.
    """
    out: List[Tuple[Optional[str], str, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            out.append((None, "", raw))
            continue
        key, _, value = line.partition("=")
        out.append((key.strip(), value.strip().strip('"').strip("'"), raw))
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_env(updates: Dict[str, str], path: Optional[Any] = None, *, backup: bool = True) -> Dict[str, Any]:
    """Apply ``updates`` to the ``.env``, preserving everything else exactly.

    Three properties this must have, in order of how badly their absence would
    hurt:

    1. **Nothing else changes.** Comments, blank lines, ordering, and keys this
       project does not own survive byte for byte. A dotenv file is shared
       territory.
    2. **A backup exists first.** ``.env.bak`` is written before any modification,
       so a wrong answer at a prompt is recoverable.
    3. **The write is atomic.** A temp file in the same directory, then a rename —
       an interruption leaves either the old file or the new one, never a torn one
       that the loader would read as half-configured credentials.

    Returns a masked summary of what changed; never echoes a value.
    """
    env_path = Path(path) if path else ENV_PATH
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = env_path.read_text().splitlines() if env_path.exists() else []

    backup_path = None
    if backup and env_path.exists():
        backup_path = env_path.with_suffix(env_path.suffix + ".bak")
        shutil.copy2(env_path, backup_path)

    remaining = dict(updates)
    out_lines: List[str] = []
    for key, _value, raw in _parse_env_lines(existing):
        if key is not None and key in remaining:
            out_lines.append(f"{key}={remaining.pop(key)}")
        else:
            out_lines.append(raw)
    if remaining:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.extend(f"{key}={value}" for key, value in remaining.items())

    temp_path = env_path.with_name(env_path.name + ".tmp")
    temp_path.write_text("\n".join(out_lines) + "\n")
    try:
        os.replace(temp_path, env_path)
    except OSError:
        # The original file is untouched, which is the point; clean up the partial
        # so a later run does not find a stray file holding a copy of a secret.
        temp_path.unlink(missing_ok=True)
        raise

    return {
        "path": str(env_path),
        "backup": str(backup_path) if backup_path else None,
        "updated": {key: mask(value) for key, value in updates.items()},
    }


# --------------------------------------------------------------------------- #
# Credential validation
# --------------------------------------------------------------------------- #
#: The three distinguishable outcomes. "Could not reach Alpaca" is not "keys
#: rejected", and telling a user on hotel wifi to regenerate working keys is a
#: worse failure than saying nothing.
CREDENTIALS_OK = "ok"
CREDENTIALS_REJECTED = "rejected"
CREDENTIALS_UNREACHABLE = "unreachable"


@dataclass
class CredentialCheck:
    status: str
    message: str

    @property
    def ok(self) -> bool:
        return self.status == CREDENTIALS_OK


def check_credentials(data_client, symbol: str = "SPY") -> CredentialCheck:
    """Validate credentials with one cheap historical-data request.

    Takes a **data** client, never a trading client — the same structural rule the
    MCP server follows. Validating a key must not require the ability to trade
    with it.

    The three outcomes are reported distinctly, and no message ever contains the
    submitted credential: an error string from a vendor SDK can echo request
    parameters, so the failure text is composed here rather than passed through.
    """
    end = datetime.now()
    start = end - timedelta(days=7)
    try:
        bars = data_client.get_bars([symbol], "1Day", start, end)
    except Exception as exc:  # noqa: BLE001 - the vendor's exception types are its own
        if _looks_like_auth_failure(exc):
            return CredentialCheck(
                CREDENTIALS_REJECTED,
                "Alpaca rejected these credentials. Check that you copied the key and "
                "secret from the Paper Account section (paper and live keys are different).",
            )
        return CredentialCheck(
            CREDENTIALS_UNREACHABLE,
            "Could not reach Alpaca — this looks like a network or rate-limit problem, "
            "not a credential problem. The keys have been saved; try a command again later.",
        )
    if not bars:
        # The request succeeded, so the credentials were accepted; an empty response
        # is a data-availability fact, not an authentication one, and must not be
        # reported as a rejection.
        return CredentialCheck(
            CREDENTIALS_UNREACHABLE,
            f"Reached Alpaca but got no bars for {symbol}. The credentials were accepted; "
            "market data may be unavailable for this window (a holiday, or a delayed feed).",
        )
    return CredentialCheck(CREDENTIALS_OK, "Credentials work — Alpaca returned market data.")


def _looks_like_auth_failure(exc: Exception) -> bool:
    """Whether an exception reads as "rejected" rather than "unreachable".

    Deliberately conservative: anything ambiguous is reported as unreachable,
    because telling someone their working keys are invalid sends them to
    regenerate credentials that were never the problem.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("401", "403", "unauthorized", "forbidden", "invalid key", "authentication")
    )


# --------------------------------------------------------------------------- #
# Doctor mode
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def run_checks(path: Optional[Any] = None, environ: Optional[Dict[str, str]] = None) -> List[Check]:
    """Diagnose the current setup: one row per check, all of them evaluated.

    A doctor, not a linter — it reports every check independently rather than
    stopping at the first miss, because the second and third problems are exactly
    what someone running a diagnostic wants to know about. It writes nothing and
    makes no network call; credential *validity* needs a request and is a separate,
    explicit step.
    """
    state = inspect_env(path, environ)
    checks = [
        Check(
            "env file",
            state.exists,
            f"{state.path} exists" if state.exists else f"{state.path} not found — run `init` to create it",
        )
    ]
    for key in CREDENTIAL_KEYS:
        value = state.resolved(key)
        source = " (from the environment)" if key in state.from_environment else ""
        checks.append(
            Check(
                key,
                not is_placeholder(value),
                f"{mask(value)}{source}"
                if not is_placeholder(value)
                else f"{mask(value)} — no command that touches the market will run",
            )
        )
    paper = state.resolved("PAPER_TRADE")
    paper_on = paper is None or str(paper).strip().lower() in {"1", "true", "yes", "on"}
    checks.append(
        Check(
            "PAPER_TRADE",
            paper_on,
            "true — orders go to the paper account"
            if paper_on
            else "FALSE — live trading is enabled and orders would use real money",
        )
    )
    checks.append(_writable_check())
    checks.extend(_extras_checks())
    return checks


def _writable_check() -> Check:
    """Whether the journal/cache directory can actually be written.

    A research session that cannot journal silently loses its own trial history,
    which corrupts every multiple-testing correction after it — worth checking
    before the first run rather than discovering later.
    """
    from src.services.audit import DEFAULT_TRIAL_JOURNAL

    directory = Path(DEFAULT_TRIAL_JOURNAL).parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-probe"
        probe.write_text("")
        probe.unlink()
    except OSError as exc:
        return Check("state directory", False, f"{directory} is not writable ({exc.strerror})")
    return Check("state directory", True, f"{directory} is writable")


#: Optional extras, what each unlocks, and the module that proves it is installed.
_EXTRAS = (
    ("store", "pyarrow", "the bar cache (`--cache`/`--offline`) and the Parquet bar store"),
    ("viz", "matplotlib", "charts in `--chart` and in HTML reports"),
    ("mcp", "mcp", "the MCP server (`python main.py mcp`)"),
    ("optimize", "sklearn", "`optimize --method bayesian`"),
    ("portfolio", "ortools", "`allocate` (the constraint solver)"),
)


def _extras_checks() -> List[Check]:
    """One row per optional extra. A missing extra is reported with the exact
    command to install it — the wizard configures, the user installs."""
    import importlib.util

    checks = []
    for extra, module, what in _EXTRAS:
        present = importlib.util.find_spec(module) is not None
        checks.append(
            Check(
                f"extra: {extra}",
                present,
                f"installed — {what}" if present else f"not installed — {what}: uv sync --extra {extra}",
            )
        )
    return checks


def build_updates(
    key: Optional[str] = None, secret: Optional[str] = None, paper_trade: Optional[bool] = None
) -> Dict[str, str]:
    """The ``.env`` updates for the answers given, skipping anything left blank.

    Only what was actually answered is written, so filling in one missing key
    never rewrites the other.
    """
    updates: Dict[str, str] = {}
    if key:
        updates["APCA_API_KEY_ID"] = key.strip()
    if secret:
        updates["APCA_API_SECRET_KEY"] = secret.strip()
    if paper_trade is not None:
        updates["PAPER_TRADE"] = "true" if paper_trade else "false"
    return updates
