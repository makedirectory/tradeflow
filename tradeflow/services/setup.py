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
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple

from tradeflow.research.provider_errors import (
    ModelProviderError,
    ProbeCategory,
    classify_provider_error,
)
from tradeflow.settings import _PLACEHOLDERS, DISTRIBUTION_NAME, ENV_PATH

logger = logging.getLogger(__name__)

#: The keys this project owns. Anything else in a ``.env`` belongs to someone else
#: and is preserved untouched.
OWNED_KEYS = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "PAPER_TRADE",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_TIMEOUT",
    "OLLAMA_NUM_CTX",
)

#: Owned keys that are secrets and so are never printed in full. Everything else owned
#: (a URL, a timeout) is shown as it is - masking it hides nothing and costs the reader
#: the value they most want to confirm.
SECRET_KEYS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")

#: The model providers ``init`` can set up, and the extra each hosted one needs.
MODEL_PROVIDERS = ("anthropic", "openai", "ollama")
PROVIDER_EXTRA = {"anthropic": "ai", "openai": "openai"}
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

#: The two that must be real for any command that touches the market.
CREDENTIAL_KEYS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")

#: A small, liquid default universe for the optional first cache warm.
DEFAULT_WARM_UNIVERSE = ("SPY", "AAPL", "MSFT", "NVDA", "AMZN")


_BASE_KEYS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "PAPER_TRADE")


def invocation(command: str, *, make_target: Optional[str] = None) -> str:
    """How to run ``command`` on the copy that is actually running.

    An installed copy has no ``main.py`` and no Makefile, so "python main.py verdict"
    and "make demo" send the reader looking for files that were never there.

    Lives here rather than in the CLI because it is not a CLI concern: a service that
    tells a reader how to fetch something — a config's campaign block naming the
    command that reads its stored trades, say — needs the same answer, and a second
    copy of this decision is how one surface starts printing instructions the other
    knows are wrong.
    """
    from tradeflow.settings import running_from_checkout

    if not running_from_checkout():
        return f"tradeflow {command}"
    return f"make {make_target}" if make_target else f"python main.py {command}"


def mcp_launch_command() -> List[str]:
    """The command an MCP client should run to start *this* copy's server.

    One definition for three readers: ``mcp --self-test`` runs it, ``init`` prints it,
    and the documented client config is the same shape. A client spawns the process
    itself - no shell, no working directory, often none of the PATH a terminal has - so
    everything here is absolute. A self-test that passed on a different command than the
    one a reader was told to configure would be reporting on something nobody uses.

    * **Installed:** the ``tradeflow`` script that sits next to the running interpreter
      (a tool environment's own ``bin``), else ``python -m tradeflow.cli mcp``.
    * **Checkout:** this interpreter and the absolute ``main.py``, which needs neither
      ``uv`` on the client's PATH nor a ``cwd``.
    """
    import sys

    from tradeflow.settings import PROJECT_ROOT, _looks_like_checkout

    python = Path(sys.executable)
    main = PROJECT_ROOT / "main.py"
    if _looks_like_checkout(PROJECT_ROOT) and main.is_file():
        return [str(python), str(main), "mcp"]
    for name in ("tradeflow", "tradeflow.exe"):
        script = python.parent / name
        if script.is_file():
            return [str(script), "mcp"]
    return [str(python), "-m", "tradeflow.cli", "mcp"]


def mcp_client_config() -> Dict[str, Any]:
    """The ``mcpServers`` block for :func:`mcp_launch_command`, ready to paste."""
    command = mcp_launch_command()
    return {"mcpServers": {"tradeflow": {"command": command[0], "args": command[1:]}}}


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
        """A printable view. Credentials are masked; settings that are not secrets
        are shown as they are — masking ``PAPER_TRADE=true`` to ``****`` hides
        nothing and costs the reader the one value they most want to confirm."""
        return {
            key: (mask(self.resolved(key)) if key in SECRET_KEYS else (self.resolved(key) or "(not set)"))
            for key in OWNED_KEYS
            # Model settings are optional; listing five "(not set)" rows would bury the
            # three that matter for a reader who never asked for a model.
            if key in _BASE_KEYS or self.resolved(key)
        }


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
        # Same rule as EnvState.summary: mask credentials, show plain settings.
        "updated": {key: (mask(value) if key in SECRET_KEYS else value) for key, value in updates.items()},
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
#: ``ready`` verified locally; ``blocked`` required by some task and missing;
#: ``optional_missing`` not needed by any task the reader asked about;
#: ``not_checked`` needs the network or was not requested; ``failed`` the check found a
#: broken state (an unwritable directory, live trading switched on).
Status = Literal["ready", "blocked", "optional_missing", "not_checked", "failed"]

#: What a reader is trying to do. ``demo`` runs on synthetic data and needs nothing.
TASKS = ("demo", "data", "research", "mcp")

PROVIDERS = ("anthropic", "openai", "ollama")

#: provider -> (extra that carries its SDK, credential name); ollama has neither.
_PROVIDER_REQUIREMENTS = {
    "anthropic": ("ai", "ANTHROPIC_API_KEY"),
    "openai": ("openai", "OPENAI_API_KEY"),
}


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    next_action: Optional[str] = None
    #: Tasks this row gates; empty means informational.
    needed_for: FrozenSet[str] = frozenset()
    #: Extras whose absence this row reports. The doctor prints one install command for
    #: all of them together, because two commands that each replace the environment
    #: undo one another.
    installs: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "ready"


def run_checks(
    path: Optional[Any] = None,
    environ: Optional[Dict[str, str]] = None,
    *,
    provider: Optional[str] = None,
) -> List[Check]:
    """Diagnose the current setup: one row per check, all of them evaluated.

    A doctor, not a linter - it reports every check independently rather than
    stopping at the first miss, because the second and third problems are exactly
    what someone running a diagnostic wants to know about. It writes nothing and
    makes no network call; credential *validity* and model reachability need a
    request and are separate, explicit steps.

    Each row says which tasks it gates, so :func:`readiness` can answer "can I run a
    backtest?" without a missing model key getting a vote.
    """
    import tradeflow

    state = inspect_env(path, environ)
    checks = [
        Check("version", "ready", f"{DISTRIBUTION_NAME} {tradeflow.__version__}"),
        Check(
            "env file",
            "ready" if state.exists else "optional_missing",
            f"{state.path} exists" if state.exists else f"{state.path} not found",
            None if state.exists else "run `init` to create it, or export the keys as environment variables",
        ),
    ]
    for key in CREDENTIAL_KEYS:
        value = state.resolved(key)
        source = " (from the environment)" if key in state.from_environment else ""
        filled = not is_placeholder(value)
        checks.append(
            Check(
                key,
                "ready" if filled else "blocked",
                f"{mask(value)}{source}"
                if filled
                else f"{mask(value)} - no command that touches the market will run",
                None if filled else "run `init` to enter it, or export it as an environment variable",
                frozenset({"data", "research", "mcp"}),
            )
        )
    paper = state.resolved("PAPER_TRADE")
    paper_on = paper is None or str(paper).strip().lower() in {"1", "true", "yes", "on"}
    checks.append(
        Check(
            "PAPER_TRADE",
            "ready" if paper_on else "failed",
            "true - orders go to the paper account"
            if paper_on
            else "FALSE - live trading is enabled and orders would use real money",
            None if paper_on else "set PAPER_TRADE=true unless live trading is deliberate",
        )
    )
    checks.append(_writable_check())
    checks.extend(_state_checks())
    checks.extend(_extras_checks())
    checks.extend(_model_checks(state, environ, provider))
    return checks


def _model_checks(state: EnvState, environ: Optional[Dict[str, str]], provider: Optional[str]) -> List[Check]:
    """One row per model provider, from local facts only: SDK importable, key present,
    endpoint configured. Nothing here opens a connection.

    The *selected* provider (``--provider``) gates ``research``; unselected ones are
    informational. With nothing selected a single ``model provider`` row gates research
    on there being at least one provider that is verifiably usable - Ollama's
    reachability cannot be known without a request, so it never counts as ready by
    itself.
    """
    from tradeflow.settings import SettingsError, ollama_base_url
    from tradeflow.utils import extras

    env = os.environ if environ is None else environ

    def resolved(key: str) -> Optional[str]:
        return env.get(key) or state.values.get(key)

    rows: List[Check] = []
    ready_providers: List[str] = []
    for name in PROVIDERS:
        selected = provider == name
        gate = frozenset({"research"}) if selected else frozenset()
        label = f"model: {name}"
        if name == "ollama":
            try:
                endpoint = ollama_base_url()
            except SettingsError as exc:
                rows.append(Check(label, "failed", str(exc), "fix the setting named above", gate))
                continue
            rows.append(
                Check(
                    label,
                    "not_checked",
                    f"endpoint {endpoint} (reachability not checked; run with --probe-model)",
                    None,
                    gate,
                )
            )
            continue
        extra, key_name = _PROVIDER_REQUIREMENTS[name]
        sdk_ok = extras.installed(extra)
        key = resolved(key_name)
        key_ok = not is_placeholder(key)
        if sdk_ok and key_ok:
            ready_providers.append(name)
            rows.append(Check(label, "ready", f"SDK installed; {key_name} {mask(key)}", None, gate))
            continue
        missing = []
        actions = []
        if not sdk_ok:
            missing.append(f"the '{extra}' extra (SDK not installed)")
        if not key_ok:
            missing.append(key_name)
            actions.append(f"set {key_name} in the environment or in .env")
        status: Status = "blocked" if selected else "optional_missing"
        rows.append(
            Check(
                label,
                status,
                "missing " + " and ".join(missing),
                "\n".join(actions) or None,
                gate,
                () if sdk_ok else (extra,),
            )
        )

    if provider is None:
        usable = bool(ready_providers)
        rows.append(
            Check(
                "model provider",
                "ready" if usable else "blocked",
                f"{', '.join(ready_providers)} usable for research"
                if usable
                else "no provider is set up for `research` (Ollama is not verified without --probe-model)",
                None
                if usable
                else "install an SDK extra and set its key, or name a local model with --provider ollama",
                frozenset({"research"}),
            )
        )
    return rows


def missing_extras(checks: List[Check]) -> List[str]:
    """Every extra some row reports absent, once each and sorted - the input to the one
    install command the doctor prints (see :attr:`Check.installs`)."""
    return sorted({name for c in checks if c.status != "ready" for name in c.installs})


def readiness(checks: List[Check]) -> Dict[str, Dict[str, Any]]:
    """Per task: whether it can run, which rows stop it, and which rows are unverified.

    ``not_checked`` rows never block - the check could not know - but they are listed so
    "ready" is not read as "confirmed".
    """
    out: Dict[str, Dict[str, Any]] = {}
    for task in TASKS:
        gating = [c for c in checks if task in c.needed_for]
        blocking = [c.name for c in gating if c.status in ("blocked", "failed", "optional_missing")]
        unverified = [c.name for c in gating if c.status == "not_checked"]
        out[task] = {"ready": not blocking, "blocking": blocking, "unverified": unverified}
    return out


def doctor_ok(checks: List[Check], task: str) -> bool:
    """Whether the check passes for ``task``: it is ready and no row is broken."""
    return readiness(checks)[task]["ready"] and not any(c.status == "failed" for c in checks)


# --------------------------------------------------------------------------- #
# Opt-in model probe
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProbeResult:
    provider: str
    model: str
    category: ProbeCategory
    detail: str

    @property
    def ok(self) -> bool:
        return self.category == "ok"


def probe_model(provider: str, model: Optional[str] = None, *, client=None) -> ProbeResult:
    """One real request, one token, no retries: does this provider answer for this model?

    Opt-in because it can be billed. Exactly one call is made even on failure; the
    clients are built with SDK retries off, so a credit error is not sent three times.
    ``client`` lets a test supply the call; production builds the same client the
    research loop uses, so the probe tests the path research will take.
    """
    from tradeflow.research.llm import DEFAULT_MODELS, build_llm_client

    key = provider.lower()
    if key not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(PROVIDERS)}.")
    chosen = model or DEFAULT_MODELS[key]
    try:
        llm = client or build_llm_client(key, chosen)
        llm.complete("Reply with one word.", "ping", max_tokens=1)
    except ModelProviderError as exc:
        return ProbeResult(key, chosen, exc.category, exc.detail)
    except ImportError:
        extra = _PROVIDER_REQUIREMENTS.get(key, ("",))[0]
        return ProbeResult(key, chosen, "unknown", f"the SDK for {key} is not installed (extra: {extra})")
    except Exception as exc:  # noqa: BLE001 - a raw exception from an injected client
        category, detail = classify_provider_error(exc)
        return ProbeResult(key, chosen, category, detail)
    return ProbeResult(key, chosen, "ok", "the provider answered one request")


def _state_checks() -> List[Check]:
    """Where this copy keeps its evidence, and whether anything is wrong with that.

    A user who arrived through PyPI or over MCP may never learn there is a working
    tree, a ``logs/`` directory or an ignore file. These rows are how the file model
    gets explained to them once, in the one command that exists to explain the setup.
    """
    from tradeflow.execution.ledger import default_ledger_path
    from tradeflow.settings import (
        git_worktree_containing,
        orphaned_checkout_state,
        state_root,
        trial_journal_path,
    )

    root = state_root()
    checks = [
        Check(
            "state root", "ready", f"{root}{' (TRADEFLOW_HOME)' if os.environ.get('TRADEFLOW_HOME') else ''}"
        ),
        Check("research journal", "ready", str(trial_journal_path())),
        Check("position ledger", "ready", str(default_ledger_path())),
    ]

    worktree = git_worktree_containing(root)
    checks.append(
        Check(
            "state outside version control",
            "ready" if worktree is None else "failed",
            "not inside a git working tree"
            if worktree is None
            else (
                f"state root is inside the git working tree at {worktree} — everything "
                "recorded here is one ignore-file edit from disclosure, and `git clean "
                "-xd` would delete it, live position ledger included"
            ),
            None if worktree is None else "set TRADEFLOW_HOME to a directory outside the working tree",
        )
    )

    orphaned = orphaned_checkout_state()
    if orphaned is not None:
        # Said out loud rather than resolved silently: a contributor whose evidence
        # lived in their checkout would otherwise deflate the next campaign against
        # none of it, with nothing erroring.
        checks.append(
            Check(
                "earlier state found",
                "failed",
                f"{orphaned / 'logs'} holds trials from when a checkout was its own "
                f"state root. It is no longer being read. To keep using it: "
                f"TRADEFLOW_HOME={orphaned}",
                f"export TRADEFLOW_HOME={orphaned} to keep using it",
            )
        )

    packs = _installed_packs()
    if packs:
        checks.append(Check("private packs installed", "ready", ", ".join(packs)))
    return checks


def example_pack_source() -> Optional[Path]:
    """The example pack's directory, or ``None`` when this copy does not carry it.

    Present in a source checkout and in the sdist; absent from the wheel, which ships
    only the package. That is deliberate - an installed copy has no use for a second
    project's source tree, and the scaffold says so plainly rather than producing an
    empty directory.
    """
    from tradeflow.settings import PROJECT_ROOT

    candidate = PROJECT_ROOT / "example"
    return candidate if (candidate / "pyproject.toml").exists() else None


def _installed_packs() -> List[str]:
    """Strategy and scanner names contributed by installed packages, not built in.

    Worth naming: it is the difference between "this is the example set" and "your own
    work is running", and it decides whether the state-root warnings above matter.
    """
    try:
        from tradeflow.services.registry import (
            BUILTIN_SCANNERS,
            BUILTIN_STRATEGIES,
            SCANNERS,
            STRATEGIES,
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not fail over its own extras
        return []
    builtin = {*BUILTIN_STRATEGIES, *BUILTIN_SCANNERS}
    return sorted({*STRATEGIES, *SCANNERS} - builtin)


def _writable_check() -> Check:
    """Whether the journal/cache directory can actually be written.

    A research session that cannot journal silently loses its own trial history,
    which corrupts every multiple-testing correction after it — worth checking
    before the first run rather than discovering later.
    """
    from tradeflow.services.audit import DEFAULT_TRIAL_JOURNAL

    directory = Path(DEFAULT_TRIAL_JOURNAL).parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-probe"
        probe.write_text("")
        probe.unlink()
    except PermissionError as exc:
        # A host permission boundary, not a credentials problem: the fix is a different
        # directory or different ownership, and no key will change it.
        return Check(
            "state directory",
            "failed",
            f"permission denied writing {directory}: {exc.strerror}",
            "point TRADEFLOW_HOME at a directory you can write to, or fix its ownership",
        )
    except OSError as exc:
        return Check(
            "state directory",
            "failed",
            f"cannot write {directory}: {exc.strerror}",
            "point TRADEFLOW_HOME at a directory you can write to",
        )
    return Check("state directory", "ready", f"{directory} is writable")


def _extras_checks() -> List[Check]:
    """One row per optional extra, with the exact command to install it - phrased
    for the copy that is running.

    The map, the probe and the recovery text all live in :mod:`tradeflow.utils.extras`;
    the wizard configures and the user installs.
    """
    from tradeflow.utils import extras

    checks = []
    for name, extra in extras.EXTRAS.items():
        if extras.installed(name):
            checks.append(Check(f"extra: {name}", "ready", f"installed - {extra.unlocks}"))
            continue
        # An extra is `blocked` only where some task cannot run without it. Everything
        # else is optional: absent is a fact about what is unlocked, not a failure.
        gated = frozenset(extra.required_for)
        checks.append(
            Check(
                f"extra: {name}",
                "blocked" if gated else "optional_missing",
                f"not installed - {extra.unlocks}",
                None,
                gated,
                (name,),
            )
        )
    return checks


def validate_model_setting(name: str, value: str) -> str:
    """A model-setting answer in the form that gets written, or ``ValueError`` naming it.

    The URL and the two numbers reuse ``settings``' own parsers, so the wizard cannot
    accept a value that a research run would later refuse.
    """
    from urllib.parse import urlparse

    from tradeflow.settings import (
        SettingsError,
        normalize_ollama_url,
        parse_ollama_num_ctx,
        parse_ollama_timeout,
    )

    text = value.strip()
    try:
        if name == "OLLAMA_BASE_URL":
            parsed = urlparse(text)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(f"OLLAMA_BASE_URL must start with http:// or https://, got {text!r}.")
            return normalize_ollama_url(text)
        if name == "OLLAMA_TIMEOUT":
            parse_ollama_timeout(text)
        elif name == "OLLAMA_NUM_CTX":
            parse_ollama_num_ctx(text)
    except SettingsError as exc:
        raise ValueError(str(exc)) from exc
    return text


def build_updates(
    key: Optional[str] = None,
    secret: Optional[str] = None,
    paper_trade: Optional[bool] = None,
    *,
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_timeout: Optional[str] = None,
    ollama_num_ctx: Optional[str] = None,
) -> Dict[str, str]:
    """The ``.env`` updates for the answers given, skipping anything left blank.

    Only what was actually answered is written, so filling in one missing key
    never rewrites the other - and a model setting left blank never overwrites a key
    that is already there with nothing. A placeholder counts as blank.
    """
    updates: Dict[str, str] = {}
    if key:
        updates["APCA_API_KEY_ID"] = key.strip()
    if secret:
        updates["APCA_API_SECRET_KEY"] = secret.strip()
    if paper_trade is not None:
        updates["PAPER_TRADE"] = "true" if paper_trade else "false"
    for name, value in (
        ("ANTHROPIC_API_KEY", anthropic_key),
        ("OPENAI_API_KEY", openai_key),
        ("OLLAMA_BASE_URL", ollama_base_url),
        ("OLLAMA_TIMEOUT", ollama_timeout),
        ("OLLAMA_NUM_CTX", ollama_num_ctx),
    ):
        if value is None or is_placeholder(value):
            continue
        updates[name] = validate_model_setting(name, value)
    return updates
