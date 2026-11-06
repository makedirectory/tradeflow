"""Every `quantred ...` command a usage page or the README shows must parse.

Prose drifts from the parser silently: a flag is renamed, a page keeps the old spelling,
and a reader pastes a command that fails with a usage error. This reads the fenced shell
blocks of the user-facing docs and hands each command to the real parser. Nothing is
executed - ``parse_args`` only parses.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from tradeflow import cli, settings

ROOT = Path(settings.PROJECT_ROOT)
USAGE = sorted((ROOT / "docs" / "content" / "usage").glob("*.md"))
PAGES = USAGE + [ROOT / "README.md"]
FENCE = re.compile(r"^```(\w*)\s*$")
SHELL = {"bash", "sh", "shell", "console"}
#: Pages that are about a contributor's or operator's stack, so their first command is
#: legitimately not the installed one.
CONTRIBUTOR_FIRST = {"docker.md"}
#: The names the docs themselves use for "your strategy" (``example_breakout`` ships in
#: the example pack, which a plain install does not have). Only these are swapped for a
#: registered one: substituting any unknown name would turn a typo in a page into a pass.
PLACEHOLDER_STRATEGIES = {"my_breakout", "example_breakout"}
DATE_FLAGS = {"--start", "--end", "--as-of"}
CHECKOUT_PREFIXES = ("python main.py", "uv run", "uv sync", "make ")


def _shell_blocks(path: Path):
    block, lang = None, ""
    for line in path.read_text().splitlines():
        m = FENCE.match(line.strip())
        if m and block is None:
            block, lang = [], m.group(1)
        elif line.strip() == "```" and block is not None:
            if lang in SHELL:
                yield block
            block = None
        elif block is not None:
            block.append(line)


def _commands(path: Path):
    """Each logical command line (continuations joined) starting with the program name."""
    programs = tuple(f"{name} " for name in (settings.CLI_NAME, *settings.LEGACY_CLI_NAMES))
    for block in _shell_blocks(path):
        joined, buffer = [], ""
        for line in block:
            stripped = line.rstrip()
            if stripped.endswith("\\"):
                buffer += stripped[:-1] + " "
                continue
            joined.append(buffer + stripped)
            buffer = ""
        if buffer:
            joined.append(buffer)
        for line in joined:
            if line.lstrip().startswith(programs):
                yield line.strip()


def _tokens(line: str):
    """Tokens ready for the parser, or None when the line is deliberately not runnable."""
    if "expected to fail" in line:
        return None
    # Comment-stripping must not eat a `#` inside quotes; shlex handles that.
    tokens = shlex.split(line, comments=True)
    if any(re.search(r"<[^>]+>|\$\w+|\$\{", tok) for tok in tokens):
        return None
    # `...` stands for "more arguments" (dropped) or, after a flag, for its value.
    out = []
    for tok in tokens:
        if tok != "...":
            out.append(tok)
        elif out and out[-1].startswith("-"):
            out.append("2024-01-02" if out[-1] in DATE_FLAGS else "x")
    # Pages teach with a strategy of the reader's own (`my_breakout`), which no install
    # registers; the parser should be asked about the flags, not about whose pack is
    # present on this machine.
    for i, tok in enumerate(out[:-1]):
        if tok == "--strategy" and out[i + 1] in PLACEHOLDER_STRATEGIES:
            out[i + 1] = "demo_trend"
    return out


CASES = [(page, line) for page in PAGES for line in _commands(page)]


@pytest.mark.parametrize("page,line", CASES, ids=[f"{p.name}:{ln[:60]}" for p, ln in CASES])
def test_a_documented_command_parses(page, line, capsys):
    tokens = _tokens(line)
    if tokens is None:
        pytest.skip("placeholder or marked as expected to fail")
    assert tokens[0] in (settings.CLI_NAME, *settings.LEGACY_CLI_NAMES)
    try:
        cli.build_parser().parse_args(tokens[1:])
    except SystemExit as exc:
        if exc.code == 0:  # --help / --version
            return
        pytest.fail(
            f"{page.name}: `{line}` does not parse: {capsys.readouterr().err.strip().splitlines()[-1]}"
        )


def test_the_scan_found_commands():
    """Guard the guard: a scan that silently finds nothing would pass forever."""
    assert len(CASES) > 100


@pytest.mark.parametrize("page", [p for p in USAGE if p.name not in CONTRIBUTOR_FIRST], ids=lambda p: p.name)
def test_primary_examples_are_installed_user_commands(page):
    for block in _shell_blocks(page):
        first = next((ln.strip() for ln in block if ln.strip() and not ln.strip().startswith("#")), None)
        if first is None:
            continue
        assert not first.startswith(CHECKOUT_PREFIXES), f"{page.name} leads with a checkout command: {first}"
        return


DOCKER_VERDICT = re.compile(r"^\s*(?:#\s*|\$\s*)?docker compose run --rm verdict\b(.*)$")


def _verdict_docs():
    """Every file that teaches a verdict command: all the docs, the READMEs (including
    the one the sdist ships inside the example pack) and the compose file."""
    files = sorted((ROOT / "docs" / "content").rglob("*.md"))
    files += [ROOT / "README.md", ROOT / "example" / "README.md", ROOT / "docker-compose.yml"]
    return [f for f in dict.fromkeys(files) if f.is_file()]


def _documented_verdict_commands():
    for page in _verdict_docs():
        if page.suffix == ".md":
            for line in _commands(page):
                if line.split()[1:2] == ["verdict"]:
                    yield page, line
        # The compose service is a verdict entrypoint, so its usage lines are verdict
        # commands even though they do not start with the program name.
        joined = ""
        for raw in page.read_text().splitlines():
            stripped = raw.rstrip()
            if stripped.endswith("\\"):
                joined += stripped[:-1] + " "
                continue
            line, joined = joined + stripped, ""
            match = DOCKER_VERDICT.match(line)
            if match:
                yield page, f"{settings.CLI_NAME} verdict{match.group(1)}"


VERDICT_CASES = list(_documented_verdict_commands())


@pytest.mark.parametrize("page,line", VERDICT_CASES, ids=[f"{p.name}:{ln[:60]}" for p, ln in VERDICT_CASES])
def test_documented_verdict_examples_can_fund_a_long_only_book(page, line):
    """A first example that cannot be funded teaches the wrong lesson: three names at
    the default weight cap read as "needs more data" for a long time. Every documented
    verdict must pass the same cap check the verdict itself makes, using the real
    parser's defaults for whatever the command leaves out.

    The scanner is part of the universe. A scanner may flag any number of the candidates
    - the book is funded from what it keeps, not from what was typed - so no name count
    can promise a fundable book through one. An example that runs a scanner must say
    ``--scanner none``; showing a scanner is for the scanning pages, not a first verdict."""
    from tradeflow.portfolio.optimizer import long_only_cap_shortfall

    tokens = _tokens(line)
    if tokens is None:
        pytest.skip("placeholder or marked as expected to fail")
    args = cli.build_parser().parse_args(tokens[1:])

    shortfall = long_only_cap_shortfall(len(args.symbols), args.max_weight, args.max_names)
    assert shortfall is None, f"{page.name}: `{line}` cannot be funded - {shortfall}"
    # A saved config names its own scanner and universe, which this check cannot see.
    assert args.config or str(args.scanner).lower() == "none", (
        f"{page.name}: `{line}` filters its candidates through scanner {args.scanner!r}, which can "
        "leave too few names to fund a book; add --scanner none"
    )


def test_the_verdict_docs_guard_found_commands():
    """Guard the guard: a scan that finds no verdict example passes forever."""
    assert len(VERDICT_CASES) >= 8
    assert {p.name for p, _ in VERDICT_CASES} >= {"README.md", "docker-compose.yml", "verdict.md"}
