#!/usr/bin/env python3
"""Verify that every relative Markdown link resolves — file *and* heading anchor.

Cross-directory links rot silently: moving a doc breaks every sibling-relative link
pointing at it, and nothing complains until a reader clicks. Anchors are worse,
because a link to a renamed heading still resolves as a file and just lands in the
wrong place.

Scope note: ``specs/`` is gitignored, so CI never sees it. Run this locally (``make
check-links``) to cover the specs too — it checks whatever of the roots below exist,
so the same command is correct in both places.

Usage:
    python scripts/check_links.py [root ...]     # defaults to the roots below
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

#: Checked when present. specs/ is local-only; the rest are tracked.
DEFAULT_ROOTS = ("docs/content", "specs", "README.md", "PROCESS.md", "CONTRIBUTING.md")

SKIP_DIRS = {"node_modules", ".venv", "build", ".git", "__pycache__", ".pytest_cache"}

#: [text](target) — the target stops at whitespace so titles like (path "t") are cut.
LINK = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
#: Docusaurus lets a heading pin its own slug: `## Title {#custom-id}`.
EXPLICIT_ID = re.compile(r"\{#([^}]+)\}\s*$")


def slugify(heading: str) -> str:
    """Approximate the GitHub/Docusaurus heading slug."""
    text = heading.strip()
    explicit = EXPLICIT_ID.search(text)
    if explicit:
        return explicit.group(1).strip().lower()
    text = EXPLICIT_ID.sub("", text)
    text = INLINE_CODE.sub(lambda m: m.group(0).strip("`"), text)
    # Emphasis markers, but NOT underscore: GitHub keeps it, and stripping it would
    # mangle every snake_case identifier in a heading (`n_trials` -> `ntrials`).
    text = re.sub(r"[*~]", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links keep their text
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)  # drop punctuation/emoji
    return re.sub(r"[\s]+", "-", text).strip("-")


def _resolve(candidate: Path) -> Path | None:
    """Resolve a link target, allowing Docusaurus's extensionless doc links.

    ``[MCP server](mcp-server)`` is a valid link to ``mcp-server.md``; requiring the
    extension would flag most of the docs site.
    """
    for option in (candidate, Path(f"{candidate}.md"), Path(f"{candidate}.mdx"), candidate / "index.md"):
        if option.exists():
            return option.resolve()
    return None


def anchors_in(path: Path) -> Set[str]:
    """Every heading slug a file offers."""
    body = FENCE.sub("", _read(path))
    return {slugify(line.lstrip("#")) for line in body.splitlines() if line.startswith("#")}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def markdown_files(roots: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    for raw in roots:
        root = Path(raw)
        if not root.exists():
            continue  # specs/ is absent in CI; that is expected, not an error
        if root.is_file():
            found.append(root)
            continue
        found.extend(p for p in root.rglob("*.md") if not SKIP_DIRS & set(p.parts))
    return sorted(found)


def check(roots: Iterable[str]) -> List[str]:
    files = markdown_files(roots)
    anchor_cache: dict = {}
    problems: List[str] = []

    for path in files:
        body = FENCE.sub("", _read(path))
        for match in LINK.finditer(body):
            target = match.group("target")
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue

            file_part, _, anchor = target.partition("#")
            if not file_part:
                resolved = path  # same-page anchor
            else:
                resolved = _resolve(path.parent / file_part)
                if resolved is None:
                    problems.append(f"{path}: missing file → [{match.group('text')}]({target})")
                    continue

            if anchor and resolved.suffix == ".md":
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = anchors_in(resolved)
                if anchor.lower() not in anchor_cache[resolved]:
                    problems.append(f"{path}: missing anchor → [{match.group('text')}]({target})")

    return problems


def main(argv: List[str]) -> int:
    roots: Tuple[str, ...] = tuple(argv[1:]) or DEFAULT_ROOTS
    problems = check(roots)
    checked = len(markdown_files(roots))
    if problems:
        print(f"Broken links ({len(problems)}) across {checked} file(s):\n")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print(f"All relative links and anchors resolve across {checked} Markdown file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
