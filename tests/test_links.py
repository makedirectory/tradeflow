"""Tests for the Markdown link checker.

A link checker that silently passes is worse than none — it converts "nobody
looked" into "CI is green". These pin the two failures it exists to catch, plus a
live pass over the repo's own docs so real rot fails the build.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_links import DEFAULT_ROOTS, check, slugify  # noqa: E402


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_detects_a_missing_file(tmp_path):
    _write(tmp_path, "a.md", "See [the other doc](b.md).")
    problems = check([str(tmp_path)])
    assert len(problems) == 1
    assert "missing file" in problems[0]


def test_detects_a_missing_anchor(tmp_path):
    _write(tmp_path, "target.md", "# Real heading\n")
    _write(tmp_path, "a.md", "See [it](target.md#not-a-heading).")
    problems = check([str(tmp_path)])
    assert len(problems) == 1
    assert "missing anchor" in problems[0]


def test_accepts_a_valid_cross_file_anchor(tmp_path):
    _write(tmp_path, "target.md", "## Why the thresholds are what they are\n")
    _write(tmp_path, "a.md", "See [it](target.md#why-the-thresholds-are-what-they-are).")
    assert check([str(tmp_path)]) == []


def test_ignores_links_inside_code_fences(tmp_path):
    _write(tmp_path, "a.md", "```\n[not a link](nope.md)\n```\n")
    assert check([str(tmp_path)]) == []


def test_allows_docusaurus_extensionless_links(tmp_path):
    _write(tmp_path, "mcp-server.md", "# MCP\n")
    _write(tmp_path, "a.md", "See [MCP server](mcp-server) and [again](./mcp-server).")
    assert check([str(tmp_path)]) == []


def test_external_links_are_not_fetched(tmp_path):
    _write(tmp_path, "a.md", "[site](https://example.invalid/nope) [mail](mailto:x@y.z)")
    assert check([str(tmp_path)]) == []


@pytest.mark.parametrize(
    "heading,slug",
    [
        ("## Why the thresholds are what they are", "why-the-thresholds-are-what-they-are"),
        ("### The `accounting` stamp", "the-accounting-stamp"),
        ("## Portfolio accounting", "portfolio-accounting"),
        ("## Custom {#pinned-id}", "pinned-id"),
        # Underscores survive: stripping them as emphasis would mangle every
        # snake_case identifier in a heading.
        ("### Known limit: `n_trials` counts a run", "known-limit-n_trials-counts-a-run"),
        ("## max_total_risk is per book", "max_total_risk-is-per-book"),
    ],
)
def test_slugify(heading, slug):
    assert slugify(heading.lstrip("#")) == slug


def test_repo_docs_have_no_broken_links():
    """The live check. Fails the build when a doc move breaks a link."""
    repo = Path(__file__).resolve().parents[1]
    import os

    cwd = os.getcwd()
    os.chdir(repo)
    try:
        problems = check(DEFAULT_ROOTS)
    finally:
        os.chdir(cwd)
    assert problems == [], "\n" + "\n".join(problems)
