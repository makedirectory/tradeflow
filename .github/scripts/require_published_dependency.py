"""Fail unless the engine version the transition package requires is already on PyPI.

The transition release of the old distribution name depends on ``quantred``. PyPI never
lets a version be replaced, so publishing it before a satisfying ``quantred`` exists would
leave a permanent release whose dependency cannot be installed. The tag order is a human
convention; this makes it a check.

Usage: require_published_dependency.py <pyproject.toml> [<releases.json>]
Without the second argument the release list is fetched from PyPI.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

TARGET = "quantred"


def satisfied(pyproject_text: str, releases: dict) -> str:
    """The newest un-yanked release meeting the dependency, or raise SystemExit."""
    deps = tomllib.loads(pyproject_text)["project"]["dependencies"]
    wanted = [r for r in map(Requirement, deps) if r.name == TARGET]
    if not wanted:
        raise SystemExit(f"the transition package declares no dependency on {TARGET}")
    spec = wanted[0].specifier
    live = [
        version
        for version, files in releases.items()
        if files and not all(f.get("yanked") for f in files) and version in spec
    ]
    if not live:
        raise SystemExit(
            f"{TARGET}{spec} is not on PyPI yet; publish it before the transition package "
            "(a published release cannot be replaced)"
        )
    return str(max(map(Version, live)))


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as handle:
        text = handle.read()
    if len(argv) > 2:
        with open(argv[2], encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{TARGET}/json", timeout=30) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            data = {"releases": {}}  # no such project yet: nothing satisfies the dependency
    print(f"found {TARGET} {satisfied(text, data['releases'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
