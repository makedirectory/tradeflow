"""The README's demo transcript is the demo's real output.

It is the first output a reader sees, and it drifted twice: an engine change moved every
number while the sample, and the prose quoting it, kept the old run. This runs the demo
at the seed the README shows (the default) and requires each line of the sample to
appear in the real output, in order.

Two kinds of line are deliberately not compared. The closing "next step" lines are
phrased for the copy that is running (a checkout says ``make init``, an installed copy
its own command), so they are a property of the reader, not of the run. And the real
output may carry *extra* strategy rows: the table lists every registered strategy, and a
developer's environment can have an extension pack installed beside the bundled one.
"""

from __future__ import annotations

from pathlib import Path

from quantred import cli, settings

README = Path(settings.PROJECT_ROOT) / "README.md"
#: The sample's last stable line; what follows is the copy-dependent next step.
LAST_STABLE = "is the product. Point QuantRed at real data:"


def _readme_demo_lines() -> list:
    lines = README.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("$ ") and line.endswith(" demo"))
    out = []
    for line in lines[start + 1 :]:
        if line.startswith("```"):
            break
        out.append(line.rstrip())
        if line.strip() == LAST_STABLE:
            return [line for line in out if line.strip()]
    raise AssertionError("the README demo sample no longer ends with the line this test anchors on")


def test_the_readme_demo_sample_matches_the_seeded_demo_output(capsys):
    expected = _readme_demo_lines()
    args = cli.parse_cli(["demo"])
    assert args.seed == 42, "the README shows a default run; the test must run the same seed"
    args.func(args)
    actual = [line.rstrip() for line in capsys.readouterr().out.splitlines()]

    position = 0
    for line in expected:
        try:
            position = actual.index(line, position) + 1
        except ValueError:
            raise AssertionError(
                f"README demo line not in the real output (in order): {line!r}\n"
                "Regenerate the sample from `quantred demo` and update the prose quoting it.\n"
                "Actual output:\n" + "\n".join(actual)
            ) from None
