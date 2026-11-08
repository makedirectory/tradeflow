"""The last release under the old distribution name says so, without breaking anything.

The notice exists so a person learns where later releases are published. It must not reach
a non-interactive stderr: scripts, CI jobs and MCP clients may treat unexpected stderr as a
failure, and this release is meant to change nothing for them.
"""

import io

import pytest

from tradeflow import cli


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_an_interactive_terminal_is_told_where_later_releases_are(monkeypatch):
    err = _Terminal()
    monkeypatch.setattr("sys.stderr", err)
    cli._rename_notice()
    text = err.getvalue()
    assert "quantred" in text
    assert "last release" in text
    assert "Uninstall tradeflow-engine" in text
    assert cli.RENAME_GUIDE_URL in text


def test_a_script_or_client_sees_nothing(monkeypatch):
    err = io.StringIO()
    monkeypatch.setattr("sys.stderr", err)
    cli._rename_notice()
    assert err.getvalue() == ""


def test_the_notice_runs_on_every_command_and_never_touches_stdout(monkeypatch, capsys):
    err = _Terminal()
    monkeypatch.setattr("sys.argv", ["tradeflow", "--version"])
    monkeypatch.setattr("sys.stderr", err)
    with pytest.raises(SystemExit):
        cli.main()
    assert "quantred" in err.getvalue()
    assert "quantred" not in capsys.readouterr().out
