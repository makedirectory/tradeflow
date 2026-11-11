"""The cost- and fill-stress tables keep every column apart at realistic magnitudes.

The cost-stress row was built from fixed-width f-strings in which the return's ``%``
and the cost's ``$`` touched, so a row printed as ``10.69%$ 906`` and could not be read
or parsed. These tests drive the real printers with the service reports faked, since
the formatting is the property, and read what reaches stdout.
"""

import re
from types import SimpleNamespace

from quantred import cli
from quantred.services import analysis


def _cost_report(points):
    return {"axis": "all", "points": points, "survives_to_multiple": 2.0}


def _run_cost(monkeypatch, capsys, points):
    monkeypatch.setattr(analysis, "run_cost_stress", lambda *a, **k: _cost_report(points))
    args = SimpleNamespace(
        start=None,
        end=None,
        capital=100000.0,
        benchmark=None,
        commission_bps=1.0,
        impact_eta=0.1,
        borrow_bps=0.0,
        cost_stress="all",
    )
    cli._print_cost_stress(None, "s", ["A"], args, {})
    return capsys.readouterr().out.splitlines()


POINTS = [
    {"multiple": 1.0, "sharpe_ratio": 1.23, "total_return": 10.69, "total_cost": 906.4},
    {"multiple": 2.0, "sharpe_ratio": -0.4, "total_return": -12.5, "total_cost": 1234567.0},
    {"multiple": 10.0, "sharpe_ratio": -3.0, "total_return": -100.0, "total_cost": 98765.0},
]


def test_cost_stress_columns_are_separated_and_aligned(monkeypatch, capsys):
    lines = _run_cost(monkeypatch, capsys, POINTS)
    table = [ln for ln in lines if ln.startswith("  ") and "multiple" in ln or re.match(r"\s+\d+\.\dx", ln)]
    assert len(table) == 4
    for ln in table[1:]:
        # Each cell is a whitespace-delimited token: nothing runs into its neighbour.
        assert len(ln.split()) == 4, ln
        assert "%$" not in ln
    # Right-aligned columns end at the same offset on every row, header included.
    ends = {len(ln) for ln in table}
    assert len(ends) == 1, table
    assert table[2].split()[3] == "$1,234,567"


def test_fill_stress_columns_are_separated_and_aligned(monkeypatch, capsys):
    report = {
        "points": [
            {"margin_bps": 0, "sharpe_ratio": 1.0, "total_return": 5.0, "trades": 40},
            {"margin_bps": 25, "sharpe_ratio": -2.5, "total_return": -1234.56, "trades": None},
        ],
        "survives_to_bps": 0,
    }
    monkeypatch.setattr(analysis, "run_fill_stress", lambda *a, **k: report)
    args = SimpleNamespace(
        start=None,
        end=None,
        capital=100000.0,
        benchmark=None,
        commission_bps=1.0,
        impact_eta=0.1,
        borrow_bps=0.0,
    )
    cli._print_fill_stress(None, "s", ["A"], args, {})
    out = capsys.readouterr().out.splitlines()
    table = [
        ln for ln in out if ln.startswith("  ") and ("through by" in ln or "touch only" in ln or " bps" in ln)
    ]
    table = [ln for ln in table if "Edge" not in ln and "historical" not in ln]
    assert len(table) == 3, out
    assert len({len(ln) for ln in table}) == 1, table
    assert table[2].split()[-3:] == ["-1234.56%", "—"][-3:] or "-1234.56%" in table[2].split()
