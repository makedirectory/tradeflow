"""The self-contained HTML report.

All offline, from fixture result dicts — the renderer is pure, so it needs no data
client at all. The properties under test are the ones that make the artifact
trustworthy rather than merely pretty: it makes no external requests, it escapes
everything, it shows its warnings before its numbers, it leaks no figures, and it
refuses a payload that is not what it was told it is.
"""

import re
from html.parser import HTMLParser

import pytest

from tradeflow.analytics.htmlreport import KINDS, ReportKindError, render_html, write_html
from tradeflow.services.analysis import VERDICT_SCHEMA

#: Any attribute that could pull a resource from off the machine.
_EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["'](?!data:|#)[^"']*|url\(\s*(?!['"]?data:)""", re.I)


class _Parses(HTMLParser):
    """Minimal well-formedness check: stdlib parses it without raising."""

    def error(self, message):  # pragma: no cover - stdlib compatibility shim
        raise AssertionError(message)


def _verdict_result(**overrides):
    result = {
        "schema": VERDICT_SCHEMA,
        "kind": "verdict",
        "run_id": "run-1",
        "memoized": False,
        "inputs": {
            "strategy": "demo_trend",
            "universe": ["AAA", "BBB"],
            "window": {"start": "2024-01-02T00:00:00", "end": "2024-12-31T00:00:00"},
            "timeframe": "1Day",
            "benchmark": "SPY",
            "cost": {"gross": False, "commission_bps": 1.0, "impact_eta": 0.3, "borrow_bps": 50.0},
        },
        "provenance": {"git_sha": "abc1234", "generated_at": "2025-01-01T00:00:00", "n_trials": 17},
        "steps": {"scan": {"status": "ok"}, "alphas": {"status": "ok"}},
        "scan": {"scanner": "demo_volume", "candidates": ["AAA", "BBB"], "flagged_count": 2},
        "alphas": {"alphas": [{"symbol": "AAA", "alpha": 0.04, "z": 1.2, "residual_vol": 0.3}]},
        "combination": None,
        "portfolio": {
            "feasible": True,
            "target_te": 0.04,
            "risk_model": "shrinkage",
            "weights": {"AAA": 0.6, "BBB": 0.4},
            "exposures": {"market": 0.02},
            "diagnostics": {
                "predicted_tracking_error": 0.037,
                "expected_active_return": 0.014,
                "expected_active_return_net": 0.006,
                "predicted_ir": 0.38,
                "transfer_coefficient": 0.71,
            },
        },
        "information": {
            "periods": 24,
            "horizon_bars": 5,
            "mean_ic": 0.018,
            "ic_tstat": 0.74,
            "rank_ic": 0.021,
            "breadth_effective": 142.0,
            "rho_bar": 0.41,
            "predicted_ir": 0.21,
            "realized_ir": 0.18,
            "ir_standard_error": 0.61,
            "multiple_testing_inflation": 0.51,
            "n_trials": 17,
        },
        "verdict": {
            "verdict": "mixed",
            "promotable": False,
            "summary": "mixed — passed: sample_size; failed: ic_tstat",
            "failed_steps": [],
            "checks": {
                "ic_tstat": {"value": 0.74, "threshold": 2.0, "passed": False, "note": "below 2 is luck"},
                "sample_size": {"value": 24, "threshold": 12, "passed": True, "note": "enough rebalances"},
            },
        },
    }
    result.update(overrides)
    return result


def _backtest_result():
    return {
        "strategy": "demo_trend",
        "symbols": ["AAA", "BBB"],
        "window": {"start": "2024-01-02T00:00:00", "end": "2024-12-31T00:00:00"},
        "initial_capital": 100_000.0,
        "final_capital": 104_200.0,
        "gross": False,
        "total_cost": 812.0,
        "cost_drag_pct": 0.81,
        "metrics": {"sharpe_ratio": 0.9, "total_return": 4.2, "max_drawdown": -6.1},
        "total_trades": 41,
    }


def _walkforward_result(promotable=False):
    return {
        "strategy": "demo_trend",
        "symbols": ["AAA"],
        "window": {"start": "2024-01-02T00:00:00", "end": "2024-12-31T00:00:00"},
        "folds": [
            {
                "index": 0,
                "is_window": {"start": "2024-01-02T00:00:00", "end": "2024-06-01T00:00:00"},
                "oos_window": {"start": "2024-06-02T00:00:00", "end": "2024-08-01T00:00:00"},
                "is_sharpe": 1.4,
                "oos_sharpe": 0.3,
                "oos_trades": 12,
                "n_trials": 20,
            }
        ],
        "oos_aggregate": {"sharpe_ratio": 0.3, "total_return": 1.1},
        "median_oos_sharpe": 0.3,
        "median_efficiency": 0.21,
        "total_oos_trades": 12,
        "gate_report": {
            "promotable": promotable,
            "checks": {"oos_sharpe": {"value": 0.3, "threshold": 0.5, "passed": promotable}},
        },
        "diagnostics": {},
    }


def _info_result():
    return {
        "strategy": "demo_trend",
        "window": {"start": "2024-01-02T00:00:00", "end": "2024-12-31T00:00:00"},
        "periods": 24,
        "mean_ic": 0.018,
        "ic_tstat": 0.74,
        "horizon_bars": 5,
        "n_trials": 3,
        "note": "IC measured as alpha-vs-forward-residual-return.",
    }


_FIXTURES = {
    "verdict": _verdict_result,
    "backtest": _backtest_result,
    "walkforward": _walkforward_result,
    "info": _info_result,
}


# --- the self-containment guarantee -----------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_renders_parseable_html_with_no_external_references(kind):
    document = render_html(_FIXTURES[kind](), kind)
    _Parses().feed(document)

    external = _EXTERNAL.findall(document)
    assert external == [], f"report reaches off-machine: {external}"
    assert "http://" not in document and "https://" not in document
    assert document.startswith("<!doctype html>")


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_carries_its_provenance(kind):
    document = render_html(_FIXTURES[kind](), kind)
    # "Code version" rather than "Git SHA": an installed copy has no repository, and
    # the packaged version is the honest answer to what produced the report.
    for field in ("Provenance", "Window", "Universe", "Code version", "Generated"):
        assert field in document


# --- honesty labelling ------------------------------------------------------
def test_a_memoized_result_leads_with_the_reuse_warning():
    document = render_html(
        _verdict_result(memoized=True, trial_id="t-9", trial_ts="2024-01-01T00:00:00+00:00"), "verdict"
    )
    assert "REUSED" in document
    assert "t-9" in document
    # Before any section: a reader must not scroll past numbers to learn they are old.
    assert document.index("REUSED") < document.index("Portfolio")


def test_gate_failures_render_as_a_warning_above_the_sections():
    document = render_html(_verdict_result(), "verdict")
    assert "Gate failures: ic_tstat" in document
    assert document.index("Gate failures") < document.index("Alphas")


def test_an_incomplete_run_says_so_before_anything_else():
    result = _verdict_result()
    result["verdict"] = {
        "verdict": "incomplete",
        "promotable": None,
        "summary": "incomplete — no verdict",
        "failed_steps": ["information"],
        "checks": {},
    }
    document = render_html(result, "verdict")
    assert "INCOMPLETE" in document
    assert "Do not act on the sections below" in document


def test_a_default_off_feature_being_enabled_is_banner_worthy():
    result = _verdict_result()
    result["portfolio"]["conditional"] = "ewma"
    document = render_html(result, "verdict")
    assert "Non-default configuration" in document
    assert "conditional risk" in document


def test_a_not_promotable_walkforward_says_which_gates_failed():
    document = render_html(_walkforward_result(promotable=False), "walkforward")
    assert "NOT PROMOTABLE" in document
    assert "failed gates: oos_sharpe" in document


def test_a_failed_leakage_probe_is_never_quiet():
    result = _walkforward_result(promotable=True)
    result["diagnostics"] = {"leakage_probe": {"passed": False}}
    assert "Leakage probe FAILED" in render_html(result, "walkforward")


# --- escaping ---------------------------------------------------------------
def test_a_hostile_symbol_renders_inert():
    result = _verdict_result()
    result["inputs"]["universe"] = ["<script>alert(1)</script>"]
    result["portfolio"]["weights"] = {"<img src=x onerror=alert(1)>": 1.0}
    document = render_html(result, "verdict")

    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;" in document
    assert "onerror=alert(1)>" not in document
    # And the escaped form must not have reintroduced an external reference.
    assert _EXTERNAL.findall(document) == []


# --- shape checking ---------------------------------------------------------
def test_a_mismatched_payload_fails_clearly_rather_than_half_rendering():
    with pytest.raises(ReportKindError, match="verdict/1"):
        render_html({"kind": "verdict", "schema": "verdict/0"}, "verdict")
    with pytest.raises(ReportKindError, match="does not look like a backtest"):
        render_html({"folds": []}, "backtest")
    with pytest.raises(ReportKindError, match="Unknown report kind"):
        render_html(_backtest_result(), "optimize")


# --- charts -----------------------------------------------------------------
def test_charts_embed_as_data_uris_when_plotting_is_available():
    pytest.importorskip("matplotlib")
    document = render_html(_verdict_result(), "verdict")
    assert "data:image/png;base64," in document
    assert _EXTERNAL.findall(document) == []


def test_the_report_renders_without_the_plotting_extra(monkeypatch):
    """The chart slot explains itself; the run does not fail."""
    from tradeflow.analytics import charts

    def missing(*args, **kwargs):
        raise RuntimeError("Charting needs matplotlib. Install the viz extra: `uv sync --extra viz`.")

    monkeypatch.setattr(charts, "render_gate_png", missing)
    monkeypatch.setattr(charts, "render_bars_png", missing)
    document = render_html(_verdict_result(), "verdict")

    assert "install the viz extra" in document.lower()
    assert "data:image/png;base64," not in document
    assert "mixed" in document  # the report is still complete in text


def test_rendering_reports_leaks_no_figures():
    plt = pytest.importorskip("matplotlib.pyplot")
    before = len(plt.get_fignums())
    for _ in range(3):
        render_html(_verdict_result(), "verdict")
        render_html(_walkforward_result(), "walkforward")
    assert len(plt.get_fignums()) == before


def test_backtest_equity_chart_comes_from_extras_not_the_result_dict():
    pytest.importorskip("matplotlib")
    without = render_html(_backtest_result(), "backtest")
    assert "Equity" not in without.split("<footer>")[0].replace("Metrics", "")

    curve = [100_000.0 + i * 12.0 for i in range(120)]
    with_chart = render_html(_backtest_result(), "backtest", extras={"equity_curve": curve})
    assert "data:image/png;base64," in with_chart


# --- determinism and size ---------------------------------------------------
def test_two_renders_of_the_same_fixture_are_identical():
    """Nothing in a render may depend on the clock or on iteration order — the
    provenance timestamp comes from the result, not from `now`."""
    assert render_html(_verdict_result(), "verdict") == render_html(_verdict_result(), "verdict")


def test_a_report_stays_comfortably_small():
    document = render_html(_verdict_result(), "verdict")
    assert len(document.encode("utf-8")) < 5_000_000


def test_write_html_writes_the_document_it_rendered(tmp_path):
    path = tmp_path / "report.html"
    written = write_html(_verdict_result(), "verdict", str(path))
    assert written == str(path)
    assert path.read_text(encoding="utf-8") == render_html(_verdict_result(), "verdict")
