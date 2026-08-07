"""Self-contained HTML rendering of a research result.

Every result in this project otherwise dies in a terminal scrollback. One HTML
file per run fixes that: openable anywhere, attachable, and honest — because it
renders the **same** result dict the CLI printed rather than recomputing anything.
Nothing in this module derives a number; a renderer that computes is a renderer
that can disagree with the report it claims to be showing.

**Self-contained is a hard requirement, not an aspiration.** No web fonts, no CDN
stylesheets, no remote images, no analytics: opening a report must issue zero
network requests. That is an offline guarantee and a privacy one at once — a
report forwarded to someone else must not phone home on their machine. The test
suite asserts it rather than trusting it.

**The medium flatters; the report must not.** Gate failures, memoized results, and
enabled evidence-gated features render as prominent warnings at the top, mirroring
the severity the terminal gives them. A pretty report that buries its own verdict
is worse than the plain-text one it replaced.
"""

import base64
import html
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Result kinds this module knows how to render.
KINDS = ("verdict", "backtest", "walkforward", "info")

#: The evidence-gated features that default off. A report where one is enabled says
#: so loudly: the reader has to know they are looking at a non-default configuration
#: whose own adoption gate has not cleared on this repo's data.
_DEFAULT_OFF = {
    "conditional": "conditional risk (Σ_t)",
    "policy": "the aim-in-front-of-the-target trading policy",
    "posterior": "the Black–Litterman posterior",
}


class ReportKindError(ValueError):
    """The result dict does not match the kind it was asked to render as.

    Raised loudly and early rather than producing a half-rendered report from a
    payload whose shape was guessed at - the failure mode a result-dict passthrough
    contract invites once two surfaces exchange these objects.
    """


def render_html(result: Dict[str, Any], kind: str, *, extras: Optional[Dict[str, Any]] = None) -> str:
    """Render ``result`` as one self-contained HTML document.

    ``kind`` is one of :data:`KINDS` and must match the payload: a composite result
    carries its own schema stamp, and a mismatch raises :class:`ReportKindError`
    rather than rendering something misleading.

    ``extras`` carries display-only material a caller already has in hand but the
    result dict does not (today: an ``equity_curve`` for a backtest, which the
    result dict deliberately omits because it can run to tens of thousands of
    points). It is never required — a report rendered without it degrades to the
    tables, which is exactly what a caller reading the dict alone gets.
    """
    if kind not in KINDS:
        raise ReportKindError(f"Unknown report kind {kind!r}; expected one of {', '.join(KINDS)}")
    _assert_shape(result, kind)
    extras = extras or {}

    sections = {
        "verdict": _verdict_sections,
        "backtest": _backtest_sections,
        "walkforward": _walkforward_sections,
        "info": _info_sections,
    }[kind](result, extras)

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head>',
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{esc(_title(result, kind))}</title>",
            f"<style>{_CSS}</style>",
            "</head><body>",
            f"<main><h1>{esc(_title(result, kind))}</h1>",
            _provenance_block(result, kind),
            _warnings_block(result, kind),
            *sections,
            _footer(),
            "</main></body></html>",
        ]
    )


def _assert_shape(result: Dict[str, Any], kind: str) -> None:
    """Fail fast when the payload is not what ``kind`` promises.

    A composite carries an explicit schema stamp, so it is checked exactly. The
    other kinds predate any stamp, so they are checked on a field only that kind
    produces - crude, but it catches the mistake that matters (a result handed to
    the wrong renderer) without inventing a version scheme for payloads that have
    none.
    """
    if not isinstance(result, dict):
        raise ReportKindError(f"Expected a result dict, got {type(result).__name__}")
    if kind == "verdict":
        from tradeflow.services.analysis import VERDICT_SCHEMA

        schema = result.get("schema")
        if schema != VERDICT_SCHEMA:
            raise ReportKindError(
                f"Expected a {VERDICT_SCHEMA} result, got schema={schema!r}. "
                "Pass the object a verdict run returned, unmodified."
            )
        return
    required = {"backtest": "metrics", "walkforward": "folds", "info": "mean_ic"}[kind]
    if required not in result:
        raise ReportKindError(
            f"This does not look like a {kind} result: no {required!r} field. "
            "Pass the object the matching run returned, unmodified."
        )


# --------------------------------------------------------------------------- #
# Escaping and small formatters
# --------------------------------------------------------------------------- #
def esc(value: Any) -> str:
    """Escape anything on its way into the document.

    Symbols, strategy names, config params, and error strings all flow into this
    report, and every one of them is ultimately user input. Everything
    interpolated goes through here - there is no "obviously safe" interpolation.
    """
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, spec: str = "{:.4g}") -> str:
    """Format a number, leaving non-numbers (and absent values) legible as-is."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        # Plain text, deliberately unescaped: formatters produce values, and
        # escaping happens once, at the point of interpolation. Escaping here too
        # would double-escape every non-numeric cell.
        return str(value)


def _pct(value: Any, spec: str = "{:+.2%}") -> str:
    return _num(value, spec)


def _short_date(value: Optional[str]) -> str:
    return (value or "—")[:10]


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _card(title: str, body: str, *, tone: str = "") -> str:
    klass = f"card {tone}".strip()
    return f'<section class="{klass}"><h2>{esc(title)}</h2>{body}</section>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, raw: bool = False) -> str:
    """A table. ``raw`` marks cells already escaped by the caller (formatters that
    emit markup, like a pass/fail pill); everything else is escaped here."""
    if not rows:
        return '<p class="empty">Not recorded.</p>'
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{c if raw else esc(c)}</td>" for c in row)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kv(pairs: Sequence[Tuple[str, Any]]) -> str:
    """A definition list. Every key and value is escaped, with no exceptions.

    There is no markup passthrough here on purpose: a "this string already looks
    like markup" heuristic is indistinguishable from "this user-supplied symbol
    contains a `<`", which is precisely how a symbol list becomes script injection.
    Callers needing markup compose it themselves, outside this helper.
    """
    items = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in pairs)
    return f"<dl class='kv'>{items}</dl>"


def _pill(passed: Optional[bool]) -> str:
    if passed is None:
        return '<span class="pill neutral">—</span>'
    return '<span class="pill pass">PASS</span>' if passed else '<span class="pill fail">FAIL</span>'


def _warning(text: str, *, level: str = "warn") -> str:
    return f'<div class="banner {esc(level)}">{esc(text)}</div>'


def _chart(png: Optional[bytes], alt: str) -> str:
    """Embed a PNG as a data URI, or explain its absence.

    A missing chart is a slot with an explanation, never a broken run: the plotting
    dependency is an optional extra, and a text-complete report without pictures is
    strictly better than no report.
    """
    if png is None:
        return (
            '<p class="empty">Chart not rendered — install the viz extra '
            "(<code>uv sync --extra viz</code>) to include charts.</p>"
        )
    encoded = base64.b64encode(png).decode("ascii")
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{encoded}">'


def _try_chart(fn) -> Optional[bytes]:
    """Render a chart, or ``None`` when plotting is unavailable.

    Only the missing-dependency case degrades silently; a genuine rendering bug
    should not be disguised as "matplotlib isn't installed".
    """
    try:
        return fn()
    except RuntimeError:  # the viz extra's own actionable "not installed" error
        return None


# --------------------------------------------------------------------------- #
# Header blocks (identical for every kind)
# --------------------------------------------------------------------------- #
def _title(result: Dict[str, Any], kind: str) -> str:
    strategy = result.get("strategy") or (result.get("inputs") or {}).get("strategy") or "result"
    return f"TradeFlow {kind} — {strategy}"


def _window(result: Dict[str, Any]) -> Dict[str, Any]:
    return (result.get("inputs") or {}).get("window") or result.get("window") or {}


def _provenance_block(result: Dict[str, Any], kind: str) -> str:
    """What a reader needs to not misread this file a month from now.

    Mandatory, at the top, never a footnote: a report that outlives the context
    that produced it carries its window, universe, cost assumptions, code version,
    and campaign trial count, or it is a number without a claim attached.
    """
    inputs = result.get("inputs") or {}
    provenance = result.get("provenance") or {}
    window = _window(result)
    universe = inputs.get("universe") or result.get("symbols") or inputs.get("candidates") or []
    pairs: List[Tuple[str, Any]] = [
        ("Kind", kind),
        ("Window", f"{_short_date(window.get('start'))} → {_short_date(window.get('end'))}"),
        ("Universe", f"{len(universe)} names: {', '.join(str(s) for s in universe) or '—'}"),
        ("Timeframe", inputs.get("timeframe") or result.get("timeframe") or "—"),
        ("Benchmark", inputs.get("benchmark") or result.get("benchmark") or "—"),
        ("Cost model", _cost_text(inputs.get("cost"), result)),
        # "Version" rather than "Git SHA": an installed copy has no repository, and
        # the packaged version is the honest answer to "what made this".
        ("Code version", provenance.get("git_sha") or _code_version()),
        ("Generated", provenance.get("generated_at") or datetime.now(timezone.utc).isoformat()),
        ("Campaign trials", provenance.get("n_trials") or result.get("n_trials") or "—"),
    ]
    if result.get("run_id"):
        pairs.append(("Run id", result["run_id"]))
    if result.get("trial_id"):
        pairs.append(("Trial id", result["trial_id"]))
    return _card("Provenance", _kv(pairs))


def _code_version() -> str:
    """The running code's identifier, for a result dict that carries none."""
    from tradeflow.services.analysis import code_version

    return code_version()


def _cost_text(cost: Optional[Dict[str, Any]], result: Dict[str, Any]) -> str:
    if cost is None:
        return "GROSS (no transaction cost charged)" if result.get("gross") else "net of cost (defaults)"
    if cost.get("gross"):
        return "GROSS (no transaction cost charged)"
    return (
        f"{cost.get('commission_bps')}bps commission · impact η={cost.get('impact_eta')} "
        f"· borrow {cost.get('borrow_bps')}bps"
    )


def _warnings_block(result: Dict[str, Any], kind: str) -> str:
    """Everything that should stop a reader, before anything that might please one."""
    banners: List[str] = []

    if result.get("memoized"):
        banners.append(
            _warning(
                f"REUSED — not re-run. Served from trial {result.get('trial_id', '?')} "
                f"recorded {result.get('trial_ts') or 'at an unknown time'} "
                f"({_age(result.get('trial_ts'))} old). Re-run with --force to re-verify.",
                level="reuse",
            )
        )

    verdict = result.get("verdict") or {}
    if verdict.get("verdict") == "incomplete":
        banners.append(
            _warning(
                "INCOMPLETE — no verdict. A step failed"
                + (
                    f" ({', '.join(verdict.get('failed_steps') or [])})"
                    if verdict.get("failed_steps")
                    else ""
                )
                + ". Do not act on the sections below.",
                level="fail",
            )
        )
    failed = [name for name, c in (verdict.get("checks") or {}).items() if not c.get("passed")]
    if failed:
        banners.append(_warning("Gate failures: " + ", ".join(sorted(failed)), level="fail"))

    gate_report = result.get("gate_report") or {}
    if gate_report and not gate_report.get("promotable"):
        failed_gates = [n for n, c in (gate_report.get("checks") or {}).items() if not c.get("passed")]
        banners.append(
            _warning("NOT PROMOTABLE — failed gates: " + ", ".join(sorted(failed_gates)), level="fail")
        )

    if result.get("low_sample"):
        banners.append(_warning("Low sample — too few rebalances to measure an IC with confidence."))
    if result.get("sanity_ceiling_breached"):
        banners.append(
            _warning("Realized IR above 2 on public data — suspect a bug or a leak, not skill.", level="fail")
        )
    leakage = (result.get("diagnostics") or {}).get("leakage_probe")
    if leakage is not None and not leakage.get("passed"):
        banners.append(_warning("Leakage probe FAILED — the result is not trustworthy.", level="fail"))

    banners.extend(_default_off_banners(result))
    return "".join(banners)


def _default_off_banners(result: Dict[str, Any]) -> List[str]:
    """Name every evidence-gated, default-off feature this run turned on."""
    portfolio = result.get("portfolio") if isinstance(result.get("portfolio"), dict) else result
    banners = []
    for key, description in _DEFAULT_OFF.items():
        value = portfolio.get(key) if isinstance(portfolio, dict) else None
        if value:
            banners.append(
                _warning(
                    f"Non-default configuration: {description} is ENABLED ({value}). "
                    "It ships off because its own adoption gate does not clear on this "
                    "repo's data — this run is not the default path.",
                    level="note",
                )
            )
    return banners


def _age(ts: Optional[str]) -> str:
    if not ts:
        return "unknown age"
    try:
        when = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (datetime.now(timezone.utc) - when).total_seconds()
    except ValueError:
        return "unknown age"
    if seconds < 3600:
        return f"{int(max(seconds, 0) // 60)} minutes"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


# --------------------------------------------------------------------------- #
# Per-kind sections
# --------------------------------------------------------------------------- #
def _verdict_sections(result: Dict[str, Any], extras: Dict[str, Any]) -> List[str]:
    from tradeflow.analytics import charts

    verdict = result.get("verdict") or {}
    checks = verdict.get("checks") or {}
    out = []

    banner_class = {
        "promotable": "pass",
        "not promotable": "fail",
        "needs more data": "warn",
        "mixed": "warn",
        "incomplete": "fail",
    }.get(verdict.get("verdict"), "warn")
    out.append(
        _card(
            "Verdict",
            f'<p class="verdict {banner_class}">{esc(verdict.get("summary", "unknown"))}</p>'
            + _table(
                ["", "Check", "Value", "Threshold", "Why it is there"],
                [
                    [
                        _pill(c.get("passed")),
                        esc(name),
                        esc(_num(c.get("value"))),
                        esc(_num(c.get("threshold"))),
                        esc(c.get("note", "")),
                    ]
                    for name, c in sorted(checks.items())
                ],
                raw=True,
            )
            + (
                _chart(
                    _try_chart(
                        lambda: charts.render_gate_png(
                            checks, promotable=bool(verdict.get("promotable")), title="RESEARCH VERDICT"
                        )
                    ),
                    "verdict gate scorecard",
                )
                if checks
                else ""
            ),
            tone=banner_class,
        )
    )

    steps = result.get("steps") or {}
    notable = [(n, s) for n, s in steps.items() if s.get("status") != "ok"]
    if notable:
        out.append(
            _card(
                "Steps",
                _table(
                    ["Step", "Status", "Detail"],
                    [[n, s.get("status"), s.get("error") or s.get("reason") or ""] for n, s in notable],
                ),
            )
        )

    scan = result.get("scan")
    if scan:
        note = (
            "<p>Nothing flagged — the full candidate list was analyzed instead.</p>"
            if scan.get("fell_back_to_candidates")
            else ""
        )
        out.append(
            _card(
                "Scan",
                f"<p>{esc(scan.get('scanner'))}: {esc(scan.get('flagged_count', 0))} of "
                f"{esc(len(scan.get('candidates') or []))} candidates flagged.</p>" + note,
            )
        )

    source = result.get("combination") or result.get("alphas")
    if source:
        rows = source.get("alphas") or []
        chart = _chart(
            _try_chart(
                lambda: charts.render_bars_png(
                    [str(r.get("symbol")) for r in rows[:20]],
                    [float(r.get("alpha", 0.0)) for r in rows[:20]],
                    title="Alpha forecast by name",
                    value_label="annualized residual return",
                )
            ),
            "alphas by name",
        )
        out.append(
            _card(
                "Alphas",
                chart
                + _table(
                    ["Symbol", "Alpha", "z", "Residual vol"],
                    [
                        [r.get("symbol"), _num(r.get("alpha")), _num(r.get("z")), _num(r.get("residual_vol"))]
                        for r in rows
                    ],
                ),
            )
        )

    out.append(_portfolio_card(result.get("portfolio")))
    out.append(_information_card(result.get("information")))
    return [s for s in out if s]


def _portfolio_card(portfolio: Optional[Dict[str, Any]]) -> str:
    from tradeflow.analytics import charts

    if not portfolio:
        return ""
    if not portfolio.get("feasible"):
        return _card(
            "Portfolio",
            f"<p>NOT FEASIBLE — {esc(portfolio.get('binding_constraint') or portfolio.get('note') or 'unknown')}</p>",
            tone="fail",
        )
    d = portfolio.get("diagnostics") or {}
    weights = portfolio.get("weights") or {}
    chart = _chart(
        _try_chart(
            lambda: charts.render_bars_png(
                list(weights)[:20],
                [float(v) for v in list(weights.values())[:20]],
                title="Proposed weights",
                value_label="portfolio weight",
            )
        ),
        "proposed portfolio weights",
    )
    exposures = portfolio.get("exposures")
    body = (
        "<p class='note'>A proposal, not an order.</p>"
        + _kv(
            [
                ("Names", len(weights)),
                ("Predicted tracking error", _pct(d.get("predicted_tracking_error"), "{:.2%}")),
                ("Target tracking error", _pct(portfolio.get("target_te"), "{:.2%}")),
                ("Expected active return (gross)", _pct(d.get("expected_active_return"))),
                ("Expected active return (net)", _pct(d.get("expected_active_return_net"))),
                ("Predicted IR", _num(d.get("predicted_ir"), "{:+.2f}")),
                ("Transfer coefficient", _num(d.get("transfer_coefficient"), "{:+.2f}")),
                ("Capacity (capital)", _num(d.get("capacity_capital"), "{:,.0f}")),
                ("Risk model", portfolio.get("risk_model")),
            ]
        )
        + chart
        + _table(["Symbol", "Weight"], [[sym, _pct(w, "{:.2%}")] for sym, w in weights.items()])
    )
    if exposures:
        body += "<h3>Factor exposures</h3>" + _table(
            ["Factor", "Exposure"], [[k, _num(v, "{:+.3f}")] for k, v in sorted(exposures.items())]
        )
    return _card("Portfolio", body)


def _information_card(information: Optional[Dict[str, Any]]) -> str:
    if not information or not information.get("periods"):
        return ""
    r = information
    return _card(
        "Information",
        _kv(
            [
                ("Rebalances measured", r.get("periods")),
                ("Horizon (bars)", r.get("horizon_bars")),
                ("IC (mean)", _num(r.get("mean_ic"), "{:+.4f}")),
                ("IC t-stat", _num(r.get("ic_tstat"), "{:+.2f}")),
                ("Rank IC", _num(r.get("rank_ic"), "{:+.4f}")),
                ("Effective breadth", _num(r.get("breadth_effective"), "{:.0f}")),
                ("Mean correlation ρ̄", _num(r.get("rho_bar"), "{:.2f}")),
                ("Predicted IR", _num(r.get("predicted_ir"), "{:+.2f}")),
                (
                    "Realized IR",
                    f"{_num(r.get('realized_ir'), '{:+.2f}')} ± {_num(r.get('ir_standard_error'), '{:.2f}')}",
                ),
                (
                    f"P(any |t|>2 across {r.get('n_trials', 1)} trials)",
                    _num(r.get("multiple_testing_inflation"), "{:.2f}"),
                ),
            ]
        ),
    )


def _backtest_sections(result: Dict[str, Any], extras: Dict[str, Any]) -> List[str]:
    from tradeflow.analytics import charts

    metrics = result.get("metrics") or {}
    equity = extras.get("equity_curve")
    out = []
    if equity:
        out.append(
            _card(
                "Equity",
                _chart(
                    _try_chart(lambda: charts.render_equity_png(equity, title="Backtest equity")),
                    "backtest equity curve",
                ),
            )
        )
    out.append(
        _card(
            "Metrics",
            _table(["Metric", "Value"], [[k, _num(v)] for k, v in sorted(metrics.items())]),
        )
    )
    out.append(
        _card(
            "Run",
            _kv(
                [
                    ("Initial capital", _num(result.get("initial_capital"), "{:,.2f}")),
                    ("Final capital", _num(result.get("final_capital"), "{:,.2f}")),
                    ("Transaction cost", _num(result.get("total_cost"), "{:,.2f}")),
                    ("Cost drag", _num(result.get("cost_drag_pct"), "{:.2f}") + "%"),
                    ("Trades", result.get("total_trades")),
                ]
            ),
        )
    )
    return out


def _walkforward_sections(result: Dict[str, Any], extras: Dict[str, Any]) -> List[str]:
    from tradeflow.analytics import charts

    gate_report = result.get("gate_report") or {}
    checks = gate_report.get("checks") or {}
    out = [
        _card(
            "Verdict",
            f'<p class="verdict {"pass" if gate_report.get("promotable") else "fail"}">'
            f"{'PROMOTABLE' if gate_report.get('promotable') else 'NOT PROMOTABLE'}</p>"
            + _table(
                ["", "Gate", "Value", "Threshold"],
                [
                    [_pill(c.get("passed")), esc(n), esc(_num(c.get("value"))), esc(_num(c.get("threshold")))]
                    for n, c in sorted(checks.items())
                ],
                raw=True,
            )
            + (
                _chart(
                    _try_chart(
                        lambda: charts.render_gate_png(
                            checks,
                            promotable=bool(gate_report.get("promotable")),
                            title="WALK-FORWARD VERDICT",
                            stats=[
                                ("OOS Sharpe (median)", result.get("median_oos_sharpe", 0.0)),
                                ("Efficiency (OOS / IS)", result.get("median_efficiency", 0.0)),
                                ("OOS trades", result.get("total_oos_trades", 0)),
                            ],
                        )
                    ),
                    "walk-forward gate scorecard",
                )
                if checks
                else ""
            ),
            tone="pass" if gate_report.get("promotable") else "fail",
        ),
        _card(
            "Folds",
            _table(
                ["#", "IS window", "OOS window", "IS Sharpe", "OOS Sharpe", "OOS trades", "Trials"],
                [
                    [
                        f.get("index"),
                        f"{_short_date((f.get('is_window') or {}).get('start'))}→"
                        f"{_short_date((f.get('is_window') or {}).get('end'))}",
                        f"{_short_date((f.get('oos_window') or {}).get('start'))}→"
                        f"{_short_date((f.get('oos_window') or {}).get('end'))}",
                        _num(f.get("is_sharpe"), "{:+.2f}"),
                        _num(f.get("oos_sharpe"), "{:+.2f}"),
                        f.get("oos_trades"),
                        f.get("n_trials"),
                    ]
                    for f in result.get("folds") or []
                ],
            ),
        ),
        _card(
            "Out-of-sample aggregate",
            _table(
                ["Metric", "Value"],
                [[k, _num(v)] for k, v in sorted((result.get("oos_aggregate") or {}).items())],
            ),
        ),
    ]
    if result.get("holdout"):
        out.append(
            _card(
                "Holdout",
                _table(["Metric", "Value"], [[k, _num(v)] for k, v in sorted((result["holdout"]).items())]),
            )
        )
    return out


def _info_sections(result: Dict[str, Any], extras: Dict[str, Any]) -> List[str]:
    card = _information_card(result)
    body = [card] if card else []
    body.append(
        _card(
            "How to read this",
            f"<p>{esc(result.get('note', ''))}</p>",
        )
    )
    return body


def _footer() -> str:
    return (
        "<footer><p>Generated by TradeFlow. This file is self-contained: opening it "
        "issues no network requests. It is a research artifact, not investment advice.</p></footer>"
    )


def write_html(result: Dict[str, Any], kind: str, path: str, **kwargs: Any) -> str:
    """Render and write the report; return the path written."""
    document = render_html(result, kind, **kwargs)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(document)
    return path


def _json_block(payload: Any) -> str:  # pragma: no cover - reserved for a detail view
    return f"<pre>{esc(json.dumps(payload, indent=2, sort_keys=True))}</pre>"


#: Inline, dependency-free, and light/dark aware. System font stacks only - a web
#: font would be an external request, which this document does not make.
_CSS = """
:root { color-scheme: light dark;
  --bg:#ffffff; --fg:#1b1f24; --muted:#6b7280; --line:#e5e7eb; --card:#ffffff;
  --pass:#16a34a; --fail:#dc2626; --warn:#b45309; --accent:#2563eb; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#0f1216; --fg:#e6e8eb; --muted:#9aa3ad; --line:#2a2f36; --card:#161a20;
  --pass:#4ade80; --fail:#f87171; --warn:#fbbf24; --accent:#60a5fa; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,
  BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 60rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
h1 { font-size:1.6rem; margin:0 0 1.25rem; }
h2 { font-size:1.05rem; margin:0 0 .75rem; letter-spacing:.02em; text-transform:uppercase;
  color:var(--muted); }
h3 { font-size:.95rem; margin:1.25rem 0 .5rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:1.1rem 1.25rem; margin:0 0 1rem; }
.card.pass { border-left:4px solid var(--pass); }
.card.fail { border-left:4px solid var(--fail); }
.card.warn { border-left:4px solid var(--warn); }
.banner { border-radius:8px; padding:.8rem 1rem; margin:0 0 .75rem; font-weight:600;
  border:1px solid var(--line); }
.banner.fail { background:color-mix(in srgb, var(--fail) 12%, transparent); border-color:var(--fail); }
.banner.warn, .banner.reuse { background:color-mix(in srgb, var(--warn) 14%, transparent);
  border-color:var(--warn); }
.banner.note { background:color-mix(in srgb, var(--accent) 10%, transparent); border-color:var(--accent); }
.verdict { font-size:1.2rem; font-weight:700; margin:.25rem 0 1rem; }
.verdict.pass { color:var(--pass); } .verdict.fail { color:var(--fail); }
.verdict.warn { color:var(--warn); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line); white-space:nowrap; }
th { color:var(--muted); font-weight:600; }
td:last-child, th:last-child { white-space:normal; }
dl.kv { display:grid; grid-template-columns:repeat(auto-fit,minmax(15rem,1fr)); gap:.4rem 1.5rem; margin:0; }
dl.kv div { display:flex; justify-content:space-between; gap:1rem; border-bottom:1px solid var(--line);
  padding:.3rem 0; }
dt { color:var(--muted); } dd { margin:0; font-variant-numeric:tabular-nums; text-align:right; }
.pill { font-size:.72rem; font-weight:700; padding:.12rem .45rem; border-radius:999px; }
.pill.pass { background:var(--pass); color:#fff; } .pill.fail { background:var(--fail); color:#fff; }
.pill.neutral { background:var(--muted); color:#fff; }
img { max-width:100%; height:auto; display:block; margin:1rem 0; border-radius:6px; }
.empty, .note { color:var(--muted); font-style:italic; }
code { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.85em; }
pre { overflow-x:auto; background:var(--card); padding:.75rem; border-radius:6px; }
footer { color:var(--muted); font-size:.82rem; margin-top:2rem; border-top:1px solid var(--line);
  padding-top:1rem; }
"""
