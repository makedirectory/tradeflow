"""Human-readable rendering of research results.

Kept separate from metric *computation* so the numbers can be consumed
programmatically (e.g. by the optimizer, or rendered to another medium) without
any formatting concerns. Nothing here computes: a renderer that derives a number
is a renderer that can disagree with the report it is summarizing.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from tradeflow.analytics import metrics as m
from tradeflow.analytics import performance

logger = logging.getLogger(__name__)

# (metric key, label, format spec), grouped into report sections.
_SECTIONS = [
    (
        "Returns",
        [
            ("total_return", "Total Return", "{:.2f}%"),
            ("cagr", "CAGR", "{:.2f}%"),
            ("buy_hold_return", "Buy & Hold (universe)", "{:.2f}%"),
            ("benchmark_buy_hold_return", "Buy & Hold (benchmark)", "{:.2f}%"),
            ("annualized_volatility", "Annualized Volatility", "{:.2f}%"),
        ],
    ),
    (
        "Risk-adjusted",
        [
            ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
            ("probabilistic_sharpe_ratio", "  Probabilistic Sharpe", "{:.2f}"),
            ("deflated_sharpe_ratio", "  Deflated Sharpe", "{:.2f}"),
            ("sortino_ratio", "Sortino Ratio", "{:.2f}"),
            ("calmar_ratio", "Calmar Ratio", "{:.2f}"),
            ("martin_ratio", "Martin Ratio (UPI)", "{:.2f}"),
            ("information_ratio", "Information Ratio", "{:.2f}"),
            ("treynor_ratio", "Treynor Ratio", "{:.2f}"),
        ],
    ),
    (
        "Drawdown & tail",
        [
            ("max_drawdown", "Max Drawdown", "{:.2f}%"),
            ("max_drawdown_duration", "Max DD Duration (days)", "{:.0f}"),
            ("ulcer_index", "Ulcer Index", "{:.2f}"),
            ("recovery_factor", "Recovery Factor", "{:.2f}"),
            ("var_95", "VaR (95%)", "{:.2f}%"),
            ("cvar_95", "CVaR (95%)", "{:.2f}%"),
            ("var_99", "VaR (99%)", "{:.2f}%"),
            ("exposure", "Time in Market", "{:.1f}%"),
        ],
    ),
    (
        "Trades",
        [
            ("total_trades", "Total Trades", "{:.0f}"),
            ("win_rate", "Win Rate", "{:.2f}%"),
            ("profit_factor", "Profit Factor", "{:.2f}"),
            ("payoff_ratio", "Payoff Ratio", "{:.2f}"),
            ("expectancy", "Expectancy", "${:.2f}"),
            ("sqn", "System Quality (SQN)", "{:.2f}"),
            ("kelly_criterion", "Kelly Fraction", "{:.2f}"),
            ("max_consecutive_wins", "Max Consecutive Wins", "{:.0f}"),
            ("max_consecutive_losses", "Max Consecutive Losses", "{:.0f}"),
            ("avg_trade_duration", "Avg Trade Duration (h)", "{:.2f}"),
            ("mae_pct", "Avg MAE", "{:.2f}%"),
            ("mfe_pct", "Avg MFE", "{:.2f}%"),
            ("avg_win", "Average Win", "${:.2f}"),
            ("avg_loss", "Average Loss", "${:.2f}"),
            ("largest_win", "Largest Win", "${:.2f}"),
            ("largest_loss", "Largest Loss", "${:.2f}"),
        ],
    ),
    (
        "Benchmark-relative",
        [
            ("alpha", "Alpha (per period)", "{:.5f}"),
            ("beta", "Beta", "{:.2f}"),
            ("r_squared", "R-squared", "{:.2f}"),
        ],
    ),
]

# Flat view for backward compatibility / simple consumers.
_ROWS = [row for _, rows in _SECTIONS for row in rows]


def _execution_lines(execution: Optional[Dict[str, Any]]) -> List[str]:
    """The gap between the book the sizer asked for and the one that could be traded.

    Silent by construction until now: whole-share rounding just happens, and it scales
    with how small the account is. Shown whenever anything was lost, and shown with
    both numbers beside their thresholds rather than as a verdict on its own - the same
    rule the promotion gates follow.
    """
    if not execution:
        return []
    verdict = performance.execution_verdict(execution)
    if verdict["executable"] is None or (verdict["executable"] and not execution.get("rounding_drag_pct")):
        return []

    mark = "(!)" if not verdict["executable"] else "(i)"
    lines = [
        "",
        f"--- Execution & cost {mark} ---",
        f"{'Intended notional':28}${execution.get('requested_notional', 0.0):,.2f}",
        f"{'Filled notional':28}${execution.get('filled_notional', 0.0):,.2f}",
    ]
    for name, check in verdict["checks"].items():
        flag = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  [{flag}] {name:22}{check['value']:.2f}% vs {check['threshold']:.2f}%")
    lost = execution.get("positions_rounded_to_zero", 0) + execution.get("positions_below_min_notional", 0)
    if lost:
        lines.append(
            f"{'  entries never opened':28}{lost} "
            f"({execution.get('positions_rounded_to_zero', 0)} rounded to zero, "
            f"{execution.get('positions_below_min_notional', 0)} below min notional)"
        )
    if not verdict["executable"]:
        lines.append("  Not the book that was validated - see the failing check above.")
    return lines


def _leg_lines(legs: Optional[Dict[str, Any]]) -> List[str]:
    """Each side of a long/short book, separately.

    Shown only when both sides actually traded: a long-only run has nothing to
    decompose, and a table with an empty short row is noise that teaches readers to
    skip the block.

    The question it exists to answer is what a net figure structurally cannot: a
    near-zero net beta is either genuinely small exposure on both sides, or a large
    long beta cancelling a large short one. Same number, opposite risk.
    """
    if not legs or not all(legs.get(side, {}).get("trades") for side in ("long", "short")):
        return []
    lines = [
        "",
        "--- Legs (diagnostic; no thresholds) ---",
        f"  {'leg':7}{'return':>9}{'vol':>8}{'max DD':>9}{'beta':>8}{'corr':>7}{'trades':>8}{'cost':>11}",
    ]
    for name in ("long", "short"):
        leg = legs[name]
        beta, corr = leg.get("beta"), leg.get("benchmark_correlation")
        lines.append(
            f"  {name:7}{leg['return_pct']:>+8.2f}%{leg['volatility_pct']:>7.3f}%"
            f"{leg['max_drawdown_pct']:>8.2f}%"
            f"{'n/a' if beta is None else format(beta, '.3f'):>8}"
            f"{'n/a' if corr is None else format(corr, '.2f'):>7}"
            f"{leg['trades']:>8}{leg['cost']:>11,.0f}"
        )
    betas = [legs[name].get("beta") for name in ("long", "short")]
    if all(b is not None for b in betas) and min(abs(b) for b in betas) > 0.3:
        lines.append(
            "  Both legs carry real market exposure - a small net beta here is two "
            "exposures cancelling, not an absence of them."
        )
    return lines


def format_universe_provenance(
    *,
    candidates: Sequence[str],
    resolved: Sequence[str],
    scanner: Optional[str],
    scan_clock: Optional[str],
    source: str,
    replayed: bool,
) -> List[str]:
    """Where this run's universe came from, and what that does and does not guarantee.

    A 61-name large-cap list is not "the market", and a report that leaves the universe
    in the background invites it to be read as one. The awkward line is the last: a
    hand-supplied candidate list is *today's* names applied to history, so anything that
    left the list - delisted, acquired, collapsed - is already absent from every
    backtest run over it.

    That is stated rather than measured, because measuring it needs point-in-time
    membership data this project does not ingest. Saying "this is survivorship-prone by
    construction" is honest; computing a survivorship number from a list that has none
    would not be.
    """
    lines = [
        "",
        "=== Universe provenance ===",
        f"  {'candidates':16}{len(candidates)} names from {source}",
    ]
    if scanner and scanner != "none":
        clock = f" as of {scan_clock}" if scan_clock else ""
        lines.append(f"  {'scanner':16}{scanner}{clock}")
        lines.append(f"  {'resolved':16}{len(resolved)} of {len(candidates)} names")
    else:
        lines.append(f"  {'scanner':16}none - candidates traded as-is")
    lines.append(f"  {'universe':16}{'replayed from config' if replayed else 'resolved this run'}")
    lines.append(
        f"  {'survivorship':16}a hand-supplied list is today's names applied to history; "
        "membership was not point-in-time"
    )
    return lines


def format_verdicts(
    *,
    statistical: Optional[Dict[str, Any]] = None,
    execution: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """The three facts about a candidate, side by side and separately labelled.

    They were already three verdicts that never collapse into one another, but each was
    printed by a different command at a different moment, so nothing ever showed a
    reader all three. That is how a backtest replay reads as "approved" when it only
    means "this saved config runs and its history looks good".

    A verdict this command could not assess is printed as **not assessed here**, with
    the command that would assess it - the same rule the prerequisites follow, because
    an unknown rendered as a blank is an unknown a reader fills in optimistically.
    """
    rows = [
        ("Statistical validation", statistical, "was the edge real, and not overfit", "walkforward"),
        ("Execution viability", execution, "can this book be traded at this capital", "backtest"),
        (
            "Evidence completeness",
            evidence,
            "what has actually been checked",
            "walkforward --bootstrap-skill",
        ),
    ]
    lines = ["", "=== Verdicts ==="]
    for label, verdict, question, command in rows:
        if verdict is None:
            lines.append(f"  {label:24}not assessed here - {question} (`{command}`)")
            continue
        lines.append(f"  {label:24}{verdict}")
    lines.append("  Three separate facts. Clearing one says nothing about the others.")
    return lines


def format_backtest_report(
    metrics: Dict[str, float],
    initial_capital: float,
    final_capital: float,
    title: str = "Backtest Results",
    execution: Optional[Dict[str, Any]] = None,
    legs: Optional[Dict[str, Any]] = None,
) -> str:
    """Render metrics as an aligned, fixed-width text block, grouped by section."""
    lines = [f"=== {title} ===", f"{'Capital':28}${initial_capital:,.2f} -> ${final_capital:,.2f}"]
    if metrics.get("low_sample"):
        lines.append(f"{'(!) low sample':28}fewer than {30} trades - treat ratios with caution")
    if not metrics.get("benchmark_available", True):
        lines.append(f"{'(i) no benchmark':28}alpha/beta/information-ratio unavailable")
    elif not metrics.get("treynor_available", True):
        lines.append(
            f"{'(i) beta near zero':28}Treynor unavailable (|beta| < "
            f"{m.MIN_ABS_BETA_FOR_TREYNOR:g}; excess return per unit of beta is not "
            f"meaningful for a book with no market exposure)"
        )
    lines.extend(_execution_lines(execution))
    lines.extend(_leg_lines(legs))
    undeflated = not metrics.get("deflation_applied", True)
    for section, rows in _SECTIONS:
        section_lines = []
        for key, label, fmt in rows:
            if key == "deflated_sharpe_ratio" and undeflated:
                # One trial deflates against nothing, so the row says what it is rather
                # than borrowing the name of a correction that was not applied.
                label = "  Sharpe (undeflated)"
            if key in metrics:
                value = fmt.format(metrics[key]) if metrics[key] != float("inf") else "inf"
                section_lines.append(f"{label:28}{value}")
        if section_lines:
            lines.append(f"--- {section} ---")
            lines.extend(section_lines)
    lines.append("=" * (len(title) + 8))
    return "\n".join(lines)


def log_backtest_report(
    metrics: Dict[str, float],
    initial_capital: float,
    final_capital: float,
    execution: Optional[Dict[str, Any]] = None,
    legs: Optional[Dict[str, Any]] = None,
) -> None:
    """Log the rendered report at INFO."""
    logger.info("\n%s", format_backtest_report(metrics, initial_capital, final_capital, execution=execution))


def _age_str(ts: Optional[str]) -> str:
    """A human ``"N days"``/``"N hours"`` age for an ISO timestamp; ``"unknown"``
    when it can't be parsed - never let a formatting failure block the notice."""
    if not ts:
        return "unknown age"
    try:
        when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - when
    except ValueError:
        return "unknown age"
    seconds = delta.total_seconds()
    if seconds < 0:
        return "0 days"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def format_cached_notice(
    row: Dict[str, Any], *, current_accounting: Optional[int] = None, vintage_available: bool = False
) -> str:
    """A prominent, unmistakable "this is a memo, not a fresh verification" banner.

    Every command that can serve a memoized trial (``backtest``/``optimize``/
    ``walkforward``, CLI and MCP alike) renders this immediately above its report
    so the reused case looks visually distinct everywhere, not just wherever it
    was first implemented. Always names the original run's id and timestamp/age -
    never let a stale number pass as freshly checked.

    ``vintage_available`` is ``True`` only when *this* lookup was itself keyed on
    a bar-cache data-vintage stamp and still matched - which, by construction,
    means the match's underlying data vintage is identical to the stored trial's
    (that's what made the dedup hash match). In that case the data caveat below is
    replaced with a positive statement rather than dropped silently; a run without
    the bar cache (``--cache``/``--offline``) still gets the original caveat, since
    for it nothing has actually changed.
    """
    ts = row.get("ts")
    lines = [
        f"=== REUSED — trial {row.get('id', '?')} from {ts or 'unknown time'} ({_age_str(ts)} old) ===",
        "Not re-run: an identical trial already exists in the trial store. Pass --force/--rerun to re-verify.",
    ]
    row_accounting = row.get("accounting")
    if current_accounting is not None and row_accounting is not None and row_accounting != current_accounting:
        lines.append(
            f"(!) stored under accounting v{row_accounting}, engine is v{current_accounting} — "
            "its metrics are NOT comparable to a fresh run"
        )
    if vintage_available:
        lines.append(
            "(i) data vintage confirmed: the bar cache's fetch stamp for this exact window matches the "
            "original run's — this reuse is vintage-safe, not just same-inputs"
        )
    else:
        lines.append(
            "(i) reused on 'same requested inputs' only — not yet guaranteed identical underlying data "
            "(no data-vintage stamp; re-run with --cache/--offline for a vintage-safe reuse); a rare vendor "
            "correction/backfill since the original run could differ"
        )
    return "\n".join(lines)


def format_verdict_report(result: Dict[str, Any]) -> str:
    """Render a composite research result as one consolidated text report.

    Pure formatting over the composite object: every number here was produced by
    the step that owns it. The verdict banner is printed **last and loudest** -
    the sections above it are evidence, and a report that buries its own conclusion
    under a pretty table is worse than no report.
    """
    inputs = result.get("inputs") or {}
    window = inputs.get("window") or {}
    universe = inputs.get("universe") or []
    provenance = result.get("provenance") or {}
    lines = [
        "",
        f"=== Research verdict: '{inputs.get('strategy', '?')}' "
        f"{_short_date(window.get('start'))}..{_short_date(window.get('end'))} ===",
        f"  universe: {len(universe)} names ({_symbol_summary(universe)})",
        f"  timeframe {inputs.get('timeframe', '?')} | benchmark {inputs.get('benchmark', '?')} | "
        f"cost {_cost_summary(inputs.get('cost'))}",
    ]
    if provenance:
        lines.append(
            f"  provenance: git {provenance.get('git_sha') or 'unknown'} | "
            f"campaign trials {provenance.get('n_trials', '?')} | "
            f"bar requests: {_fetch_summary(provenance.get('bar_requests'))}"
        )

    lines.extend(_verdict_step_lines(result))
    lines.extend(_verdict_scan_lines(result))
    lines.extend(_verdict_alpha_lines(result))
    lines.extend(_verdict_portfolio_lines(result))
    lines.extend(_verdict_information_lines(result))
    lines.extend(_verdict_banner_lines(result))
    return "\n".join(lines)


def _short_date(value: Optional[str]) -> str:
    return (value or "?")[:10]


def _symbol_summary(symbols, limit: int = 8) -> str:
    listed = list(symbols)[:limit]
    suffix = f", +{len(symbols) - limit} more" if len(symbols) > limit else ""
    return ", ".join(str(s) for s in listed) + suffix if listed else "none"


def _cost_summary(cost: Optional[Dict[str, Any]]) -> str:
    if not cost:
        return "unknown"
    if cost.get("gross"):
        return "GROSS (no transaction cost charged)"
    return (
        f"{cost.get('commission_bps')}bps commission, impact η={cost.get('impact_eta')}, "
        f"borrow {cost.get('borrow_bps')}bps"
    )


def format_offline_scan_notice(as_of) -> str:
    """Say plainly that an offline scan saw only what the cache already holds.

    A scan picks a universe at a clock. Offline it reads local Parquet and nothing
    else, so the selection is exactly as current as the cache and no more - and
    nothing errors when coverage ends before that clock: the newest cached bar simply
    becomes "the latest", and a universe chosen from stale bars is indistinguishable
    from one chosen from fresh ones. That is precisely the case that has to announce
    itself.
    """
    return (
        f"OFFLINE: universe resolved at {as_of.isoformat()} from cached bars only - "
        "as current as the cache and no more. `cache status` shows the coverage."
    )


def _fetch_summary(stats: Optional[Dict[str, Any]]) -> str:
    """How much of the "one shared fetch" claim actually held, as measured.

    This counts *in-run sharing*, not provider access. ``fetches`` is the number of
    requests that missed this run's own memo and reached the data client underneath -
    and on an ``--offline`` run that client is the local bar cache, so the old wording
    ("hit the provider") reported a network round trip on a run that made none. A
    provenance line that overstates where data came from is worse than no line, since
    provenance is the one thing a reader cannot check for themselves.
    """
    if not stats:
        return "not measured"
    fetches, requests = stats.get("fetches", "?"), stats.get("requests", "?")
    return f"{fetches} of {requests} reached the data client, the rest shared within this run"


def _verdict_step_lines(result: Dict[str, Any]) -> list:
    """One line per step that did not simply succeed - silence means everything ran."""
    steps = result.get("steps") or {}
    notable = [(name, s) for name, s in steps.items() if s.get("status") != "ok"]
    if not notable:
        return []
    lines = ["", "  Steps:"]
    for name, state in notable:
        detail = state.get("error") or state.get("reason") or ""
        marker = "(!)" if state.get("status") == "failed" else "  -"
        lines.append(f"  {marker} {name}: {state.get('status')}{f' — {detail}' if detail else ''}")
    return lines


def _verdict_scan_lines(result: Dict[str, Any]) -> list:
    scan = result.get("scan")
    if not scan:
        return []
    lines = [
        "",
        f"  Scan ({scan.get('scanner')}): {scan.get('flagged_count', 0)} of "
        f"{len(scan.get('candidates') or [])} candidates flagged",
    ]
    if scan.get("fell_back_to_candidates"):
        lines.append("  (i) nothing flagged — the full candidate list was analyzed instead")
    return lines


def _verdict_alpha_lines(result: Dict[str, Any], top: int = 5) -> list:
    combination = result.get("combination")
    alphas = result.get("alphas")
    source = combination or alphas
    if not source:
        return []
    lines = [""]
    if combination:
        weights = combination.get("signal_weights") or {}
        lines.append(
            f"  Alphas (combined, measured IC {combination.get('combined_ic', 0.0):+.4f}): "
            + ", ".join(f"{k} {v:+.2f}" for k, v in weights.items())
        )
    else:
        flag = "  (low confidence)" if alphas.get("low_confidence") else ""
        lines.append(
            f"  Alphas ({alphas.get('scaling', '?')} scaling, assumed IC "
            f"{alphas.get('ic', 0.0):+.4f}): {alphas.get('universe_size', 0)} names{flag}"
        )
    table = (source.get("alphas") or [])[:top]
    for row in table:
        lines.append(f"    {row.get('symbol', '?'):<8} alpha {float(row.get('alpha', 0.0)):+.4f}")
    return lines


def _verdict_portfolio_lines(result: Dict[str, Any], top: int = 5) -> list:
    pf = result.get("portfolio")
    if not pf:
        return []
    if not pf.get("feasible"):
        return [
            "",
            f"  Portfolio: NOT FEASIBLE — {pf.get('binding_constraint') or pf.get('note') or 'unknown'}",
        ]
    d = pf.get("diagnostics") or {}
    lines = [
        "",
        f"  Portfolio (proposal, not an order): {len(pf.get('weights') or {})} names, "
        f"TE {float(d.get('predicted_tracking_error', 0.0)):.2%} (target {pf.get('target_te', 0):.2%})",
        f"    expected active return {float(d.get('expected_active_return', 0.0)):+.2%} gross"
        f" / {float(d.get('expected_active_return_net', d.get('expected_active_return', 0.0))):+.2%} net",
        f"    predicted IR {float(d.get('predicted_ir', 0.0)):+.2f} | "
        f"transfer coefficient {float(d.get('transfer_coefficient', 0.0)):+.2f}",
    ]
    for symbol, weight in list((pf.get("weights") or {}).items())[:top]:
        lines.append(f"    {symbol:<8} {float(weight):>7.2%}")
    exposures = pf.get("exposures")
    if exposures:
        lines.append(
            "    exposures: " + ", ".join(f"{k} {float(v):+.2f}" for k, v in sorted(exposures.items()))
        )
    return lines


def _verdict_information_lines(result: Dict[str, Any]) -> list:
    inf = result.get("information")
    if not inf or not inf.get("periods"):
        return []
    flag = "  (!) low sample" if inf.get("low_sample") else ""
    return [
        "",
        f"  Information ({inf.get('periods')} rebalances, horizon {inf.get('horizon_bars')} bars):{flag}",
        f"    IC {float(inf.get('mean_ic', 0.0)):+.4f}  t-stat {float(inf.get('ic_tstat', 0.0)):+.2f}  "
        f"rank-IC {float(inf.get('rank_ic', 0.0)):+.4f}",
        f"    breadth {float(inf.get('breadth_effective', 0.0)):.0f} effective "
        f"(ρ̄ {float(inf.get('rho_bar', 0.0)):.2f}, {inf.get('n_names', 0)} names)",
        f"    IR predicted {float(inf.get('predicted_ir', 0.0)):+.2f} vs realized "
        f"{float(inf.get('realized_ir', 0.0)):+.2f} ± {float(inf.get('ir_standard_error', 0.0)):.2f}",
        f"    P(any |t|>2 across {inf.get('n_trials', 1)} campaign trials) = "
        f"{float(inf.get('multiple_testing_inflation', 0.0)):.2f}",
    ]


def _verdict_banner_lines(result: Dict[str, Any]) -> list:
    """The conclusion, with every gate that produced it shown underneath.

    Checks are listed whatever the verdict says, passes included: a verdict whose
    supporting numbers are only visible when it is bad teaches the reader to trust
    the good ones unexamined.
    """
    verdict = result.get("verdict") or {}
    lines = ["", f"  VERDICT: {verdict.get('summary', 'unknown')}"]
    for name, check in sorted((verdict.get("checks") or {}).items()):
        passed = bool(check.get("passed"))
        value, threshold = check.get("value"), check.get("threshold")
        detail = f"{_num(value)} vs {_num(threshold)}"
        # Every note states the *failing* condition ("...is indistinguishable from
        # zero"), so printing it beside a PASS produced a line that contradicted its
        # own verdict. On a pass the value and its threshold already say everything.
        note = f" — {check['note']}" if not passed and check.get("note") else ""
        lines.append(f"    [{'PASS' if passed else 'FAIL'}] {name}: {detail}{note}")
    if verdict.get("verdict") == "incomplete":
        lines.append("    No verdict is offered for a partial run — do not act on the sections above.")
    lines.append("")
    return lines


def _num(value: Any) -> str:
    """Format a gate's value/threshold without pretending a bool or a None is a float."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


# --------------------------------------------------------------------------- #
# Trial-store browsing
# --------------------------------------------------------------------------- #
#: Rendered wherever a stored field is absent. A trial recorded before a field
#: existed, or of a kind that never produces one, did not score zero and did not
#: fail — it has nothing recorded, and the two must never look alike.
NOT_RECORDED = "—"


def _cell(value: Any, spec: str = "{:.3f}") -> str:
    """A metric cell, or :data:`NOT_RECORDED` when there is genuinely nothing stored."""
    if value is None:
        return NOT_RECORDED
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _plural(count: int, noun: str) -> str:
    """``1 name`` / ``2 names`` — a detail view is read closely enough for this to
    grate."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def format_trials_table(rows: List[Dict[str, Any]], *, total: Optional[int] = None) -> str:
    """The trial listing: one row per trial, absences rendered as absences.

    Always states how many rows matched versus how many are shown — a listing that
    silently truncates reads as "this is everything" when it is not.
    """
    if not rows:
        return "No trials matched."
    lines = [
        f"{'ID':14}{'KIND':12}{'STRATEGY':16}{'SHARPE':>9}{'DSR':>8}{'PROMO':>7}{'ACCT':>6}  TS",
    ]
    for r in rows:
        promo = NOT_RECORDED if r.get("promotable") is None else ("yes" if r["promotable"] else "no")
        lines.append(
            f"{str(r.get('id', ''))[:14]:14}{str(r.get('kind', ''))[:12]:12}"
            f"{str(r.get('strategy') or '')[:16]:16}"
            f"{_cell(r.get('oos_sharpe')):>9}{_cell(r.get('deflated_sharpe')):>8}"
            f"{promo:>7}{r.get('accounting', ''):>6}  {(r.get('ts') or '')[:19]}"
        )
    if total is not None and total > len(rows):
        lines.append(f"\nShowing {len(rows)} of {total} matching trials (use --limit/--offset for more).")
    elif total is not None:
        lines.append(f"\n{total} matching trial(s).")
    return "\n".join(lines)


def format_trial_detail(trial: Dict[str, Any]) -> str:
    """Everything the store knows about one trial.

    Companion records report their *presence and shape* rather than their contents:
    a detail view exists to tell you what is recoverable, not to dump a return
    series into a terminal.
    """
    lines = [
        "",
        f"=== Trial {trial.get('id')} ({trial.get('kind')}) ===",
        f"  recorded : {trial.get('ts') or NOT_RECORDED}",
        f"  strategy : {trial.get('strategy') or NOT_RECORDED}",
        f"  window   : {(trial.get('window_start') or NOT_RECORDED)[:19]} → "
        f"{(trial.get('window_end') or NOT_RECORDED)[:19]}",
        f"  universe : {trial.get('universe_hash', NOT_RECORDED)[:16]} (hash)",
        f"  accounting v{trial.get('accounting')} | git {trial.get('git_sha') or NOT_RECORDED}",
        "",
        "  Params (the run's identity, including the folded cost and vintage keys):",
    ]
    params = trial.get("params") or {}
    for key in sorted(params):
        lines.append(f"    {key} = {params[key]}")
    if not params:
        lines.append(f"    {NOT_RECORDED}")

    lines.append("")
    lines.append("  Headline metrics:")
    for label, key, spec in (
        ("OOS Sharpe", "oos_sharpe", "{:.3f}"),
        ("Deflated Sharpe", "deflated_sharpe", "{:.3f}"),
        ("Profit factor", "oos_profit_factor", "{:.3f}"),
        ("Max drawdown", "oos_max_dd", "{:.3f}"),
        ("Efficiency", "efficiency", "{:.3f}"),
        ("OOS trades", "oos_trades", "{:.0f}"),
    ):
        lines.append(f"    {label:<18}{_cell(trial.get(key), spec)}")
    promo = trial.get("promotable")
    lines.append(f"    {'Promotable':<18}{NOT_RECORDED if promo is None else ('yes' if promo else 'no')}")

    lines.append("")
    lines.append("  Stored alongside this trial:")
    returns = trial.get("returns")
    lines.append(
        f"    return series : {NOT_RECORDED} (not recorded)"
        if not returns
        else f"    return series : {returns['periods']} periods, "
        f"{str(returns['start'])[:10]} → {str(returns['end'])[:10]}"
    )
    weights = trial.get("weights")
    lines.append(
        f"    proposed book : {NOT_RECORDED} (not recorded)"
        if not weights
        else f"    proposed book : {_plural(len(weights.get('weights') or {}), 'name')}"
        + (", with factor exposures" if weights.get("exposures") else "")
        + (", with active weights" if weights.get("active_weights") else "")
    )
    trades = trial.get("trades")
    lines.append(
        f"    trade table   : {NOT_RECORDED} (not recorded — pass --record-trades on the run)"
        if trades is None
        else f"    trade table   : {_plural(len(trades.get('rows') or []), 'trade')}"
    )

    reused = trial.get("reused_by") or []
    lines.append("")
    if reused:
        lines.append(f"  Reused by {len(reused)} later trial(s) with the same identity:")
        for row in reused:
            lines.append(f"    {row['id']}  {(row.get('ts') or '')[:19]}  ({row.get('kind')})")
    else:
        lines.append("  Not reused by any later trial.")
    lines.append("")
    return "\n".join(lines)


def format_trial_trades(trades: Optional[Dict[str, Any]], limit: int = 25) -> str:
    """The stored trade table, truncated loudly rather than quietly."""
    if not trades:
        return f"  trade table: {NOT_RECORDED} (not recorded)"
    columns = trades.get("columns") or []
    rows = trades.get("rows") or []
    lines = ["", "  Trades:", "    " + "  ".join(f"{str(c)[:12]:>12}" for c in columns)]
    for row in rows[:limit]:
        lines.append("    " + "  ".join(f"{str(v)[:12]:>12}" for v in row))
    if len(rows) > limit:
        lines.append(f"    … {len(rows) - limit} more trade(s) not shown (--trades-limit to raise).")
    return "\n".join(lines)


def format_leaderboard(board: Dict[str, Any]) -> str:
    """The leaderboard, and the multiple-testing context that makes it honest.

    The family's trial count sits on every row and the caveat is always printed —
    a ranking without its selection context is the exact trap this project's
    evaluation machinery exists to close, and it would be worse coming from our own
    tooling, which lends it authority.
    """
    rows = board.get("rows") or []
    if not rows:
        return "No trials matched."
    ranked_by = "deflated Sharpe" if board.get("rank_by") == "dsr" else "RAW Sharpe"
    lines = [
        "",
        # Name the population, not just the sort key. "Top 3 by deflated Sharpe"
        # still reads as "the best of everything I have run" unless it says what it
        # ranked over.
        f"  Top {len(rows)} by {ranked_by}"
        + ("" if board.get("in_sample_included") else " (validated runs only)")
        + ":",
        # KIND is not decoration. Without it a reader cannot tell a validated
        # walk-forward from a search's winning candidate, and those mean opposite
        # things about whether a number is evidence.
        f"    {'#':<3}{'ID':14}{'KIND':12}{'STRATEGY':16}{'DSR':>8}{'SHARPE':>9}{'FAMILY n_trials':>17}",
    ]
    for i, r in enumerate(rows, 1):
        family = r.get("family_n_trials")
        lines.append(
            f"    {i:<3}{str(r.get('id', ''))[:14]:14}{str(r.get('kind') or '')[:12]:12}"
            f"{str(r.get('strategy') or '')[:16]:16}"
            f"{_cell(r.get('deflated_sharpe')):>8}{_cell(r.get('oos_sharpe')):>9}"
            f"{(NOT_RECORDED if family is None else family):>17}"
        )
    lines.append("")
    if board.get("in_sample_included"):
        lines.append(
            "  (!) IN-SAMPLE rows included. An 'optimize' row is the winner of a search — "
            "best-of-N by construction — so its rank measures selection, not skill."
        )
    elif board.get("in_sample_excluded"):
        lines.append(
            f"  {board['in_sample_excluded']} in-sample row(s) excluded (search candidates, "
            "not track records). Pass --include-in-sample to see them."
        )
    lines.append(f"  {board.get('caveat', '')}")
    if board.get("max_family_n_trials", 0) >= 50:
        lines.append(
            f"  This family has tried {board['max_family_n_trials']} configs. At that count "
            "the best raw Sharpe is largely selection — read the deflated column, not the rank."
        )
    lines.append("")
    return "\n".join(lines)
