"""MCP server: a thin adapter exposing QuantRed's deterministic core as tools, so
an agent can do the research and the engine can stay boringly deterministic.

No business logic lives here - every tool calls a function in :mod:`quantred.services`
and logs the call to the audit trail. The server constructs **only** a data
client (never a broker/trading client), so it is structurally incapable of
placing an order. The hard wall is the *absence* of any
order/live/account-mutation tool; it cannot be prompt-injected around because the
capability is not wired in.

Run with ``quantred mcp`` when installed, or ``python main.py mcp`` from a checkout
(needs the ``mcp`` extra either way).
"""

import contextlib
import logging
import math
import sys
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

try:
    # pydantic (hence every install with the 'mcp' extra) needs this TypedDict on
    # Python < 3.12; the module must still import without the extra.
    from typing_extensions import TypedDict
except ImportError:  # pragma: no cover - only without the extra
    from typing import TypedDict

from quantred.analytics.series_comparison import MIN_OVERLAP as SERIES_MIN_OVERLAP
from quantred.services import analysis, campaign, configs, glossary, registry
from quantred.services.audit import audit_log, new_run_id
from quantred.settings import BRAND, MCP_SERVER_NAME
from quantred.utils.timeutils import parse_market_date

logger = logging.getLogger(__name__)

SERVER_NAME = MCP_SERVER_NAME

#: The complete, intended tool surface (used by tests to assert the wall).
EXPOSED_TOOLS = (
    "list_strategies",
    "list_scanners",
    "get_param_ranges",
    "run_scan",
    "run_backtest",
    "run_screen",
    "confirm_screen_point",
    "run_causality_probes",
    "run_optimization",
    "run_walk_forward",
    "validate_draft_strategy_code",
    "validate_draft_scanner_code",
    "run_draft_walk_forward",
    "run_verdict",
    "compute_alphas",
    "combine_alphas",
    "compute_risk",
    "construct_portfolio",
    "compute_information",
    "compute_attribution",
    "compute_horizon",
    "render_report",
    "list_trials",
    "get_trial",
    "analyze_trial",
    "compare_trials",
    "get_campaign_material",
    "best_trials",
    "get_metrics_glossary",
    "summarize_bars",
    "save_config",
    "load_config",
    "list_configs",
)

#: Tools that record a trial. An agent that does not know a call costs a trial will
#: burn a campaign's multiple-testing budget at machine speed, so every one of these
#: says so in its description - asserted by a test, not left to good intentions.
JOURNALING_TOOLS = frozenset(
    {
        "run_backtest",
        "confirm_screen_point",
        "run_optimization",
        "run_walk_forward",
        "run_draft_walk_forward",
        "run_verdict",
    }
)

#: The inverse, and it has to be said just as explicitly. A tool that spends no
#: statistical budget is useless to an agent that assumes every call costs a trial:
#: it will ration the cheap question exactly as hard as the expensive one, which is
#: the behaviour a screen exists to remove.
NON_JOURNALING_NOTE = (
    "Journals NOTHING. This is reconnaissance, not evidence: no trial is recorded, "
    "the campaign's multiple-testing total is unchanged, and nothing here is "
    "promotable or citable as validation. Ask freely — that is the point. To turn one "
    "screened point into recorded evidence, call `confirm_screen_point` with exactly "
    "that point's parameters."
)

#: Evidence-gated features that ship **off**. Where a tool exposes one, its
#: description must name the gate and its current verdict rather than presenting the
#: flag as a neutral option - an agent reads a description as fact and acts on it.
EVIDENCE_GATED = ("conditional", "policy", "posterior")

#: Service parameters this surface deliberately does **not** expose, and why.
#:
#: MCP parity says anything a run can be configured with should be reachable from both
#: surfaces. Evidence gating is the stronger rule, and where the two disagree this is the
#: list that records which won. A feature whose own adoption gate does not clear must not
#: find in the agent surface an easier way to be switched on than the one a human reads a
#: warning before using — an agent acts on a description at machine speed, and being
#: reachable is not the same as being validated.
#:
#: Keyed by service parameter, because that is what the parity test enumerates: a
#: parameter absent from a tool must appear here with a reason, or the test fails. That
#: makes "not wired yet" impossible to confuse with "deliberately withheld".
DEFERRED_PARAMS = {
    "conditional": "evidence-gated: the conditional-risk adoption gate does not clear",
    "conditional_lambda": "evidence-gated: configures conditional risk",
    "conditional_method": "evidence-gated: configures conditional risk",
    "posterior": "evidence-gated: the Bayesian posterior gate does not clear",
    "posterior_ic": "evidence-gated: configures the posterior",
    "posterior_t_eff": "evidence-gated: configures the posterior",
    "posterior_tau": "evidence-gated: configures the posterior",
    "policy": "evidence-gated: the multi-period aim policy gate does not clear",
    # Inert rather than dangerous, and withheld for that reason: the service passes it
    # only when `policy == "aim"`, so exposing it without the gated policy would be a
    # knob that silently reaches nothing — the defect `_refuse_inert_flags` exists to
    # stop on the CLI.
    "trade_rate": "evidence-gated: reaches nothing unless `policy` is 'aim', which is gated",
    "decay_lookback_days": "evidence-gated: only feeds the aim policy's decay estimate",
}

#: Service parameters withheld for a reason that is not an evidence gate, kept apart so
#: ``DEFERRED_PARAMS`` stays exactly the evidence-gated list its own test pins.
#:
#: Keyed by tool, because the answer is per tool: ``run_verdict``'s ``journal`` is the
#: CLI's ``--no-journal``, and it is withheld there because the multiple-testing
#: correction counts attempts - an agent able to run unrecorded verdicts could try as
#: many variants as it liked without the campaign's total ever noticing. (The draft
#: walk-forward does expose its own ``journal``, for smoke tests that must not consume a
#: test; that is a decision about that tool, not a precedent for this one.)
WITHHELD_PARAMS = {
    # Reached by the cost stress, which scales it with the other turnover costs; there is
    # no standalone spread flag on the CLI either, so withholding it is not a miss.
    "run_backtest": {
        "spread_bps": "scaled only by the cost stress; no surface exposes a standalone spread",
    },
    "run_verdict": {
        "journal": "every verdict run over this surface is recorded, so the campaign's "
        "multiple-testing count stays honest",
    },
}

#: Deliberately CLI-only, and not an oversight. Quarantining evidence and retiring an
#: era are operator decisions about a campaign's record, not run configuration: one
#: changes what every later leaderboard and memo reports, the other moves a user's files.
#: Both are cheap to ask a human for and expensive to have done on your behalf by
#: something that cannot be asked why afterwards. An agent that believes a trial is
#: contaminated should say so and let a person act.
#: ``small-real`` is here for a different and stronger reason than the other two, and
#: it is also covered by ``FORBIDDEN_TOOLS`` below: it is the one mode that places real
#: orders. The read-only wall means an agent physically cannot reach it, and this entry
#: records that the absence is a decision. Deciding to spend real money on a book is not
#: a research step an agent takes on somebody's behalf; an agent that thinks execution
#: telemetry is worth gathering should say so and let a person start the run.
OPERATOR_ONLY = ("trials archive", "trials mark-contaminated", "small-real")

#: Capabilities that must NEVER be exposed over MCP (the safety model).
FORBIDDEN_TOOLS = frozenset(
    {
        "place_order",
        "submit_order",
        "submit_market_order",
        "submit_bracket_order",
        "start_live",
        "run_live",
        # Every spelling of the mode that can place real orders. Named rather than left
        # to the general wall, because a list of forbidden capabilities that omits the
        # newest one reads as a list somebody checked.
        "small_real",
        "run_small_real",
        "start_small_real",
        "cancel_order",
        "cancel_all_orders",
        "close_position",
        "close_all_positions",
        "set_paper_trade",
        "set_config",
        "get_account",
        "list_positions",
    }
)


def _parse_date(value: str) -> datetime:
    """Parse an ISO date/datetime string from a tool argument.

    The CLI's own reading (``utils.timeutils.parse_market_date``): an exchange-zone
    instant, so a request means the same window, and the same recorded trial, over
    either surface.
    """
    return parse_market_date(value)


#: Appended to `run_optimization` only: it is the one tool that journals one trial per
#: evaluated config. Every other journaling tool states its own count in its own words,
#: because a shared sentence was false under three of them. An agent that misreads its
#: own experiment history misreads every result after it.
_OPTIMIZATION_JOURNALING_NOTE = (
    "Journals one trial per evaluated config, so this call counts toward the "
    "campaign's multiple-testing total (what the Deflated Sharpe deflates against). "
    "An identical prior run is returned from the trial store labeled `memoized`, with "
    "the original run's timestamp, instead of being re-run; pass `force=true` to "
    "re-verify and append a new trial."
)

#: The glossary's deflated-Sharpe definition is appended to every tool that reports the
#: metric, and it describes the deflated figure. A single backtest or an optimization row
#: is computed for one trial, so the definition would be read as a claim about a number
#: that was never deflated.
_NOT_DEFLATED_NOTE = (
    "Caution on `deflated_sharpe_ratio` in THIS tool's results: it is computed for a single "
    "trial (`deflation_applied: false` where reported), so it is NOT deflated for the number "
    "of configs the campaign has tried, whatever the definition above says. For the "
    "deflated figure use `run_walk_forward`, `list_trials` or `best_trials`."
)

#: `confirm_screen_point` journals exactly what `run_backtest` does: one trial.
_CONFIRM_JOURNALING_NOTE = (
    "Journals ONE trial (the same record `run_backtest` writes), so this call counts "
    "once toward the campaign's multiple-testing total. An identical prior run is "
    "returned from the trial store labeled `memoized: true` with `journaled: false` - "
    "nothing new was recorded - unless `force=true`, which re-runs and appends a new trial."
)

#: Appended where a tool exposes an evidence-gated feature.
_GATED_NOTE = (
    "Evidence-gated features ship OFF and are not neutral options: conditional risk, "
    "the aim-in-front-of-the-target trading policy, and the Bayesian posterior "
    "each have an adoption gate (a predictive-accuracy test or a net-of-cost A/B), and "
    "none of them clears on this repository's own demo data. Enabling one is a "
    "deliberate departure from the validated default, not a tuning choice."
)


#: Returned instead of rows when the store cannot be opened. The store is derived and
#: passive by contract: a broken one degrades a read, it never fails a session.
_NO_STORE = "The trial store could not be opened; no campaign history is available for this call."


@contextlib.contextmanager
def _trial_store():
    """A read-only handle on the trial store, or ``None`` if it cannot be opened.

    Delegates. This was a third copy of the CLI's and the service's opener - three
    places deciding which journal to index, which is the one decision they must never
    disagree about.
    """
    from quantred.services.audit import open_trial_store

    with open_trial_store() as store:
        yield store


#: Said after every refusal of an unknown trial id, so the agent has a next step.
_USE_LIST_TRIALS = "Use list_trials to see which trial ids are recorded."


def _trade_recovery_reason(report: Dict[str, Any]) -> Optional[str]:
    """What an agent can actually do to get a trade table for this kind of trial.

    The shared reason says to re-run "with trades recorded", which only two tools can
    do: `run_backtest` and `run_walk_forward` take `record_trades`. `run_verdict`,
    `run_optimization` and `run_draft_walk_forward` do not, so advice naming the
    parameter sent an agent to a tool that rejects it - and it acts on the advice.
    """
    kind = report.get("kind")
    if kind == "verdict":
        return (
            "a verdict trial carries no trade table - it scores a cross-sectional book, not "
            "a sequence of trades, and `run_verdict` cannot record one. To see trades, run "
            "`run_backtest` with `record_trades=true` over the same window and universe "
            "and analyze that trial instead."
        )
    if kind == "optimize":
        return (
            "an optimization trial carries no trade table - `run_optimization` evaluates many "
            "configs and cannot record any of their trades. To see one config's trades, run "
            "`run_backtest` with `record_trades=true` on its parameters over the same window "
            "and universe, and analyze that trial instead."
        )
    if kind == "walkforward" and str(report.get("strategy") or "").startswith("draft:"):
        return (
            "a draft walk-forward trial carries no trade table - `run_draft_walk_forward` "
            "cannot record one. Run `run_backtest` with `record_trades=true` on an installed "
            "strategy, or install the draft and run `run_walk_forward` with `record_trades=true`."
        )
    return None


def _trade_recovery_note(report: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the shared "re-run with trades recorded" reason with one the trial's own
    kind can act on. Only an unavailable table is touched, and only when the kind has a
    tailored answer; a capped or unrecorded-completeness table keeps its own reason."""
    if report.get("status") != "unavailable":
        return report
    reason = _trade_recovery_reason(report)
    if reason is None:
        return report
    if report.get("kind") != "verdict" and not str(report.get("reason", "")).startswith(
        "no trade table was recorded"
    ):
        return report
    return {**report, "reason": reason}


#: Appended where a tool accepts `workers`. Parallelism is a throughput choice, and
#: an agent should know it changes nothing else about the answer.
_WORKERS_NOTE = (
    "`workers` (default 1) evaluates candidates across that many processes. It changes "
    "wall-clock only: the same seed produces the same trials, the same winner, and the "
    "same campaign trial count as a sequential run, because workers only execute — this "
    "server still does every journal write itself. Memory scales with workers x the "
    "per-worker bar footprint, and parallel runs read from the local bar cache."
)


def _describe(doc: Optional[str], *metric_keys: str, notes: Optional[List[str]] = None) -> str:
    """A tool's description: its docstring, plus the glossary's own words for any
    metric it reports.

    Metric definitions are *pulled* from :mod:`quantred.services.glossary` rather than
    restated here, so a description can never drift from what the metric actually
    means — the same one-source-of-truth rule the layer table applies to logic.
    Descriptions matter more than docs do: an agent cannot notice that a
    description is stale, it can only act on it, and every action costs a trial.
    """
    parts = [(doc or "").strip()]
    if metric_keys:
        definitions = glossary.definitions_for(metric_keys)
        if definitions:
            parts.append(
                "Metric definitions (from the shared glossary — call get_metrics_glossary "
                "for the full set and its pitfalls):\n"
                + "\n".join(f"- {name}: {text}" for name, text in definitions.items())
            )
    parts.extend(notes or [])
    return "\n\n".join(p for p in parts if p)


class StrategyEntry(TypedDict):
    """One row of ``list_strategies``; declared so a client can validate it."""

    name: str
    description: str
    #: None means the class declares no timeframe - absent, not an empty string.
    timeframe: Optional[str]
    #: Shipped demonstration scaffolding, not a validated strategy.
    demo: bool


class ScannerEntry(TypedDict):
    """One row of ``list_scanners``."""

    name: str
    description: str
    demo: bool


def _keys_missing_message() -> str:
    """What a tool that needs bars says when the server has no keys to fetch them with.

    Spelled for the copy that is running, and aimed at a person at a terminal: the agent
    cannot fix this itself, and a client that only shows "Connection closed" (which is
    what a server dying at startup used to look like) says nothing at all.
    """
    from quantred.settings import invocation

    return (
        "This tool needs market data and the server has no broker keys, so it cannot fetch "
        f"bars. Add them from a terminal with `{invocation('init')}` and restart the server, "
        f"or start it with `{invocation('mcp --offline')}` to read only the local bar cache, "
        "which needs no keys. Catalogue and trial-history tools work without keys."
    )


def build_server(data_client=None, *, offline: bool = False, cache_dir=None):
    """Construct the FastMCP server with all read/analyze/propose tools registered.

    ``data_client`` may be injected (offline tests); otherwise a data-only client is
    built **lazily**, on the first tool that needs bars. Building it here made a server
    with no keys exit at startup, which a client reports only as a closed connection - and
    took the catalogue and trial-history tools, which need no market data, down with it.
    ``offline`` builds the keyless, cache-only client instead. Either way the server holds
    only a data client, never a broker. Importing ``mcp`` is deferred to here so the
    package and the rest of the CLI never require the extra.
    """
    from mcp.server.fastmcp import FastMCP

    if data_client is not None:
        _assert_no_trading_client(data_client)
    held: List[Any] = [data_client]

    def _data():
        """The data client, built on first use. A failed build is retried, not cached."""
        if held[0] is None:
            from quantred.services.data import build_data_client
            from quantred.settings import SettingsError

            try:
                client = build_data_client(offline=offline, cache_dir=cache_dir)
            except SettingsError as exc:
                raise RuntimeError(_keys_missing_message()) from exc
            _assert_no_trading_client(client)
            held[0] = client
        return held[0]

    def _worker_cache() -> Dict[str, Any]:
        """The bar cache this server reads, for the tools whose workers build their own
        client. Server configuration, not a tool argument: without it a server started
        with ``--offline`` handed its workers a keyed client over the default cache, so
        every dispatched candidate failed on credentials or read other bars."""
        return {"offline": offline, "cache_dir": cache_dir}

    server = FastMCP(SERVER_NAME)
    # FastMCP takes no version, so a client would display this app as the SDK's own
    # ("quantred 1.30.0"). The low-level server's `version` is what `initialize` reports.
    import quantred

    server._mcp_server.version = quantred.__version__

    def _logged(tool: str, inputs: Dict[str, Any], result: Any) -> Any:
        run_id = result.get("run_id") if isinstance(result, dict) else None
        _SESSION["tool_calls"] += 1
        audit_log(tool, inputs, run_id=run_id or new_run_id(), result_summary=_summary(result))
        return _json_safe(result)

    def tool(*metric_keys: str, notes: Optional[List[str]] = None):
        """Register a tool whose description is *composed*, not hand-written twice.

        The docstring stays next to the code, and the glossary's own definitions for
        the metrics this tool reports are appended mechanically — so the description
        an agent reads cannot drift from what the metric means, and shared caveats
        (journaling, evidence-gated defaults) are stated identically everywhere.
        """

        def register(fn):
            server.tool(description=_describe(fn.__doc__, *metric_keys, notes=notes))(fn)
            return fn

        return register

    # ---------------- Discovery (safe) ---------------- #
    @tool()
    def list_strategies() -> List[StrategyEntry]:
        """List the trading strategies this engine can run. Each row has `name`, a
        one-line `description`, `timeframe` (the bar timeframe it is designed for, or null
        when the class declares none) and `demo` (true for shipped demonstration
        scaffolding, which shows how the pieces fit and is not a validated strategy).

        Start here. A strategy is a `bar -> score` policy: each defines one continuous
        conviction score, and both the discrete BUY/SELL/HOLD signal and the continuous
        alpha forecast are derived from that one score. The `name` values are what every
        other tool's `strategy` argument accepts. Read-only.
        """
        return _logged("list_strategies", {}, registry.list_strategies())

    @tool()
    def list_scanners() -> List[ScannerEntry]:
        """List the universe scanners. Each row has `name`, a one-line `description` of
        what it selects for, and `demo` (true for shipped demonstration scaffolding, not
        a validated scanner).

        A scanner narrows a candidate symbol list to the names worth analyzing on a
        given day; the `name` values are what every other tool's `scanner` argument
        accepts, and `"none"` skips scanning and uses the candidates as given. Choosing
        a universe by looking at outcomes is look-ahead — treat universe selection as a
        research decision, not something to tune. Read-only.
        """
        return _logged("list_scanners", {}, registry.list_scanners())

    @tool()
    def get_param_ranges(kind: str, name: str) -> Dict[str, Any]:
        """The tunable parameters of one strategy or scanner: bounds and defaults.

        `kind` is "strategy" or "scanner". This is the map of what `run_optimization`
        and `run_walk_forward` are allowed to search over, and what a `config` override
        on any other tool may contain — a parameter outside its declared range is
        rejected rather than silently clamped. Fewer parameters searched means fewer
        trials burned and a less deflated Sharpe to clear. Read-only.
        """
        return _logged(
            "get_param_ranges", {"kind": kind, "name": name}, registry.get_param_ranges(kind, name)
        )

    # ---------------- Read / analyze (safe) ---------------- #
    @tool()
    def run_scan(
        scanner: str,
        symbols: List[str],
        config: Optional[Dict[str, Any]] = None,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a universe scanner over `symbols` and return the names it flags.

        Narrows a candidate list to the symbols worth analyzing, with each one's
        scanner signal strength. This is the same universe resolution every other tool
        performs internally — call it when you want to see or reuse the selection,
        not as a prerequisite.

        `as_of` can pin the scan to a historical ISO date/datetime. Omit it only when
        you genuinely want "now"; historical backtests and verdicts resolve scanner
        state at their own window end to avoid mixing today's universe with an older
        evaluation window.

        Scanning is a research decision, not a tuning knob: picking a universe by
        which one produces the best backtest is look-ahead, and it will not survive
        out-of-sample. Read-only, journals nothing.
        """
        inputs = {"scanner": scanner, "symbols": symbols, "config": config, "as_of": as_of}
        return _logged(
            "run_scan",
            inputs,
            analysis.run_scan(_data(), scanner, symbols, config, as_of=_parse_date(as_of) if as_of else None),
        )

    @tool(notes=[NON_JOURNALING_NOTE])
    def run_causality_probes(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        capital: float = 100_000.0,
        gross: bool = False,
        commission_bps: float = 1.0,
        impact_eta: float = 0.3,
        borrow_bps: float = 50.0,
        benchmark: Optional[str] = None,
        position_limits: Optional[Dict[str, Any]] = None,
        scanner: Optional[str] = None,
        scan_as_of: Optional[str] = None,
        sample: int = 3,
    ) -> Dict[str, Any]:
        """Check whether this strategy's decisions could have been made when they were.

        Four probes, all by perturbation: withhold something that only becomes knowable
        after a fill, re-run, and require the decision at that instant to be unchanged.
        `execution_clock` (was every input available strictly before the fill priced),
        `same_bar_ranking` (entry ordering must not consult the bar it transacts on),
        `benchmark_alignment` (the benchmark must not be paired out of step with the
        strategy), and `as_of_scanner` (universe selection must not read past its clock).

        **This is NOT the leakage probe and does not replace it.** That one shifts the
        feed forward to test for *future data*. It cannot detect a one-bar look-ahead at
        all, because a shift moves signal and price together and leaves the relationship
        intact — it cleared a candidate whose every signal was executed one bar early.
        A passing leakage probe says nothing about anything checked here, and vice versa.

        Read `verdict` carefully. `incomplete` means at least one probe had nothing to
        examine — no trades, no contention for slots, no benchmark — and that is not a
        pass: a run that never traded has not been cleared by a probe about trading.
        `not_exercised` names which ones. Each probe's `passed` is true, false, or null
        for the same reason.
        """
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "benchmark": benchmark,
            "position_limits": position_limits,
            "scanner": scanner,
            "sample": sample,
        }
        result = analysis.run_causality_probes(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            capital=capital,
            gross=gross,
            commission_bps=commission_bps,
            impact_eta=impact_eta,
            borrow_bps=borrow_bps,
            benchmark=benchmark,
            position_limits=position_limits,
            scanner=scanner,
            scan_as_of=_parse_date(scan_as_of) if scan_as_of else None,
            sample=sample,
        )
        return _logged("run_causality_probes", inputs, result)

    @tool("sharpe_ratio", "sortino_ratio", "profit_factor", notes=[NON_JOURNALING_NOTE])
    def run_screen(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        method: str = "grid",
        objective: str = "sharpe_ratio",
        max_evals: int = 50,
        seed: int = 42,
        capital: float = 100_000.0,
        gross: bool = False,
        commission_bps: float = 1.0,
        impact_eta: float = 0.3,
        borrow_bps: float = 50.0,
        position_limits: Optional[Dict[str, Any]] = None,
        param_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
        workers: int = 1,
    ) -> Dict[str, Any]:
        """Sweep a parameter space cheaply to ask whether a family holds anything at all.

        The reconnaissance step *before* optimization. One data fetch, N evaluations,
        and nothing recorded — so "does any parameterisation of this idea survive?" is
        a question you can afford to ask. `workers` (default 1) spreads the evaluations
        across that many processes, which read the local bar cache; it changes
        wall-clock only, and the report is the one a sequential screen gives.

        Read the `distribution` block, not `best_point`. `best_point` is the winning
        row flattened with its metrics; its `params` field is just the parameter values,
        and is what `confirm_screen_point` takes. The best of N points is the
        maximum of N draws, which is a positive number even when nothing searched has
        any edge, and it grows with N. `noise_baseline.expected_best_under_null` is
        what the best of that many draws is worth if none of them had any edge: a
        `best_point` below it is indistinguishable from noise no matter how good the
        number looks. That baseline assumes the points are independent, which
        neighbouring grid points are not, so it sits high; it is a reference for
        reading the table, not a test. It is omitted, with a reason, for any objective
        whose null is not zero.

        `gradients` is the other half. A positive rate that moves coherently across a
        parameter is different evidence from the same count of positive points
        scattered at random — and it can point off the edge of the range you searched,
        which a best-point report structurally cannot show.

        `param_ranges` narrows the search per parameter, e.g.
        {"lookback": {"min": 10, "max": 30, "step": 5}}; each supplies any of
        min/max/step and inherits the rest. It can only narrow: a bound outside the
        strategy's declared range is refused. `position_limits` is the book each point is
        evaluated against (e.g. {"max_positions": 8}); without it every point runs at
        whatever the strategy class declares, which is not what would be deployed.
        Parameter combinations a strategy declares invalid are never drawn.
        """
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "method": method,
            "objective": objective,
            "max_evals": max_evals,
            "seed": seed,
            "gross": gross,
            "position_limits": position_limits,
            "param_ranges": param_ranges,
        }
        result = analysis.run_screen(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            method=method,
            objective=objective,
            max_evals=max_evals,
            seed=seed,
            capital=capital,
            gross=gross,
            commission_bps=commission_bps,
            impact_eta=impact_eta,
            borrow_bps=borrow_bps,
            position_limits=position_limits,
            param_ranges=param_ranges,
            workers=workers,
            **_worker_cache(),
        )
        return _logged("run_screen", inputs, result)

    @tool("sharpe_ratio", "deflated_sharpe_ratio", notes=[_CONFIRM_JOURNALING_NOTE, _NOT_DEFLATED_NOTE])
    def confirm_screen_point(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        params: Dict[str, Any],
        capital: float = 100_000.0,
        gross: bool = False,
        commission_bps: float = 1.0,
        impact_eta: float = 0.3,
        borrow_bps: float = 50.0,
        position_limits: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Re-run EXACTLY ONE screened point as a proper, journaled trial.

        `params` is one parameter set — one point: only parameters the strategy declares
        (see `get_param_ranges`), such as a screen's `best_point.params`. Any other key,
        including the metrics beside `best_point`'s parameters, is refused rather than
        journaled into the trial's config. This is the only way a screen's
        finding becomes evidence, and it is deliberately singular: confirming a set
        would make the screen itself journal, which is the statistical-budget problem
        the screen exists to avoid. It would also record the best of N, which is the
        one selection a sweep cannot support.

        Identical to running `run_backtest` with those parameters when the extra
        cost and book knobs here (`commission_bps`, `impact_eta`, `borrow_bps`,
        `position_limits`) are left at their defaults — same dedup identity, same
        memoization, same journal record — so a confirmed point is indistinguishable
        from a backtest anyone else ran, and counts once. Set one and the run is a
        different config from a plain backtest of those parameters.
        """
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "params": params,
            "gross": gross,
            "position_limits": position_limits,
            "force": force,
        }
        result = analysis.confirm_screen_point(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            params,
            capital=capital,
            gross=gross,
            commission_bps=commission_bps,
            impact_eta=impact_eta,
            borrow_bps=borrow_bps,
            position_limits=position_limits,
            force=force,
        )
        return _logged("confirm_screen_point", inputs, result)

    @tool(
        "sharpe_ratio",
        "deflated_sharpe_ratio",
        "max_drawdown",
        "profit_factor",
        "low_sample",
        notes=[_NOT_DEFLATED_NOTE],
    )
    def run_backtest(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        capital: float = 100_000.0,
        config: Optional[Dict[str, Any]] = None,
        beta_sizing: bool = False,
        benchmark: str = "SPY",
        gross: bool = False,
        take_profit_margin_bps: float = 0.0,
        force: bool = False,
        record_trades: bool = False,
    ) -> Dict[str, Any]:
        """Backtest `strategy` on `symbols` over [start, end] (YYYY-MM-DD).

        Returns the full metrics dict (NET of transaction cost by default — commission
        + half-spread + square-root impact; pass gross=True to disable), the trade
        count, total cost, and a path to the trades CSV (trades are not inlined).
        `config` overrides default params.

        `take_profit_margin_bps` requires the price to trade that far *through* a
        take-profit before it counts as filled. Zero — the default — fills a target the
        moment a bar touches it, which models a resting limit always first in the queue.
        For a strategy whose gain concentrates in target exits that assumption is the
        result rather than a detail, so raising it is how you find out which you have.

        The result carries `excursion`: how bad the book got *inside* a bar against how
        bad its closing marks ever showed. The equity curve is marked at each bar's
        close, so a shallow `max_drawdown` can sit over a period the book spent in more
        trouble — and per-trade MAE cannot tell you, because a position deep underwater
        may be a small part of the book. Read both bounds and quote the pair: the
        closing mark is a lower bound on the pain, and the intra-bar figure marks every
        position at its own worst tick and so assumes they all got there at once, which
        makes it an upper bound and never a measurement. The realized worst lies
        between them. Nothing is gated on either.

        Journals this as a trial (the same research journal and trial store the
        `backtest` command uses) so it counts toward the campaign's multiple-testing
        total. An exact prior trial is served instead (result has `memoized: true`)
        unless `force=True`, which re-runs and appends a new trial rather than
        overwriting the memoized one. A memoized result says `journaled: false`: nothing
        new was recorded, and `trial_id` is the original's. Memoization is scoped to the engine's accounting
        version, so results from an older engine are never served to a newer one.

        `record_trades=true` stores the full trade table with the journaled trial, which
        `analyze_trial` reads; without it the trial keeps no table and it reports it as
        not recorded. `compare_trials` does not read that table: it reads the dated
        return series every journaled backtest stores either way. A prior identical trial that kept no table
        does not answer a `record_trades=true` request: the backtest is re-run and a new
        trial (with the table) is journaled, counting toward the multiple-testing total.

        A memoized result is NOT the full result: the trial store keeps the headline
        metrics, not total cost, execution checks, excursion, the trades CSV or the
        resolved config. It carries `not_recorded` naming every field a fresh run has
        that this one lacks (absent, not zero), `from_request` for values filled from
        your request, and `derived` for values computed from stored ones. Pass
        `force=True` to recompute the missing fields.
        """
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "capital": capital,
            "config": config,
            "beta_sizing": beta_sizing,
            "gross": gross,
            "take_profit_margin_bps": take_profit_margin_bps,
            "force": force,
            **({"record_trades": True} if record_trades else {}),
        }
        result = analysis.run_backtest(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            capital,
            config,
            beta_sizing,
            benchmark,
            gross=gross,
            take_profit_margin_bps=take_profit_margin_bps,
            force=force,
            **({"record_trades": True} if record_trades else {}),
        )
        return _logged("run_backtest", inputs, result)

    @tool(
        "sharpe_ratio",
        "deflated_sharpe_ratio",
        notes=[_OPTIMIZATION_JOURNALING_NOTE, _NOT_DEFLATED_NOTE, _WORKERS_NOTE],
    )
    def run_optimization(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        method: str = "grid",
        objective: str = "sharpe_ratio",
        max_evals: int = 50,
        seed: int = 42,
        capital: float = 100_000.0,
        gross: bool = False,
        force: bool = False,
        workers: int = 1,
    ) -> Dict[str, Any]:
        """Search a strategy's parameters IN-SAMPLE (grid|random|bayesian).

        Returns best_params, best_score, and the top-10 configs (full grid -> CSV).
        NET of transaction cost by default (commission + half-spread + square-root
        impact; pass gross=True to disable) — gross search reliably favors the
        highest-turnover config. IMPORTANT: in-sample results from picking the
        best of many configs are inflated and are NOT evidence of edge. Always
        validate with run_walk_forward before trusting them.

        Each evaluated config is journaled as its own trial (same journal/trial
        store the CLI uses). A candidate identical to one already scored this
        campaign is served from the trial store instead of re-run (`n_memoized`
        in the result) unless `force=True`.
        """
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "method": method,
            "objective": objective,
            "max_evals": max_evals,
            "seed": seed,
            "gross": gross,
            "force": force,
        }
        result = analysis.run_optimization(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            method,
            objective,
            max_evals,
            seed,
            capital,
            gross=gross,
            force=force,
            workers=workers,
            **_worker_cache(),
        )
        return _logged("run_optimization", inputs, result)

    @tool(
        "sharpe_ratio",
        "deflated_sharpe_ratio",
        "profit_factor",
        "max_drawdown",
        notes=[_WORKERS_NOTE],
    )
    def run_walk_forward(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        mode: str = "anchored",
        n_folds: Optional[int] = None,
        embargo_days: Optional[int] = None,
        holdout_days: int = 0,
        method: str = "grid",
        objective: str = "sharpe_ratio",
        max_evals: int = 50,
        seed: int = 42,
        capital: float = 100_000.0,
        include_pbo: bool = False,
        parameter_sensitivity: bool = False,
        leakage_probe: bool = False,
        gross: bool = False,
        force: bool = False,
        workers: int = 1,
        benchmark: Optional[str] = None,
        position_limits: Optional[Dict[str, Any]] = None,
        record_trades: bool = False,
        cost_stress: Optional[Literal["all", "turnover", "borrow"]] = "all",
    ) -> Dict[str, Any]:
        """Honest out-of-sample evaluation across folds - your advancement criterion.

        Optimizes in-sample per fold, scores out-of-sample, and returns the OOS
        aggregate, walk-forward efficiency, IS->OOS degradation, per-fold summary,
        the holdout score (if holdout_days>0), the Deflated Sharpe (corrected for
        all configs tried), and the PROMOTION-GATE verdict (pass/fail per gate +
        overall "promotable"). A config advances only if it is promotable - never
        on in-sample Sharpe. include_pbo adds a heuristic IS-best fold OOS rank-failure rate (not a true
        probability of backtest overfitting) and is expensive; leave it off unless needed.
        NET of transaction cost by default, in-sample and out (pass gross=True to
        disable) — gross validation systematically promotes turnover the
        strategy could not afford live.

        `promotable` is the statistical gates ONLY and does not mean clear for paper.
        The `evidence` block beside it is what the CLI prints as "Promotion
        prerequisites" and "Evidence completeness": cost stress, family bootstrap
        (not run over this tool, so it stays unevaluated), and benchmark-relative
        information ratio and excess return, each pass/fail/unevaluated, plus excess
        return per fold and `evidence.clear`. Read `evidence.evidence_completeness`
        before acting on `promotable`; an unevaluated check is unknown, not passed.
        `holdout` is the holdout's metrics on a fresh run (null when none was requested); a
        memoized answer carries `holdout_record` instead ({state, days, metrics, params}:
        recorded, none requested, or requested before the journal kept it).
        When `evidence` is null, `evidence_unavailable` says why (a memoized answer keeps no
        per-fold detail). An unknown `cost_stress` is refused rather than stressing nothing.
        `cost_stress` ("all"/"turnover"/"borrow", null to skip) re-runs the chosen
        config at worse costs and at the same `position_limits` book, unjournaled, and is on by default as in the CLI.

        `benchmark` is a symbol to measure against (e.g. "SPY"). Without it the
        benchmark-relative prerequisites are left unevaluated rather than failed.
        It is scored per fold and aggregated as a median: `oos_aggregate`
        (concatenated trades) deliberately carries no benchmark figures, so its
        `benchmark_available: false` is not a missing benchmark. `position_limits` is the book the config says it will trade
        (e.g. {"max_positions": 8}); without it the validation runs at whatever the
        strategy class declares, which is not what would be deployed.

        Journals the OOS aggregate as one validated trial. An identical prior
        validation (same recipe: mode/folds/method/objective/max_evals/seed/cost/
        book over the same window) is served instead (result has `memoized: true` and
        `journaled: false`: nothing new was recorded) unless `force=True`. The trial
        counts toward the campaign's multiple-testing total once.

        `record_trades=true` stores the out-of-sample trade table with the validated
        trial, which `analyze_trial` reads; without it the trial keeps no table.
        `compare_trials` reads the stitched out-of-sample return series instead, which
        every journaled validation stores either way. A prior identical validation that kept no table does not
        answer a `record_trades=true` request: the validation is re-run and a new trial
        (with the table) is journaled, adding every configuration its inner search
        evaluates to the family's multiple-testing total - as large a cost as the first
        validation. Ask for the table on the first run you mean to analyze.
        """
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "mode": mode,
            "n_folds": n_folds,
            "embargo_days": embargo_days,
            "holdout_days": holdout_days,
            "method": method,
            "objective": objective,
            "max_evals": max_evals,
            "seed": seed,
            "include_pbo": include_pbo,
            "gross": gross,
            "force": force,
            "benchmark": benchmark,
            "position_limits": position_limits,
            **({"record_trades": True} if record_trades else {}),
            **({"cost_stress": cost_stress} if cost_stress != "all" else {}),
        }
        result = analysis.run_walk_forward(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            mode=mode,
            n_folds=n_folds,
            embargo_days=embargo_days,
            holdout_days=holdout_days,
            method=method,
            objective=objective,
            max_evals=max_evals,
            seed=seed,
            capital=capital,
            include_pbo=include_pbo,
            parameter_sensitivity=parameter_sensitivity,
            leakage_probe=leakage_probe,
            gross=gross,
            force=force,
            workers=workers,
            benchmark=benchmark,
            position_limits=position_limits,
            **_worker_cache(),
            **({"record_trades": True} if record_trades else {}),
            **({"cost_stress": cost_stress} if cost_stress != "all" else {}),
        )
        return _logged("run_walk_forward", inputs, result)

    @tool()
    def validate_draft_strategy_code(code: str, class_name: Optional[str] = None) -> Dict[str, Any]:
        """Validate draft Strategy source without registering or running it.

        Use this for agent-authored or private-package-bound strategy code before
        spending market-data calls on it. The code is compiled in the research
        sandbox with restricted imports, then checked for the Strategy contract:
        concrete subclass, docstring, valid PARAM_RANGES with no more than five
        searchable params, default construction, required sizing/risk config, and
        one continuous score source. It writes no files and journals no trials.

        Always answers, never raises: a rejection comes back as `valid: false` with
        `error_kind` either `invalid_draft` (rewrite the source - `error` names what
        to fix) or `validator_error` (the validator itself failed on this input; the
        draft may be fine, so rewriting it is the wrong response).

        This is a drafting aid, not promotion. To make validated code available by
        name, ship it in a package exposing the `quantred.strategies` entry-point
        group, or keep using `run_draft_walk_forward` with source attached.
        """
        inputs = {"class_name": class_name, "code_hash": analysis.draft_code_hash(code)}
        return _logged(
            "validate_draft_strategy_code",
            inputs,
            analysis.validate_draft_strategy_code(code, class_name=class_name),
        )

    @tool()
    def validate_draft_scanner_code(code: str, class_name: Optional[str] = None) -> Dict[str, Any]:
        """Validate draft ScannerStrategy source without registering or running it.

        Use this for agent-authored or private-package-bound scanner code before it
        becomes part of a universe-selection workflow. The code is compiled in the
        research sandbox with restricted imports, then checked for the scanner
        contract: concrete subclass, docstring, valid PARAM_RANGES with no more than
        five searchable params, default construction, and a generated signal frame
        containing `signal` plus numeric `signal_strength` using the scanner signal
        vocabulary. It writes no files and journals no trials. It answers with the
        same verdict shape as `validate_draft_strategy_code`, including on rejection.

        To make validated scanner code available by name, ship it in a package
        exposing the `quantred.scanners` entry-point group.
        """
        inputs = {"class_name": class_name, "code_hash": analysis.draft_code_hash(code)}
        return _logged(
            "validate_draft_scanner_code",
            inputs,
            analysis.validate_draft_scanner_code(code, class_name=class_name),
        )

    @tool("sharpe_ratio", "deflated_sharpe_ratio", "profit_factor", "max_drawdown")
    def run_draft_walk_forward(
        code: str,
        symbols: List[str],
        start: str,
        end: str,
        class_name: Optional[str] = None,
        mode: str = "anchored",
        n_folds: Optional[int] = None,
        embargo_days: Optional[int] = None,
        holdout_days: int = 0,
        method: str = "grid",
        objective: str = "sharpe_ratio",
        max_evals: int = 50,
        seed: int = 42,
        capital: float = 100_000.0,
        include_pbo: bool = False,
        include_monte_carlo: bool = False,
        parameter_sensitivity: bool = False,
        leakage_probe: bool = False,
        gross: bool = False,
        journal: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Validate draft Strategy source, then run normal walk-forward gates.

        This is the bridge from "agent can propose or modify strategy code" to
        "QuantRed can validate it" without putting proprietary code in this repo.
        The source is compiled in-memory through the same sandbox as
        `validate_draft_strategy_code`, never registered globally, and never made
        live. Results are regular walk-forward results with a `draft` block carrying
        class name, source hash, and whether the run was journaled.

        Source that does not validate comes back as the same `valid: false` verdict
        the two validators return, rather than as an error: nothing ran and no trial
        was spent, so a rejected draft costs nothing against the campaign's budget.
        Validating first with `validate_draft_strategy_code` is still cheaper.

        By default this journals one validated trial under
        `draft:<ClassName>:<code_hash>` so campaign history still reflects that the
        source consumed a test. Set `journal=false` only for smoke tests that should
        not enter the campaign record (`draft.journaled` is then false). One
        journaled run counts once toward the campaign's multiple-testing total. An
        identical prior draft validation is served from the trial store, labeled
        `memoized: true` with `draft.journaled: false` (nothing new was recorded),
        unless `force=true`.
        """
        inputs = {
            "code_hash": analysis.draft_code_hash(code),
            "symbols": symbols,
            "start": start,
            "end": end,
            "class_name": class_name,
            "mode": mode,
            "n_folds": n_folds,
            "holdout_days": holdout_days,
            "method": method,
            "objective": objective,
            "max_evals": max_evals,
            "seed": seed,
            "gross": gross,
            "journal": journal,
            "force": force,
        }
        result = analysis.run_draft_walk_forward(
            _data(),
            code,
            symbols,
            _parse_date(start),
            _parse_date(end),
            class_name=class_name,
            mode=mode,
            n_folds=n_folds,
            embargo_days=embargo_days,
            holdout_days=holdout_days,
            method=method,
            objective=objective,
            max_evals=max_evals,
            seed=seed,
            capital=capital,
            include_pbo=include_pbo,
            include_monte_carlo=include_monte_carlo,
            parameter_sensitivity=parameter_sensitivity,
            leakage_probe=leakage_probe,
            gross=gross,
            journal=journal,
            force=force,
        )
        return _logged("run_draft_walk_forward", inputs, result)

    @tool()
    def compute_alphas(
        strategy: str,
        symbols: List[str],
        as_of: str,
        source: str = "strategy",
        scanner: str = "demo_volume",
        ic: float = 0.03,
        benchmark: str = "SPY",
        neutralize: bool = False,
        lookback_days: int = 180,
        scaling: str = "case1",
        neutralize_factors: Optional[List[str]] = None,
        price_derived: Optional[bool] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rank `symbols` by continuous alpha (residual-return forecast) as of a date.

        Turns each name's view into a comparable, annualized residual-return
        forecast. `scaling` picks the per-name scaling: "case1" =
        sigma*IC*z (the default; correct when per-name signal vol is constant across
        names), "case2" = IC*c_g*z (no per-name vol multiply, for signals whose vol is
        proportional to the name's vol — most price signals), or "auto" to let a
        Std_TS-vs-omega regression decide (echoed under `case`). source: "strategy"
        uses the strategy's continuous conviction; "signal" uses its BUY/SELL/HOLD as
        +1/-1/0; "scanner" uses the scanner's continuous strength. Read-only. The
        absolute scale is only as good as the assumed IC; relative ranking is not.

        `neutralize_factors` names risk-model factors to regress out of the alpha, and
        the factors actually removed come back in `neutralized_against`. Without it that
        field is empty — which is what this surface reported for every call before the
        parameter existed, so a caller could ask for neutralization and be told, in a
        field it echoed back, that none had happened.

        `scaling` is one of case1/case2/auto; anything else is refused. On a thin
        universe (fewer than ten scored names) neutralization and the case choice are
        both skipped - `neutralized_against` stays empty and case2 gives case1's alphas -
        and `low_confidence_reason` says which of your requests were skipped.
        """
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "as_of": as_of,
            "source": source,
            "scanner": scanner,
            "ic": ic,
            "benchmark": benchmark,
            "neutralize": neutralize,
            "scaling": scaling,
        }
        result = analysis.compute_alphas(
            _data(),
            strategy,
            symbols,
            _parse_date(as_of),
            source=source,
            scanner=scanner,
            ic=ic,
            benchmark=benchmark,
            neutralize=neutralize,
            lookback_days=lookback_days,
            scaling=scaling,
            **{
                k: v
                for k, v in (
                    ("neutralize_factors", neutralize_factors),
                    ("price_derived", price_derived),
                    ("timeframe", timeframe),
                )
                if v is not None
            },
        )
        return _logged("compute_alphas", inputs, result)

    @tool()
    def combine_alphas(
        signals: List[str],
        symbols: List[str],
        as_of: str,
        benchmark: str = "SPY",
        neutralize: bool = False,
        lookback_days: int = 365,
        horizon: int = 5,
        n_points: int = 12,
    ) -> Dict[str, Any]:
        """Combine several strategies' signals into one alpha by IC + correlation.

        `signals` is a list of strategy names. Measures each signal's IC and the
        signal correlation matrix over a trailing window of realized residual returns,
        shrinks the ICs by estimation confidence, and combines them with GLS weights
        (Ω⁻¹·IC) so redundant signals split a weight instead of double-counting.
        Returns the ranked combined alphas plus measured ICs, shrunk ICs, weights, and
        the correlation matrix. Read-only; measure on out-of-sample data for honesty.
        """
        inputs = {"signals": signals, "symbols": symbols, "as_of": as_of, "neutralize": neutralize}
        result = analysis.compute_combined_alphas(
            _data(),
            signals,
            symbols,
            _parse_date(as_of),
            benchmark=benchmark,
            neutralize=neutralize,
            lookback_days=lookback_days,
            horizon=horizon,
            n_points=n_points,
        )
        return _logged("combine_alphas", inputs, result)

    @tool()
    def compute_horizon(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        source: str = "strategy",
        benchmark: str = "SPY",
        max_lag: int = 10,
    ) -> Dict[str, Any]:
        """Measure an alpha's decay/half-life and recommend cadence + lagged blend.

        Measures the IC-vs-lag profile (alpha at t vs residual return n periods later),
        fits the per-period decay δ and half-life, derives the rebalance cadence that
        maximizes IC·√(1/Δt), and computes the IR-maximizing current/lagged blend from
        δ and the signal autocorrelation (diversify if δ>ρ, hedge if δ<ρ). The
        half-life is the holding period to amortize cost over. Read-only.

        The report carries no recommendation to switch trading policy: the policy that
        would replace the lagged blend is evidence-gated and not offered on this surface.
        """
        inputs = {"strategy": strategy, "symbols": symbols, "start": start, "end": end}
        result = analysis.compute_horizon(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            source=source,
            benchmark=benchmark,
            max_lag=max_lag,
        )
        # The CLI's report points at its `--policy aim` flag, which this server does not
        # expose (evidence-gated) and which is no tuning suggestion for an agent to act
        # on, so the pointer is left off this payload rather than reworded.
        result = {k: v for k, v in result.items() if k != "blend_superseded_by"}
        return _logged("compute_horizon", inputs, result)

    @tool(notes=[_GATED_NOTE])
    def compute_risk(
        symbols: List[str],
        as_of: str,
        model: str = "shrinkage",
        timeframe: str = "1Day",
        lookback_days: int = 365,
    ) -> Dict[str, Any]:
        """Estimate the universe's covariance Σ and summarize its risk structure.

        Returns the shrinkage intensity δ, condition number, mean correlation, the
        equal-weight portfolio volatility, and the top risk contributors as of a date.
        model: "shrinkage" = sample covariance shrunk toward constant correlation (default), "sample" = raw, "factor" =
        structural XFXᵀ+Δ (adds the factor-vs-specific risk split). Read-only: Σ sizes
        conviction, it never places an order. Risk is not additive — correlated names
        are one bet, which is what this matrix captures.
        """
        inputs = {
            "symbols": symbols,
            "as_of": as_of,
            "model": model,
            "timeframe": timeframe,
            "lookback_days": lookback_days,
        }
        result = analysis.compute_risk(
            _data(),
            symbols,
            _parse_date(as_of),
            model=model,
            timeframe=timeframe,
            lookback_days=lookback_days,
        )
        return _logged("compute_risk", inputs, result)

    @tool("information_ratio", "turnover", notes=[_GATED_NOTE])
    def construct_portfolio(
        strategy: str,
        symbols: List[str],
        as_of: str,
        source: str = "strategy",
        target_te: float = 0.04,
        max_weight: float = 0.25,
        max_names: Optional[int] = None,
        min_weight: Optional[float] = None,
        benchmark: str = "SPY",
        capital: Optional[float] = None,
        scanner: Optional[str] = None,
        neutralize: Optional[bool] = None,
        neutralize_factors: Optional[List[str]] = None,
        book: Optional[str] = None,
        gross_leverage: Optional[float] = None,
        short_max_weight: Optional[float] = None,
        benchmark_holdings: Optional[Dict[str, float]] = None,
        benchmark_premium: Optional[float] = None,
        current_weights: Optional[Dict[str, float]] = None,
        holding_period_years: Optional[float] = None,
        lookback_days: Optional[int] = None,
        timeframe: Optional[str] = None,
        risk_model: Optional[str] = None,
        cost_aware: Optional[bool] = None,
        commission_bps: Optional[float] = None,
        impact_eta: Optional[float] = None,
        participation_cap: Optional[float] = None,
        borrow_bps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Construct the mean-variance optimal portfolio from alphas and Σ.

        Maximizes αᵀw − λ·wᵀΣw over long-only, box-bounded, budgeted (optionally
        cardinality-capped) weights, calibrating λ to `target_te`. Cost-aware by
        default: the objective carries name-specific turnover and square-root impact,
        so a no-trade band emerges from the cost itself rather than being imposed.

        Returns the proposed weights, the active weights and factor exposures of the
        resulting book, and the Fundamental-Law report: IR* (the best achievable
        information ratio), predicted tracking error and IR, the transfer coefficient
        (how much of IR* survives the constraints), turnover, expected active return
        gross and net of cost, and the capacity at which impact erases the alpha.

        A PROPOSAL, never an order. This is research-clock only: it cannot trade, and
        a human promotes any config that comes out of it.

        `book="market_neutral"` builds a long/short book and REQUIRES `gross_leverage` (the
        cap is never defaulted); `short_max_weight` sizes the short side.
        `gross_leverage` with the default long-only book is refused rather than ignored; `benchmark_holdings` (plus `benchmark_premium`) makes the solve
        benchmark-relative; `neutralize_factors` regresses named risk-model factors out
        of the alpha first; `current_weights` makes turnover measured from a real book
        rather than from cash. The cost knobs set the assumptions the objective prices
        turnover with.

        Scope note, so you do not assume more than this exposes: the evidence-gated
        features — conditional risk, the Bayesian posterior, and the multi-period
        aim policy — are **not** arguments here and will not become so while their own
        adoption gates do not clear. That is deliberate, not an omission: being reachable
        from an agent is not the same as being validated, and this surface must not be
        the easier way to switch on a feature the command line warns about.
        """
        # Only what the caller actually set is forwarded, so an omitted knob keeps the
        # service's own default rather than this surface restating it — a second copy of
        # a default is a second thing to keep in step.
        optional = {
            "min_weight": min_weight,
            "scanner": scanner,
            "neutralize": neutralize,
            "neutralize_factors": neutralize_factors,
            "book": book,
            "gross_leverage": gross_leverage,
            "short_max_weight": short_max_weight,
            "benchmark_holdings": benchmark_holdings,
            "benchmark_premium": benchmark_premium,
            "current_weights": current_weights,
            "holding_period_years": holding_period_years,
            "lookback_days": lookback_days,
            "timeframe": timeframe,
            "risk_model": risk_model,
            "cost_aware": cost_aware,
            "commission_bps": commission_bps,
            "impact_eta": impact_eta,
            "participation_cap": participation_cap,
            "borrow_bps": borrow_bps,
        }
        # Every knob the caller set, not just the four this used to record. The audit
        # log exists so a human can replay what an agent did, and the tool grew from
        # nine parameters to twenty-seven while the record stayed at four — a
        # market-neutral, leveraged proposal audited identically to a default one.
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "as_of": as_of,
            "source": source,
            "target_te": target_te,
            "max_weight": max_weight,
            "max_names": max_names,
            "benchmark": benchmark,
            "capital": capital,
            **{k: v for k, v in optional.items() if v is not None},
        }
        result = analysis.construct_portfolio(
            _data(),
            strategy,
            symbols,
            _parse_date(as_of),
            source=source,
            target_te=target_te,
            max_weight=max_weight,
            max_names=max_names,
            benchmark=benchmark,
            capital=capital,
            **{k: v for k, v in optional.items() if v is not None},
        )
        return _logged("construct_portfolio", inputs, result)

    @tool("information_ratio", "deflated_sharpe_ratio")
    def compute_information(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        source: str = "strategy",
        benchmark: str = "SPY",
        horizon: int = 5,
        n_trials: int = 1,
    ) -> Dict[str, Any]:
        """Measure a strategy's IC, breadth, and predicted-vs-realized IR.

        Pairs the alpha known at each rebalance with the subsequent realized residual
        return (strict forward alignment) to measure the information coefficient
        (Pearson + rank) and its t-stat, the effective breadth (deflated by ρ̄), and the
        predicted vs realized information ratio — with guardrails (IR standard-error
        band, multiple-testing inflation for `n_trials`, sanity ceiling). An IC t-stat
        below 2 means the skill is not distinguishable from luck. Read-only.
        """
        inputs = {"strategy": strategy, "symbols": symbols, "start": start, "end": end, "n_trials": n_trials}
        result = analysis.compute_information(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            source=source,
            benchmark=benchmark,
            horizon=horizon,
            n_trials=n_trials,
        )
        return _logged("compute_information", inputs, result)

    @tool("information_ratio")
    def compute_attribution(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        source: str = "strategy",
        scanner: str = "demo_volume",
        benchmark: str = "SPY",
        neutralize_factors: Optional[List[str]] = None,
        signals: Optional[List[str]] = None,
        horizon: int = 5,
        n_points: int = 24,
        n_trials: int = 1,
        timeframe: Optional[str] = None,
        benchmark_holdings: Optional[Dict[str, float]] = None,
        benchmark_premium: Optional[float] = None,
        detail: Optional[bool] = None,
        min_obs: Optional[int] = None,
        risk_model: Optional[str] = None,
        bootstrap_skill: Optional[bool] = None,
        bootstrap_b: Optional[int] = None,
        bootstrap_block_length: Optional[float] = None,
        bootstrap_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Attribute realized active return to timing, risk factors, signals and picking.

        Splits each sampled rebalance's active return by an exact regression identity
        into the systematic benchmark-timing bucket, each risk factor
        (market/momentum/volatility/size), the strategy's own alpha as a signal column
        (plus any additional `signals`), and a stock-picking remainder — then confronts
        every attributed t-stat with the same research-integrity guardrails
        `compute_information` applies to ICs, so a bucket that looks significant on a
        short sample is not read as skill.

        Answers the question `compute_information` cannot: not "does the signal rank
        names" but "where did the realized return actually come from". A book whose
        return is mostly the factor bucket is a factor bet wearing a strategy's name.

        Read-only, research-clock, leakage-safe by construction: each cross-section is
        rebuilt from bars strictly at or before its own instant. Journals nothing.

        `n_trials` deflates for the search that produced this candidate — pass the
        number of configurations tried, not 1, or the t-stats flatter themselves.
        """
        optional = {
            "neutralize_factors": neutralize_factors,
            "signals": signals,
            "timeframe": timeframe,
            "benchmark_holdings": benchmark_holdings,
            "benchmark_premium": benchmark_premium,
            "detail": detail,
            "min_obs": min_obs,
            "risk_model": risk_model,
            "bootstrap_skill": bootstrap_skill,
            "bootstrap_b": bootstrap_b,
            "bootstrap_block_length": bootstrap_block_length,
            "bootstrap_seed": bootstrap_seed,
        }
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "source": source,
            "scanner": scanner,
            "benchmark": benchmark,
            "horizon": horizon,
            "n_points": n_points,
            "n_trials": n_trials,
            **{k: v for k, v in optional.items() if v is not None},
        }
        result = analysis.compute_attribution(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            source=source,
            scanner=scanner,
            benchmark=benchmark,
            horizon=horizon,
            n_points=n_points,
            n_trials=n_trials,
            **{k: v for k, v in optional.items() if v is not None},
        )
        return _logged("compute_attribution", inputs, result)

    # Not the shared journaling note: "one trial per evaluated config" is wrong here,
    # where an incomplete run can journal none. The docstring states the whole rule.
    @tool("deflated_sharpe_ratio", "information_ratio", notes=[_GATED_NOTE])
    def run_verdict(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        scanner: str = "demo_volume",
        signals: Optional[List[str]] = None,
        source: str = "strategy",
        benchmark: str = "SPY",
        timeframe: str = "1Day",
        capital: Optional[float] = 100_000.0,
        horizon: int = 5,
        target_te: float = 0.04,
        risk_model: str = "shrinkage",
        gross: bool = False,
        commission_bps: float = 1.0,
        impact_eta: float = 0.3,
        borrow_bps: float = 50.0,
        force: bool = False,
        max_weight: Optional[float] = None,
        max_names: Optional[int] = None,
        lookback_days: Optional[int] = None,
        neutralize_factors: Optional[List[str]] = None,
        participation_cap: Optional[float] = None,
        n_points: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the whole cross-sectional pipeline once and return one composite answer.

        Scan the universe, refine the signal into alphas (combining several when
        `signals` names more than one), construct the cost-aware portfolio, and measure
        the information content - all against ONE resolved universe, ONE window, and
        ONE cost model. Running the four steps as separate tool calls gives no such
        guarantee: each re-resolves its own universe and applies its own defaults, and
        the joined-up story can silently be four different stories. Prefer this when
        you want the answer rather than one stage's detail.

        Returns `{schema, inputs, steps, scan, alphas, combination, portfolio,
        information, verdict, provenance}`. `verdict.verdict` is one of promotable /
        not promotable / needs more data / mixed / infeasible / incomplete, assembled from the gates
        the steps themselves computed (IC t-stat vs 2, realized IR vs its own standard
        error, the sanity ceiling, expected active return net of cost) - every check is
        in `verdict.checks` with its value and threshold. A run where any step failed
        is `incomplete` and carries NO verdict, whatever the completed sections say:
        do not act on a partial run's weights.

        `infeasible` means the caps cannot fund a fully invested long-only book from
        this universe (for example three names under the default `max_weight`): the
        binding constraint is in `verdict.summary` and in `portfolio.binding_constraint`,
        the later steps are reported as skipped, and more dates will not help. Nothing
        is relaxed for you. Recover by passing a wider `symbols` list, a larger
        `max_weight`, or a larger `max_names` explicitly, and read the result's
        `inputs` to confirm which caps were applied. An unknown `risk_model` is refused
        up front, listing the valid models. A history too short to estimate a
        book at all names no constraint and is `needs more data`, not `infeasible`.

        The allocation knobs (`max_weight`, `max_names`, `lookback_days`,
        `neutralize_factors`, `participation_cap`, `n_points`) are the CLI's own; leave
        one out and the service's default applies; the result's `inputs` block reports
        the value each one took. The evidence-gated construction knobs are deliberately not offered here.

        This answers "what does the pipeline say about this universe as of `end`" -
        a forecast and a proposed book. The scanner is resolved at `end`, not at
        wall-clock now, so a historical verdict does not mix today's universe into an
        older evaluation window. For "did this ever work", use `run_backtest` or
        `run_walk_forward`.

        Read-only research clock: proposes a portfolio, never places an order. Journals
        exactly ONE trial (kind `verdict`), so it counts once toward the campaign's
        multiple-testing total - except that an `incomplete` run in which no step measured an
        IC journals none, and so does an `infeasible` run refused by the allocation preflight
        (before any step read the data): `journaled: false`, with
        `not_journaled_reason`.
        An `infeasible` run found only after the alphas read the data does journal, since
        it was a look at the data. A non-journaled run also reports `provenance.n_trials`
        without itself (`null` when the store could not say). Candidates with no bars are
        dropped by the alphas step, after the preflight counted them; they are listed in
        `dropped_symbols` (empty when none). A step that failed for lack of cached bars says so, and `cache_miss` holds
        one warm command covering every such step. An identical prior run is returned
        from the trial store labeled `memoized: true`, with the original run's timestamp,
        rather than re-run; pass `force=true` to re-verify and append a new trial.
        """
        # Only what the caller set is forwarded, so an omitted knob keeps the service's
        # own default instead of this surface restating it.
        optional = {
            "max_weight": max_weight,
            "max_names": max_names,
            "lookback_days": lookback_days,
            "neutralize_factors": neutralize_factors,
            "participation_cap": participation_cap,
            "n_points": n_points,
        }
        optional = {k: v for k, v in optional.items() if v is not None}
        inputs = {"strategy": strategy, "symbols": symbols, "start": start, "end": end, **optional}
        result = analysis.run_verdict(
            _data(),
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            scanner=scanner,
            signals=signals,
            source=source,
            benchmark=benchmark,
            timeframe=timeframe,
            capital=capital,
            horizon=horizon,
            target_te=target_te,
            risk_model=risk_model,
            gross=gross,
            commission_bps=commission_bps,
            impact_eta=impact_eta,
            borrow_bps=borrow_bps,
            force=force,
            **optional,
        )
        return _logged("run_verdict", inputs, result)

    @tool()
    def render_report(kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Render a result you already have as one self-contained HTML document.

        `kind` is one of verdict/backtest/walkforward/info, and `result` is the dict
        that kind's run tool returned, **unmodified** — pass it through rather than
        rebuilding it, or the report will describe something the run did not produce.
        A payload that does not match its kind is rejected with a clear error rather
        than half-rendered.

        Returns `{kind, html, bytes}`. The document embeds its CSS and charts, so
        opening it issues no network requests at all — it is safe to save, attach, or
        forward. Provenance (window, universe, cost model, git SHA, campaign
        n_trials) is a mandatory header, and memoized results, gate failures, a backtest
        that is not executable at its capital, a failed leakage probe, and any enabled
        evidence-gated feature render as warnings above the sections rather than as
        footnotes.

        This computes nothing: it is the same renderer the CLI's `--html` flag uses,
        over the same object. Charts need the plotting extra on the server; without it
        the tables still render and the chart slots say so.

        Note that a backtest result carries no equity curve (it is omitted from the
        payload deliberately), so a backtest report rendered here shows the metrics
        table without the equity chart.
        """
        from quantred.analytics.htmlreport import render_html

        document = render_html(result, kind)
        return _logged(
            "render_report",
            {"kind": kind, "run_id": result.get("run_id") if isinstance(result, dict) else None},
            {"kind": kind, "html": document, "bytes": len(document.encode("utf-8"))},
        )

    # ---------------- Campaign memory (read-only) ---------------- #
    @tool("sharpe_ratio", "deflated_sharpe_ratio")
    def list_trials(
        strategy: Optional[str] = None,
        kind: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        gates_passed: Optional[bool] = None,
        sort: str = "date",
        limit: int = 20,
        offset: int = 0,
        include_in_sample: bool = False,
        all_accounting: bool = False,
    ) -> Dict[str, Any]:
        """What has this campaign already tried? Filtered rows from the trial store.

        Ask this BEFORE running anything expensive. Every backtest, optimization,
        walk-forward, and verdict ever run — over the CLI or over this server — is
        recorded here, and re-running something the campaign already answered costs a
        trial without buying information.

        `symbols` matches on the normalized universe, so order and case never change
        what is found. `since`/`until` take a date (a New York market date) or an ISO
        datetime; both ends include the dates they name, so `until="2026-11-06"` includes
        that evening. Each row's `ts` is the UTC instant it was recorded. `sort` is
        date/sharpe/dsr; a trial with no recorded metric
        sorts last rather than as a zero. Returns `{rows, total, in_sample_excluded, in_sample_included, undated_excluded}` —
        `total` is how many matched, which is usually more than were returned.

        **Ranking leaves in-sample kinds out.** Sorting by `sharpe`/`dsr`, or setting
        `min_sharpe`, makes this a leaderboard, and `optimize` (best of N candidates),
        `backtest` (one pass over its window) and `alpha` rows are not validated out of
        sample, so they are excluded unless `include_in_sample=true` — as `best_trials`
        does. `in_sample_excluded` says how many that hid. Every row carries
        `sharpe_basis` (what its Sharpe measures: out-of-sample aggregate, median fold,
        full window, or best of N) and `in_sample`; read them before comparing two Sharpes.

        Absent fields are null, and null means *not recorded*: a trial predating a
        field did not fail to record it, and a forecast kind did not score zero.
        An unknown `kind` is refused with the valid ones rather than listing nothing.
        Read-only: this cannot modify or delete anything.
        """
        inputs = {"strategy": strategy, "kind": kind, "symbols": symbols, "limit": limit}
        with _trial_store() as store:
            if store is None:
                return _logged("list_trials", inputs, {"rows": [], "total": 0, "error": _NO_STORE})
            filters = {
                "strategy": strategy,
                "kind": kind,
                "symbols": symbols,
                "since": _parse_date(since) if since else None,
                "until": _parse_date(until) if until else None,
                "min_sharpe": min_sharpe,
                "promotable": gates_passed,
                "all_accounting": all_accounting,
            }
            listing = store.listing(
                sort=sort, limit=limit, offset=offset, include_in_sample=include_in_sample, **filters
            )
            listing.pop("exclude_kinds", None)
            return _logged("list_trials", inputs, listing)

    @tool()
    def get_trial(trial_id: str) -> Dict[str, Any]:
        """Everything the trial store knows about one trial.

        Its full params (the run's dedup identity, including the folded cost and
        data-vintage keys), provenance, headline metrics, and what was stored
        alongside it: whether its out-of-sample return series was persisted (and over
        what window), the book it proposed, its trade table if the run opted into
        keeping one, and which later trials were served from this one.

        `null` for a companion means *not recorded* — the run predates that storage,
        or did not opt in. It never means the run produced nothing. A walk-forward also
        carries `holdout_record`: `state` is `recorded` (with `metrics`), `none_requested`,
        `not_retained` (asked for, but the trial predates the journal keeping the result)
        or `unknown`. Read-only.
        """
        with _trial_store() as store:
            if store is None:
                return _logged("get_trial", {"trial_id": trial_id}, {"error": _NO_STORE})
            from quantred.store.trials import AmbiguousTrialId

            try:
                resolved = store.resolve_id(trial_id)
            except AmbiguousTrialId as exc:
                raise ValueError(f"{exc} {_USE_LIST_TRIALS}") from exc
            trial = store.get_trial(resolved) if resolved else None
            if trial is None:
                raise ValueError(f"No trial with id {trial_id!r}. {_USE_LIST_TRIALS}")
            if trial.get("kind") == "walkforward":
                # The index never kept the holdout; the journal line is the only place it
                # is, and `state` says whether it is there, was never asked for, or
                # predates the journal keeping it.
                from quantred.services.audit import holdout_for_trial

                trial["holdout_record"] = holdout_for_trial(trial["id"])
            return _logged("get_trial", {"trial_id": trial_id}, trial)

    @tool()
    def analyze_trial(trial_id: str, allow_partial: bool = False) -> Dict[str, Any]:
        """What one recorded run's trades actually did: exit-reason P&L, win and loss
        by reason, holding period, and per-trade excursion.

        Read `status` before any number. `complete` means the figures cover every trade
        the run made. `unavailable` means there are none and `reason` says why — most
        often that the run did not record trades (`record_trades=true` on `run_backtest` or
        `run_walk_forward`; no other tool takes it), which is NOT the same as a run that made
        no trades. Verdict, optimization and draft walk-forward trials never carry a trade
        table; `reason` says which tool to run instead for the trial's own kind. `truncated`
        appears only with `allow_partial=true`,
        and then every count and total is a partial and must be reported as one.

        A trade table capped at the storage ceiling is refused by default rather than
        summed: a total over a prefix of a run's trades is not a smaller number than
        the truth, it is a wrong one. So is a table whose completeness was never
        recorded.

        The excursion figures are **per trade** — how far one position went against its
        own entry — and are not the book's aggregate open drawdown. A position deep
        underwater that is a small fraction of the book did not put the book that far
        underwater. Journals nothing; read-only.
        """
        inputs = {"trial_id": trial_id, "allow_partial": allow_partial}
        with _trial_store() as store:
            if store is None:
                return _logged("analyze_trial", inputs, {"error": _NO_STORE})
            report = analysis.analyze_trial(store, trial_id, allow_partial=allow_partial)
            if report.get("error"):
                raise ValueError(f"{report['error']} {_USE_LIST_TRIALS}")
            return _logged("analyze_trial", inputs, _trade_recovery_note(report))

    @tool()
    def compare_trials(
        trial_ids: List[str],
        min_overlap: int = SERIES_MIN_OVERLAP,
        across_accounting: bool = False,
    ) -> Dict[str, Any]:
        """Correlate the recorded return series of two or more trials — are they in
        fact one result?

        Ask this before reporting two candidates as separate evidence. Results
        correlating near 1.0 are one bet held twice however differently they were
        parameterised, and counting both inflates what a campaign appears to have
        found.

        Pairs are **refused**, not caveated, when they share fewer than `min_overlap`
        dates or were recorded under different accounting versions — a correlation over
        a handful of dates is an error bar, and two engines that compute different
        things produce series whose correlation is partly about the engines.
        `across_accounting=true` computes those anyway and marks them
        `comparable: false`.

        Read `pairs` alongside `matrix`. The matrix holds `null` where nothing was
        computed, because in a correlation matrix a zero is the strong claim that two
        results move independently — exactly what a refusal cannot say. Every
        correlation carries a Fisher-z `interval`; quote it, since one resting on a
        short overlap is wide and a bare coefficient hides that.

        Only trials that store a dated return series can be compared: walk-forwards (the
        stitched out-of-sample folds, drafts included) and backtests (daily returns from
        realized trade P&L over the window, in-sample). Optimizations, verdicts and other
        kinds store none, and neither do backtests recorded over MCP before this server
        began storing one; those are reported as having no series. Each series carries its
        trial `kind`, and `kinds_differ` / a pair's `unlike_basis` flag a comparison of
        unlike things. Journals nothing; read-only.
        """
        # `across_accounting` decides whether cross-era pairs were computed or refused,
        # so an audit trail without it cannot tell a default comparison from one that
        # deliberately crossed an accounting boundary.
        inputs = {
            "trial_ids": trial_ids,
            "min_overlap": min_overlap,
            "across_accounting": across_accounting,
        }
        with _trial_store() as store:
            if store is None:
                return _logged("compare_trials", inputs, {"error": _NO_STORE})
            report = analysis.compare_trials(
                store,
                trial_ids,
                min_overlap=min_overlap,
                across_accounting=across_accounting,
            )
            if report.get("error"):
                raise ValueError(f"{report['error'].rstrip('.')}. {_USE_LIST_TRIALS}")
            unknown = report.get("unknown_trial_ids") or []
            if len(trial_ids) - len(unknown) < 2:
                raise ValueError(
                    f"Fewer than two of these trials are recorded (unknown: {', '.join(map(repr, unknown))}). "
                    f"{_USE_LIST_TRIALS}"
                )
            return _logged("compare_trials", inputs, report)

    @tool()
    def get_campaign_material(trial_id: str) -> Dict[str, Any]:
        """What validated a trial: its recipe, its evidence, and what is recoverable.

        Read this before proposing a config from a trial, and pass it through as
        `provenance.campaign` when you call `save_config` — that is the same block
        `trials promote` writes, and a second shape invented here would be a second
        provenance format, which is the thing this project has been bitten by more
        than once.

        Three sections, labelled because they age differently. `recipe` is how it was
        validated — the folds, the embargo, the objective, the search method, and the
        cost model and book limits folded into its identity. It stays true. `evidence`
        is what was measured under one accounting version and is worthless outside it;
        check `comparable_with_current_engine` before quoting a single number from it.
        `metadata` is about the record: when, which git sha, and which stored artifacts
        still exist.

        A section that could not be assembled says so with a reason rather than coming
        back empty. Most trial kinds have no separate recipe — a backtest's identity
        *is* its parameters — and the recipe section says that instead of implying the
        run was validated with no settings. Journals nothing; read-only.
        """
        with _trial_store() as store:
            if store is None:
                return _logged("get_campaign_material", {"trial_id": trial_id}, {"error": _NO_STORE})
            material = campaign.campaign_material(store, trial_id)
            if material.get("not_indexed"):
                # The service's reason names the CLI's reindex command, which an agent
                # cannot run; say what it can do instead.
                raise ValueError(
                    f"No trial with id {trial_id!r} is indexed. Its journal line may still exist, but "
                    "reindexing the store from the journal is an operator action this server does not "
                    f"offer. {_USE_LIST_TRIALS}"
                )
            if not material.get("available"):
                raise ValueError(f"{material.get('reason') or 'No such trial.'} {_USE_LIST_TRIALS}")
            return _logged("get_campaign_material", {"trial_id": trial_id}, material)

    @tool("deflated_sharpe_ratio", "sharpe_ratio")
    def best_trials(
        strategy: Optional[str] = None,
        kind: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        rank_by: str = "dsr",
        limit: int = 10,
        include_in_sample: bool = False,
        all_accounting: bool = False,
    ) -> Dict[str, Any]:
        """The campaign's leaderboard, ranked by DEFLATED Sharpe by default.

        Read the caveat this returns before reporting a winner. Ranking a research
        campaign's trials and presenting the top row is selection bias by
        construction: the best of N tried configs is a biased estimate of its own
        future performance. That is why the default ranking is the deflated Sharpe
        (which already discounts for N), why every row carries its family's
        `n_trials`, and why `rank_by="sharpe"` returns a caveat saying in as many
        words that the ordering does not correct for how many configs were tried.

        In-sample kinds (`optimize`, `alpha`, `backtest`) are excluded by default: an
        `optimize` row is the winner of a search, best-of-N by construction, so ranking
        one ranks the selection bias rather than any skill, and a plain `backtest` is one
        pass over its window, never validated out of sample (its `oos_sharpe` is the
        full-window Sharpe). `include_in_sample=true`
        opts back in, and `in_sample_excluded` reports how many were dropped.

        A row with no value for the ranking metric (a verdict has neither a Sharpe nor a
        DSR) is not ranked and not listed; `unranked_excluded` counts them. A row's DSR
        was deflated by the trials of its own run, while `family_n_trials` is the
        strategy-and-universe count now, so DSRs from runs of different size are not
        comparable.

        Returns `{rank_by, rows, max_family_n_trials, in_sample_included,
        in_sample_excluded, unranked_excluded, caveat}`. The caveat and the per-row family counts are
        part of the payload, not decoration — quote them when you report a result.
        Read-only.
        """
        inputs = {"strategy": strategy, "symbols": symbols, "rank_by": rank_by, "limit": limit}
        with _trial_store() as store:
            if store is None:
                return _logged("best_trials", inputs, {"rows": [], "error": _NO_STORE})
            board = store.best(
                rank_by=rank_by,
                limit=limit,
                include_in_sample=include_in_sample,
                strategy=strategy,
                kind=kind,
                symbols=symbols,
                all_accounting=all_accounting,
            )
            return _logged("best_trials", inputs, board)

    @tool()
    def get_metrics_glossary() -> Dict[str, Any]:
        """Definitions + pitfalls of every reported metric, plus global caveats.

        Read this before interpreting results - it spells out the Deflated Sharpe /
        multiple-testing trap and the closed-trade equity-curve caveat so you don't
        over-trust in-sample Sharpe.
        """
        return _logged("get_metrics_glossary", {}, glossary.metrics_glossary())

    @tool()
    def summarize_bars(
        symbols: List[str], timeframe: str = "1Day", lookback_days: int = 90
    ) -> Dict[str, Any]:
        """Compact OHLCV stats per symbol (return, vol, trend, volume) - no raw bars.

        Descriptive only. Picking symbols by these stats then backtesting them is
        look-ahead; treat universe choice as a research decision, not an input to
        optimize.
        """
        inputs = {"symbols": symbols, "timeframe": timeframe, "lookback_days": lookback_days}
        return _logged(
            "summarize_bars", inputs, analysis.summarize_bars(_data(), symbols, timeframe, lookback_days)
        )

    # ---------------- Propose (writes a file, never live state) ---------------- #
    @tool()
    def save_config(
        name: str,
        strategy: str,
        params: Dict[str, Any],
        scanner: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        capital: Optional[float] = None,
        position_limits: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Propose a candidate config by writing configs/<name>.json, with provenance.

        This is the end of what automation may do. It writes a file for a HUMAN to
        read, judge, and promote; it does not affect any running process, does not
        change any default, and cannot enable live trading. Promotion is a manual
        step by design — the research clock proposes, a person disposes.

        `name` is a plain file name (no directories, no `..`); the file is always written
        under the configs directory. A name already in use is refused unless
        `overwrite=true`: a saved config may be one a person has promoted. The config is
        checked before anything is written: the strategy and scanner must exist, `params`
        must construct the strategy (a key that is a near-miss of a declared parameter is
        refused as a typo), and `position_limits` may only name limits that are enforced.
        A config that could not run is refused, not saved.

        Include the evidence in `provenance` (the walk-forward or verdict run behind
        the proposal, its window and universe), because a config with no recorded
        reason for existing is a config nobody can safely promote. It accepts only these
        fields: `method`, `objective`, `windows`, `oos_metrics`, `n_trials`, `seed`,
        `git_sha`, `timestamp`, `notes`, `accounting`, `campaign`; any other key is
        refused. There is no trial-id field: put the trial id in `notes`, or pass the
        `get_campaign_material` block as `campaign`, which carries it.

        For a config proposed from a recorded trial, put `get_campaign_material`'s
        block in `provenance.campaign` verbatim rather than summarising it. It carries
        the validation recipe the parameters alone cannot express, and which accounting
        era its numbers belong to. One format, so a config written here and one written
        by `trials promote` read identically.

        **Pass `position_limits`, and pass the book that was actually validated** —
        `recipe.folded_into_identity._limits` in that same block. Omitting it does not
        leave the book unspecified: a run from the config inherits the strategy class's
        default of one position, so a config whose provenance says eight were validated
        silently trades one. `symbols` and `capital` the same way, when you know them.
        The validated cost model is taken from that same campaign block when it records
        one, so the config trades at the costs it was validated under.
        """
        inputs = {
            "name": name,
            "strategy": strategy,
            "params": params,
            "scanner": scanner,
            "position_limits": position_limits,
            "overwrite": overwrite,
        }
        result = configs.save_config(
            name,
            strategy=strategy,
            params=params,
            scanner=scanner,
            symbols=symbols,
            capital=capital,
            position_limits=position_limits,
            provenance=provenance,
            overwrite=overwrite,
        )
        return _logged("save_config", inputs, result)

    @tool()
    def load_config(name: str) -> Dict[str, Any]:
        """Read back one saved candidate config: its strategy, params, symbols, capital,
        position_limits, and provenance.

        `symbols`, `capital` and `position_limits` are absent when the config never
        recorded them (absent, not zero or a default); a config with no
        `position_limits` runs at the strategy class's own book.

        Reading a config does not activate it — nothing here is running the config,
        and loading it changes no behavior anywhere. Use it to check what a past
        proposal actually contained before re-testing or superseding it. Read-only.
        """
        try:
            loaded = configs.load_config(name)
        except FileNotFoundError:
            raise ValueError(
                f"No saved config named {name!r}. Use list_configs to see what exists."
            ) from None
        return _logged("load_config", {"name": name}, loaded)

    @tool()
    def list_configs() -> List[Dict[str, Any]]:
        """List every saved candidate config with a compact summary of each.

        These are proposals awaiting human review, not active settings: none of them
        is in use by anything until a person promotes it. Read-only.
        """
        return _logged("list_configs", {}, configs.list_configs())

    _refuse_unknown_arguments(server)
    return server


def _refuse_unknown_arguments(server) -> None:
    """Make an argument this surface does not accept an error rather than a silence.

    The framework validates a call against the tool's schema and then **drops** every
    key the schema does not declare. Nothing raises, nothing is logged, and the tool
    returns a perfectly good result computed without the argument — so an agent told to
    "turn on conditional risk" passes `conditional="ewma"`, gets a portfolio back, and
    reports that it did. The wall held; its account of what it did did not.

    That is the failure this whole surface is designed against: an agent reads a
    description as fact and acts on it, and every action costs a journaled trial. A
    dropped argument is the same class of lie as a stale description, arriving at call
    time instead of read time.

    A plain typo has the identical shape — `targt_te` silently leaves the default in
    place — so both are refused. Naming the near-miss matters more here than for a
    human, who would see the default in the output and wonder; an agent has nothing to
    wonder at.

    Wrapped at dispatch rather than declared per tool: `**kwargs` cannot express this,
    because the framework turns it into a *required* schema property called `extra`,
    which changes every tool's contract to fix a problem in none of them.

    **Wrapped on the tool manager, not on the server.** The obvious place —
    reassigning ``server.call_tool`` after the tools are registered — does not work, and
    fails in the worst possible way: the framework registers its low-level protocol
    handler during ``FastMCP.__init__``, capturing the *bound method* as it was then, so
    a later reassignment is invisible to every real client while a test that calls
    ``server.call_tool`` directly still sees it. That is a guard which passes its own
    suite and protects nobody, and it is what the first version of this did.
    ``FastMCP.call_tool`` looks its manager up as ``self._tool_manager`` at call time, so
    wrapping the manager is on both paths.
    """
    manager = server._tool_manager
    original = manager.call_tool

    async def call_tool(name, arguments, *args, **kwargs):
        declared = _declared_arguments(server, name)
        # An unknown tool, or a schema we could not read, is the framework's business:
        # pass it through rather than inventing a second way for a call to fail.
        if declared is not None and isinstance(arguments, dict):
            unknown = sorted(set(arguments) - declared)
            if unknown:
                raise ValueError(_unknown_argument_message(name, unknown, declared))
        return await original(name, arguments, *args, **kwargs)

    manager.call_tool = call_tool


def _declared_arguments(server, name: str):
    """The argument names one tool accepts, or ``None`` if that cannot be established.

    Read from the registered tool's own schema rather than from a signature, so it is
    the same thing the client was shown. ``None`` on any doubt: refusing a call on a
    guess about what a tool accepts would be worse than the silence this replaces.
    """
    try:
        tool = server._tool_manager.get_tool(name)
        schema = getattr(tool, "parameters", None) or {}
        properties = schema.get("properties")
    except Exception:  # noqa: BLE001 - never turn a lookup problem into a call failure
        return None
    return set(properties) if isinstance(properties, dict) else None


def _unknown_argument_message(name: str, unknown, declared) -> str:
    """Say what was ignored, and — where it can be known — why.

    A withheld knob and a mistyped one need different actions, so they get different
    sentences. `DEFERRED_PARAMS` already records why each gated family is absent; this
    is the moment an agent actually needs that reason, so it is quoted rather than
    referred to.
    """
    import difflib

    lines = [f"{name} does not accept: {', '.join(unknown)}."]
    for argument in unknown:
        reason = DEFERRED_PARAMS.get(argument) or WITHHELD_PARAMS.get(name, {}).get(argument)
        if reason:
            lines.append(f"  {argument}: withheld from this surface — {reason}.")
            continue
        close = difflib.get_close_matches(argument, sorted(declared), n=1)
        if close:
            lines.append(f"  {argument}: not a parameter of this tool. Did you mean {close[0]}?")
        else:
            lines.append(f"  {argument}: not a parameter of this tool.")
    lines.append(
        "Refused rather than ignored: an argument dropped in silence leaves you believing it was applied."
    )
    return "\n".join(lines)


#: What this process has done, for the line it prints when it stops. A plain dict
#: mutated in place: the count is read from a signal handler, which must not take a lock.
_SESSION: Dict[str, int] = {"tool_calls": 0}


def _client_gone(exc: BaseException) -> bool:
    """True when every leaf of ``exc`` is the client's pipe having closed under us.

    A client that exits without closing stdin first surfaces as a ``BrokenPipeError``
    inside an exception group from the SDK's task group. That is a disconnect, not a
    crash, and it should not print a traceback to someone who just closed a chat.
    """
    leaves = getattr(exc, "exceptions", None)
    if leaves is not None:
        return bool(leaves) and all(_client_gone(leaf) for leaf in leaves)
    return isinstance(exc, (BrokenPipeError, ConnectionResetError))


def _stop_line(why: str) -> str:
    """The one line printed when the server stops, saying what that did and did not do.

    Not the CLI's generic interrupt text, which says no trial was recorded - true of a
    command interrupted part-way, false of a server that may have journaled trials all
    session. Every tool call here appends to the journal before it returns, so work
    already done is on disk and only a request in progress is lost.
    """
    from quantred.services.audit import default_trial_journal

    n = _SESSION["tool_calls"]
    return (
        f"MCP server stopped ({why}). It handled {n} tool call(s) this session; trials "
        f"already journaled are kept in {default_trial_journal()}. A request in progress, "
        "if any, was not completed."
    )


def stop_now(why: str, code: int) -> None:
    """Say what stopping did, then leave without interpreter finalization.

    ``os._exit`` is deliberate, not tidy-minded: the SDK reads stdin on a worker thread
    blocked in ``read(2)`` that nothing can cancel or wake, so ``anyio.run`` never
    returns after an interrupt and, once it is forced out, finalization joins that
    thread - hanging, or aborting the interpreter if interrupted again. Leaving
    directly is the only exit that is bounded. Repeats are ignored first so a second
    Ctrl-C cannot re-enter this.
    """
    import signal

    from quantred.utils import streaming

    for sig in streaming.STOP_SIGNALS:
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
    try:
        sys.stderr.write(_stop_line(why) + "\n")
    except Exception:  # noqa: BLE001 - a closed stderr must not stop the exit
        pass
    streaming.exit_now(code)


def _install_stop_handlers() -> None:
    """Route SIGINT/SIGTERM to :func:`stop_now` instead of the loop.

    ``signal.signal``, not an event-loop handler: tool functions here are synchronous
    and run on the loop's own thread, so a loop callback would not fire until the call
    in progress returned. A no-op off the main thread, where signals cannot be set.
    """
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return
    from quantred.utils import streaming

    for sig in streaming.STOP_SIGNALS:
        code = 128 + int(sig)
        name = "interrupted" if sig == signal.SIGINT else "terminated"
        signal.signal(sig, lambda _n, _f, _name=name, _code=code: stop_now(_name, _code))


def serve(*, offline: bool = False, cache_dir=None) -> None:
    """Build the server and run it over stdio (what Claude Desktop/Code launch).

    Starts without broker keys: bars are fetched lazily, so a keyless server still
    answers the catalogue and trial-history tools. ``offline`` reads only the bar cache.
    """
    # Said at startup, not on request: an agent driving these tools cannot see
    # where its evidence is landing, and every journaled trial is permanent.
    from quantred.services.setup import invocation
    from quantred.settings import git_worktree_containing, state_root

    root = state_root()
    logger.info("MCP state root: %s", root)
    worktree = git_worktree_containing(root)
    if worktree is not None:
        logger.warning(
            "State root %s is inside the git working tree at %s - trials, configs and "
            "any private strategy's evidence are being written into a repository.",
            root,
            worktree,
        )

    server = build_server(offline=offline, cache_dir=cache_dir)
    # After the build, not before: a missing extra stops the server there, and "waiting
    # for a client" from a process about to exit is false.
    logger.info(
        "%s MCP server (stdio) is waiting for an MCP client on stdin/stdout. Live "
        "trading is NOT exposed. Started by hand in a terminal it is attached to no chat: "
        "a client launches its own copy (check the command with `%s`). Ctrl-C stops it.",
        BRAND,
        invocation("mcp --self-test"),
    )
    _install_stop_handlers()
    try:
        server.run()
    except KeyboardInterrupt:
        # Only reachable where the handlers could not be installed (not the main thread).
        stop_now("interrupted", 130)
    except BaseException as exc:  # noqa: BLE001 - classified, and re-raised unless a disconnect
        if not _client_gone(exc):
            raise
        stop_now("client disconnected", 0)


def _json_safe(value: Any) -> Any:
    """``value`` with every non-finite float spelled ``"nan"``, ``"inf"`` or ``"-inf"``.

    JSON has no NaN or infinity. Some services already spelled them as strings and the
    rest reached the wire as bare ``NaN`` / ``Infinity``, which a strict client refuses to
    parse, so one payload could carry ``"nan"`` in one field and an unparseable ``NaN``
    in the next. One convention, applied once at the boundary every tool returns through.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if type(value).__module__ == "numpy":
        if getattr(value, "ndim", 0):
            return _json_safe(value.tolist())
        if hasattr(value, "item"):
            return _json_safe(value.item())
    return value


def _assert_no_trading_client(data_client) -> None:
    """Guardrail: the MCP process must hold only a data client."""
    from quantred.marketdata.client import MarketDataClient

    if not isinstance(data_client, MarketDataClient):
        raise RuntimeError("MCP server requires a MarketDataClient (data only); refusing to start.")
    for attr in ("broker", "trading_client", "_broker"):
        if getattr(data_client, attr, None) is not None:
            raise RuntimeError(f"MCP data client unexpectedly exposes {attr!r}; refusing to start.")


def _summary(result: Any) -> Dict[str, Any]:
    """A tiny, log-friendly digest of a tool result (no large arrays)."""
    if not isinstance(result, dict):
        return {"type": type(result).__name__, "len": len(result) if hasattr(result, "__len__") else None}
    keep = (
        "run_id",
        "best_score",
        "total_trades",
        "n_trials",
        "n_trials_total",
        "flagged_count",
        "memoized",
        "trial_id",
    )
    digest = {k: result[k] for k in keep if k in result}
    if "gate_report" in result and isinstance(result["gate_report"], dict):
        digest["promotable"] = result["gate_report"].get("promotable")
    return digest
