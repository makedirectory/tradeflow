"""MCP server: a thin adapter exposing TradeFlow's deterministic core as tools, so
an agent can do the research and the engine can stay boringly deterministic.

No business logic lives here - every tool calls a function in :mod:`tradeflow.services`
and logs the call to the audit trail. The server constructs **only** a data
client (never a broker/trading client), so it is structurally incapable of
placing an order. The hard wall is the *absence* of any
order/live/account-mutation tool; it cannot be prompt-injected around because the
capability is not wired in.

Run with ``tradeflow mcp`` when installed, or ``python main.py mcp`` from a checkout
(needs the ``mcp`` extra either way).
"""

import contextlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from tradeflow.services import analysis, configs, glossary, registry
from tradeflow.services.audit import audit_log, new_run_id

logger = logging.getLogger(__name__)

SERVER_NAME = "tradeflow"

#: The complete, intended tool surface (used by tests to assert the wall).
EXPOSED_TOOLS = (
    "list_strategies",
    "list_scanners",
    "get_param_ranges",
    "run_scan",
    "run_backtest",
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
    "compute_horizon",
    "render_report",
    "list_trials",
    "get_trial",
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
    {"run_backtest", "run_optimization", "run_walk_forward", "run_draft_walk_forward", "run_verdict"}
)

#: Evidence-gated features that ship **off**. Where a tool exposes one, its
#: description must name the gate and its current verdict rather than presenting the
#: flag as a neutral option - an agent reads a description as fact and acts on it.
EVIDENCE_GATED = ("conditional", "policy", "posterior")

#: Capabilities that must NEVER be exposed over MCP (the safety model).
FORBIDDEN_TOOLS = frozenset(
    {
        "place_order",
        "submit_order",
        "submit_market_order",
        "submit_bracket_order",
        "start_live",
        "run_live",
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
    """Parse an ISO date/datetime string from a tool argument."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%d")


#: Appended to every journaling tool's description. One sentence, one place: an
#: agent that misreads its own experiment history misreads every result after it.
_JOURNALING_NOTE = (
    "Journals one trial per evaluated config, so this call counts toward the "
    "campaign's multiple-testing total (what the Deflated Sharpe deflates against). "
    "An identical prior run is returned from the trial store labeled `memoized`, with "
    "the original run's timestamp, instead of being re-run; pass `force=true` to "
    "re-verify and append a new trial."
)

#: Appended where a tool exposes an evidence-gated feature.
_GATED_NOTE = (
    "Evidence-gated features ship OFF and are not neutral options: conditional risk, "
    "the aim-in-front-of-the-target trading policy, and the Black-Litterman posterior "
    "each have an adoption gate (a predictive-accuracy test or a net-of-cost A/B), and "
    "none of them clears on this repository's own demo data. Enabling one is a "
    "deliberate departure from the validated default, not a tuning choice."
)


#: Returned instead of rows when the store cannot be opened. The store is derived and
#: passive by contract: a broken one degrades a read, it never fails a session.
_NO_STORE = "The trial store could not be opened; no campaign history is available for this call."


@contextlib.contextmanager
def _trial_store():
    """A read-only handle on the trial store, or ``None`` if it cannot be opened."""
    from tradeflow.services import audit
    from tradeflow.store.trials import TrialStore, db_path_for_journal

    journal_path = audit.DEFAULT_TRIAL_JOURNAL
    try:
        store = TrialStore(db_path_for_journal(journal_path), journal_path=journal_path)
    except Exception:  # noqa: BLE001 - a passive store never breaks its caller
        logger.warning("Trial store unavailable for this MCP call", exc_info=True)
        yield None
        return
    try:
        yield store
    finally:
        store.close()


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

    Metric definitions are *pulled* from :mod:`tradeflow.services.glossary` rather than
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


def build_server(data_client=None):
    """Construct the FastMCP server with all read/analyze/propose tools registered.

    ``data_client`` may be injected (offline tests); otherwise a data-only Alpaca
    client is built. Importing ``mcp`` is deferred to here so the package and the
    rest of the CLI never require the extra.
    """
    from mcp.server.fastmcp import FastMCP

    if data_client is None:
        from tradeflow.services.data import build_data_client

        data_client = build_data_client()
    _assert_no_trading_client(data_client)

    server = FastMCP(SERVER_NAME)
    dc = data_client

    def _logged(tool: str, inputs: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        run_id = result.get("run_id") if isinstance(result, dict) else None
        audit_log(tool, inputs, run_id=run_id or new_run_id(), result_summary=_summary(result))
        return result

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
    def list_strategies() -> List[Dict[str, str]]:
        """List the trading strategies this engine can run: name, what each one
        measures, and the bar timeframe it is designed for.

        Start here. A strategy is a `bar -> score` policy: each defines one continuous
        conviction score, and both the discrete BUY/SELL/HOLD signal and the continuous
        alpha forecast are derived from that one score. The `name` values are what every
        other tool's `strategy` argument accepts. Read-only.
        """
        return _logged("list_strategies", {}, registry.list_strategies())

    @tool()
    def list_scanners() -> List[Dict[str, str]]:
        """List the universe scanners: name and what each one selects for.

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
            analysis.run_scan(dc, scanner, symbols, config, as_of=_parse_date(as_of) if as_of else None),
        )

    @tool(
        "sharpe_ratio",
        "deflated_sharpe_ratio",
        "max_drawdown",
        "profit_factor",
        "low_sample",
        notes=[_JOURNALING_NOTE],
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

        Journals this as a trial (the same research journal and trial store the
        `backtest` command uses) so it counts toward the campaign's multiple-testing
        total. An exact prior trial is served instead (result has `memoized: true`)
        unless `force=True`, which re-runs and appends a new trial rather than
        overwriting the memoized one. Memoization is scoped to the engine's accounting
        version, so results from an older engine are never served to a newer one.
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
        }
        result = analysis.run_backtest(
            dc,
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
        )
        return _logged("run_backtest", inputs, result)

    @tool("sharpe_ratio", "deflated_sharpe_ratio", notes=[_JOURNALING_NOTE, _WORKERS_NOTE])
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
            dc,
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
        )
        return _logged("run_optimization", inputs, result)

    @tool(
        "sharpe_ratio",
        "deflated_sharpe_ratio",
        "profit_factor",
        "max_drawdown",
        notes=[_JOURNALING_NOTE, _WORKERS_NOTE],
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
    ) -> Dict[str, Any]:
        """Honest out-of-sample evaluation across folds - your advancement criterion.

        Optimizes in-sample per fold, scores out-of-sample, and returns the OOS
        aggregate, walk-forward efficiency, IS->OOS degradation, per-fold summary,
        the holdout score (if holdout_days>0), the Deflated Sharpe (corrected for
        all configs tried), and the PROMOTION-GATE verdict (pass/fail per gate +
        overall "promotable"). A config advances only if it is promotable - never
        on in-sample Sharpe. include_pbo is expensive; leave it off unless needed.
        NET of transaction cost by default, in-sample and out (pass gross=True to
        disable) — gross validation systematically promotes turnover the
        strategy could not afford live.

        `benchmark` is a symbol to measure against (e.g. "SPY"). Without it every
        fold reports `benchmark_available: false`, information ratio 0 and no betas,
        so any benchmark-relative promotion prerequisite is left unevaluated rather
        than failed. `position_limits` is the book the config says it will trade
        (e.g. {"max_positions": 8}); without it the validation runs at whatever the
        strategy class declares, which is not what would be deployed.

        Journals the OOS aggregate as one validated trial. An identical prior
        validation (same recipe: mode/folds/method/objective/max_evals/seed/cost/
        book over the same window) is served instead (result has `memoized: true`)
        unless `force=True`.
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
        }
        result = analysis.run_walk_forward(
            dc,
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
        name, ship it in a package exposing the `tradeflow.strategies` entry-point
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
        exposing the `tradeflow.scanners` entry-point group.
        """
        inputs = {"class_name": class_name, "code_hash": analysis.draft_code_hash(code)}
        return _logged(
            "validate_draft_scanner_code",
            inputs,
            analysis.validate_draft_scanner_code(code, class_name=class_name),
        )

    @tool("sharpe_ratio", "deflated_sharpe_ratio", "profit_factor", "max_drawdown", notes=[_JOURNALING_NOTE])
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
        "TradeFlow can validate it" without putting proprietary code in this repo.
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
        not enter the campaign record. An identical prior draft validation is served
        from the trial store unless `force=true`.
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
            dc,
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
            dc,
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
            dc,
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
        """
        inputs = {"strategy": strategy, "symbols": symbols, "start": start, "end": end}
        result = analysis.compute_horizon(
            dc,
            strategy,
            symbols,
            _parse_date(start),
            _parse_date(end),
            source=source,
            benchmark=benchmark,
            max_lag=max_lag,
        )
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
        model: "shrinkage" = Ledoit–Wolf (default), "sample" = raw, "factor" =
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
            dc, symbols, _parse_date(as_of), model=model, timeframe=timeframe, lookback_days=lookback_days
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
        benchmark: str = "SPY",
        capital: Optional[float] = None,
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

        Scope note, so you do not assume more than this exposes: this tool solves the
        long-only, cash-relative book with the default cost assumptions. Factor
        neutralization, a portfolio-level benchmark, market-neutral books, and the
        evidence-gated features below are reachable from the command line but are not
        arguments here — `run_verdict` is the composite path with coherent defaults.
        """
        inputs = {"strategy": strategy, "symbols": symbols, "as_of": as_of, "target_te": target_te}
        result = analysis.construct_portfolio(
            dc,
            strategy,
            symbols,
            _parse_date(as_of),
            source=source,
            target_te=target_te,
            max_weight=max_weight,
            max_names=max_names,
            benchmark=benchmark,
            capital=capital,
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
            dc,
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

    @tool("deflated_sharpe_ratio", "information_ratio", notes=[_JOURNALING_NOTE, _GATED_NOTE])
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
        not promotable / needs more data / mixed / incomplete, assembled from the gates
        the steps themselves computed (IC t-stat vs 2, realized IR vs its own standard
        error, the sanity ceiling, expected active return net of cost) - every check is
        in `verdict.checks` with its value and threshold. A run where any step failed
        is `incomplete` and carries NO verdict, whatever the completed sections say:
        do not act on a partial run's weights.

        This answers "what does the pipeline say about this universe as of `end`" -
        a forecast and a proposed book. The scanner is resolved at `end`, not at
        wall-clock now, so a historical verdict does not mix today's universe into an
        older evaluation window. For "did this ever work", use `run_backtest` or
        `run_walk_forward`.

        Read-only research clock: proposes a portfolio, never places an order. Journals
        exactly ONE trial (kind `verdict`), so it counts once toward the campaign's
        multiple-testing total. An identical prior run is returned from the trial store
        with `memoized: true` and the original run's timestamp rather than re-run; pass
        `force=true` to re-verify and append a new trial.
        """
        inputs = {"strategy": strategy, "symbols": symbols, "start": start, "end": end}
        result = analysis.run_verdict(
            dc,
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
        n_trials) is a mandatory header, and memoized results, gate failures, a failed
        leakage probe, and any enabled evidence-gated feature render as warnings above
        the sections rather than as footnotes.

        This computes nothing: it is the same renderer the CLI's `--html` flag uses,
        over the same object. Charts need the plotting extra on the server; without it
        the tables still render and the chart slots say so.

        Note that a backtest result carries no equity curve (it is omitted from the
        payload deliberately), so a backtest report rendered here shows the metrics
        table without the equity chart.
        """
        from tradeflow.analytics.htmlreport import render_html

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
        all_accounting: bool = False,
    ) -> Dict[str, Any]:
        """What has this campaign already tried? Filtered rows from the trial store.

        Ask this BEFORE running anything expensive. Every backtest, optimization,
        walk-forward, and verdict ever run — over the CLI or over this server — is
        recorded here, and re-running something the campaign already answered costs a
        trial without buying information.

        `symbols` matches on the normalized universe, so order and case never change
        what is found. `sort` is date/sharpe/dsr; a trial with no recorded metric
        sorts last rather than as a zero. Returns `{rows, total}` — `total` is how
        many matched, which is usually more than were returned.

        Absent fields are null, and null means *not recorded*: a trial predating a
        field did not fail to record it, and a forecast kind did not score zero.
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
            rows = store.list_trials(sort=sort, limit=limit, offset=offset, **filters)
            return _logged("list_trials", inputs, {"rows": rows, "total": store.count_trials(**filters)})

    @tool()
    def get_trial(trial_id: str) -> Dict[str, Any]:
        """Everything the trial store knows about one trial.

        Its full params (the run's dedup identity, including the folded cost and
        data-vintage keys), provenance, headline metrics, and what was stored
        alongside it: whether its out-of-sample return series was persisted (and over
        what window), the book it proposed, its trade table if the run opted into
        keeping one, and which later trials were served from this one.

        `null` for a companion means *not recorded* — the run predates that storage,
        or did not opt in. It never means the run produced nothing. Read-only.
        """
        with _trial_store() as store:
            if store is None:
                return _logged("get_trial", {"trial_id": trial_id}, {"error": _NO_STORE})
            trial = store.get_trial(trial_id)
            if trial is None:
                return _logged(
                    "get_trial",
                    {"trial_id": trial_id},
                    {"error": f"No trial with id {trial_id!r}. Use list_trials to see what exists."},
                )
            return _logged("get_trial", {"trial_id": trial_id}, trial)

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

        In-sample kinds (`optimize`, `alpha`) are excluded by default: an `optimize`
        row is the winner of a search, best-of-N by construction, so ranking one
        ranks the selection bias rather than any skill. `include_in_sample=true`
        opts back in, and `in_sample_excluded` reports how many were dropped.

        Returns `{rank_by, rows, max_family_n_trials, in_sample_included,
        in_sample_excluded, caveat}`. The caveat and the per-row family counts are
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
            "summarize_bars", inputs, analysis.summarize_bars(dc, symbols, timeframe, lookback_days)
        )

    # ---------------- Propose (writes a file, never live state) ---------------- #
    @tool()
    def save_config(
        name: str,
        strategy: str,
        params: Dict[str, Any],
        scanner: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Propose a candidate config by writing configs/<name>.json, with provenance.

        This is the end of what automation may do. It writes a file for a HUMAN to
        read, judge, and promote; it does not affect any running process, does not
        change any default, and cannot enable live trading. Promotion is a manual
        step by design — the research clock proposes, a person disposes.

        Include the evidence in `provenance` (the walk-forward or verdict run behind
        the proposal, its trial id, its window and universe), because a config with
        no recorded reason for existing is a config nobody can safely promote.
        """
        inputs = {"name": name, "strategy": strategy, "params": params, "scanner": scanner}
        result = configs.save_config(
            name, strategy=strategy, params=params, scanner=scanner, provenance=provenance
        )
        return _logged("save_config", inputs, result)

    @tool()
    def load_config(name: str) -> Dict[str, Any]:
        """Read back one saved candidate config: its strategy, params, and provenance.

        Reading a config does not activate it — nothing here is running the config,
        and loading it changes no behavior anywhere. Use it to check what a past
        proposal actually contained before re-testing or superseding it. Read-only.
        """
        return _logged("load_config", {"name": name}, configs.load_config(name))

    @tool()
    def list_configs() -> List[Dict[str, Any]]:
        """List every saved candidate config with a compact summary of each.

        These are proposals awaiting human review, not active settings: none of them
        is in use by anything until a person promotes it. Read-only.
        """
        return _logged("list_configs", {}, configs.list_configs())

    return server


def serve() -> None:
    """Build the server and run it over stdio (what Claude Desktop/Code launch)."""
    # Said at startup, not on request: an agent driving these tools cannot see
    # where its evidence is landing, and every journaled trial is permanent.
    from tradeflow.settings import git_worktree_containing, state_root

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

    logger.info("Starting TradeFlow MCP server (stdio). Live trading is NOT exposed.")
    build_server().run()


def _assert_no_trading_client(data_client) -> None:
    """Guardrail: the MCP process must hold only a data client."""
    from tradeflow.marketdata.client import MarketDataClient

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
