"""Command-line entry point.

A thin adapter that wires the layers together per command - all the real work
lives in ``tradeflow/`` (and the shared service core in ``tradeflow/services/``, so the CLI,
the MCP server, and the research agent run the same code):

    demo         run the whole pipeline on synthetic data (no keys, no network)
    scan         run the universe scanner and print flagged symbols
    backtest     scan universe -> BacktestEngine -> performance report
    live         scan universe -> LiveEngine -> LiveTrader (paper/live orders)
    optimize     search a strategy's parameters by backtest objective
    allocate     weight a portfolio across scanned symbols (OR-Tools)
    alphas       rank a universe by continuous alpha (residual-return forecast)
    risk         estimate the universe covariance Σ and summarize its risk structure
    info         information report: IC, breadth, predicted-vs-realized IR
    horizon      measure alpha decay / half-life; recommend cadence + lagged blend
    walkforward  out-of-sample validation with promotion gates
    mcp          serve TradeFlow to an agent over MCP (read-only)
    research     autonomous, offline research loop -> shortlist of configs
    trials       inspect the trial store: campaign n_trials, recent trials, drift check
    cache        inspect/warm the persistent bar cache (backtest/optimize/walkforward --cache)

Run ``python main.py <command> --help`` for options, or use the Makefile targets
for preconfigured combos (``make demo``, ``make backtest``, ...).
"""

import argparse
import asyncio
import contextlib
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradeflow.marketdata.base import MarketDataProvider
from tradeflow.services.registry import STRATEGIES
from tradeflow.settings import DATA_FEEDS
from tradeflow.utils.logging_config import setup_logging
from tradeflow.utils.timeutils import NEW_YORK

logger = logging.getLogger(__name__)

# A reasonable default candidate list for the scanner to filter.
DEFAULT_UNIVERSE = ["NVDA", "RIVN", "NFLX", "META", "BAC", "MS", "TSLA", "GS", "AMD", "AAPL"]


# ---------------------------------------------------------------------------- #
# Wiring
# ---------------------------------------------------------------------------- #
def build_data_and_broker(
    cache: bool = False,
    offline: bool = False,
    cache_dir: Optional[Any] = None,
    feed: Optional[str] = None,
):
    """Construct the Alpaca-backed broker and market-data client from settings.

    ``cache``/``offline``/``cache_dir`` are forwarded to
    :func:`tradeflow.services.data.build_data_client`, which owns the actual provider
    construction (and its opt-in bar-cache wrapping) - kept in one place so the
    CLI and the read-only MCP/research path never diverge on how a data client
    gets built.
    """
    from tradeflow.brokers.alpaca.factory import build_broker
    from tradeflow.services.data import build_data_client
    from tradeflow.settings import load_settings

    settings = load_settings()
    broker = build_broker(settings.alpaca_key, settings.alpaca_secret, settings.paper_trade)
    data_client = build_data_client(cache=cache, offline=offline, cache_dir=cache_dir, feed=feed)
    return broker, data_client


def resolve_universe(
    data_client, scanner_name: Optional[str], candidates: List[str], as_of: Optional[datetime] = None
) -> List[str]:
    """Filter ``candidates`` through the scanner, falling back to them if none flag.

    Delegates to the shared service core so the CLI and MCP server use one path.
    """
    from tradeflow.services.data import resolve_universe as _resolve

    return _resolve(data_client, scanner_name, candidates, as_of=as_of)


def build_cost_model(args):
    """Build the parametric cost model from ``--gross``/``--commission-bps``/etc.,
    or ``None`` for ``--gross``.

    Shared by backtest/optimize/walkforward so a search or validation prices trades
    the same way a live backtest does. Gross-by-default optimization silently
    favors the highest-turnover config (see ``ParameterOptimizer``'s and
    ``WalkForwardValidator``'s own docstrings), so this must reach every
    research-clock entrypoint that can run a search or a validation, not just
    ``backtest``.
    """
    from tradeflow.costs import ParametricCostModel

    if args.gross:
        return None
    return ParametricCostModel(
        commission_bps=args.commission_bps, impact_eta=args.impact_eta, annual_borrow_bps=args.borrow_bps
    )


def _cost_key(args, vintage: Optional[str] = None) -> Dict[str, Any]:
    """The cost-model assumptions (plus, when given, a data-vintage stamp) folded
    into a trial's dedup key (see :func:`_dedup_params`) - two runs differing only
    in ``--commission-bps`` (or any other cost flag) must never collide as "the
    same trial", and - once the bar cache is in play - neither must two runs whose
    underlying data actually differs (a corporate-action backfill since the
    original run). ``vintage`` is omitted whenever the data client isn't
    cache-backed (see :func:`_vintage_stamp`), so a non-cached run's dedup hash is
    byte-identical to before this existed - a non-breaking addition."""
    key = (
        {"gross": True}
        if args.gross
        else {
            "gross": False,
            "commission_bps": args.commission_bps,
            "impact_eta": args.impact_eta,
            "borrow_bps": args.borrow_bps,
        }
    )
    if vintage is not None:
        key["_vintage"] = vintage
    return key


def _dedup_params(params: Dict[str, Any], args, vintage: Optional[str] = None) -> Dict[str, Any]:
    """``params`` plus the cost-model assumptions, folded under a reserved key so
    the dedup hash (and the journaled/displayed config) reflects everything that
    can change a trial's outcome, not just the strategy's own tunable params."""
    return {**params, "_cost": _cost_key(args, vintage)}


def _vintage_stamp(data_client, universe: List[str], timeframe: str, start: Any, end: Any) -> Optional[str]:
    """The bar cache's data-vintage stamp for this exact fetch, or ``None`` when
    the data client isn't cache-backed (today's behavior - no vintage guarantee).

    Calling this ensures ``[start, end]`` is cached for ``universe`` (see
    :meth:`~tradeflow.store.bars.CachedMarketData.vintage_stamp`) - it warms the cache
    as a side effect, which is why callers compute it once, up front, and reuse
    the same value for both the memoization lookup and the eventual record: the
    two must use an identical dedup key or a matching prior trial would never be
    found.
    """
    from tradeflow.store.bars import CachedMarketData

    provider = getattr(data_client, "provider", None)
    if not isinstance(provider, CachedMarketData):
        return None
    return provider.vintage_stamp(universe, timeframe, start, end)


@contextlib.contextmanager
def _open_trial_store(journal_path: Optional[Any] = None):
    """A trial store against ``journal_path`` (default: the current
    ``audit.DEFAULT_TRIAL_JOURNAL``), or ``None`` on any failure to open one.

    v1 of the trial store is passive and derived (see ``tradeflow.store.trials``): a
    broken store must never break the command it's attached to, memoization
    included - every caller here treats ``None`` as "skip memoization, run
    normally," never as an error to propagate.
    """
    from tradeflow.services import audit
    from tradeflow.store.trials import TrialStore, db_path_for_journal

    path = journal_path or audit.DEFAULT_TRIAL_JOURNAL
    try:
        store = TrialStore(db_path_for_journal(path), journal_path=path)
    except Exception:  # noqa: BLE001
        logger.warning("Trial store unavailable; memoization disabled for this run", exc_info=True)
        yield None
        return
    try:
        yield store
    finally:
        store.close()


def _find_cached_trial(
    strategy: str,
    params: Dict[str, Any],
    symbols: List[str],
    start: Any,
    end: Any,
    accounting: int,
) -> Optional[Dict[str, Any]]:
    """Look up an exact prior trial via the trial store; ``None`` if none exists
    (including when the store itself is unavailable - see :func:`_open_trial_store`).
    """
    from tradeflow.optimization.config_store import current_git_sha

    with _open_trial_store() as store:
        if store is None:
            return None
        return store.find(
            strategy=strategy,
            params=params,
            symbols=symbols,
            window_start=start,
            window_end=end,
            accounting=accounting,
            git_sha=current_git_sha(),
        )


def _write_html(args, result: Dict[str, Any], kind: str, extras: Optional[Dict[str, Any]] = None) -> None:
    """Write the ``--html`` report when the flag was given, else do nothing.

    Renders the dict the command already produced - never a second computation, so
    the file and the terminal can never disagree. A rendering failure is reported
    and swallowed: the report is an artifact of the run, and losing it must not
    lose the run.
    """
    path = getattr(args, "html", None)
    if not path:
        return
    from tradeflow.analytics.htmlreport import write_html

    try:
        print(f"HTML report written to {write_html(result, kind, path, extras=extras)}")
    except (OSError, ValueError) as exc:
        print(f"HTML report skipped: {exc}")


def _flags_given(parser, argv: List[str], command: Optional[str]) -> set:
    """Which options the user actually typed, as argparse dests.

    Needed because a saved config fills in what the command line left unsaid, and
    "left unsaid" cannot be inferred from the parsed value: a flag passed explicitly
    with its default value is indistinguishable from one omitted. Reading the tokens
    is the only way to tell, and getting it wrong means a config silently overriding
    something the user typed.
    """
    sub = parser
    if command is not None:
        choices = parser._subparsers._group_actions[0].choices
        sub = choices.get(command, parser)
    by_option = {opt: action.dest for action in sub._actions for opt in action.option_strings}
    given = set()
    for token in argv:
        if token.startswith("-") and (dest := by_option.get(token.split("=", 1)[0])) is not None:
            given.add(dest)
    return given


def parse_cli(argv: Optional[List[str]] = None):
    """Parse argv and record which flags were typed, for config layering."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.flags_given = _flags_given(parser, argv, getattr(args, "command", None))
    return args


def _load_strategy_from_config(path: str):
    """Load a saved config and construct the strategy directly from its params.

    Returns ``(strategy, strategy_name, scanner_name)``.

    Construction goes through ``Strategy.__init__`` exactly as
    ``create_with_defaults()`` does, so out-of-range/unrecognized params raise
    loudly (:meth:`Strategy._validate_parameters`) rather than silently
    trading on a config an older strategy version can no longer honor.
    """
    from tradeflow.optimization.config_store import load_config
    from tradeflow.services.registry import resolve_strategy_class

    payload = load_config(path)
    strategy_name = payload["strategy"]
    cls = resolve_strategy_class(strategy_name)
    strategy = cls(dict(payload.get("params") or {}))
    return strategy, strategy_name, payload.get("scanner")


def _settle_universe_replay(args, sources, given) -> None:
    """Replay the config's universe unless a re-resolution is explicitly asked for.

    A saved config means "this is the book we validated". Re-scanning it on use turns
    that into "a new book from an old recipe" - a different experiment, which can move
    results in either direction without the reader knowing the universe changed. Worse,
    the scanner would run over the *resolved* names, applying the filter a second time
    rather than repeating the original decision.

    Replay is expressed by pinning the scanner off, so every command that resolves a
    universe - directly or through a service - honours it identically instead of each
    growing its own replay branch.
    """
    if args.universe_source != "config":
        if getattr(args, "re_resolve_universe", False) and args.universe_source == "flag":
            # The collision: --symbols already replaced the saved book, so there is
            # nothing left to re-resolve. Saying so beats letting a flag look honoured.
            sources.append("universe=<--symbols given; --re-resolve-universe has nothing to re-resolve>")
        return  # no saved universe in play: behave exactly as before

    if getattr(args, "re_resolve_universe", False):
        candidates = args.candidate_symbols
        if candidates:
            args.symbols = list(candidates)
            sources.append(f"universe=<re-resolved from {len(candidates)} saved candidates>")
        else:
            sources.append("universe=<re-resolved over the saved book; no candidates recorded>")
        return

    if "scanner" in given and getattr(args, "scanner", None) not in (None, "none"):
        # An explicitly typed scanner is a decision too, and flags win. Say that the
        # two instructions disagreed rather than silently honouring one of them.
        sources.append(f"universe=<--scanner {args.scanner} given; saved book re-scanned>")
        return
    if getattr(args, "scanner", None) not in (None, "none"):
        args.scanner = "none"
    sources.append(f"universe=<replayed from config, {len(args.symbols)} symbols>")


def _strategy_from(args, tuned):
    """The strategy a run should use, from the resolved name and any tuned params.

    Construction goes through ``Strategy.__init__`` exactly as
    ``create_with_defaults()`` does, so params an older strategy can no longer honour
    raise loudly rather than trading on a config it cannot actually run.
    """
    cls = STRATEGIES[args.strategy]
    strategy = cls(dict(tuned)) if tuned else cls.create_with_defaults()
    limits = getattr(args, "config_position_limits", None)
    if limits:
        # After construction: `position_limits` is not a tunable parameter, so it does
        # not go through PARAM_RANGES validation, and a strategy built from defaults
        # would otherwise silently keep the defaults the file exists to override.
        strategy.config["position_limits"] = {**strategy.position_limits(), **limits}
    return strategy


def apply_run_config(args):
    """Layer a saved config under the command line; explicit flags win.

    One file configures a run whatever its type, so this is shared by every command
    that accepts ``--config`` rather than reimplemented per command. Only fields the
    command actually has are filled - ``risk`` takes a universe but no strategy - so
    a config saved from a walk-forward is usable everywhere without being meaningful
    everywhere.

    Returns the config's tuned params (``{}`` when there is no ``--config``), because
    the analysis services take a strategy *name* plus a params overlay rather than a
    constructed strategy. A command that ignores that return would run the config's
    universe with the strategy's *default* params while reporting that it loaded the
    config, which is the failure this whole surface exists to avoid.

    A ``--strategy`` that contradicts the config is refused rather than resolved. The
    params in the file belong to the strategy in the file, and quietly handing one
    strategy's tuned params to another is not an outcome worth guessing at.
    """
    from tradeflow.optimization.config_store import load_config

    if not getattr(args, "config", None):
        return {}

    given = getattr(args, "flags_given", None)
    if given is None:
        # Parsed without parse_cli - a direct parse_args, as tests and embedders do.
        # Fall back to comparing against the parser's own defaults. Less precise than
        # reading the tokens (a flag typed with its default value reads as absent), but
        # it is the only signal available, and it errs toward letting the file fill in
        # rather than inventing a contradiction the user never wrote.
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices
        sub = choices.get(getattr(args, "command", ""), parser)
        given = {
            action.dest
            for action in sub._actions
            if action.dest != "help" and getattr(args, action.dest, None) != sub.get_default(action.dest)
        }

    payload = load_config(args.config)
    name = payload["strategy"]
    if "strategy" in given and getattr(args, "strategy", name) != name:
        raise SystemExit(
            f"--config {args.config} holds strategy {name!r} and its tuned params, but "
            f"--strategy {args.strategy!r} was given. Those params belong to {name!r}; "
            f"drop --strategy, or drop --config and tune {args.strategy!r} on its own."
        )

    sources = []
    if hasattr(args, "strategy"):
        sources.append(f"strategy={name!r}")
        # Construct it here purely to validate: params an older strategy can no longer
        # honour must fail now, not four steps into a pipeline.
        _load_strategy_from_config(args.config)
        args.strategy = name

    for field, flag in (("scanner", "--scanner"), ("symbols", "--symbols"), ("capital", "--capital")):
        if not hasattr(args, field):
            continue
        value = payload.get(field)
        if field in given:
            sources.append(f"{field}=<{flag} given>")
        elif value is not None:
            setattr(args, field, value)
            sources.append(f"{field}=<config>")

    # Limits recorded in the file win over the strategy class's defaults, because the
    # file is what was validated. Merged rather than replaced so a config written before
    # this still gets the defaults for keys it never recorded.
    limits = payload.get("position_limits")
    if limits:
        tuned_limits = dict(limits)
        sources.append("position_limits=<config>")
    else:
        tuned_limits = None
    args.config_position_limits = tuned_limits

    args.candidate_symbols = payload.get("candidate_symbols")
    args.universe_source = "flag" if "symbols" in given else ("config" if payload.get("symbols") else None)
    _settle_universe_replay(args, sources, given)

    cost = payload.get("cost") or {}
    applied_cost = [
        field
        for field in ("gross", "commission_bps", "impact_eta", "borrow_bps")
        if hasattr(args, field) and field not in given and field in cost
    ]
    for field in applied_cost:
        setattr(args, field, cost[field])
    if applied_cost:
        sources.append("cost=<config>")

    print(f"Config {args.config}: {', '.join(sources) or 'nothing this command can use'}")
    # The window is never stored - see save_config - so it is always this run's own,
    # and saying so is what stops a reader assuming the config pinned it.
    if hasattr(args, "start") and hasattr(args, "end"):
        print(f"  window {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d} (from this run, not the config)")
    return dict(payload.get("params") or {})


# ---------------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------------- #
def cmd_backtest(args) -> None:
    import json

    from tradeflow.analytics.reporting import (
        format_backtest_report,
        format_cached_notice,
        log_backtest_report,
    )
    from tradeflow.engine.backtest import ACCOUNTING_VERSION, BacktestEngine
    from tradeflow.services.sizing import build_beta_sizer

    tuned = apply_run_config(args)
    strategy_name = args.strategy
    strategy = _strategy_from(args, tuned)

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    universe = resolve_universe(data_client, args.scanner, args.symbols, as_of=args.scan_as_of or args.end)

    # Computed once, up front: it warms the cache as a side effect (when
    # cache-backed) and must match exactly between the lookup below and the
    # eventual journal_trial() record, or a matching prior trial would never be
    # found - see _vintage_stamp's own docstring.
    vintage = _vintage_stamp(data_client, universe, strategy.config["timeframe"], args.start, args.end)
    tunable = {k: strategy.config[k] for k in strategy.PARAM_RANGES if k in strategy.config}
    dedup_params = _dedup_params(tunable, args, vintage)

    if not args.force:
        cached = _find_cached_trial(
            strategy_name, dedup_params, universe, args.start, args.end, ACCOUNTING_VERSION
        )
        if cached is not None:
            print(
                format_cached_notice(
                    cached, current_accounting=ACCOUNTING_VERSION, vintage_available=vintage is not None
                )
            )
            metrics = json.loads(cached["metrics_json"])
            final_capital = args.capital * (1.0 + metrics.get("total_return", 0.0) / 100.0)
            print(format_backtest_report(metrics, args.capital, final_capital, title="Backtest Results"))
            _write_html(
                args,
                {
                    "strategy": strategy_name,
                    "symbols": universe,
                    "window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
                    "initial_capital": args.capital,
                    "final_capital": final_capital,
                    "gross": args.gross,
                    "metrics": metrics,
                    "memoized": True,
                    "trial_id": cached["id"],
                    "trial_ts": cached["ts"],
                },
                "backtest",
            )
            return

    sizer = None
    if args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark, as_of=args.start)

    # Metrics are net of transaction cost by default; --gross disables the charge.
    cost_model = build_cost_model(args)
    result = BacktestEngine(strategy, data_client, sizer=sizer, cost_model=cost_model).run(
        universe, args.start, args.end, args.capital, benchmark=args.benchmark
    )

    if not args.no_journal:
        from tradeflow.analytics.metrics import returns_from_equity
        from tradeflow.analytics.performance import build_dated_equity_curve
        from tradeflow.services.audit import journal_trial

        # Persist this trial's own dated return series (daily-resampled,
        # from realized trade P&L - the same construction every persisted trial
        # kind uses) so it can later join a Reality Check family panel.
        dated_equity = build_dated_equity_curve(result.trades, args.capital)
        returns_series = returns_from_equity(dated_equity) if not dated_equity.empty else None
        from tradeflow.services.analysis import trades_payload

        journal_trial(
            "backtest",
            strategy=strategy_name,
            symbols=universe,
            candidate_symbols=args.symbols,
            start=args.start,
            end=args.end,
            params=dedup_params,
            metrics=result.metrics,
            returns=returns_series,
            # Opt-in: a campaign's worth of trade tables is storage nobody asked
            # for, so only a run you intend to inspect keeps one.
            trades=trades_payload(result.trades) if args.record_trades else None,
        )

    log_backtest_report(
        result.metrics,
        result.initial_capital,
        result.final_capital,
        execution=result.execution,
        legs=result.legs,
    )
    _print_universe_provenance(args, universe)
    _print_verdicts_for_backtest(result)
    if getattr(args, "cost_stress", False):
        _print_cost_stress(data_client, strategy_name, universe, args, tuned)
    if not args.gross and result.total_cost:
        print(
            f"Transaction cost: ${result.total_cost:,.2f} "
            f"({result.total_cost / result.initial_capital * 100:.2f}% of capital); "
            f"gross final ${result.gross_final_capital:,.2f}"
        )

    if getattr(args, "chart", None):
        from tradeflow.analytics.charts import render_backtest_chart

        try:
            path = render_backtest_chart(result, args.chart, title=f"{strategy_name} — backtest")
            print(f"Chart saved to {path}")
        except RuntimeError as exc:  # matplotlib (viz extra) not installed
            print(f"Chart skipped: {exc}")

    if getattr(args, "html", None):
        from tradeflow.services.analysis import backtest_payload
        from tradeflow.services.audit import new_run_id

        _write_html(
            args,
            backtest_payload(
                result,
                run_id=new_run_id(),
                strategy=strategy_name,
                symbols=universe,
                start=args.start,
                end=args.end,
                capital=args.capital,
                gross=args.gross,
                benchmark=args.benchmark,
            ),
            "backtest",
            # The equity curve is not in the result dict (it can run to tens of
            # thousands of points), but the CLI is holding it, so the chart can be
            # drawn here without changing what any surface returns.
            extras={"equity_curve": result.equity_curve},
        )

    _maybe_print_attribution_verdict(
        data_client, strategy_name, universe, args.start, args.end, args.benchmark
    )


def cmd_scan(args) -> None:
    from tradeflow.analytics.reporting import format_offline_scan_notice
    from tradeflow.scanners.symbol_scanner import SymbolScanner, resolve_scan_clock

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    # Before the result, not after it: a caveat under a list of symbols is a caveat
    # most readers have already scrolled past.
    if args.offline:
        print(format_offline_scan_notice(resolve_scan_clock(args.as_of)))
    flagged = SymbolScanner(data_client, args.scanner).scan(args.symbols, as_of=args.as_of)
    if not flagged:
        print("No symbols flagged.")
        return
    print(f"{'SYMBOL':10}SIGNAL")
    for symbol, signal in flagged:
        print(f"{symbol:10}{signal}")
    if getattr(args, "drift", False):
        _print_scanner_drift(data_client, args)


def _print_scanner_drift(data_client, args) -> None:
    """How much this universe moves when the scan clock moves.

    A config records the universe its scanner *resolved*, so an unstable scan means the
    book a deployment gets is not the book that was validated - and no promotion gate
    would notice, because the gates never see the scan twice.
    """
    from tradeflow.services.analysis import run_scanner_drift

    report = run_scanner_drift(data_client, args.scanner, args.symbols, args.as_of)
    print(f"\n=== Scanner drift ({report['baseline_size']} flagged at {report['as_of'][:10]}) ===")
    print(f"  {'clock':>10}{'flagged':>9}{'overlap':>9}{'turnover':>10}")
    for row in report["comparisons"]:
        print(f"  {row['offset_days']:>9}d{row['size']:>9}{row['overlap']:>9}{row['turnover_pct']:>9.1f}%")
    worst = report["max_turnover_pct"]
    print(
        f"  Universe turnover peaks at {worst:.1f}% across these clocks."
        if worst
        else "  Universe is identical across these clocks."
    )


def cmd_allocate(args) -> None:
    # The scalar-weight path scores no strategy, so only the config's universe,
    # scanner and capital reach it; the utility path can use the tuned params too.
    tuned = apply_run_config(args)

    if getattr(args, "objective", "weights") == "utility":
        _allocate_utility(args, tuned)
        return

    from tradeflow.scanners.symbol_scanner import SymbolScanner
    from tradeflow.services.sizing import allocate_portfolio

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    scanner = SymbolScanner(data_client, args.scanner)
    flagged = [symbol for symbol, _ in scanner.scan(args.symbols)]
    if not flagged:
        print("No symbols flagged; nothing to allocate.")
        return

    allocations = allocate_portfolio(
        data_client, flagged, scanner.timeframe, args.capital, args.max_positions, args.max_weight
    )
    if not allocations:
        print("Solver produced no allocation (no positively-scored candidates).")
        return

    print(f"{'SYMBOL':10}{'WEIGHT':>8}{'DOLLARS':>14}{'SHARES':>10}")
    for a in allocations:
        print(f"{a.symbol:10}{a.weight:>7.1%}{a.dollars:>14,.2f}{a.shares:>10.0f}")


def _allocate_utility(args, tuned=None) -> None:
    """Mean-variance portfolio construction (alpha + Σ) — a read-only proposal."""
    from tradeflow.services.analysis import construct_portfolio, longshort_report

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    book = args.book.replace("-", "_")

    if args.longshort_report:
        report = longshort_report(
            data_client,
            args.strategy,
            args.symbols,
            args.as_of,
            config=tuned,
            source=args.source,
            scanner=args.scanner,
            target_te=args.target_te,
            max_weight=args.max_weight,
            benchmark=args.benchmark,
            neutralize_factors=args.neutralize_factors,
            capital=args.capital,
            holding_period_years=args.holding_period,
            cost_aware=not args.gross_objective,
            gross_leverage=args.gross_leverage or 2.0,
            short_max_weight=args.short_max_weight,
        )
        _print_longshort_report(report, args)
        return

    result = construct_portfolio(
        data_client,
        args.strategy,
        args.symbols,
        as_of=args.as_of,
        config=tuned,
        source=args.source,
        scanner=args.scanner,
        target_te=args.target_te,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        max_names=args.max_names,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
        capital=args.capital,
        holding_period_years=args.holding_period,
        cost_aware=not args.gross_objective,
        benchmark_holdings=args.benchmark_holdings,
        benchmark_premium=args.benchmark_premium,
        book=book,
        gross_leverage=args.gross_leverage,
        short_max_weight=args.short_max_weight,
        conditional=getattr(args, "conditional", None),
        conditional_lambda=getattr(args, "conditional_lambda", None),
        posterior=getattr(args, "posterior", None),
        posterior_ic=getattr(args, "posterior_ic", None),
        posterior_t_eff=getattr(args, "posterior_t_eff", None),
        posterior_tau=getattr(args, "posterior_tau", None),
        policy=getattr(args, "policy", None),
        trade_rate=getattr(args, "trade_rate", None),
    )
    if not result["feasible"]:
        print(f"Infeasible: {result.get('binding_constraint') or result.get('note')}")
        return

    d = result["diagnostics"]
    mode = "cost-aware" if d.get("cost_aware") else "gross (cost-blind)"
    te_label = "active TE (vs w_B)" if d.get("has_benchmark") else "predicted TE"
    print(
        f"\nPortfolio for '{args.strategy}' as of {args.as_of:%Y-%m-%d} "
        f"(target TE {args.target_te:.0%}, {mode})"
    )
    print(
        f"  IR* {d['ir_star']:.2f}  {te_label} {d['predicted_tracking_error']:.1%}  "
        f"predicted IR {d['predicted_ir']:.2f}  transfer coef {d['transfer_coefficient']:.2f}  "
        f"turnover {d['turnover']:.1%}"
    )
    bp = result.get("benchmark_portfolio")
    if bp:
        print(
            f"  benchmark '{bp['source']}': coverage {bp['coverage']:.0%}  "
            f"active beta {d['active_beta']:+.2f}  residual risk {d['residual_risk']:.1%}  "
            f"(ψ² = β_a²σ_B² + ω²: {d['active_beta'] ** 2 * d['benchmark_variance']:.5f} "
            f"+ {d['residual_risk'] ** 2:.5f} = {d['predicted_tracking_error'] ** 2:.5f})"
        )
        if bp["uncovered_weight"] > 1e-6:
            print(
                f"  ! {bp['uncovered_weight']:.0%} of the benchmark file's weight isn't in the "
                "trading universe — dropped and renormalized away, not reflected in TE"
            )
        if d.get("self_benchmark_warning"):
            print("  ! current holdings ≈ benchmark — tracking error measures distance from yourself")
        vai = bp.get("value_added_identity")
        if vai:
            print(
                f"  value added: SR_B {vai['sr_benchmark']:.2f}  IR {vai['ir']:.2f}  "
                f"=>  SR_P ≈ {vai['sr_portfolio_predicted']:.2f}  (SR_P² ≈ SR_B² + IR², predicted)"
            )
        print(
            "  consensus returns (mu_B={:.0%}/yr; the zero-skill baseline your alpha deviates from):".format(
                bp["premium"]
            )
        )
        for sym, mu in sorted(bp["consensus_returns"].items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"    {sym:10}mu {mu:+.2%}")
    if d.get("book") == "market_neutral":
        print(
            f"  market-neutral: Σw residual {d['dollar_neutral_residual']:.2e}  "
            f"gross leverage {d['gross_leverage']:.2f} / cap {d['gross_leverage_cap']:.2f}  "
            f"borrow cost {d['borrow_cost']:.2%}/yr"
        )
        if not bp:
            print("  ! no benchmark supplied — dollar-neutral is enforced, beta-neutrality is unverified")
    if "round_trip_cost" in d:
        # Headline: the conservative round-trip haircut (enter + exit, amortized) - the
        # same cost model as capacity, so the two agree.
        print(
            f"  net active return {d['expected_active_return_net']:.2%}/yr (round-trip)  "
            f"= gross {d['expected_active_return']:.2%} − round-trip cost {d['round_trip_cost']:.2%}"
        )
        # Detail: the one-way cost of this rebalance's turnover (prior reporting).
        if d.get("cost_aware"):
            print(
                f"    this rebalance: turnover cost {d['cost_drag']:.2%}/yr one-way "
                f"(linear {d['linear_cost']:.2%} + √-impact {d['impact_cost']:.2%}); "
                f"one-way net {d['expected_active_return_net_oneway']:.2%}"
            )
        else:
            print(
                f"    this rebalance: turnover cost {d['cost_drag']:.2%}/yr one-way; "
                f"one-way net {d['expected_active_return_net_oneway']:.2%}"
            )
    if "capacity_capital" in d:
        print(f"  capacity ≈ ${d['capacity_capital']:,.0f} (where √-impact erases the alpha)")
    if "sigma_regime" in result:
        regime = result["sigma_regime"]
        print(
            f"  conditional Σ ({regime['method']}, λ={regime['lambda']:.2f}): "
            f"mean σ_t/σ_unconditional = {regime['mean_sigma_regime']:.2f}"
        )
    if d.get("policy") == "aim":
        pr = result.get("policy_report") or {}
        if d.get("aim_degraded"):
            print(f"  policy 'aim': degraded to the plain myopic solve — {d.get('fallback_reason')}")
        else:
            print(
                f"  policy 'aim': κ {d['kappa']:.3f}"
                + (" (overridden)" if d.get("kappa_overridden") else f" (derived {d['kappa_derived']:.3f})")
                + f"  trading half-life {d['trading_half_life_rebalances']:.1f} rebalances  "
                f"φ {d['phi_per_rebalance']:.3f}/rebalance  discount {d['signal_discount']:.2f}"
            )
            hl = pr.get("decay_half_life_bars")
            hl_upper = pr.get("decay_half_life_upper_bars")
            hl_str = f"{hl:.1f}" if isinstance(hl, (int, float)) else str(hl)
            hl_upper_str = f"{hl_upper:.1f}" if isinstance(hl_upper, (int, float)) else str(hl_upper)
            if hl is not None:
                print(
                    f"    decay half-life {hl_str} bars (upper CI bound {hl_upper_str}, "
                    f"fit R² {pr.get('decay_r_squared') or 0.0:.2f}, used conservatively)"
                )
    post = result.get("posterior")
    if post:
        propagated = sum(1 for row in post["per_name"] if row["source"] == "propagated")
        print(
            f"  Black-Litterman posterior (IC {post['ic']:.3f}, T_eff {post['t_eff']:.0f}, "
            f"tau {post['tau']:.4f}): {propagated} name(s) propagated from correlated views"
        )
        for row in sorted(post["per_name"], key=lambda r: abs(r["posterior_mu"]), reverse=True)[:10]:
            pi = f"{row['consensus_pi']:+.2%}" if row["consensus_pi"] is not None else "  —  "
            q = f"{row['view_q']:+.2%}" if row["view_q"] is not None else "  —  "
            print(
                f"    {row['symbol']:10}pi {pi}  q {q}  mu_post {row['posterior_mu']:+.2%}  ({row['source']})"
            )
    print(f"\n{'SYMBOL':10}{'WEIGHT':>8}" + (f"{'DOLLARS':>14}{'SHARES':>10}" if result["holdings"] else ""))
    if result["holdings"]:
        for h in result["holdings"]:
            print(f"{h['symbol']:10}{h['weight']:>7.1%}{h['dollars']:>14,.2f}{h['shares']:>10.0f}")
    else:
        for sym, w in result["weights"].items():
            print(f"{sym:10}{w:>7.1%}")


def _print_longshort_report(report, args) -> None:
    """The same alphas/Σ/costs solved long-only vs market-neutral."""
    if not report["feasible"]:
        print(f"Infeasible: {report.get('note')}")
        return
    print(f"\nLong-only price report for '{args.strategy}' as of {args.as_of:%Y-%m-%d}")
    print(
        f"  IR_LS (long/short) {report['ir_long_short']:.2f}   IR_LO (long-only) {report['ir_long_only']:.2f}   "
        f"shrinkage {report['shrinkage_measured']:.2f}  "
        f"(reference line ≈ {report['shrinkage_reference_curve']:.2f})"
    )
    print(
        f"  transfer coefficient: long-only {report['transfer_coefficient_long_only']:.2f}   "
        f"long/short {report['transfer_coefficient_long_short']:.2f}"
    )
    lo_size, ls_size = report["size_exposure_long_only"], report["size_exposure_long_short"]
    if lo_size is not None:
        print(f"  size exposure: long-only {lo_size:+.3f}   long/short {ls_size:+.3f}")
    print(
        f"  long-only binding fraction {report['binding_fraction']:.0%}   "
        f"long/short gross leverage {report['gross_leverage']:.2f}   "
        f"Σw residual {report['dollar_neutral_residual']:.2e}   "
        f"borrow cost {report['borrow_cost']:.2%}/yr"
    )
    print(f"  note: {report['note']}")


def _worker_data_spec(args):
    """How worker processes should build their own data clients, or ``None`` for a
    sequential run.

    Parallel execution is cache-backed by construction: a live data client cannot be
    pickled to a spawned worker, and N workers independently fetching identical bars
    is strictly worse than one warmed local cache. Asking for workers therefore opts
    into the bar cache whether or not `--cache` was passed, and says so.
    """
    from tradeflow.optimization.parallel import DataSpec, resolve_workers

    if resolve_workers(getattr(args, "workers", None)) <= 1:
        return None
    if not args.cache and not args.offline:
        print("--workers implies the bar cache (workers read local Parquet, not the API).")
    return DataSpec(kind="cache", cache_dir=args.cache_dir, offline=args.offline)


def cmd_optimize(args) -> None:
    from tradeflow.engine.backtest import ACCOUNTING_VERSION
    from tradeflow.optimization.optimizer import ParameterOptimizer

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    universe = resolve_universe(data_client, args.scanner, args.symbols, as_of=args.scan_as_of or args.end)
    timeframe = STRATEGIES[args.strategy].create_with_defaults().config["timeframe"]
    vintage = _vintage_stamp(data_client, universe, timeframe, args.start, args.end)
    # Workers, when asked for, read bars from the local cache rather than the API —
    # warmed once here, before dispatch, so N cold workers cannot stampede the vendor
    # for the same ranges simultaneously.
    data_spec = _worker_data_spec(args)
    if data_spec is not None:
        from tradeflow.optimization.parallel import warm_for

        warm_for(data_spec, universe, timeframe, args.start, args.end)
    # Net of transaction cost by default; --gross searches on gross returns, which
    # reliably favors the highest-turnover config.
    with _open_trial_store() as trial_store:
        # Per-candidate memoization: a candidate this campaign already
        # scored is served from the store instead of re-simulated - real with
        # random sampling, or a resumed/extended search.
        optimizer = ParameterOptimizer(
            STRATEGIES[args.strategy],
            data_client,
            initial_capital=args.capital,
            cost_model=build_cost_model(args),
            trial_store=trial_store,
            strategy_name=args.strategy,
            cost_key=_cost_key(args, vintage),
            accounting=ACCOUNTING_VERSION,
            force=args.force,
            workers=args.workers,
            data_spec=data_spec,
            seed=args.seed,
        )

        if args.method == "grid":
            result = optimizer.grid_search(
                universe, args.start, args.end, args.objective, max_evals=args.max_evals
            )
        elif args.method == "random":
            result = optimizer.random_search(
                universe, args.start, args.end, args.objective, n_samples=args.max_evals
            )
        else:  # bayesian
            result = optimizer.optimize_bayesian(universe, args.start, args.end, args.objective)

    n_memoized = 0
    if not args.no_journal and not result.results.empty:
        from tradeflow.services.audit import journal_trial

        # Each evaluated config is a distinct trial — a 50-point search is 50 trials,
        # not one. Recording them per-config is what makes a campaign-level Deflated
        # Sharpe honest; the search columns are the params, the rest metrics.
        searchable = optimizer.space.searchable
        defaults = optimizer.space.defaults
        for row in result.results.to_dict("records"):
            if "_memoized_from" in row:
                # Already exists as its own trial; journaling it again here would
                # double-count the exact repeat this spec exists to stop.
                n_memoized += 1
                continue
            searched = {k: row[k] for k in searchable if k in row}
            metrics = {k: v for k, v in row.items() if k not in searchable}
            journal_trial(
                "optimize",
                strategy=args.strategy,
                symbols=universe,
                start=args.start,
                end=args.end,
                params={**defaults, **searched, "_cost": _cost_key(args, vintage)},
                metrics=metrics,
                objective=args.objective,
            )

    if n_memoized:
        print(f"{n_memoized}/{len(result.results)} candidate(s) reused from the trial store (not re-run)")
    print(f"\nBest {result.objective}: {result.best_score:.4f}")
    print(f"Best parameters: {result.best_params}")
    if not result.results.empty:
        result.results.to_csv(args.output, index=False)
        print(f"Full results written to {args.output}")


def cmd_walkforward(args) -> None:
    from tradeflow.analytics.reporting import format_cached_notice
    from tradeflow.engine.backtest import ACCOUNTING_VERSION
    from tradeflow.optimization.config_store import build_provenance, current_git_sha, save_config
    from tradeflow.optimization.walk_forward import WalkForwardValidator

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    universe = resolve_universe(data_client, args.scanner, args.symbols, as_of=args.scan_as_of or args.end)
    timeframe = STRATEGIES[args.strategy].create_with_defaults().config["timeframe"]
    vintage = _vintage_stamp(data_client, universe, timeframe, args.start, args.end)

    # Top-level memoization key: the *validation recipe*, not the chosen params —
    # those aren't known until the search runs. Same seed + same recipe + same
    # window is deterministic, so serving a prior result is honest, not a
    # shortcut.
    recipe = {
        "mode": args.mode,
        "n_folds": args.folds,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "embargo_days": args.embargo_days,
        "holdout_days": args.holdout_days,
        "method": args.method,
        "objective": args.objective,
        "max_evals": args.max_evals,
        "seed": args.seed,
        "_cost": _cost_key(args, vintage),
    }

    with _open_trial_store() as trial_store:
        if not args.force and trial_store is not None:
            cached = trial_store.find(
                strategy=args.strategy,
                params=recipe,
                symbols=universe,
                window_start=args.start,
                window_end=args.end,
                accounting=ACCOUNTING_VERSION,
                git_sha=current_git_sha(),
            )
            if cached is not None:
                print(
                    format_cached_notice(
                        cached, current_accounting=ACCOUNTING_VERSION, vintage_available=vintage is not None
                    )
                )
                _print_cached_walkforward(cached)
                return

        # Net of transaction cost by default; --gross validates on gross returns, which
        # systematically promotes turnover the strategy could not afford live.
        # Per-candidate memoization (a fold's IS search) is threaded through the
        # same trial store as top-level.
        data_spec = _worker_data_spec(args)
        if data_spec is not None:
            from tradeflow.optimization.parallel import warm_for

            warm_for(data_spec, universe, timeframe, args.start, args.end)
        validator = WalkForwardValidator(
            STRATEGIES[args.strategy],
            data_client,
            initial_capital=args.capital,
            seed=args.seed,
            cost_model=build_cost_model(args),
            trial_store=trial_store,
            strategy_name=args.strategy,
            cost_key=_cost_key(args, vintage),
            accounting=ACCOUNTING_VERSION,
            force=args.force,
            workers=args.workers,
            data_spec=data_spec,
            benchmark=getattr(args, "benchmark", None),
        )
        result = validator.run(
            universe,
            args.start,
            args.end,
            mode=args.mode,
            n_folds=args.folds,
            train_days=args.train_days,
            test_days=args.test_days,
            embargo_days=args.embargo_days,
            holdout_days=args.holdout_days,
            method=args.method,
            objective=args.objective,
            max_evals=args.max_evals,
            pbo=args.pbo,
            monte_carlo=args.monte_carlo,
            parameter_sensitivity=args.param_sensitivity,
            leakage_probe=args.leakage_probe,
        )
    _print_walkforward(result, args.objective)

    if getattr(args, "html", None):
        from tradeflow.services.analysis import walk_forward_payload
        from tradeflow.services.audit import new_run_id

        _write_html(
            args,
            walk_forward_payload(
                result,
                run_id=new_run_id(),
                strategy=args.strategy,
                symbols=universe,
                start=args.start,
                end=args.end,
                mode=args.mode,
                objective=args.objective,
                method=args.method,
                gross=args.gross,
            ),
            "walkforward",
        )

    if not args.no_journal and result.folds:
        from tradeflow.services.analysis import trades_payload
        from tradeflow.services.audit import journal_trial

        # One walk-forward is one *validated* config — the OOS aggregate is the
        # headline. The many IS-optimization configs it evaluated internally are
        # already counted in n_trials_total (which the DSR used); record that count
        # rather than one row per inner config, matching how the agent journals.
        chosen = result.holdout_params or result.folds[-1].is_best_params
        report = result.gate_report()
        journal_trial(
            "walkforward",
            strategy=args.strategy,
            symbols=universe,
            candidate_symbols=args.symbols,
            start=args.start,
            end=args.end,
            params=dict(chosen),
            metrics=result.oos_aggregate,
            objective=args.objective,
            extra={
                "n_trials": result.n_trials_total,
                "promotable": report["promotable"],
                "efficiency": result.median_efficiency(),
            },
            returns=result.oos_returns,
            trades=trades_payload(result.oos_trade_table) if args.record_trades else None,
            dedup_params=recipe,
        )

    bootstrap_report = None
    if getattr(args, "bootstrap_skill", False) and result.folds:
        from tradeflow.services.analysis import compute_bootstrap_skill

        bootstrap_report = compute_bootstrap_skill(
            result.oos_returns,
            args.strategy,
            universe,
            result.n_trials_total,
            result.oos_aggregate,
            B=args.bootstrap_b,
            block_length=args.bootstrap_block_length,
            seed=args.bootstrap_seed,
        )
        _print_bootstrap_skill(bootstrap_report)

    if result.folds:
        _print_promotion_prerequisites(data_client, args, result, universe, bootstrap_report)

    if getattr(args, "chart", None):
        from tradeflow.analytics.charts import render_walkforward_chart

        try:
            path = render_walkforward_chart(result, args.chart, title=f"{args.strategy} — walk-forward")
            print(f"Chart saved to {path}")
        except RuntimeError as exc:  # matplotlib (viz extra) not installed
            print(f"Chart skipped: {exc}")

    if args.results_csv:
        rows = [
            {
                "fold": fr.fold.index,
                **{f"is_{k}": v for k, v in fr.is_metrics.items()},
                **{f"oos_{k}": v for k, v in fr.oos_metrics.items()},
                "oos_trades": fr.oos_trades,
            }
            for fr in result.folds
        ]
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.results_csv, index=False)
        print(f"\nPer-fold results written to {args.results_csv}")

    if args.save_config and result.folds:
        chosen = result.holdout_params or result.folds[-1].is_best_params
        provenance = build_provenance(
            method=args.method,
            objective=args.objective,
            windows={
                "start": args.start,
                "end": args.end,
                "mode": args.mode,
                "folds": len(result.folds),
                "holdout_days": args.holdout_days,
                "embargo_days": args.embargo_days,
            },
            oos_metrics=result.oos_aggregate,
            n_trials=result.n_trials_total,
            seed=args.seed,
        )
        path = save_config(
            args.save_config,
            strategy=args.strategy,
            scanner=args.scanner,
            params=chosen,
            # Both universes, because they record different decisions. `symbols` is
            # the book that was validated and what a replay trades; `candidate_symbols`
            # is what it was resolved from, and the only thing a genuine re-resolution
            # can re-scan - running a scanner over the resolved names is a second filter
            # over an already-filtered set, not the original decision repeated.
            symbols=universe,
            candidate_symbols=args.symbols,
            capital=args.capital,
            # Written out in full so a frozen config states what it risks rather than
            # inheriting it. The strategy's own defaults are a decision nobody made.
            position_limits=STRATEGIES[args.strategy].create_with_defaults().position_limits(),
            # _cost_key(args) without the vintage: that stamp fingerprints the *data*
            # a run read, and pinning a reusable config to one data snapshot is the
            # opposite of what it is for.
            cost=_cost_key(args),
            provenance=provenance,
        )
        print(f"Chosen config saved to {path} (a human promotes it to live; nothing auto-flips)")

    _maybe_print_attribution_verdict(data_client, args.strategy, universe, args.start, args.end)


def _print_cached_walkforward(row: Dict[str, Any]) -> None:
    """The reduced report served for a memoized walk-forward: the OOS aggregate
    and verdict the trial store denormalized, not the full per-fold table (that
    detail isn't persisted). Re-run with --force for it."""
    print("\n=== Walk-Forward Validation (cached — per-fold detail not stored) ===")
    print(
        f"  OOS Sharpe {row.get('oos_sharpe') or 0:.3f}  MaxDD {row.get('oos_max_dd') or 0:.2f}%  "
        f"PF {row.get('oos_profit_factor') or 0:.2f}  DSR {row.get('deflated_sharpe') or 0:.3f}  "
        f"trades {row.get('oos_trades') or 0}"
    )
    efficiency = row.get("efficiency")
    n_trials = row.get("n_trials_in_session")
    print(
        f"  Efficiency (OOS/IS): {'n/a' if efficiency is None else f'{efficiency:.3f}'}  "
        f"trials total: {'n/a' if n_trials is None else n_trials}"
    )
    promotable = row.get("promotable")
    verdict = "PROMOTABLE" if promotable else ("NOT promotable" if promotable is not None else "unknown")
    print(f"\nVerdict: {verdict} (from the original run — pass --force to re-verify the promotion gates)")


def _print_walkforward(result, objective: str) -> None:
    print("\n=== Walk-Forward Validation ===")
    print(
        f"{'FOLD':>4} {'IS ' + objective:>16} {'OOS ' + objective:>16} {'OOS Sharpe':>12} "
        f"{'OOS PF':>8} {'OOS trades':>11}"
    )
    for fr in result.folds:
        print(
            f"{fr.fold.index:>4} {fr.is_metrics.get(objective, 0):>16.3f} "
            f"{fr.oos_metrics.get(objective, 0):>16.3f} {fr.oos_metrics.get('sharpe_ratio', 0):>12.3f} "
            f"{fr.oos_metrics.get('profit_factor', 0):>8.2f} {fr.oos_trades:>11}"
        )

    agg = result.oos_aggregate
    print("\n--- OOS aggregate (concatenated trades) ---")
    print(
        f"  Sharpe {agg.get('sharpe_ratio', 0):.3f}  CAGR {agg.get('cagr', 0):.2f}%  "
        f"MaxDD {agg.get('max_drawdown', 0):.2f}%  PF {agg.get('profit_factor', 0):.2f}  "
        f"DSR {agg.get('deflated_sharpe_ratio', 0):.3f}  trades {agg.get('total_trades', 0)}"
    )
    print(
        f"  Efficiency (OOS/IS {objective}): {result.efficiency:.3f}  trials total: {result.n_trials_total}"
    )
    if result.degradation:
        deg = "  ".join(f"{k} {v:+.3f}" for k, v in result.degradation.items())
        print(f"  Degradation (IS-OOS): {deg}")
    if result.holdout is not None:
        print("\n--- Holdout (scored once) ---")
        print(
            f"  Sharpe {result.holdout.get('sharpe_ratio', 0):.3f}  "
            f"CAGR {result.holdout.get('cagr', 0):.2f}%  trades {result.holdout.get('total_trades', 0)}"
        )
    if result.pbo is not None:
        print(f"\nPBO (prob. of backtest overfitting): {result.pbo:.2f}")
    if result.monte_carlo:
        mc = result.monte_carlo
        print(
            f"Monte Carlo OOS Sharpe p05/p50: {mc.get('sharpe_p05', 0):.3f} / {mc.get('sharpe_p50', 0):.3f}"
        )

    report = result.gate_report()
    print("\n--- Promotion gates ---")
    for name, check in report["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {check['value']} (threshold {check['threshold']})")
    verdict = "PROMOTABLE" if report["promotable"] else "NOT promotable"
    median_sharpe = result.median_oos("sharpe_ratio")
    print(
        f"\nVerdict: {verdict} — OOS Sharpe {median_sharpe:.2f}, efficiency "
        f"{result.median_efficiency():.2f}, {result.total_oos_trades()} OOS trades, "
        f"DSR {agg.get('deflated_sharpe_ratio', 0):.2f}"
    )


def _print_promotion_prerequisites(data_client, args, result, universe, bootstrap_report) -> None:
    """What a candidate should clear before paper, beside `promotable` rather than in it.

    `promotable` stays statistical and keeps meaning for every trial already recorded.
    These ask the questions that come after: does the edge survive worse cost
    assumptions, and does it still look like skill once the whole family is priced in?

    Computed here because the family test needs this trial already journaled, which
    happens above - `gate_report` runs before that and structurally cannot see it.
    """
    from tradeflow.analytics.performance import promotion_prerequisites
    from tradeflow.services.analysis import run_cost_stress

    stress = None
    if getattr(args, "cost_stress", None):
        chosen = result.holdout_params or result.folds[-1].is_best_params
        stress = run_cost_stress(
            data_client,
            args.strategy,
            universe,
            args.start,
            args.end,
            config=dict(chosen),
            capital=args.capital,
            commission_bps=args.commission_bps,
            impact_eta=args.impact_eta,
            borrow_bps=args.borrow_bps,
            axis=args.cost_stress,
        )

    benchmark_ir = None
    if getattr(args, "benchmark", None) and any(
        fold.oos_metrics.get("benchmark_available") for fold in result.folds
    ):
        benchmark_ir = result.median_oos("information_ratio")

    benchmark_excess = result.median_oos_excess_return() if benchmark_ir is not None else None

    prereq = promotion_prerequisites(
        cost_stress=stress,
        bootstrap=bootstrap_report,
        benchmark_ir=benchmark_ir,
        benchmark_excess_pct=benchmark_excess,
    )
    _print_fold_disagreement(result)
    _print_leg_stability(result)
    _print_walkforward_verdicts(result, prereq)
    if not prereq["evaluated"] and prereq["checks"]["family_bootstrap"]["n_used"] == 0:
        return  # nothing to say: no input was produced by this run

    print("\n=== Promotion prerequisites (separate from `promotable`) ===")
    for name, check in prereq["checks"].items():
        if not check["evaluated"]:
            print(f"  [ -- ] {name:18}not evaluated - {check['note']}")
            continue
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {name:18}{check['value']:.4g} vs {check['threshold']:.4g}")
    # The count leads, and the verdict word never stands alone: "Prerequisites: clear"
    # can be skimmed as all-clear even with a caveat on the next line, and this is a
    # display someone reads immediately before risking money.
    ready, done, total = prereq["ready"], prereq["evaluated"], prereq["total"]
    unknown = total - done
    verdict = {None: "nothing assessed", True: "clear so far" if unknown else "clear", False: "NOT clear"}[
        ready
    ]
    tail = f"; {unknown} unknown" if unknown else ""
    print(f"  Prerequisites: {done} of {total} evaluated - {verdict}{tail}")
    if unknown:
        print("  An unevaluated check is not a passed one - what is unknown stays unknown.")


def _print_fold_disagreement(result) -> None:
    """The spread the median is hiding.

    A median is what the prerequisite gates on, and a median is exactly where regime
    failure hides: excess returns of +0.2%, -2.0% and +3.2% have a positive median and
    a five-point spread, and only one of those two numbers would make anyone look
    closer. Reported rather than gated - the median already gates, and nobody yet knows
    what "too much disagreement" is worth failing a candidate over.
    """
    excess = result.excess_return_by_fold()
    if len(excess) < 2:
        return
    negative = [value for value in excess if value < 0]
    spread = max(excess) - min(excess)
    print("\n=== Excess return by fold (diagnostic) ===")
    print("  " + "  ".join(f"{value:+.2f}%" for value in excess))
    print(f"  median {result.median_oos_excess_return():+.2f}%   spread {spread:.2f}pp")
    if negative and len(negative) < len(excess):
        print(
            f"  {len(negative)} of {len(excess)} folds lost to the benchmark - the median "
            "is an average over folds that disagree, not a typical fold."
        )


def _print_leg_stability(result) -> None:
    """Each leg's beta fold by fold, when the book trades both sides.

    A book that is neutral on average and directional within folds is a different
    proposition from one that is neutral throughout, and no aggregate separates them.
    Diagnostic only - it gates nothing.
    """
    by_fold = result.leg_beta_by_fold()
    if len(by_fold) < 2 or not any(any(b is not None for b in betas) for betas in by_fold.values()):
        return  # long-only, or no benchmark was given for the folds to score against
    print("\n=== Leg beta by fold (diagnostic) ===")
    for name, betas in sorted(by_fold.items()):
        rendered = "  ".join("n/a" if b is None else f"{b:+.2f}" for b in betas)
        print(f"  {name:6}{rendered}")
    print("  A book neutral on average can still be directional inside a fold.")


def _print_walkforward_verdicts(result, prereq) -> None:
    """Statistical validation and evidence completeness. Execution is not measurable
    here - it is a property of a book at a capital, which a backtest reports."""
    from tradeflow.analytics.reporting import format_verdicts

    gates = result.gate_report()
    failed = [name for name, check in gates["checks"].items() if not check.get("passed")]
    statistical = "PASS" if gates["promotable"] else f"FAIL - {', '.join(failed)}"

    done, total = prereq["evaluated"], prereq["total"]
    unknown = total - done
    ready = {None: "nothing assessed", True: "clear so far" if unknown else "clear", False: "NOT clear"}[
        prereq["ready"]
    ]
    evidence = f"{done} of {total} evaluated - {ready}" + (f"; {unknown} unknown" if unknown else "")
    print("\n".join(format_verdicts(statistical=statistical, evidence=evidence)))


def _print_bootstrap_skill(report: Dict[str, Any]) -> None:
    """The bootstrap-skill report (`walkforward --bootstrap-skill`):
    own p next to family p, ALWAYS together — a great own p and a terrible
    family p is exactly the selection-luck signature this test exists to catch."""
    print("\n--- Bootstrap skill ---")
    if not report.get("available"):
        print(f"  {report.get('note', 'unavailable')}")
        return
    own = report["own"]
    if own.get("insufficient_data"):
        print(f"  {report['verdict']}")
        return
    print(
        f"  this config: IR {own['ir_observed']:+.2f}, p={own['p_value']:.3f} (own, "
        f"B={own['B']}, L={own['block_length']:.1f}, SE(p)≈{own['p_se']:.3f})"
    )
    family = report["family"]
    if family.get("available"):
        print(
            f"  family: best of K={family['k_trials']} trials, p={family['family_p']:.3f} "
            f"(used {family['n_used']}/{family['n_attempted']} attempted trials — "
            f"{family['n_with_returns']} had a stored return series, "
            f"{family['n_excluded_short']} excluded for too little date overlap)"
        )
        if family["block_sensitivity_flag"] or own["block_sensitivity_flag"]:
            print(
                "  ⚠ significance flips across L/2..2L — see block_sensitivity "
                "(a p that flips there is not a result)"
            )
    else:
        print(f"  family: unavailable — {family.get('note', 'see n_used/n_attempted')}")
    cross = report["parametric_cross_check"]
    print(
        f"  parametric cross-check: PSR {cross['probabilistic_sharpe_ratio']:.2f}, "
        f"DSR {cross['deflated_sharpe_ratio']:.2f} (n_trials={report['n_trials_total']})"
    )
    print(f"  verdict: {report['verdict']}")


def cmd_research(args) -> None:
    """Run the autonomous research loop (opt-in; needs the ``ai`` extra).

    Offline research clock only: proposes hypotheses, validates them out-of-sample
    via walk-forward, and writes a shortlist of provenance-stamped candidate
    configs for a human to review. Never touches live trading.
    """
    import importlib.util

    # Each provider pulls in a different optional dependency (Ollama needs none).
    required = {"anthropic": "anthropic", "openai": "openai"}.get(args.provider)
    if required and importlib.util.find_spec(required) is None:
        extra = {"anthropic": "ai", "openai": "openai"}[args.provider]
        sys.exit(
            f"Provider '{args.provider}' needs the '{extra}' extra. Install it:\n    uv sync --extra {extra}"
        )

    from tradeflow.research.agent import ResearchAgent, ResearchConfig
    from tradeflow.research.proposer import build_proposer
    from tradeflow.services.data import build_data_client

    data_client = build_data_client()
    universe = resolve_universe(data_client, args.scanner, args.symbols, as_of=args.scan_as_of or args.end)
    cfg = ResearchConfig(
        goal=args.goal,
        mode=args.mode,
        n_folds=args.folds,
        embargo_days=args.embargo_days,
        holdout_days=args.holdout_days,
        method=args.method,
        objective=args.objective,
        max_evals=args.max_evals,
        capital=args.capital,
        max_trials=args.max_trials,
        max_dry_rounds=args.max_dry_rounds,
        max_tokens=args.max_tokens,
        shortlist_size=args.shortlist_size,
        allow_code_gen=args.allow_code_gen,
    )
    proposer = build_proposer(args.provider, args.model, allow_code_gen=args.allow_code_gen)
    agent = ResearchAgent(args.strategy, data_client, proposer, cfg, seed=args.seed)
    result = agent.run(universe, args.start, args.end)

    print(f"\n=== Research session {agent.session_id} ===")
    print(
        f"Stopped: {result.stopped_reason} after {result.rounds} rounds, "
        f"{result.n_trials_total} cumulative trials"
    )
    print(f"Holdout (scored once): {result.holdout_window}")
    if not result.shortlist:
        print("No candidate cleared the promotion gates.")
    for c in result.shortlist:
        oos = c.oos_metrics.get("sharpe_ratio", 0.0)
        hold = (c.holdout_metrics or {}).get("sharpe_ratio", 0.0)
        print(f"  [{c.id}] OOS Sharpe {oos:.2f} | holdout Sharpe {hold:.2f} | {c.saved_path}")
        print(f"        hypothesis: {c.hypothesis[:100]}")
    print(f"\nJournal: {result.journal_path}  (a human reviews and promotes; nothing is live)")


def _journal_alpha(args, strategy_label: str, source: str, result: dict) -> None:
    """Record one alpha run as a read-only trial.

    An alpha run is a point-in-time forecast, not a backtest — it has no Sharpe to
    deflate, so it is journaled under ``kind="alpha"`` for dedup and future IC /
    bootstrap work, and a multiple-testing count skips this kind.
    The window collapses to the single as-of date.
    """
    if args.no_journal:
        return
    from tradeflow.services.audit import journal_trial

    knobs = {
        "source": source,
        "ic": getattr(args, "ic", None),
        "scaling": getattr(args, "scaling", None),
        "neutralize": args.neutralize,
        "neutralize_factors": list(args.neutralize_factors or []),
        "lookback_days": args.lookback_days,
    }
    journal_trial(
        "alpha",
        strategy=strategy_label,
        symbols=args.symbols,
        start=args.as_of,
        end=args.as_of,
        params={k: v for k, v in knobs.items() if v is not None},
        metrics={},  # a forecast has no realized-edge metric
        extra={
            "universe_size": result.get("universe_size", len(result.get("alphas", []))),
            "benchmark_available": result.get("benchmark_available"),
            "low_confidence": result.get("low_confidence"),
        },
    )


def cmd_init(args) -> None:
    """Guided first-run setup: write a valid `.env`, check it, and say what to try.

    Three modes over one set of service functions: `--check` diagnoses what exists
    and writes nothing; `--non-interactive` builds a `.env` from environment
    variables with no prompts (for scripts and containers); the default is the
    interactive wizard.
    """
    from tradeflow.services import setup

    if args.check:
        _print_doctor(setup.run_checks())
        return
    if args.non_interactive:
        _init_non_interactive(args, setup)
        return
    _init_interactive(args, setup)


def _print_doctor(checks) -> None:
    """Every check, pass or fail, with an exit code that reflects the essentials.

    A doctor reports the whole picture rather than stopping at the first problem —
    but a missing credential is not the same as a missing optional extra, so only
    the former fails the run.
    """
    from tradeflow.settings import state_root

    print("\nTradeFlow setup check\n")
    print(f"  state root: {state_root()}")
    print("  (journal, trial store, bar cache, and configs all live here)\n")
    for check in checks:
        print(f"  [{'ok' if check.passed else 'FAIL'}] {check.name:<22} {check.detail}")
    essential = [c for c in checks if not c.passed and not c.name.startswith("extra:")]
    if essential:
        print(f"\n{len(essential)} problem(s) to fix. Run `python main.py init` for the guided setup.")
        raise SystemExit(1)
    print("\nSetup looks good. `make demo` needs nothing; `python main.py verdict` needs the keys above.")


def _init_non_interactive(args, setup) -> None:
    """Build a `.env` purely from environment variables — no prompts, no retries."""
    import os

    updates = setup.build_updates(
        key=os.environ.get("APCA_API_KEY_ID"),
        secret=os.environ.get("APCA_API_SECRET_KEY"),
        paper_trade=None if "PAPER_TRADE" not in os.environ else os.environ["PAPER_TRADE"] != "false",
    )
    if not updates:
        raise SystemExit(
            "Nothing to write: set APCA_API_KEY_ID / APCA_API_SECRET_KEY in the environment first."
        )
    result = setup.write_env(updates, args.env_path)
    print(f"Wrote {result['path']}: {', '.join(f'{k}={v}' for k, v in result['updated'].items())}")
    if result["backup"]:
        print(f"Previous file backed up to {result['backup']}")


def _init_interactive(args, setup) -> None:
    """The guided path. Prompts are the only thing that lives here; every decision
    it makes is a service function that a test can drive directly."""
    import getpass

    print("\nTradeFlow setup\n")
    state = setup.inspect_env(args.env_path)

    if state.exists:
        print(f"Found {state.path}:")
        for key, masked in state.summary().items():
            print(f"  {key:<22} {masked}")
        if state.complete and not _confirm("\nCredentials already look set. Replace them?", default=False):
            print("Left unchanged.")
            _print_next_steps()
            return
    else:
        print(f"No {state.path} yet — this will create one.")

    print(
        "\nAlpaca paper-trading keys come from https://app.alpaca.markets/"
        " → Paper Account → API Keys."
        "\nPress Enter at both prompts to skip and stay in keyless demo mode.\n"
    )
    # getpass, not input: a key echoed into a terminal ends up in scrollback and
    # shell history, which is exactly where a secret should never be.
    key = getpass.getpass("  APCA_API_KEY_ID (hidden): ").strip()
    secret = getpass.getpass("  APCA_API_SECRET_KEY (hidden): ").strip() if key else ""

    if not key or not secret:
        print("\nSkipped — no keys written. `make demo` runs offline with no credentials at all.")
        _print_next_steps()
        return

    paper = True
    if not _confirm("\nKeep PAPER_TRADE=true (orders go to the paper account)?", default=True):
        print(
            "\n  Live trading means real money. The research clock proposes; a human promotes —\n"
            "  turning this off removes the safety net that makes that separation matter."
        )
        paper = not _typed_confirmation("ENABLE LIVE")

    result = setup.write_env(setup.build_updates(key, secret, paper), args.env_path)
    print(f"\nWrote {result['path']} ({', '.join(f'{k}={v}' for k, v in result['updated'].items())}).")
    if result["backup"]:
        print(f"Previous file backed up to {result['backup']} — nothing else in it was changed.")

    print("\nChecking the credentials against Alpaca…")
    check = _check_credentials_now(setup)
    print(f"  {check.message}")

    if check.ok and _confirm("\nWarm the local bar cache now (a small universe, ~1 year daily)?", False):
        _warm_cache(args)

    _print_next_steps()


def _check_credentials_now(setup):
    """Validate through the data-only client factory — never a trading client."""
    from tradeflow.services.data import build_data_client

    try:
        return setup.check_credentials(build_data_client())
    except Exception as exc:  # noqa: BLE001 - a setup failure is a message, not a traceback
        return setup.CredentialCheck(
            setup.CREDENTIALS_UNREACHABLE, f"Could not build a data client: {type(exc).__name__}"
        )


def _warm_cache(args) -> None:
    """Optional first cache warm. Interruptible: the cache writes per partition, so
    Ctrl-C leaves a partial cache that is safe and resumable, and says so."""
    from tradeflow.services import setup as setup_service
    from tradeflow.services.data import build_data_client

    end = datetime.now()
    start = end - timedelta(days=365)
    universe = list(setup_service.DEFAULT_WARM_UNIVERSE)
    try:
        provider = build_data_client(cache=True, cache_dir=args.cache_dir).provider
        summary = provider.warm(universe, "1Day", start, end)
        cached = sum(1 for s in summary.values() if not s["already_cached"])
        print(f"  Cached {cached} of {len(universe)} symbols. `--offline` now works for this universe.")
    except KeyboardInterrupt:
        print("\n  Interrupted — the partial cache is safe and resumable (`cache warm` continues it).")
    except Exception as exc:  # noqa: BLE001
        print(f"  Cache warm skipped: {type(exc).__name__}. Nothing else is affected.")


def _confirm(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _typed_confirmation(phrase: str) -> bool:
    """Require the exact phrase. A yes/no prompt is too easy to answer wrongly for
    a choice that moves real money."""
    return input(f"  Type {phrase!r} to confirm, or anything else to keep paper trading: ") == phrase


def _print_next_steps() -> None:
    print(
        "\nNext:"
        "\n  make demo                 the whole pipeline on synthetic data — no keys, no network"
        "\n  python main.py verdict    scan → alphas → portfolio → information, one verdict"
        "\n  python main.py backtest   did the idea ever work?"
        "\n  python main.py init --check   re-run these checks any time\n"
    )


def cmd_verdict(args) -> None:
    """Run the whole cross-sectional pipeline once and print one consolidated answer.

    Scan, alphas (combined when several signals are given), portfolio construction,
    and information analysis over one universe, one window, and one cost model - the
    joined-up story the individual commands can only approximate by agreeing with
    each other by hand. Read-only research clock: proposes a book, places no orders.

    Exits non-zero when a step failed, so a scripted caller can tell a partial run
    from a complete one without parsing the report.
    """
    import json

    from tradeflow.analytics.reporting import format_cached_notice, format_verdict_report
    from tradeflow.services.analysis import run_verdict

    tuned = apply_run_config(args)
    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)

    result = run_verdict(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        config=tuned,
        scanner=args.scanner,
        signals=args.combine,
        source=args.source,
        benchmark=args.benchmark,
        timeframe=args.timeframe,
        capital=args.capital,
        horizon=args.horizon,
        target_te=args.target_te,
        max_weight=args.max_weight,
        max_names=args.max_names,
        neutralize_factors=args.neutralize_factors,
        risk_model=args.risk_model,
        lookback_days=args.lookback_days,
        gross=args.gross,
        commission_bps=args.commission_bps,
        impact_eta=args.impact_eta,
        borrow_bps=args.borrow_bps,
        force=args.force,
        journal=not args.no_journal,
    )

    if result.get("memoized"):
        print(
            format_cached_notice(
                {"id": result.get("trial_id"), "ts": result.get("trial_ts")},
                vintage_available=False,
            )
        )
    print(format_verdict_report(result))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"Structured result written to {args.json}")
    _write_html(args, result, "verdict")

    # Any incomplete run exits non-zero, not just one with a named step failure: a
    # run that produced no gate at all is equally not something to act on, and a
    # caller checking the exit code should not have to know the difference.
    if (result.get("verdict") or {}).get("verdict") == "incomplete":
        raise SystemExit(1)


def cmd_alphas(args) -> None:
    """Print the ranked alpha table (residual-return forecasts) for a universe.

    Read-only research-clock flow: scores each name as of --as-of, scales the
    cross-section into comparable annualized-return forecasts, and ranks them.
    Produces no orders and saves no config.
    """
    tuned = apply_run_config(args)

    from tradeflow.services.analysis import compute_alphas, compute_combined_alphas

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)

    if args.combine:
        combined = compute_combined_alphas(
            data_client,
            args.combine,
            args.symbols,
            as_of=args.as_of,
            benchmark=args.benchmark,
            neutralize=args.neutralize,
            neutralize_factors=args.neutralize_factors,
            lookback_days=args.lookback_days,
        )
        _journal_alpha(args, "+".join(args.combine), "combine", combined)
        _print_combined_alphas(combined, args)
        return

    result = compute_alphas(
        data_client,
        args.strategy,
        args.symbols,
        as_of=args.as_of,
        config=tuned,
        source=args.source,
        scanner=args.scanner,
        ic=args.ic,
        benchmark=args.benchmark,
        neutralize=args.neutralize,
        neutralize_factors=args.neutralize_factors,
        lookback_days=args.lookback_days,
        scaling=args.scaling,
    )

    _journal_alpha(args, args.strategy, args.source, result)

    src_desc = {
        "strategy": f"'{args.strategy}' score",
        "signal": f"'{args.strategy}' signal",
        "scanner": f"scanner '{args.scanner}' strength",
    }[args.source]
    print(f"\nAlphas from {src_desc} as of {args.as_of:%Y-%m-%d} (IC={args.ic}, benchmark={args.benchmark})")
    _print_neutralization(result)
    _print_case(result)
    if not result["benchmark_available"]:
        print("  ! benchmark unavailable — residual vol falls back to total volatility")
    if result["low_confidence"]:
        print(f"  ! thin universe ({result['universe_size']} names) — demean-only, low confidence")
    if not result["alphas"]:
        print("No scorable names.")
        return

    print(f"\n{'SYMBOL':10}{'SCORE':>10}{'Z':>9}{'BETA':>8}{'RESID_VOL':>11}{'ALPHA':>10}")
    for row in result["alphas"]:
        print(
            f"{row['symbol']:10}{row['score']:>10.3f}{row['z']:>9.2f}{row['beta']:>8.2f}"
            f"{row['residual_vol']:>10.1%}{row['alpha']:>10.2%}"
        )


def _print_neutralization(result) -> None:
    """One honest line about what the alphas were actually regressed against.

    Reports what was *applied* (from the refinement's meta), not what was requested —
    and warns when a requested factor exposure was unavailable, so "factor-neutral"
    is never claimed for un-neutralized output.
    """
    requested = result.get("neutralize_factors") or []
    applied = result.get("neutralized_against") or []
    if applied:
        print(f"  neutralized against: {', '.join(applied)}")
    missing = [f for f in requested if f not in applied]
    if missing:
        print(
            f"  ! requested factor(s) NOT neutralized (exposures unavailable — "
            f"insufficient history?): {', '.join(missing)}"
        )


def _print_case(result) -> None:
    """One line on the chosen scaling and the Case test, when it ran."""
    case = result.get("case")
    if not case:
        if result.get("scaling") and result["scaling"] != "case1":
            print(f"  scaling: {result['scaling']}")
        return
    amb = " (ambiguous → base rate)" if case.get("ambiguous") else ""
    print(
        f"  scaling: {result['scaling']} — Case {case['case']}{amb} "
        f"(R²={case['r_squared']:.2f}, t={case['t_stat']:+.1f}, "
        f"candidate corr {case.get('candidate_correlation', float('nan')):.2f})"
    )


def _print_combined_alphas(result, args) -> None:
    """Print the multi-signal combination: per-signal weights/ICs + the combined alphas."""
    if not result.get("universe_size"):
        print(result.get("note", "No combined alphas produced."))
        return

    print(f"\nCombined alpha from {', '.join(result['signals'])} as of {args.as_of:%Y-%m-%d}")
    _print_neutralization(result)
    print(f"  measured over {result['n_periods']} rebalances  |  combined IC {result['combined_ic']:.4f}")
    print(f"\n{'SIGNAL':16}{'IC':>9}{'SHRUNK':>9}{'WEIGHT':>9}")
    for sig in result["signals"]:
        print(
            f"{sig:16}{result['signal_ics'][sig]:>9.4f}{result['signal_shrunk_ics'][sig]:>9.4f}"
            f"{result['signal_weights'][sig]:>9.3f}"
        )
    if result["low_confidence"]:
        print(f"\n  ! thin universe ({result['universe_size']} names) — demean-only, low confidence")
    print(f"\n{'SYMBOL':10}{'SCORE':>10}{'Z':>9}{'BETA':>8}{'RESID_VOL':>11}{'ALPHA':>10}")
    for row in result["alphas"]:
        print(
            f"{row['symbol']:10}{row['score']:>10.3f}{row['z']:>9.2f}{row['beta']:>8.2f}"
            f"{row['residual_vol']:>10.1%}{row['alpha']:>10.2%}"
        )


def cmd_risk(args) -> None:
    """Print the universe's covariance risk summary (read-only).

    Estimates an annualized, well-conditioned Σ as of --as-of and reports the
    shrinkage intensity, conditioning, mean correlation, equal-weight portfolio
    volatility, and the top risk contributors. Produces no orders.
    """
    apply_run_config(args)

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)

    if getattr(args, "evaluate_conditional", False):
        _print_conditional_evidence_gate(data_client, args)
        return

    from tradeflow.services.analysis import compute_risk

    result = compute_risk(
        data_client,
        args.symbols,
        as_of=args.as_of,
        model=args.model,
        benchmark=args.benchmark,
        lookback_days=args.lookback_days,
        timeframe=args.timeframe,
        conditional=getattr(args, "conditional", None),
        conditional_lambda=getattr(args, "conditional_lambda", None),
    )
    if not result.get("universe_size"):
        print(result.get("note", "No risk matrix produced."))
        return

    delta = result["shrinkage"]
    delta_str = f"  shrinkage δ {delta:.3f}" if delta is not None else ""
    print(f"\nRisk model '{result['model']}' as of {args.as_of:%Y-%m-%d} ({result['timeframe']} returns)")
    print(f"  names {result['universe_size']}{delta_str}")
    print(
        f"  condition number {result['condition_number']:.1f}  PD {result['positive_definite']}  "
        f"mean corr {result['mean_correlation']:.2f}  eq-weight vol {result['equal_weight_volatility']:.1%}"
    )
    if "factor_risk_share" in result:
        print(
            f"  risk split: {result['factor_risk_share']:.0%} factor / "
            f"{result['specific_risk_share']:.0%} specific  (factors: {', '.join(result['factor_names'])})"
        )
    if "sigma_regime" in result:
        regime = result["sigma_regime"]
        print(
            f"  conditional ({regime['method']}, λ={regime['lambda']:.2f}): "
            f"mean σ_t/σ_unconditional = {regime['mean_sigma_regime']:.2f}"
        )
    print(f"\n{'SYMBOL':10}{'VOL':>9}{'RISK CONTRIB':>14}")
    for row in result["top_risk_contributors"]:
        print(f"{row['symbol']:10}{row['volatility']:>8.1%}{row['risk_contribution']:>14.2%}")


def _print_conditional_evidence_gate(data_client, args) -> None:
    """The MZ/QLIKE evidence gate (`risk --evaluate-conditional`) — the report
    that decides whether conditioning is worth turning on."""
    from datetime import timedelta

    from tradeflow.services.analysis import evaluate_conditional_risk

    end = args.as_of
    start = end - timedelta(days=args.lookback_days)
    r = evaluate_conditional_risk(
        data_client,
        args.symbols,
        start,
        end,
        timeframe=args.timeframe,
        conditional_lambda=getattr(args, "conditional_lambda", None),
    )
    if not r.get("n_names"):
        print(r.get("note", "No evidence-gate report produced."))
        return

    print(f"\nConditional-vol evidence gate: {start:%Y-%m-%d}..{end:%Y-%m-%d} ({r['n_names']} names)")
    print(f"  {'method':16}{'QLIKE':>10}{'MZ a':>10}{'MZ b':>10}{'r2':>8}")
    for method, stats in r["pooled"].items():
        mz = stats["mincer_zarnowitz"]
        print(f"  {method:16}{stats['qlike']:>10.4f}{mz['a']:>10.6f}{mz['b']:>10.3f}{mz['r2']:>8.3f}")
    verdict = (
        "PASSED — a conditional method beats unconditional on QLIKE"
        if r["gate_passed"]
        else ("NOT passed — unconditional wins on QLIKE; the honest call is to leave --conditional off")
    )
    print(f"\n  best by pooled QLIKE: {r['best_method_pooled_qlike']}")
    print(f"  Gate: {verdict}")


def cmd_info(args) -> None:
    """Print the information report: IC, breadth, and the predicted-vs-realized IR.

    Read-only research diagnostic: measures the strategy's information coefficient and
    effective breadth over [start, end] and reconciles predicted IR with realized,
    surfacing the research-integrity guardrails. Produces no orders.
    """
    tuned = apply_run_config(args)

    from tradeflow.services.analysis import compute_information

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)

    if getattr(args, "scaling_ab", False):
        _print_scaling_ab(data_client, args)
        return
    if getattr(args, "conditional_ab", False):
        _print_conditional_ab(data_client, args)
        return
    if getattr(args, "policy_ab", False):
        _print_policy_ab(data_client, args)
        return
    if getattr(args, "attribution", False):
        _print_attribution_report(data_client, args)
        return

    r = compute_information(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        config=tuned,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
        horizon=args.horizon,
        n_trials=args.n_trials,
    )
    if not r.get("periods"):
        print(r.get("note", "No information report produced."))
        return

    print(f"\nInformation report: '{args.strategy}' {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}")
    print(f"  measured over {r['periods']} rebalances (horizon {r['horizon_bars']} bars)")
    flag = "  ⚠ low sample" if r["low_sample"] else ""
    print(f"  IC mean {r['mean_ic']:+.4f}  t-stat {r['ic_tstat']:+.2f}  rank-IC {r['rank_ic']:+.4f}{flag}")
    print(
        f"  breadth: {r['breadth_effective']:.0f} effective vs {r['breadth_naive']:.0f} naive "
        f"(ρ̄ {r['rho_bar']:.2f}, {r['n_names']} names)"
    )
    print(
        f"  IR: predicted {r['predicted_ir']:+.2f}  realized {r['realized_ir']:+.2f} "
        f"± {r['ir_standard_error']:.2f} (SE)"
    )
    print(
        f"  guardrails: P(any |t|>2 in {r['n_trials']} trials) = {r['multiple_testing_inflation']:.2f}"
        + ("  | ⚠ realized IR > 2 — suspect a bug/leak" if r["sanity_ceiling_breached"] else "")
    )
    print(
        f"  attribution: factor {r['factor_return']:+.4f} / specific {r['specific_return']:+.4f} "
        f"per rebalance (factor tilts vs name selection)"
    )
    # IC-uncertainty level shrink: how much of the measured IC level survives its own
    # estimation error, with an overlap-honest effective T.
    if "level_shrink_factor" in r:
        print(
            f"  level shrink: keep {r['level_shrink_factor']:.0%} of the naive level "
            f"(T_eff {r['effective_t']:.1f}, IC {r['recommended_ic']:+.4f})"
        )
    _print_bucket_diagnostic(r.get("risk_bucket_diagnostic"))
    verdict = "distinguishable from luck" if abs(r["ic_tstat"]) >= 2 else "NOT distinguishable from luck"
    print(f"\n  Verdict: skill is {verdict} (IC t-stat {r['ic_tstat']:+.2f}).")
    if abs(r["ic_tstat"]) >= 2:
        print(f"  Recommended IC for alpha scaling (a human applies it): {r['recommended_ic']:.4f}")
        print(
            f"  After the IC-uncertainty level shrink, deploy IC ≈ "
            f"{r['recommended_ic'] * r.get('level_shrink_factor', 1.0):.4f}"
        )
    _write_html(args, r, "info")


def _print_bucket_diagnostic(diag) -> None:
    """The equal-risk-contribution check, when it engaged."""
    if not diag or not diag.get("engaged"):
        if diag and diag.get("reason"):
            print(f"  risk buckets: suppressed — {diag['reason']}")
        return
    marker = "⚠ tilt" if diag["tilt_detected"] else "ok"
    print(
        f"  risk buckets ({diag['n_buckets']}, by residual vol): {marker} — "
        f"variance-share gradient {diag['variance_share_gradient']:+.2f} "
        f"vs band ±{diag['sampling_band']:.2f}"
    )
    if diag["tilt_detected"]:
        print(f"    → {diag['verdict']}")


def _print_scaling_ab(data_client, args) -> None:
    """Walk-forward A/B of the two scalings (the `--scaling-ab` research mode)."""
    from tradeflow.services.analysis import run_scaling_ab

    r = run_scaling_ab(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
        horizon=args.horizon,
    )
    if not r.get("periods"):
        print(r.get("note", "No scaling A/B produced."))
        return
    print(f"\nScaling A/B: '{args.strategy}' {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}")
    print(f"  measured over {r['periods']} rebalances (horizon {r['horizon_bars']} bars)")
    print(f"  Case 1 (σ·IC·z) realized IR: {r['case1_realized_ir']:+.2f}")
    print(f"  Case 2 (IC·c_g·z) realized IR: {r['case2_realized_ir']:+.2f}")
    agree = "agree" if r["agree"] else "DISAGREE"
    print(f"  regression picks {r['regression_pick']}, A/B picks {r['ab_pick']} — {agree}")
    print("  (compare the IR gap to its standard-error band before acting — a small gap is noise)")


def _print_attribution_report(data_client, args) -> None:
    """Full performance-attribution report (`info --attribution`)."""
    from tradeflow.services.analysis import compute_attribution

    r = compute_attribution(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
        horizon=args.horizon,
        n_trials=args.n_trials,
        signals=getattr(args, "attribution_signals", None),
        conditional=getattr(args, "conditional", None),
        conditional_lambda=getattr(args, "conditional_lambda", None),
        bootstrap_skill=getattr(args, "bootstrap_skill", False),
    )
    _print_attribution(r, args.strategy, args.start, args.end)
    te_by_regime = r.get("te_by_regime")
    if te_by_regime:
        print(f"\n  predicted-vs-realized TE by regime (conditional={r.get('conditional')}):")
        print(f"    {'regime':8}{'n':>5}{'predicted TE':>14}{'realized TE':>14}{'gap':>10}")
        for label in ("low", "mid", "high"):
            row = te_by_regime.get(label)
            if not row or not row.get("n"):
                continue
            print(
                f"    {label:8}{row['n']:>5}{row['predicted_te']:>13.1%}"
                f"{row['realized_te']:>14.1%}{row['gap']:>+10.1%}"
            )


def _print_conditional_ab(data_client, args) -> None:
    """The net-of-cost A/B (`info --conditional-ab`)."""
    from tradeflow.services.analysis import run_conditional_risk_ab

    r = run_conditional_risk_ab(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
        horizon=args.horizon,
        conditional_method=getattr(args, "conditional", None) or "ewma",
        conditional_lambda=getattr(args, "conditional_lambda", None),
    )
    if not r.get("periods"):
        print(r.get("note", "No conditional-risk A/B produced."))
        return

    print(f"\nConditional-risk net-of-cost A/B: '{args.strategy}' {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}")
    print(
        f"  measured over {r['periods']} rebalances (horizon {r['horizon_bars']} bars, method {r['conditional_method']})"
    )
    print(f"  {'variant':14}{'net IR':>9}{'realized TE':>13}{'pred TE':>10}{'turnover':>10}")
    for name, s in r["summaries"].items():
        if s.get("periods", 0) < 2:
            continue
        print(
            f"  {name:14}{s['net_ir']:>+9.2f}{s['realized_te']:>13.1%}"
            f"{s['mean_predicted_te']:>10.1%}{s['mean_turnover']:>10.1%}"
        )
    print(f"  winner (net IR): {r['winner_net_ir']}")
    print("  (net of the real transaction cost — a Σ that tracks TE better but churns the book loses here)")


def _print_policy_ab(data_client, args) -> None:
    """The net-of-cost A/B (`info --policy-ab`): myopic vs aim policy."""
    from tradeflow.services.analysis import run_policy_ab

    r = run_policy_ab(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
        horizon=args.horizon,
        trade_rate=getattr(args, "trade_rate", None),
    )
    if not r.get("periods"):
        print(r.get("note", "No policy A/B produced."))
        return

    print(
        f"\nMulti-period policy net-of-cost A/B: '{args.strategy}' {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}"
    )
    print(f"  measured over {r['periods']} rebalances (horizon {r['horizon_bars']} bars)")
    print(f"  {'variant':10}{'net IR':>9}{'realized TE':>13}{'pred TE':>10}{'turnover':>10}")
    for name, s in r["summaries"].items():
        if s.get("periods", 0) < 2:
            continue
        print(
            f"  {name:10}{s['net_ir']:>+9.2f}{s['realized_te']:>13.1%}"
            f"{s['mean_predicted_te']:>10.1%}{s['mean_turnover']:>10.1%}"
        )
    print(f"  winner (net IR): {r['winner_net_ir']}")
    if r.get("over_damped"):
        print("  ⚠ over-damped: aim traded less AND scored a lower net IR — not an improvement")
    print(
        "  (net of the real transaction cost — an aim policy that tracks decay but churns less "
        "yet still loses net IR should, and does, lose here)"
    )


def _print_universe_provenance(args, resolved) -> None:
    """Where the universe came from, printed with the result rather than inferred.

    A 61-name large-cap list is not "the market", and a report that leaves the universe
    in the background invites it to be read as one.
    """
    from tradeflow.analytics.reporting import format_universe_provenance
    from tradeflow.scanners.symbol_scanner import resolve_scan_clock

    source = {
        "config": "the saved config",
        "flag": "--symbols",
    }.get(getattr(args, "universe_source", None), "--symbols")
    if getattr(args, "universe_source", None) is None and args.symbols == DEFAULT_UNIVERSE:
        source = "the built-in default list"
    candidates = getattr(args, "candidate_symbols", None) or args.symbols
    replayed = getattr(args, "universe_source", None) == "config" and not getattr(
        args, "re_resolve_universe", False
    )
    clock = resolve_scan_clock(args.scan_as_of or args.end).isoformat() if args.scanner != "none" else None
    print(
        "\n".join(
            format_universe_provenance(
                candidates=candidates,
                resolved=resolved,
                scanner=args.scanner,
                scan_clock=clock,
                source=source,
                replayed=replayed,
            )
        )
    )


def _print_verdicts_for_backtest(result) -> None:
    """A backtest can speak to execution and to nothing else.

    Saying so is the point: the statistical verdict comes from a walk-forward and the
    evidence one from a campaign, and a backtest that stayed silent about both is how a
    replay reads as approval.
    """
    from tradeflow.analytics.performance import execution_verdict
    from tradeflow.analytics.reporting import format_verdicts

    verdict = execution_verdict(getattr(result, "execution", None))
    executable = verdict["executable"]
    summary = {None: "UNKNOWN - nothing was attempted", True: "PASS", False: f"FAIL - {verdict['reason']}"}[
        executable
    ]
    print("\n".join(format_verdicts(execution=summary)))


def _print_cost_stress(data_client, strategy_name: str, universe, args, tuned) -> None:
    """Re-run this config under worse cost assumptions and show where the edge dies.

    A single cost assumption produces a single number and no way to tell how much of
    the result was the assumption. The curve separates a strategy that survives five
    times its assumed cost from one that clears by a hair at 1x - both of which are
    "profitable at 1bp" and are not the same proposition.
    """
    from tradeflow.services.analysis import run_cost_stress

    report = run_cost_stress(
        data_client,
        strategy_name,
        universe,
        args.start,
        args.end,
        config=tuned or None,
        capital=args.capital,
        benchmark=args.benchmark,
        commission_bps=args.commission_bps,
        impact_eta=args.impact_eta,
        borrow_bps=args.borrow_bps,
        axis=args.cost_stress,
    )
    print(f"\n=== Cost stress ({report['axis']} axis) ===")
    print(f"  {'multiple':>10}{'Sharpe':>10}{'return':>10}{'cost':>14}")
    for point in report["points"]:
        print(
            f"  {point['multiple']:>9.1f}x{point['sharpe_ratio']:>10.2f}"
            f"{point['total_return']:>9.2f}%${point['total_cost']:>12,.0f}"
        )
    survives = report["survives_to_multiple"]
    if survives:
        print(f"  Edge survives to {survives:.0f}x its assumed cost.")
    else:
        print("  Edge does not survive its own cost assumptions.")
    print("  Nothing journaled: one candidate under stated assumptions, not new candidates.")


def _print_attribution(r, strategy: str, start, end) -> None:
    """Render a ``compute_attribution`` report: per-row mean/IR/t/share-of-variance,
    honest cumulation, and the skill-vs-luck verdict."""
    if not r.get("periods"):
        print(r.get("note", "No attribution report produced."))
        return

    print(f"\nPerformance attribution: '{strategy}' {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    print(f"  measured over {r['periods']} rebalances (horizon {r['horizon_bars']} bars)")
    # Named for what it is on every line that carries a level. These come from a paper
    # book built from the signal's own cross-section, scaled to unit gross - not from
    # anything the execution engine did, which is why a strategy the engine declined to
    # trade at all still has rows here.
    print("  paper book at unit gross — signal quality, not an executed track record;")
    print("  a strategy the engine takes zero trades in can still score here.")
    print(f"  {'row':28}{'p-b mean/yr':>13}{'p-b IR':>9}{'t':>8}{'share ψ²':>10}")

    rows = r["rows"]

    def _line(label: str, key: str, not_skill: bool = False) -> None:
        row = rows[key]
        if not_skill:
            print(f"  {label:28}{'—':>13}{'—':>9}{'—':>8}{'(not skill)':>12}")
        else:
            print(
                f"  {label:28}{row['annualized_mean'] * 100:>12.2f}%{row['ir']:>9.2f}"
                f"{row['t_stat']:>8.2f}{row['share_of_variance'] * 100:>9.1f}%"
            )

    _line("active beta · expected", "beta_expected", not_skill=True)
    _line("active beta · surprise", "beta_surprise", not_skill=True)
    _line("timing (δβ·δr)", "timing")
    for name in r["risk_factor_names"]:
        _line(f"{name} factor", name)
    for name in r["signal_names"]:
        _line(name, name)
    _line("specific (stock-picking)", "specific")

    c = r["cumulation"]
    warn = (
        "  ⚠ large relative to attributed terms — degrade to per-period detail"
        if r["cumulation_unreliable"]
        else ""
    )
    print(
        f"\n  cumulative active return: {c['honest_car'] * 100:+.2f}% "
        f"(top-down parts {sum(c['linked_components'].values()) * 100:+.2f}% + "
        f"δ_CP {c['delta_cp'] * 100:+.2f}%){warn}"
    )
    print(
        f"  guardrails: {r['n_rows']} attributed rows, P(any |t|>2 in {r['n_trials']} trials) = "
        f"{r['multiple_testing_inflation']:.2f}; best row '{r['best_row']}' t={r['best_row_t_stat']:+.2f}"
        + ("  | ⚠ realized IR > 2 — suspect a bug/leak" if r["sanity_ceiling_breached"] else "")
    )
    print(
        f"\n  Verdict: IR {r['total_active_ir']:+.2f} ± {r['total_active_ir_se']:.2f} "
        f"(Y={r['years']:.1f}yr): {r['verdict']}; Y*≈{r['years_to_significance']:.0f}yr at this IR"
    )


def _maybe_print_attribution_verdict(
    data_client, strategy: str, symbols, start, end, benchmark: str = "SPY"
) -> None:
    """A compact attribution verdict appended to backtest/walk-forward output once
    there's enough history to trust it (the verdict line belongs
    on every backtest with >= 1 year of history, not gated behind a separate
    `info --attribution` call). Silently skipped below a year or on insufficient data."""
    if (end - start).days < 365:
        return
    from tradeflow.services.analysis import compute_attribution

    r = compute_attribution(data_client, strategy, symbols, start, end, benchmark=benchmark, n_points=16)
    if not r.get("periods"):
        return
    print(
        f"\nAttribution verdict: IR {r['total_active_ir']:+.2f} ± {r['total_active_ir_se']:.2f} "
        f"(Y={r['years']:.1f}yr): {r['verdict']}; Y*≈{r['years_to_significance']:.0f}yr at this IR"
    )
    print(
        f"  best row '{r['best_row']}' t={r['best_row_t_stat']:+.2f} of {r['n_rows']} attributed rows "
        f"(P(any |t|>2) = {r['multiple_testing_inflation']:.2f}) — run `info --attribution` for the full breakdown"
    )


def cmd_horizon(args) -> None:
    """Print the alpha-decay curve, half-life, recommended cadence, and lagged blend.

    Read-only research diagnostic: measures how fast the signal's IC decays and turns
    that into a rebalance cadence and a current/lagged blend. Produces no orders.
    """
    tuned = apply_run_config(args)

    from tradeflow.services.analysis import compute_horizon

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    r = compute_horizon(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
        config=tuned,
        source=args.source,
        scanner=args.scanner,
        benchmark=args.benchmark,
        neutralize_factors=args.neutralize_factors,
        max_lag=args.max_lag,
        timeframe=args.timeframe,
    )
    if not r.get("ic_by_lag"):
        print(r.get("note", "No horizon report produced."))
        return

    print(f"\nInformation horizon: '{args.strategy}' {args.start:%Y-%m-%d}..{args.end:%Y-%m-%d}")
    print("  IC by lag: " + "  ".join(f"{n}:{ic:+.3f}" for n, ic in sorted(r["ic_by_lag"].items())))

    def _hl_str(v) -> str:
        return f"{v:.1f}" if isinstance(v, (int, float)) and v == v and v != float("inf") else "∞"

    hl = r["half_life"]
    hl_str = f"{_hl_str(hl)} periods" if hl == hl and hl != float("inf") else "∞ (no decay detected)"
    print(f"  decay δ {r['decay_delta']:.3f}  half-life {hl_str}  fit R² {r['decay_r_squared']:.2f}")
    print(
        f"    CI (±1.96 SE on the fit): [{_hl_str(r.get('half_life_lower'))}, "
        f"{_hl_str(r.get('half_life_upper'))}] periods — the aim policy discounts "
        "using the upper bound (conservative against killing a good signal)"
    )
    print(
        f"  recommended cadence: every {r['recommended_cadence']} periods  "
        f"(best return horizon ≈ {r['peak_return_horizon']:.1f})"
    )
    rec = "recommended" if r["blend_recommended"] else "not worth the turnover cost"
    print(
        f"  lagged blend [{r['blend_regime']}]: w_now {r['blend_weight_now']:+.2f}  "
        f"w_lagged {r['blend_weight_lagged']:+.2f}  (ρ {r['signal_autocorrelation']:.2f}, "
        f"cost {r['blend_annual_cost']:.2%}/yr → {rec})"
    )
    print(f"  ({r['blend_superseded_by']})")


def cmd_cache(args) -> None:
    """Inspect/warm the bar cache: persistent OHLCV bars behind
    ``MarketDataProvider``, so a repeated backtest/optimize/walkforward request
    reuses previously-fetched data instead of re-hitting Alpaca.
    """
    from tradeflow.services.data import build_data_client
    from tradeflow.store.bars import CachedMarketData

    if args.cache_command == "status":
        # No network needed - open the cache directly.
        cached = CachedMarketData(_NullProvider(), cache_dir=args.cache_dir, offline=True)
        info = cached.status(symbols=args.symbols, timeframe=args.timeframe)
        print(f"Cache dir : {info['cache_dir']}")
        print(f"Coverage DB: {info['db_path']}")
        if not info["entries"]:
            print("No cached symbols.")
        else:
            print(f"{'SYMBOL':10}{'TIMEFRAME':12}{'COVERED FROM':26}{'COVERED TO':26}LAST FETCHED")
            for e in info["entries"]:
                print(f"{e['symbol']:10}{e['timeframe']:12}{e['lo']:26}{e['hi']:26}{e['last_fetch']}")
        if info["drift"]:
            print(
                f"\nDRIFT DETECTED — coverage claims data no longer on disk for: {', '.join(info['drift'])}"
            )
            print("Run `cache refresh` for those symbols.")
        else:
            print("\nOK — no drift detected.")
        return

    data_client = build_data_client(cache=True, cache_dir=args.cache_dir)
    provider = data_client.provider
    assert isinstance(provider, CachedMarketData)  # build_data_client(cache=True) guarantees this

    if args.cache_command == "warm":
        universe = resolve_universe(
            data_client, args.scanner, args.symbols, as_of=args.scan_as_of or args.end
        )
        summary = provider.warm(universe, args.timeframe, args.start, args.end)
        for symbol, s in summary.items():
            state = "already cached" if s["already_cached"] else f"fetched {s['gaps_fetched']} gap(s)"
            print(f"{symbol}: {state}")
        return

    # refresh
    summary = provider.refresh(args.symbols, args.timeframe, args.start, args.end)
    for symbol, s in summary.items():
        if s["refreshed"]:
            print(f"{symbol}: refreshed [{s['start']}, {s['end']}]")
        else:
            print(f"{symbol}: skipped — {s['reason']}")


class _NullProvider(MarketDataProvider):
    """A stand-in upstream for `cache status`, which never fetches (offline=True
    on the ``CachedMarketData`` it's wrapped in) - so inspecting the local cache
    needs no Alpaca credentials at all."""

    def get_bars(self, symbols, timeframe, start, end):  # pragma: no cover - never called
        raise RuntimeError("cache status does not fetch")

    async def stream_bars(self, symbols, handler):  # pragma: no cover - never called
        raise RuntimeError("cache status does not stream")

    def supports_streaming(self) -> bool:
        return False


def cmd_trials(args) -> None:
    """Inspect the trial store: the queryable index over the research
    journal that lets a campaign-level Deflated Sharpe count every config you've
    ever tried, not just the ones from this process.
    """
    from tradeflow.store.trials import DEFAULT_JOURNAL_PATH, TrialStore

    # `query` has no --journal flag; status/rebuild do. Passing it here too means
    # a schema-mismatch rebuild (which can fire inside the constructor) replays
    # the journal the user actually pointed at, not the global default.
    with TrialStore(args.db, journal_path=getattr(args, "journal", None)) as store:
        if args.trials_command == "status":
            info = store.status(args.journal)
            print(f"DB      : {info['db_path']}")
            print(f"Journal : {info['journal_path']}")
            print(f"Schema  : v{info['schema_version']}")
            print(f"Rows    : {info['rows']}")
            print(f"Journal : {info['journal_lines']} lines ({info['journal_trial_lines']} trials)")
            if info["orphaned_rows"]:
                print(f"Orphaned: {info['orphaned_rows']} row(s) with no strategy (session_start missing)")
            if info["drift"]:
                print(
                    "\nDRIFT DETECTED — rows undercount journaled trials, or rows are orphaned. Run `trials rebuild`."
                )
            else:
                print("\nOK — no drift detected.")
            return

        if args.trials_command == "rebuild":
            journal = args.journal or DEFAULT_JOURNAL_PATH
            stats = store.rebuild(args.journal)
            print(
                f"Rebuilt {stats['rows']} trial rows from {stats['journal_lines']} journal lines ({journal})."
            )
            return

        if args.trials_command == "promote":
            _promote_trial(store, args)
            return

        if args.trials_command == "show":
            _print_trial_detail(store, args)
            return

        if args.trials_command == "best":
            _print_leaderboard(store, args)
            return

        # list (and the `query` alias) — defaults to the current accounting version;
        # --all-accounting opts into a listing that spans versions.
        _print_trials_list(store, args)


def _trial_filters(args) -> Dict[str, Any]:
    """The filter kwargs shared by `trials list` and `trials best`."""
    return {
        "strategy": args.strategy,
        "kind": args.kind,
        "symbols": args.symbols,
        "since": args.since,
        "until": args.until,
        "min_sharpe": args.min_sharpe,
        "promotable": args.gates_passed,
        "accounting": args.accounting,
        "all_accounting": args.all_accounting,
    }


def _print_trials_list(store, args) -> None:
    import json

    from tradeflow.analytics.reporting import format_trials_table

    filters = _trial_filters(args)
    rows = store.list_trials(sort=args.sort, limit=args.limit, offset=args.offset, **filters)
    total = store.count_trials(**filters)
    if args.json:
        print(json.dumps({"rows": rows, "total": total}, indent=2, default=str))
        return
    print(format_trials_table(rows, total=total))
    if args.strategy and args.symbols:
        from tradeflow.engine.backtest import ACCOUNTING_VERSION

        accounting = args.accounting if args.accounting is not None else ACCOUNTING_VERSION
        n = store.family_count(args.strategy, args.symbols, accounting)
        print(
            f"\nCampaign n_trials for '{args.strategy}' over {', '.join(args.symbols)} "
            f"(accounting v{accounting}): {n}"
        )


def _promote_trial(store, args) -> None:
    """Write a portable config from a trial the campaign already validated.

    `--save-config` writes the chosen config *after* a walk-forward, so saving one you
    have already validated meant validating it again - and the memo only serves an
    identical recipe, which it is not once a seed has changed to ask a different
    question. This reads the recorded trial instead.

    Deliberately *not* a fast path through validation. The trial store holds what a
    config needs because a real validation put it there; reading from it cannot bless
    state that was never validated, whereas a `--skip-validation` flag would - and
    would be reached for exactly when someone is in a hurry, which is when it matters
    most that a saved config means what it says.
    """
    from tradeflow.optimization.config_store import Provenance, save_config
    from tradeflow.services.audit import universe_for_trial

    trial = store.get_trial(args.trial_id)
    if trial is None:
        sys.exit(f"No trial with id {args.trial_id!r}. Try `trials list` to see what is recorded.")

    promotable = trial.get("promotable")
    if not promotable and not args.force:
        sys.exit(
            f"Trial {args.trial_id} is not promotable (promotable={promotable!r}). Promoting it "
            "would put a config on disk whose own provenance says it did not clear the gates. "
            "Re-run `trials show` to see why, or pass --force to save it with that verdict recorded."
        )

    universe = universe_for_trial(args.trial_id) or {}
    symbols = universe.get("symbols")
    if not symbols:
        sys.exit(
            f"Trial {args.trial_id} has no universe in the journal, so a config promoted from it "
            "could not name the book it validated. The trial store records a universe *hash*, not "
            "the symbols; the journal is where they live."
        )

    # The journaled params carry the dedup key's reserved entries (`_cost` and
    # friends). They are not strategy parameters - the schema has its own `cost` field -
    # and passing them through would leave a stray key in every constructed strategy.
    recorded = dict(trial.get("params") or {})
    cost = recorded.pop("_cost", None)
    params = {name: value for name, value in recorded.items() if not name.startswith("_")}

    path = save_config(
        args.save_config,
        strategy=trial.get("strategy"),
        scanner=None,
        params=params,
        cost=cost,
        symbols=symbols,
        candidate_symbols=universe.get("candidate_symbols"),
        provenance=Provenance(
            objective="",
            windows={"start": trial.get("window_start"), "end": trial.get("window_end")},
            oos_metrics=trial.get("metrics") or {},
            n_trials=int(trial.get("n_trials_in_session") or 1),
            seed=trial.get("seed"),
            git_sha=trial.get("git_sha"),
            timestamp=trial.get("ts"),
            accounting=int(trial.get("accounting") or 1),
            notes=f"promoted from trial {args.trial_id}"
            + ("" if promotable else " (NOT promotable at the time of promotion)"),
        ),
    )
    print(f"Promoted trial {args.trial_id} -> {path}")
    print(f"  strategy {trial.get('strategy')!r}  universe {len(symbols)} symbols", end="")
    if universe.get("candidate_symbols"):
        print(f" resolved from {len(universe['candidate_symbols'])} candidates")
    else:
        # Distinguishable from a config whose candidates equal its universe, because
        # `--re-resolve-universe` can only be honest about one of those.
        print(" (no candidate list recorded - --re-resolve-universe will say so)")
    if not promotable:
        print("  WARNING: this trial did not clear its gates; the config records that verdict.")
    print("  Saving a config never trades it - a human promotes it to live.")


def _print_trial_detail(store, args) -> None:
    import json

    from tradeflow.analytics.reporting import format_trial_detail, format_trial_trades

    trial = store.get_trial(args.trial_id)
    if trial is None:
        sys.exit(f"No trial with id {args.trial_id!r}. Try `trials list` to see what is recorded.")
    if args.json:
        print(json.dumps(trial, indent=2, default=str))
        return
    print(format_trial_detail(trial))
    if trial.get("trades"):
        print(format_trial_trades(trial["trades"], limit=args.trades_limit))


def _print_leaderboard(store, args) -> None:
    import json

    from tradeflow.analytics.reporting import format_leaderboard

    board = store.best(
        rank_by=args.rank_by,
        limit=args.limit,
        include_in_sample=args.include_in_sample,
        **_trial_filters(args),
    )
    if args.json:
        # The honesty rules live in the payload, not only in the formatting: an agent
        # reading this over a wire must see the family counts and the caveat too.
        print(json.dumps(board, indent=2, default=str))
        return
    print(format_leaderboard(board))


def cmd_mcp(args) -> None:
    """Serve TradeFlow over MCP (stdio). Opt-in; requires the ``mcp`` extra.

    Live trading is intentionally not exposed: the server builds
    only a data client, so it cannot place orders.
    """
    # The guard has to wrap `serve()`, not just the import above it. The server
    # module imports fine without the extra — it pulls in the MCP SDK lazily, inside
    # build_server — so guarding only the import let the real failure escape as a
    # traceback from several frames down, which is precisely the moment someone is
    # trying to connect a client for the first time.
    try:
        from tradeflow.mcp.server import serve

        serve()
    except ImportError:
        sys.exit(_missing_extra_message("mcp", "The MCP server"))


def _missing_extra_message(extra: str, what: str) -> str:
    """How to install an optional extra, phrased for the copy that is running.

    An installed copy has no checkout to `uv sync` in, so telling it to is a dead
    end — the same failure the missing-credentials message used to have.
    """
    from tradeflow.settings import _looks_like_checkout, state_root

    if _looks_like_checkout(state_root()):
        command = f"uv sync --extra {extra}"
    else:
        command = f'uv tool install --force "tradeflow-engine[{extra}]"'
    return f"{what} needs the '{extra}' extra. Install it:\n    {command}"


# The book limits a live run may override from the command line, and the
# `position_limits` key each one sets.
_LIMIT_OVERRIDES = (
    ("max_positions", "max_positions", "--max-positions"),
    ("max_position_size", "max_position_size", "--max-position-size"),
    ("max_gross_exposure", "max_gross_exposure", "--max-gross-exposure"),
    ("max_total_risk", "max_total_risk", "--max-total-risk"),
    ("min_notional", "min_notional", "--min-notional"),
)


def _live_feed(args) -> Optional[str]:
    """The feed this run pins, from the flag or the environment.

    Unset stays unset: pinning a default would mean an entitled account silently
    trading a partial venue or a delayed tape with nothing in the output to say so.
    """
    from tradeflow.settings import data_feed

    return getattr(args, "feed", None) or data_feed()


def _apply_limit_overrides(args, strategy) -> None:
    """Let the command line override the book limits a saved config or strategy declares.

    Only flags actually typed apply. `--max-positions` carries a default, so applying it
    unconditionally would let an untyped default silently overrule the limit a frozen
    config exists to pin.
    """
    given = getattr(args, "flags_given", set())
    overrides = {
        key: getattr(args, dest)
        for key, dest, _ in _LIMIT_OVERRIDES
        if dest in given and getattr(args, dest, None) is not None
    }
    if overrides:
        strategy.config["position_limits"] = {**strategy.position_limits(), **overrides}


def _refuse_inert_flags(args) -> None:
    """Stop a run whose typed flag cannot reach anything.

    A sizing flag that needs `--portfolio` or `--beta-sizing` is otherwise parsed and
    discarded in silence, so the preflight prints the limits the run really has while
    the operator reads the ones they asked for and concludes the two agree.
    """
    given = getattr(args, "flags_given", set())
    if "max_weight" in given and not args.portfolio:
        equivalent = ""
        capital = _live_capital(args)
        if capital:
            equivalent = (
                f"\n  For a per-position ceiling on this book, that weight is "
                f"--max-position-size {args.max_weight * capital:.0f} "
                f"({args.max_weight:.0%} of ${capital:,.0f})."
            )
        sys.exit(
            f"--max-weight {args.max_weight} sizes the --portfolio allocator, which this "
            f"run does not use, so it would have been ignored.\n"
            f"  Add --portfolio to allocate, or drop the flag.{equivalent}"
        )
    if "benchmark" in given and not args.beta_sizing:
        sys.exit(
            f"--benchmark {args.benchmark} selects the symbol beta sizing measures against, "
            f"and this run does not size by beta, so it would have been ignored.\n"
            f"  Add --beta-sizing to use it, or drop the flag."
        )


def cmd_live(args) -> None:
    from tradeflow.engine.live import BlindStartError, LiveEngine
    from tradeflow.execution.live_trader import LiveTrader
    from tradeflow.services.sizing import build_beta_sizer, build_portfolio_weight_sizer

    tuned = apply_run_config(args)
    strategy = _strategy_from(args, tuned)
    _apply_limit_overrides(args, strategy)
    _refuse_inert_flags(args)
    capital = _live_capital(args)

    if getattr(args, "no_reaffirm_entries", False):
        strategy.config["reaffirm_entries"] = False
        logger.info("Entry re-affirmation off: waiting for a fresh crossing before entering")

    if args.portfolio:
        _refuse_contradictory_portfolio_cardinality(strategy, args.max_positions)

    broker, data_client = build_data_and_broker(feed=_live_feed(args))
    universe = resolve_universe(data_client, args.scanner, args.symbols)

    sizer = None
    if args.portfolio:
        from tradeflow.brokers.errors import BrokerError

        try:
            account = broker.get_account()
        except BrokerError as exc:
            logger.warning("Could not read the account (%s); sizing against a nominal $100k", exc)
            account = None
        equity = account.equity if account else 100_000.0
        sizer = build_portfolio_weight_sizer(
            data_client, equity, universe, "1Day", args.max_positions, args.max_weight
        )
        if sizer is not None:
            universe = sizer.symbols  # trade only the funded names
    elif args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark)

    from tradeflow.engine.barcheck import BarChecks, BarQualityFilter
    from tradeflow.execution.ledger import PositionLedger
    from tradeflow.marketdata.timeframe import Timeframe

    # Guards and the ledger are on by default: the live path is the only place a
    # bad bar or a missed fill costs money, and both are opt-*out* for that reason.
    bar_filter = None
    if not args.no_bar_checks:
        timeframe = Timeframe.parse(strategy.config["timeframe"])
        bar_filter = BarQualityFilter(
            checks=BarChecks(max_return=args.max_bar_return),
            interval=timeframe.duration(),
        )
    ledger = None if args.no_ledger else PositionLedger()

    _print_live_preflight(args, strategy, broker, universe, capital, ledger)
    _refuse_ambiguous_broker_mode(args)
    if getattr(args, "preflight", False):
        print("\n--preflight: nothing was started and no order path ran.")
        return

    engine = LiveEngine(
        strategy,
        data_client,
        LiveTrader(broker, strategy, sizer=sizer, capital=capital),
        bar_filter=bar_filter,
        ledger=ledger,
        reconcile_every=args.reconcile_every,
        allow_blind_start=args.allow_blind_start,
    )
    try:
        asyncio.run(engine.start(universe))
    except BlindStartError as exc:
        sys.exit(f"Refusing to start: {exc}")
    except KeyboardInterrupt:
        # Ctrl-C is how a live session is meant to end. A traceback here reads as a
        # crash, and buries whether anything was left open.
        print("\nInterrupted - live engine stopped. No new positions were opened.")
    finally:
        if bar_filter is not None:
            report = bar_filter.report()
            logger.info(
                "Bar quality: %d of %d rejected (%.1f%%) %s",
                report["rejected"],
                report["seen"],
                report["rate"] * 100,
                report["by_reason"] or "",
            )
            if report["elevated"]:
                logger.error(
                    "ELEVATED bar-rejection rate — this is a data-feed problem wearing "
                    "a quiet market's clothes. Investigate before trusting these results."
                )


def _refuse_ambiguous_broker_mode(args) -> None:
    """Refuse to reach the order path on real money without it being said out loud.

    ``PAPER_TRADE`` defaults to true, which is the right default and also the reason
    this check exists: a default nobody set is indistinguishable from a decision
    somebody made, right up until it is wrong. A live run therefore has to be asserted
    on the command line as well as in the environment - two independent statements of
    the same intent, because one of them can be inherited from a shell nobody remembers
    exporting.
    """
    from tradeflow.settings import paper_trade_mode

    if paper_trade_mode():
        return
    if getattr(args, "live_money", False):
        print("  LIVE MONEY confirmed by --live-money. Orders will use real capital.")
        return
    raise SystemExit(
        "PAPER_TRADE is false, so this would place orders with real money. Refusing to "
        "start on an environment variable alone: pass --live-money to say so on the "
        "command line, or set PAPER_TRADE=true to trade the paper account."
    )


def _live_capital(args) -> Optional[float]:
    """How much of the account this run may deploy.

    A paper account arrives with whatever equity the venue handed it - typically far
    more than the capital a config was validated at - and sizing against that balance
    trades a different book from the one that was tested. That does not merely flatter
    the result; it invalidates the execution telemetry the run exists to gather, because
    fills, slippage and rounding are all properties of a book at a size.

    ``None`` means "the whole account", which is the historical behaviour and stays the
    default for a run that never mentioned capital.
    """
    capital = getattr(args, "capital", None)
    if capital:
        return float(capital)
    return None


def _print_live_preflight(args, strategy, broker, universe, capital, ledger) -> None:
    """The exact contract this run will trade under, printed before any order logic.

    Everything here is a read. It places nothing, and it is printed on every live run
    rather than only under ``--preflight`` - the point is that the contract is never
    implicit, and a check you have to remember to ask for is one that gets skipped
    exactly when it matters.
    """
    from tradeflow.execution.halt import HaltState
    from tradeflow.services.audit import DEFAULT_TRIAL_JOURNAL
    from tradeflow.settings import paper_trade_mode

    limits = strategy.position_limits()
    account = None
    try:
        account = broker.get_account()
    except Exception as exc:  # noqa: BLE001 - a preflight reports, it does not fail the run
        print(f"  (account unreadable: {exc})")

    print("\n=== Live preflight ===")
    mode = "PAPER" if paper_trade_mode() else "LIVE - REAL MONEY"
    print(f"  {'broker mode':22}{mode}")
    if account is not None:
        print(f"  {'account':22}equity ${account.equity:,.2f}  cash ${account.cash:,.2f}")
    deployable = f"${capital:,.2f}" if capital else "the whole account (no --capital set)"
    print(f"  {'capital this run':22}{deployable}")
    print(
        f"  {'universe':22}{len(universe)} symbols ({'replayed' if args.scanner == 'none' else args.scanner})"
    )
    # Warm-up and the stream must agree. Left unset the SDK defaults disagree - the
    # historical half resolves to the full tape, the stream to a single venue - so an
    # account entitled to one and not the other warms up on nothing and streams fine.
    feed = _live_feed(args)
    print(f"  {'data feed':22}{feed or 'SDK default (full tape for history, IEX for the stream)'}")

    # Absent is not zero: a limit nobody declared is unbounded, and must not read as 0.
    def _limit(value, render):
        return render(value) if value is not None else "unset"

    print(f"  {'max positions':22}{_limit(limits.get('max_positions'), str)}")
    size = limits.get("max_position_size")
    # A per-position ceiling above the whole deployable book cannot bind. Saying so
    # here stops it reading as a limit somebody chose.
    inert = (
        "  (above this run's capital - not a binding limit)" if size and capital and size >= capital else ""
    )
    print(f"  {'max position size':22}{_limit(size, lambda v: f'${v:,.2f}')}{inert}")
    # Gross exposure is a fraction of deployable capital, so show the dollars it means
    # here - the fraction alone is the one number an operator cannot sanity-check.
    gross = limits.get("max_gross_exposure")
    if gross is not None and capital:
        print(f"  {'max gross exposure':22}{gross:g} ({gross:.0%} of capital = ${gross * capital:,.2f})")
    else:
        print(f"  {'max gross exposure':22}{_limit(gross, lambda v: f'{v:g} ({v:.0%} of capital)')}")
    risk = limits.get("max_total_risk")
    if risk is not None and capital:
        print(f"  {'max total risk':22}{risk:g} ({risk:.0%} of capital = ${risk * capital:,.2f})")
    else:
        print(f"  {'max total risk':22}{_limit(risk, lambda v: f'{v:g} ({v:.0%} of capital)')}")
    print(f"  {'min notional':22}{_limit(limits.get('min_notional'), lambda v: f'${v:,.2f}')}")
    print(
        f"  {'entries':22}{'re-affirmed' if strategy.config.get('reaffirm_entries', True) else 'fresh crossings only'}"
    )
    print(f"  {'bar guards':22}{'off' if args.no_bar_checks else 'on'}")
    print(f"  {'reconcile every':22}{args.reconcile_every:g}s")
    # Where the drift data lands. Telemetry nobody can find is telemetry nobody checks.
    print(f"  {'ledger':22}{ledger.path if ledger is not None else 'DISABLED (--no-ledger)'}")
    print(f"  {'journal':22}{DEFAULT_TRIAL_JOURNAL}")
    print(f"  {'halt state':22}{HaltState().path}")


def _refuse_contradictory_portfolio_cardinality(strategy, allocator_max_positions) -> None:
    """Refuse a live portfolio run whose two position caps cannot both be honored.

    ``--max-positions`` bounds what the allocator may fund; the strategy's
    ``position_limits.max_positions`` bounds what the live book will hold. When the
    first exceeds the second, the book is truncated to the smaller number and which
    names survive is decided by which signals arrive first, not by the allocation -
    so what trades is not the book that was allocated, and not the one that was
    validated. There is no reading of that configuration worth starting.

    Checked against the declared caps rather than the funded set, and before the
    broker is reached. The allocator hard-caps its own cardinality, so a run that
    happens to fund few enough names today would otherwise start and the same
    configuration would truncate tomorrow; and the operator learns this before a
    market-data fetch and a solve rather than after.
    """
    limit = (strategy.position_limits() or {}).get("max_positions")
    if not limit or not allocator_max_positions:
        return
    limit, funded_cap = int(limit), int(allocator_max_positions)
    if funded_cap <= limit:
        return
    raise SystemExit(
        f"--portfolio would fund up to {funded_cap} names, but this strategy holds at most "
        f"{limit} (position_limits.max_positions), so {funded_cap - limit} of them would never "
        f"be traded. Reconcile the two before trading: raise position_limits.max_positions to "
        f"{funded_cap} in the strategy config, or pass --max-positions {limit}."
    )


def cmd_halt(args) -> None:
    """Record a halt. Blocks new entries; never blocks an exit."""
    from tradeflow.execution.halt import HaltState

    halt = HaltState().set(args.reason, actor="cli", scope=args.scope)
    print(f"Halted. {halt}")
    print("New entries are refused while this stands. Exits are still allowed.")
    print(f"Lift it with:  tradeflow resume {args.scope}")


def cmd_resume(args) -> None:
    """Lift a halt."""
    from tradeflow.execution.halt import HaltState

    if HaltState().clear(args.scope):
        print(f"Halt lifted for {args.scope!r}. Trading may resume.")
    else:
        print(f"No halt was in force for {args.scope!r}; nothing to lift.")


def cmd_halts(args) -> None:
    """Show what is currently halted."""
    from tradeflow.execution.halt import HaltState

    halts = HaltState().list()
    if not halts:
        print("Nothing is halted.")
        return
    print("Active halts:")
    for halt in halts:
        print(f"  {halt}")


def cmd_flatten(args) -> None:
    """Halt, cancel every open order, and close every position.

    Deliberately goes straight to the broker rather than through the engine, so it
    works when the engine is wedged or holding state you no longer believe.
    """
    from tradeflow.execution.flatten import flatten

    if not args.confirm:
        raise SystemExit(
            "flatten closes every position and halts trading. Re-run with --confirm once you are sure."
        )
    broker, _ = build_data_and_broker()
    report = flatten(broker, reason=args.reason, actor="cli")
    if args.json:
        import json

        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.summary())
    if not report.complete:
        raise SystemExit(1)


def cmd_reconcile(args) -> None:
    """Compare the position ledger against the broker's actual account state.

    Answers "is my book what I think it is" on demand. Read-only and advisory: the
    broker is authoritative, and nothing is ever corrected automatically — an
    automated fix for a missed fill is how a position gets doubled unattended.
    """
    import json

    from tradeflow.execution.ledger import PositionLedger

    broker, _ = build_data_and_broker()
    report = PositionLedger(args.ledger).reconcile(broker)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.summary())
    if not report.clean:
        raise SystemExit(1)


def cmd_demo(args) -> None:
    """Run the whole pipeline on synthetic data - no Alpaca keys, no network.

    The point isn't the numbers; it's the *shape* of the workflow. We backtest
    every registered strategy (in-sample, where everything looks plausible), then
    walk-forward one of them out-of-sample and let the promotion gates deliver the
    verdict. The data is a seeded random walk with no real edge - so a healthy run
    ends in "NOT promotable", which is exactly the honesty the engine exists for.
    """
    from tradeflow.engine.backtest import BacktestEngine
    from tradeflow.marketdata.client import MarketDataClient
    from tradeflow.marketdata.synthetic import SyntheticMarketData
    from tradeflow.marketdata.timeframe import Timeframe
    from tradeflow.optimization.walk_forward import WalkForwardValidator

    data_client = MarketDataClient(SyntheticMarketData(seed=args.seed))
    symbols = ["SYNW", "SYNX", "SYNY", "SYNZ"]
    anchor = datetime(2024, 12, 31)

    def window_for(timeframe: str) -> tuple:
        """A window wide enough for the timeframe (intraday needs fewer days)."""
        unit = Timeframe.parse(timeframe).unit
        days = 60 if unit in ("min", "hour") else 730
        return anchor - timedelta(days=days), anchor

    print("\n" + "=" * 70)
    print("  TradeFlow demo — synthetic data, no API keys, no network")
    print("  (a seeded random walk: realistic-looking, no actual edge)")
    print("=" * 70)

    print("\n1) In-sample backtest of every registered strategy")
    print("   In-sample, almost anything looks tradeable. That's the trap.\n")
    print(f"   {'STRATEGY':18}{'RETURN':>10}{'SHARPE':>9}{'TRADES':>8}")
    print(f"   {'-' * 43}")
    chart_result = None  # the wf strategy's in-sample BacktestResult, for the chart
    wf_strategy = "ma_crossover"
    for name, cls in STRATEGIES.items():
        try:
            strategy = cls.create_with_defaults()
            start, end = window_for(strategy.config["timeframe"])
            result = BacktestEngine(strategy, data_client).run(symbols, start, end, 100_000.0)
            if name == wf_strategy:
                chart_result = result
            m = result.metrics
            print(
                f"   {name:18}{m.get('total_return', 0.0):>9.2f}%"
                f"{m.get('sharpe_ratio', 0.0):>9.2f}{int(m.get('total_trades', 0)):>8}"
            )
        except Exception as exc:  # noqa: BLE001 - demo should never hard-crash
            print(f"   {name:18}{'(skipped: ' + str(exc)[:30] + ')':>30}")

    print(f"\n2) Walk-forward validation of '{wf_strategy}' (the honest scorecard)")
    print("   Optimize in-sample, score out-of-sample across folds, then gate it.\n")
    start, end = window_for(STRATEGIES[wf_strategy].create_with_defaults().config["timeframe"])
    wf_result = WalkForwardValidator(
        STRATEGIES[wf_strategy], data_client, initial_capital=100_000.0, seed=args.seed
    ).run(
        symbols,
        start,
        end,
        n_folds=3,
        holdout_days=90,
        method="grid",
        objective="sharpe_ratio",
        max_evals=12,
    )
    report = wf_result.gate_report()

    print(
        f"   OOS Sharpe (median): {wf_result.median_oos('sharpe_ratio'):.2f}   "
        f"efficiency (OOS/IS): {wf_result.median_efficiency():.2f}   "
        f"OOS trades: {wf_result.total_oos_trades()}"
    )
    print("\n   Promotion gates:")
    for gate_name, check in report["checks"].items():
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"     [{mark}] {gate_name}: {check['value']} (threshold {check['threshold']})")
    verdict = "PROMOTABLE" if report["promotable"] else "NOT promotable"
    print(f"\n   Verdict: {verdict}")
    # The next step differs by how they got here: an installed copy has no Makefile
    # and no .env.example to copy, and telling them to use one is a dead end.
    print(
        "\n   No edge in a random walk → the gates refuse to promote it. That refusal\n"
        "   is the product. Point TradeFlow at real data:\n"
        f"{_next_step_hint()}\n"
    )

    if getattr(args, "chart", None):
        if chart_result is None:
            print("   Chart skipped: no backtest result was captured.")
        else:
            from tradeflow.analytics.charts import render_demo_summary

            try:
                path = render_demo_summary(
                    chart_result, wf_result, args.chart, strategy=wf_strategy, seed=args.seed
                )
                print(f"   Saved demo chart → {path}\n")
            except RuntimeError as exc:  # matplotlib (viz extra) not installed
                print(f"   Chart skipped: {exc}\n")


def cmd_demo_agent(args) -> None:
    """Narrate one research session end-to-end on live Alpaca data.

    The arc the guardrails produce: an AI proposes a strategy, the sandbox admits
    or rejects it, survivors are validated out-of-sample by walk-forward, the
    promotion gates deliver a verdict, and whatever is left is scored once on a
    holdout reserved before the search began. Nothing here can place an order.

    ``--provider replay`` (default) replays a curated proposal set, so the demo is
    deterministic and needs no LLM key. Any other provider drives a live model
    through the identical loop.
    """
    from tradeflow.costs import ParametricCostModel
    from tradeflow.research.agent import ResearchAgent, ResearchConfig
    from tradeflow.research.demo_proposals import DEMO_PROPOSALS
    from tradeflow.research.proposer import FixedProposer, build_proposer
    from tradeflow.services.data import build_data_client

    rule = "=" * 74
    print(f"\n{rule}")
    print("  TradeFlow demo-agent — AI proposal → sandbox → walk-forward → verdict")
    print("  Live Alpaca market data · research clock only · no order path exists")
    print(rule)

    end = args.end or (datetime.now() - timedelta(days=2))
    start = args.start or (end - timedelta(days=args.lookback_days))

    if args.provider == "replay":
        proposer = FixedProposer(DEMO_PROPOSALS)
        source = f"replayed proposal set ({len(DEMO_PROPOSALS)} proposals, deterministic)"
    else:
        proposer = build_proposer(args.provider, args.model, allow_code_gen=True)
        source = f"live model — {args.provider}/{proposer.client.model}"

    # Costs are charged on every simulated fill, in-sample and out. Validating on
    # gross returns systematically promotes turnover a strategy could not afford.
    cost_model = None if args.no_costs else ParametricCostModel()

    data_client = build_data_client()
    symbols = args.symbols

    state = {"round": 0}

    def narrate(event: str, payload: dict) -> None:
        if event == "session_start":
            research = payload["research_window"]
            holdout = payload["holdout_window"]
            print(f"\n  Proposals from : {source}")
            print(f"  Universe       : {', '.join(symbols)}  (bars fetched live from Alpaca)")
            costs = (
                "gross returns (--no-costs)"
                if cost_model is None
                else "commission + half-spread + sqrt impact, charged per fill"
            )
            print(f"  Costs          : {costs}")
            print(f"  Base strategy  : {payload['strategy']}")
            print(f"  Goal           : {payload['goal']}")
            print(f"\n  Research window: {research['start'][:10]} → {research['end'][:10]}")
            print(f"  Sacred holdout : {holdout['start'][:10]} → {holdout['end'][:10]}")
            print("                   reserved now, never searched, scored once at the end")
            return

        if event == "reject":
            state["round"] += 1
            print(f"\n  ── Round {state['round']} " + "─" * 52)
            hypothesis = payload.get("hypothesis") or "(no hypothesis supplied)"
            print(f"     Hypothesis  {_wrap_field(hypothesis)}")
            print(f"     Sandbox     REJECTED — {payload['reason']}")
            print("                 ↳ no bars loaded, no backtest run, no trial consumed")
            return

        if event == "trial":
            state["round"] += 1
            print(f"\n  ── Round {state['round']} " + "─" * 52)
            kind = "new strategy implementation" if payload["kind"] == "code" else "parameter config"
            print(f"     Proposal    [{payload['kind']}] {kind}")
            print(f"     Hypothesis  {_wrap_field(payload['hypothesis'])}")
            if payload["kind"] == "code":
                print("     Sandbox     ADMITTED — imports clean, contract valid, params within cap")
            print(
                f"     Walk-forward  in-sample Sharpe {payload['is_sharpe']:>6.2f}"
                f"   →   out-of-sample {payload['oos_sharpe']:>6.2f}"
                f"   (efficiency {payload['efficiency']:.2f})"
            )
            print(f"     Multiple-testing correction applied over {payload['n_trials_cumulative']} trials")
            print("     Promotion gates:")
            for name, check in payload["gate_report"]["checks"].items():
                mark = "PASS" if check["passed"] else "FAIL"
                print(
                    f"       [{mark}] {name:<24} {check['value']:>10.2f}   threshold {check['threshold']:.2f}"
                )
            if payload["advanced"]:
                print("     Verdict     PROMOTABLE — enters the shortlist")
            elif payload["promotable"]:
                print("     Verdict     passes gates but does not beat the incumbent — discarded")
            else:
                print("     Verdict     NOT promotable — discarded")
            return

        if event == "holdout_score":
            metrics = payload["holdout_metrics"] or {}
            print(f"\n  Holdout score (first and only look) — candidate {payload['candidate']}")
            print(f"     Sharpe {metrics.get('sharpe_ratio', 0.0):.2f}   config → {payload['saved']}")

    cfg = ResearchConfig(
        goal=args.goal,
        n_folds=args.folds,
        holdout_days=args.holdout_days,
        max_evals=args.max_evals,
        max_trials=len(DEMO_PROPOSALS) if args.provider == "replay" else args.max_trials,
        # The replay is built so the early rounds are *meant* to be rejected; the
        # default dryness stop would end the session before the last proposal ran.
        max_dry_rounds=len(DEMO_PROPOSALS) if args.provider == "replay" else args.max_dry_rounds,
        capital=args.capital,
        allow_code_gen=True,
        cost_model=cost_model,
    )
    agent = ResearchAgent(args.strategy, data_client, proposer, cfg, seed=args.seed, observer=narrate)
    result = agent.run(symbols, start, end)

    print(f"\n{rule}")
    print(f"  Session {agent.session_id} — stopped: {result.stopped_reason}")
    print(f"  {result.rounds} rounds, {result.n_trials_total} cumulative trials")
    print(rule)

    if not result.shortlist:
        print("\n  Shortlist: EMPTY — nothing cleared the gates.")
        print("\n  This is the intended outcome, not a failed run. Every proposal was either")
        print("  rejected before evaluation or failed out-of-sample. An agent that cannot")
        print("  produce a losing strategy quietly is an agent you cannot trust with a")
        print("  winning one.")
    else:
        print(f"\n  Shortlist: {len(result.shortlist)} candidate(s) for HUMAN review")
        for c in result.shortlist:
            oos = c.oos_metrics.get("sharpe_ratio", 0.0)
            hold = (c.holdout_metrics or {}).get("sharpe_ratio", 0.0)
            print(f"    [{c.id}] OOS Sharpe {oos:.2f} | holdout Sharpe {hold:.2f}")
            print(f"          {c.saved_path}")
        print("\n  These are provenance-stamped config files on disk. Nothing is live.")

    print("\n  What did NOT happen — structurally, not by policy:")
    print("    · No order was placed. This command builds a data client only.")
    print("    · PAPER_TRADE was never read, let alone flipped.")
    print("    · No model output reached the trade clock. Promotion is a human step.")
    print(f"\n  Full audit trail (every proposal, rejection, and gate): {result.journal_path}\n")


def _wrap_field(text: str, width: int = 58, indent: int = 17) -> str:
    """Wrap a long narration field under a fixed label column."""
    import textwrap

    lines = textwrap.wrap(text.strip(), width=width) or [""]
    pad = " " * indent
    return f"\n{pad}".join(lines)


# ---------------------------------------------------------------------------- #
# Argument parsing
# ---------------------------------------------------------------------------- #
def _symbols(value: str) -> List[str]:
    return [s.strip().upper() for s in value.split(",") if s.strip()]


#: The bare-flag default for --neutralize-factors: the risk-control factors.
#: Momentum is deliberately excluded — a momentum tilt is a return bet the alphas
#: may intend; regress it out explicitly if that's the goal.
DEFAULT_NEUTRAL_FACTORS = ["market", "volatility", "size"]


def _factors(value: str) -> List[str]:
    from tradeflow.risk.exposures import FACTOR_NAMES

    factors = [s.strip().lower() for s in value.split(",") if s.strip()]
    unknown = [f for f in factors if f not in FACTOR_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown factor(s) {', '.join(unknown)}; available: {', '.join(FACTOR_NAMES)}"
        )
    return factors


def _add_neutralize_factors_flag(parser, note: str = "") -> None:
    parser.add_argument(
        "--neutralize-factors",
        dest="neutralize_factors",
        type=_factors,
        nargs="?",
        const=DEFAULT_NEUTRAL_FACTORS,
        default=[],
        help="Factor-neutral alphas: regress out these risk-model exposures "
        f"(comma-separated; bare flag = {','.join(DEFAULT_NEUTRAL_FACTORS)} — momentum kept){note}",
    )


def _add_cost_flags(parser) -> None:
    """--gross/--commission-bps/--impact-eta/--borrow-bps, shared by every command
    that can price a fill (backtest/optimize/walkforward) so a search or
    validation is never silently gross by omission."""
    parser.add_argument(
        "--gross",
        action="store_true",
        help="Use GROSS returns (disable transaction cost; net is the default)",
    )
    parser.add_argument("--commission-bps", dest="commission_bps", type=float, default=1.0)
    parser.add_argument(
        "--impact-eta", dest="impact_eta", type=float, default=0.3, help="Market-impact coefficient"
    )
    parser.add_argument(
        "--borrow-bps", dest="borrow_bps", type=float, default=50.0, help="Annual short borrow rate (bps)"
    )


def _add_config_flag(parser) -> None:
    """``--config``: a saved run configuration, filling what the command line omits.

    One file for every run type, so a tuned config can be versioned beside the
    strategies it belongs to and then used to backtest, allocate, or produce a verdict
    without restating the universe each time.
    """
    parser.add_argument(
        "--config",
        default=None,
        help="Saved run config (walkforward --save-config). Supplies strategy, params, "
        "scanner, universe, capital and cost settings; any flag you pass wins over it. "
        "The saved universe is replayed as-is - see --re-resolve-universe.",
    )
    parser.add_argument(
        "--re-resolve-universe",
        dest="re_resolve_universe",
        action="store_true",
        help="Re-run the scanner instead of replaying the config's universe. Uses the "
        "saved candidate list where one was recorded, since re-scanning the resolved "
        "book would filter an already-filtered set rather than repeat the decision.",
    )


def _add_cache_flags(parser) -> None:
    """--cache/--offline/--cache-dir, on every read-only command that fetches bars.

    Not on `live`, which must read the market as it is, and not on `research`, which
    drives the others. Everything else that reads history takes them: a warm cache is
    useless if the command you want to run against it has no way to say so, and the
    commands that lacked these degraded into empty results when the network was down
    rather than reading the bars already on disk.

    ``--cache`` wraps the data client in the persistent bar cache
    (``tradeflow.store.bars.CachedMarketData``): a repeated request for the same
    symbols/window is served from local Parquet instead of re-fetched. ``--offline``
    additionally forbids any network call and implies ``--cache`` on its own — a
    byte-reproducible, cache-only run.
    """
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache fetched bars locally and reuse them on repeat requests (see `cache warm/status/refresh`)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Cache-only: never touch the network; fail loudly if the requested range isn't cached "
        "(implies --cache)",
    )
    parser.add_argument(
        "--cache-dir", dest="cache_dir", default=None, help="Bar cache directory (default: cache/bars)"
    )


def _date(value: str) -> datetime:
    """Parse a date or ISO datetime into a New York instant.

    **One date contract.** ``2026-08-22`` means that market date in the exchange's own
    zone, everywhere in this program. That is a domain decision: someone typing a date
    at a trading tool means the session, not UTC midnight.

    Leaving the value naive and letting each consumer localize it is what produced the
    bug this replaced. ``cache warm --end 2026-08-22`` recorded coverage through
    00:00Z (the store reads a naive datetime as UTC) while ``scan --as-of 2026-08-22``
    asked for 04:00Z (the scanner reads one as New York), so a cache holding exactly
    the right daily bar reported a four-hour hole and refused an offline run whose
    data was complete. Two readings of one string is a semantic fork; it surfaced in
    the cache first and would have surfaced next in reports, or in live/backtest
    parity.

    Returning an *aware* value is what stops the contract re-forking: a naive datetime
    carries no zone to disagree about because it carries no zone at all.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    if parsed.tzinfo is None:
        return NEW_YORK.localize(parsed)
    return parsed.astimezone(NEW_YORK)


def _now() -> datetime:
    """Now, on the clock every parsed date uses.

    The window defaults have to share the contract or they reintroduce the mixed-
    awareness comparison the contract exists to remove.
    """
    return datetime.now(NEW_YORK)


def _next_step_hint() -> str:
    """The "now try real data" instruction, phrased for the copy that is running.

    An installed copy has no Makefile and no `.env.example` to copy, so pointing at
    either is a dead end for exactly the user this path exists to serve.
    """
    from tradeflow.settings import _looks_like_checkout, state_root

    if _looks_like_checkout(state_root()):
        return (
            "     make init        add your free Alpaca paper keys (or see .env.example)\n"
            "     make backtest    then run it on real market data"
        )
    return (
        "     tradeflow init       add your free Alpaca paper keys\n"
        "     tradeflow verdict    then run the whole pipeline on real market data"
    )


def _version_banner() -> str:
    """The version, which copy is running, and where its state lives."""
    import tradeflow
    from tradeflow.settings import state_root

    # The package's own __version__, not importlib.metadata: the code that is
    # actually running is what the reader needs to know, and a metadata lookup can
    # resolve a stale or entirely different distribution that happens to share a
    # name.
    return (
        f"tradeflow {tradeflow.__version__}\n"
        f"  running from : {Path(tradeflow.__file__).parent}\n"
        f"  state root   : {state_root()}"
    )


class _VersionAction(argparse.Action):
    """Print the banner verbatim.

    argparse's built-in ``version`` action runs the text through the help formatter,
    which re-wraps it into one line — collapsing exactly the layout that makes the
    provenance readable.
    """

    def __init__(self, option_strings, dest=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(_version_banner())
        parser.exit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradeflow",
        description="TradeFlow — a broker-agnostic trading engine that's refreshingly "
        "honest about how hard making money actually is",
    )
    # Version *and* provenance. Someone with both an installed command and a checkout
    # will eventually run one thinking it is the other, and the difference is
    # invisible until a result disagrees — so say which copy this is and where its
    # state lives.
    parser.add_argument(
        "--version", action=_VersionAction, help="Show the version, which copy is running, and its state root"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p, *, with_dates: bool) -> None:
        p.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
        p.add_argument("--scanner", default="volume", help="Universe scanner ('none' to skip)")
        p.add_argument(
            "--symbols", type=_symbols, default=DEFAULT_UNIVERSE, help="Comma-separated candidate symbols"
        )
        if with_dates:
            p.add_argument("--start", type=_date, default=_now() - timedelta(days=30))
            p.add_argument("--end", type=_date, default=_now())
            p.add_argument("--capital", type=float, default=100_000.0)
            p.add_argument(
                "--scan-as-of",
                dest="scan_as_of",
                type=_date,
                default=None,
                help="Resolve the scanner at this date/datetime; defaults to --end for historical runs",
            )

    def add_no_journal(p) -> None:
        # Trials are journaled so a campaign-level Deflated Sharpe can count them
        # Opt out for throwaway/reproducibility runs you don't want
        # inflating the multiple-testing total.
        p.add_argument(
            "--no-journal",
            dest="no_journal",
            action="store_true",
            help="Do not record this run's trial(s) in the research journal",
        )

    def add_force(p) -> None:
        # An identical prior trial is served from the trial store instead of
        # re-run by default (see `trials query`); --force/--rerun re-verifies and
        # APPENDS a new trial rather than overwriting the memoized one.
        p.add_argument(
            "--force",
            "--rerun",
            dest="force",
            action="store_true",
            help="Re-run even if an identical trial already exists, instead of serving it from the trial "
            "store; appends a new trial rather than overwriting the prior one",
        )

    def add_config_flag(p) -> None:
        # One definition, shared. Two of these existed and drifted: the newer flags
        # reached the analysis commands and not `backtest`/`live`, which are the two
        # that had `--config` first. The old help was also stale - a contradictory
        # `--strategy` is refused now, not silently overridden.
        _add_config_flag(p)

    def add_workers_flag(p) -> None:
        # Default 1: sequential is not "a pool of one", it is the original code path
        # untouched. Memory scales with workers x per-worker bar frames, so the cap is
        # deliberately conservative and raising it is an explicit act.
        p.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Evaluate candidates across N worker processes (default 1 = sequential). "
            "Implies the bar cache; memory scales with N x the per-worker bar footprint",
        )

    def add_record_trades_flag(p) -> None:
        # Opt-in, not default: a long optimization campaign multiplying thousands of
        # candidates by hundreds of trades each is exactly the storage nobody asked
        # for. This is for the runs you intend to open again.
        p.add_argument(
            "--record-trades",
            dest="record_trades",
            action="store_true",
            help="Persist this run's trade table with its trial, so `trials show` can render it",
        )

    def add_html_flag(p) -> None:
        # One self-contained file per run: inline CSS, charts embedded as data URIs,
        # zero external requests when opened. Renders the same result object the
        # terminal report does, so the two can never disagree.
        p.add_argument(
            "--html",
            metavar="PATH",
            default=None,
            help="Write a self-contained HTML report of this run (no external requests; "
            "charts need the viz extra, and degrade to tables without it)",
        )

    bt = subparsers.add_parser("backtest", help="Run a historical backtest (did the idea ever work?)")
    add_common(bt, with_dates=True)
    add_config_flag(bt)
    bt.add_argument(
        "--beta-sizing",
        dest="beta_sizing",
        action="store_true",
        help="Scale position sizing inversely by each symbol's beta",
    )
    bt.add_argument("--benchmark", default="SPY", help="Benchmark symbol for beta")
    _add_cost_flags(bt)
    bt.add_argument(
        "--cost-stress",
        dest="cost_stress",
        nargs="?",
        const="all",
        default=None,
        choices=["all", "turnover", "borrow"],
        help="Re-run under scaled cost assumptions and show where the edge dies. "
        "'borrow' scales only the borrow rate, which a long-short book is exposed to "
        "differently: it is carry on inventory, not a toll on turnover.",
    )
    _add_cache_flags(bt)
    bt.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="render the equity curve + metrics to an image (needs the viz extra: matplotlib)",
    )
    add_html_flag(bt)
    add_record_trades_flag(bt)
    add_no_journal(bt)
    add_force(bt)
    bt.set_defaults(func=cmd_backtest)

    live = subparsers.add_parser("live", help="Run live/paper trading (paper by default, for your own good)")
    add_common(live, with_dates=False)
    add_config_flag(live)
    live.add_argument(
        "--portfolio",
        action="store_true",
        help="Size positions by OR-Tools portfolio weights instead of per-trade risk",
    )
    live.add_argument(
        "--beta-sizing",
        dest="beta_sizing",
        action="store_true",
        help="Scale position sizing inversely by each symbol's beta",
    )
    live.add_argument("--benchmark", default="SPY", help="Benchmark symbol for beta")
    live.add_argument(
        "--capital",
        type=float,
        default=None,
        help="How much of the account this run may deploy. A saved config supplies it; "
        "without either, sizing uses the whole account balance - which on a paper "
        "account is whatever the venue handed out, not what was validated.",
    )
    live.add_argument(
        "--preflight",
        action="store_true",
        help="Print the run contract and exit without starting the order path",
    )
    live.add_argument(
        "--live-money",
        dest="live_money",
        action="store_true",
        help="Acknowledge on the command line that PAPER_TRADE=false means real capital",
    )
    live.add_argument(
        "--max-positions",
        dest="max_positions",
        type=int,
        default=5,
        help="How many positions the book may hold at once. Overrides the strategy's and "
        "the saved config's limit, and bounds the allocator under --portfolio",
    )
    live.add_argument(
        "--max-position-size",
        dest="max_position_size",
        type=float,
        default=None,
        help="Dollar ceiling on any one position. Overrides the strategy's and the saved config's limit",
    )
    live.add_argument(
        "--max-gross-exposure",
        dest="max_gross_exposure",
        type=float,
        default=None,
        help="Ceiling on gross exposure as a fraction of deployable capital (0.9 = 90%%). "
        "Overrides the strategy's and the saved config's limit",
    )
    live.add_argument(
        "--max-total-risk",
        dest="max_total_risk",
        type=float,
        default=None,
        help="Ceiling on total risk across the book as a fraction of deployable capital. "
        "Overrides the strategy's and the saved config's limit",
    )
    live.add_argument(
        "--min-notional",
        dest="min_notional",
        type=float,
        default=None,
        help="Skip an entry whose sized order falls below this dollar value, rather than "
        "sending an order whose costs swamp it",
    )
    live.add_argument(
        "--feed",
        choices=DATA_FEEDS,
        default=None,
        help="Pin the Alpaca market-data feed for both warm-up and the live stream. "
        "Unset leaves the SDK's defaults, which is the right choice for an entitled "
        "account; unentitled keys generally need 'iex'",
    )
    live.add_argument(
        "--allow-blind-start",
        dest="allow_blind_start",
        action="store_true",
        help="Start even when no symbol warmed up. Indicators then begin from no history "
        "and emit confident-looking signals nothing has validated",
    )
    live.add_argument(
        "--max-weight",
        dest="max_weight",
        type=float,
        default=0.25,
        help="Largest weight the --portfolio allocator may give one name. Sizes the "
        "allocation only; use --max-position-size to bound what the book will hold",
    )
    # Guards are opt-OUT. The live loop is the only place a corrupt bar or a missed
    # fill costs money, so the safe configuration is the default one.
    live.add_argument(
        "--no-bar-checks",
        dest="no_bar_checks",
        action="store_true",
        help="Disable bar-quality guards (staleness, spikes, inconsistent OHLC, out-of-order). "
        "Guards reject a bad bar; they never repair one",
    )
    live.add_argument(
        "--max-bar-return",
        dest="max_bar_return",
        type=float,
        default=0.35,
        help="Reject a single-bar move larger than this fraction (default 0.35). Deliberately "
        "loose: a guard that vetoes real moves removes the strategy's best opportunities",
    )
    live.add_argument(
        "--no-ledger",
        dest="no_ledger",
        action="store_true",
        help="Do not record intended orders and observed fills for reconciliation",
    )
    live.add_argument(
        "--no-reaffirm-entries",
        dest="no_reaffirm_entries",
        action="store_true",
        help="Wait for a fresh crossing instead of opening a position whose entry signal "
        "fired before this process saw it — so starting into an established trend stays "
        "flat. Exits are never gated: a position the strategy no longer wants still closes",
    )
    live.add_argument(
        "--reconcile-every",
        dest="reconcile_every",
        type=float,
        default=300.0,
        help="Seconds between position-reconciliation sweeps (0 disables). Reports divergence "
        "from the broker's account state; never corrects it",
    )
    live.set_defaults(func=cmd_live)

    halt = subparsers.add_parser(
        "halt", help="Stop opening new positions (exits still allowed) — durable until resumed"
    )
    halt.add_argument("scope", nargs="?", default="all", help="'all', or a strategy class name")
    halt.add_argument("--reason", required=True, help="Why — recorded with the halt")
    halt.set_defaults(func=cmd_halt)

    resume = subparsers.add_parser("resume", help="Lift a halt")
    resume.add_argument("scope", nargs="?", default="all", help="'all', or a strategy class name")
    resume.set_defaults(func=cmd_resume)

    halts = subparsers.add_parser("halts", help="Show what is currently halted — read-only")
    halts.set_defaults(func=cmd_halts)

    flat = subparsers.add_parser(
        "flatten",
        help="Emergency: halt, cancel all orders, close all positions (bypasses the engine)",
    )
    flat.add_argument("--confirm", action="store_true", help="Required — this closes real positions")
    flat.add_argument("--reason", required=True, help="Why — recorded with the halt")
    flat.add_argument("--json", action="store_true", help="Emit the report as JSON")
    flat.set_defaults(func=cmd_flatten)

    reconcile = subparsers.add_parser(
        "reconcile",
        help="Check the position ledger against the broker's actual account state — read-only",
    )
    reconcile.add_argument("--ledger", default=None, help="Ledger path (default: logs/position_ledger.jsonl)")
    reconcile.add_argument("--json", action="store_true", help="Emit the report as JSON")
    reconcile.set_defaults(func=cmd_reconcile)

    scan = subparsers.add_parser("scan", help="Run the universe scanner only")
    scan.add_argument("--scanner", default="volume")
    scan.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    scan.add_argument(
        "--drift",
        action="store_true",
        help="Also re-scan at nearby clocks and report how much the universe moves. "
        "A config records the universe its scanner resolved, so drift is the gap "
        "between the validated book and the one a deployment would get.",
    )
    scan.add_argument(
        "--as-of",
        dest="as_of",
        type=_date,
        default=None,
        help="Resolve scanner state at this date/datetime instead of now",
    )
    _add_cache_flags(scan)
    scan.set_defaults(func=cmd_scan)

    alloc = subparsers.add_parser("allocate", help="Weight a portfolio over scanned symbols")
    alloc.add_argument("--scanner", default="volume")
    alloc.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    alloc.add_argument("--capital", type=float, default=100_000.0)
    alloc.add_argument("--max-positions", dest="max_positions", type=int, default=5)
    alloc.add_argument("--max-weight", dest="max_weight", type=float, default=0.25)
    alloc.add_argument(
        "--objective",
        choices=["weights", "utility"],
        default="weights",
        help="'weights' = trailing-return scalar sizing (OR-Tools); "
        "'utility' = mean-variance construction from alpha + Σ (a research proposal)",
    )
    alloc.add_argument(
        "--strategy", choices=STRATEGIES, default="volume_spike", help="Alpha source (utility)"
    )
    alloc.add_argument(
        "--source",
        choices=["strategy", "signal", "scanner"],
        default="strategy",
        help="Score origin (utility)",
    )
    alloc.add_argument("--as-of", dest="as_of", type=_date, default=_now(), help="Rebalance date (utility)")
    alloc.add_argument(
        "--target-te", dest="target_te", type=float, default=0.04, help="Target tracking error"
    )
    alloc.add_argument(
        "--max-names", dest="max_names", type=int, default=None, help="Cardinality cap (utility)"
    )
    alloc.add_argument(
        "--min-weight",
        dest="min_weight",
        type=float,
        default=0.0,
        help="Dust floor: min weight if held (utility)",
    )
    alloc.add_argument("--benchmark", default="SPY", help="Benchmark for residual vol / beta (utility)")
    _add_neutralize_factors_flag(alloc, note=" (utility)")
    alloc.add_argument(
        "--gross-objective",
        dest="gross_objective",
        action="store_true",
        help="Cost-blind solve (utility): drop the transaction cost from the objective, report it ex-post only. "
        "Default is cost-aware (name-specific turnover + √-impact in the objective).",
    )
    alloc.add_argument(
        "--holding-period",
        dest="holding_period",
        type=float,
        default=1.0 / 12.0,
        help="Expected holding period in years, to annualize the in-objective cost (utility; default 1/12)",
    )
    alloc.add_argument(
        "--benchmark-holdings",
        dest="benchmark_holdings",
        default=None,
        help="Portfolio-level benchmark w_B (utility): 'equal' over the covered universe, "
        "or a symbol,weight CSV/JSON holdings file. Moves TE/alpha-neutrality/transfer coefficient "
        "into active space (w_a = w - w_B); omit for the cash-relative behavior.",
    )
    alloc.add_argument(
        "--benchmark-premium",
        dest="benchmark_premium",
        type=float,
        default=0.05,
        help="mu_B: assumed annual benchmark excess return, for the reverse-optimization "
        "consensus-returns report (utility; only used with --benchmark-holdings)",
    )
    alloc.add_argument(
        "--book",
        choices=["long-only", "market-neutral"],
        default="long-only",
        help="'long-only' (default) is the standard box [0,cap]/budget "
        "Σw=1 solve, unchanged. 'market-neutral' relaxes to box [-short-max-weight,cap], "
        "budget Σw=0, and REQUIRES --gross-leverage (an unconstrained long/short book on "
        "a noisy Σ is a leverage machine).",
    )
    alloc.add_argument(
        "--gross-leverage",
        dest="gross_leverage",
        type=float,
        default=None,
        help="‖w‖1 <= L cap (utility, mandatory with --book market-neutral)",
    )
    alloc.add_argument(
        "--short-max-weight",
        dest="short_max_weight",
        type=float,
        default=0.25,
        help="Per-name short-side box magnitude s_i (utility; --book market-neutral only, default 0.25)",
    )
    alloc.add_argument(
        "--longshort-report",
        dest="longshort_report",
        action="store_true",
        help="Also solve long-only and market-neutral on the SAME alphas/Σ/costs and report the "
        "IR shrinkage, both transfer coefficients, and the long-only book's size exposure "
        "Uses --gross-leverage/--short-max-weight for the market-neutral leg "
        "(defaults 2.0 / 0.25 if not set).",
    )
    alloc.add_argument(
        "--conditional",
        choices=["ewma", "har"],
        default=None,
        help="Condition Σ's volatilities before the solve, so "
        "target_te is measured against current, not trailing-average, risk. Default "
        "off — see 'risk --evaluate-conditional' before turning this on.",
    )
    alloc.add_argument(
        "--conditional-lambda",
        dest="conditional_lambda",
        type=float,
        default=None,
        help="Override the EWMA decay λ (utility; default RiskMetrics 0.94 daily / 0.97 weekly)",
    )
    alloc.add_argument(
        "--posterior",
        choices=["bl"],
        default=None,
        help="Black-Litterman-blend the alphas with the consensus "
        "prior before the solve — uncovered names get a real, Σ-propagated posterior "
        "instead of being excluded. Default off until validated OOS. Requires "
        "--posterior-t-eff.",
    )
    alloc.add_argument(
        "--posterior-ic",
        dest="posterior_ic",
        type=float,
        default=None,
        help="IC behind the BL view precision (utility; default: same assumed IC the refine step used)",
    )
    alloc.add_argument(
        "--posterior-t-eff",
        dest="posterior_t_eff",
        type=float,
        default=None,
        help="Effective independent observations behind the BL view (utility; pins "
        "tau=1/T_eff — required with --posterior bl; see 'info' "
        "--> effective_t for a measured value)",
    )
    alloc.add_argument(
        "--tau",
        dest="posterior_tau",
        type=float,
        default=None,
        help="Override the pinned BL tau=1/T_eff (utility; sensitivity knob)",
    )
    alloc.add_argument(
        "--policy",
        choices=["aim"],
        default=None,
        help="'aim' replaces the myopic jump-to-target with "
        "Gârleanu-Pedersen partial adjustment - alpha discounted by κ/(κ+φ) (φ from "
        "the strategy's own measured decay), aim solved cost-free, traded κ of the "
        "gap each rebalance, banded by the optimizer's own no-trade band. Default off until "
        "validated OOS net of cost — see 'info --policy-ab'. Long-only only; "
        "incompatible with --book market-neutral or --benchmark-holdings.",
    )
    alloc.add_argument(
        "--trade-rate",
        dest="trade_rate",
        type=float,
        default=None,
        help="Override the derived κ (utility; only with --policy aim)",
    )
    _add_cache_flags(alloc)
    _add_config_flag(alloc)
    alloc.set_defaults(func=cmd_allocate)

    opt = subparsers.add_parser(
        "optimize", help="Tune strategy parameters via backtest (in-sample — trust nothing yet)"
    )
    add_common(opt, with_dates=True)
    opt.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    opt.add_argument("--objective", default="sharpe_ratio")
    opt.add_argument("--max-evals", dest="max_evals", type=int, default=50)
    opt.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for candidate sampling. Affects --method random and bayesian, and "
        "grid too once --max-evals caps the grid (the cap is sampled, not truncated). "
        "Matches walkforward's --seed, so the same value reproduces its inner search.",
    )
    opt.add_argument("--output", default="optimization_results.csv")
    _add_cost_flags(opt)
    _add_cache_flags(opt)
    add_workers_flag(opt)
    add_no_journal(opt)
    add_force(opt)
    opt.set_defaults(func=cmd_optimize)

    wf = subparsers.add_parser(
        "walkforward",
        help="Out-of-sample validation: optimize IS, score OOS, across folds (the honest scorecard)",
    )
    add_common(wf, with_dates=True)
    wf.add_argument("--mode", choices=["anchored", "rolling"], default="anchored")
    wf.add_argument("--folds", type=int, default=None, help="Number of folds (or use --train/--test-days)")
    wf.add_argument("--train-days", dest="train_days", type=int, default=None)
    wf.add_argument("--test-days", dest="test_days", type=int, default=None)
    wf.add_argument(
        "--embargo-days",
        dest="embargo_days",
        type=int,
        default=None,
        help="IS->OOS gap; defaults to required lookback in calendar days",
    )
    wf.add_argument("--holdout-days", dest="holdout_days", type=int, default=0)
    wf.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    wf.add_argument("--objective", default="sharpe_ratio")
    wf.add_argument("--max-evals", dest="max_evals", type=int, default=50)
    wf.add_argument("--seed", type=int, default=42)
    wf.add_argument(
        "--benchmark",
        default=None,
        help="Score each fold against this symbol and check the median per-fold "
        "information ratio as a paper prerequisite. Per fold, then median - the same "
        "shape as every other fold statistic here.",
    )
    wf.add_argument(
        "--cost-stress",
        dest="cost_stress",
        nargs="?",
        const="all",
        default="all",
        choices=["all", "turnover", "borrow"],
        help="Stress the chosen config's costs and check the paper prerequisite "
        "(survives >= 3x its assumed cost). On by default here: this is where a "
        "promotion decision is made, and cost sensitivity belongs in that story rather "
        "than in an optional follow-up.",
    )
    wf.add_argument(
        "--no-cost-stress",
        dest="cost_stress",
        action="store_const",
        const=None,
        help="Skip the cost stress (it re-runs the chosen config once per multiple)",
    )
    wf.add_argument("--pbo", action="store_true", help="Estimate probability of backtest overfitting")
    wf.add_argument(
        "--monte-carlo",
        dest="monte_carlo",
        action="store_true",
        help="Block-bootstrap the OOS trade sequence",
    )
    wf.add_argument(
        "--param-sensitivity",
        dest="param_sensitivity",
        action="store_true",
        # `%%`, not `%`: argparse %-formats help text against the action's own
        # attributes, so a literal percent renders the action's __dict__ mid-sentence.
        help="Perturb chosen params +-10%% and re-test",
    )
    wf.add_argument(
        "--leakage-probe",
        dest="leakage_probe",
        action="store_true",
        help="Shift the data feed forward to detect future-data leakage",
    )
    wf.add_argument(
        "--bootstrap-skill",
        dest="bootstrap_skill",
        action="store_true",
        help="Nonparametric skill check — this config's own zero-alpha "
        "stationary bootstrap p, next to the FAMILY p from White's Reality Check "
        "over every OOS return series the trial store has recorded for this "
        "(strategy, universe, accounting). Advisory only (not a promotion gate).",
    )
    wf.add_argument(
        "--bootstrap-b",
        dest="bootstrap_b",
        type=int,
        default=2000,
        help="Bootstrap resamples B (default 2000)",
    )
    wf.add_argument(
        "--bootstrap-block-length",
        dest="bootstrap_block_length",
        type=float,
        default=None,
        help="Override the stationary bootstrap's expected block length "
        "(default: the Politis-White rule, auto-computed and reported)",
    )
    wf.add_argument("--bootstrap-seed", dest="bootstrap_seed", type=int, default=0)
    wf.add_argument("--results-csv", dest="results_csv", default=None, help="Write per-fold table to CSV")
    wf.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="render the verdict + promotion-gate scorecard to an image (needs the viz extra)",
    )
    wf.add_argument(
        "--save-config",
        dest="save_config",
        default=None,
        help="Save the chosen config (with provenance) to this path",
    )
    _add_cost_flags(wf)
    _add_cache_flags(wf)
    add_html_flag(wf)
    add_workers_flag(wf)
    add_record_trades_flag(wf)
    add_no_journal(wf)
    add_force(wf)
    wf.set_defaults(func=cmd_walkforward)

    init = subparsers.add_parser(
        "init",
        help="Guided first-run setup: write a valid .env, check it, and say what to try next",
    )
    init.add_argument(
        "--check",
        action="store_true",
        help="Diagnose the current setup and exit — writes nothing, makes no network call",
    )
    init.add_argument(
        "--non-interactive",
        dest="non_interactive",
        action="store_true",
        help="Build the .env from environment variables with no prompts (scripts, containers)",
    )
    init.add_argument(
        "--env-path", dest="env_path", default=None, help="Write to this .env instead of the default"
    )
    init.add_argument(
        "--cache-dir", dest="cache_dir", default=None, help="Bar cache directory (default: cache/bars)"
    )
    init.set_defaults(func=cmd_init)

    # The shared knobs, once. Step-specific tuning stays on the individual commands:
    # `verdict` is the honest default path through the pipeline, not a superset of
    # every flag the five steps between them accept.
    verdict = subparsers.add_parser(
        "verdict",
        help="The whole pipeline in one command: scan → alphas → portfolio → information, "
        "one universe, one window, one consolidated verdict — read-only",
    )
    verdict.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    verdict.add_argument("--scanner", default="volume", help="Universe scanner ('none' to skip)")
    verdict.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    verdict.add_argument("--start", type=_date, default=_now() - timedelta(days=365))
    verdict.add_argument("--end", type=_date, default=_now())
    verdict.add_argument("--capital", type=float, default=100_000.0)
    verdict.add_argument(
        "--source", choices=["strategy", "signal", "scanner"], default="strategy", help="Alpha score origin"
    )
    verdict.add_argument(
        "--combine",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=None,
        help="Combine several strategies' signals into one alpha (comma-separated); "
        "with one or none, the single-signal path runs",
    )
    verdict.add_argument("--benchmark", default="SPY", help="Benchmark for residual vol / beta")
    verdict.add_argument("--timeframe", default="1Day", help="Bar timeframe")
    verdict.add_argument("--horizon", type=int, default=5, help="Forward-return horizon in bars")
    verdict.add_argument("--lookback-days", dest="lookback_days", type=int, default=365)
    verdict.add_argument(
        "--risk-model",
        dest="risk_model",
        choices=["shrinkage", "sample", "factor"],
        default="shrinkage",
        help="Covariance model: shrinkage (Ledoit–Wolf), sample (raw), or factor (XFXᵀ+Δ)",
    )
    verdict.add_argument("--target-te", dest="target_te", type=float, default=0.04)
    verdict.add_argument("--max-weight", dest="max_weight", type=float, default=0.25)
    verdict.add_argument(
        "--max-names", dest="max_names", type=int, default=None, help="Cardinality cap on the book"
    )
    _add_neutralize_factors_flag(verdict)
    _add_cost_flags(verdict)
    _add_cache_flags(verdict)
    verdict.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Also write the structured result object (the same one the report renders) to a JSON file",
    )
    add_html_flag(verdict)
    add_no_journal(verdict)
    add_force(verdict)
    _add_config_flag(verdict)
    verdict.set_defaults(func=cmd_verdict)

    alphas = subparsers.add_parser(
        "alphas",
        help="Rank a universe by continuous alpha (residual-return forecast) — read-only",
    )
    alphas.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    alphas.add_argument("--scanner", default="volume", help="Scanner used as the --source score metric")
    alphas.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    alphas.add_argument(
        "--as-of", dest="as_of", type=_date, default=_now(), help="Rebalance date (YYYY-MM-DD)"
    )
    alphas.add_argument(
        "--source",
        choices=["strategy", "signal", "scanner"],
        default="strategy",
        help="Score origin: 'strategy' = continuous conviction; 'signal' = BUY/SELL/HOLD; "
        "'scanner' = scanner strength",
    )
    alphas.add_argument("--ic", type=float, default=0.03, help="Assumed information coefficient")
    alphas.add_argument(
        "--combine",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=None,
        help="Combine several strategies' signals into one alpha (comma-separated, "
        "e.g. volume_spike,ma_crossover,mean_reversion). Measures + shrinks their ICs.",
    )
    alphas.add_argument("--benchmark", default="SPY", help="Benchmark for residual vol / beta")
    alphas.add_argument(
        "--neutralize",
        action="store_true",
        help="Make alphas beta-neutral (regress out benchmark beta)",
    )
    _add_neutralize_factors_flag(alphas)
    alphas.add_argument("--lookback-days", dest="lookback_days", type=int, default=180)
    alphas.add_argument(
        "--scaling",
        choices=["case1", "case2", "auto"],
        default="case1",
        help="Per-name scaling: 'case1' = σ·IC·z (default); 'case2' = IC·c_g·z "
        "(no per-name vol multiply); 'auto' = let a Std_TS-vs-ω regression decide",
    )
    add_no_journal(alphas)
    _add_cache_flags(alphas)
    _add_config_flag(alphas)
    alphas.set_defaults(func=cmd_alphas)

    info = subparsers.add_parser(
        "info",
        help="Information report: measure IC, breadth, and predicted-vs-realized IR — read-only",
    )
    info.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    info.add_argument(
        "--source", choices=["strategy", "signal", "scanner"], default="strategy", help="Alpha score origin"
    )
    info.add_argument("--scanner", default="volume", help="Scanner used when --source scanner")
    info.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    info.add_argument("--start", type=_date, default=_now() - timedelta(days=365))
    info.add_argument("--end", type=_date, default=_now())
    info.add_argument("--benchmark", default="SPY", help="Benchmark for residual returns")
    info.add_argument("--horizon", type=int, default=5, help="Forward-return horizon in bars")
    info.add_argument(
        "--n-trials",
        dest="n_trials",
        type=int,
        default=1,
        help="Configs tried (for multiple-testing inflation)",
    )
    info.add_argument(
        "--scaling-ab",
        dest="scaling_ab",
        action="store_true",
        help="Research mode: walk-forward the realized IR under Case-1 vs "
        "Case-2 scaling and compare against the regression's pick",
    )
    info.add_argument(
        "--attribution",
        action="store_true",
        help="Performance attribution: split realized active return into "
        "systematic timing, risk factors, signals, and stock-picking, with per-row "
        "t-stats and the skill-vs-luck verdict, instead of the IC/breadth report",
    )
    info.add_argument(
        "--attribution-signals",
        dest="attribution_signals",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=None,
        help="Extra strategies to attribute as additional signal columns "
        "(comma-separated; the strategy's own alpha is always included) — "
        "compare a --combine weight against its realized counterpart",
    )
    info.add_argument(
        "--conditional",
        choices=["ewma", "har"],
        default=None,
        help="With --attribution, condition the per-period Σ(t) and add a "
        "predicted-vs-realized TE by regime table; with --conditional-ab, the "
        "conditional method the A/B compares against unconditional (default ewma).",
    )
    info.add_argument(
        "--conditional-lambda",
        dest="conditional_lambda",
        type=float,
        default=None,
        help="Override the EWMA decay λ (default RiskMetrics 0.94 daily / 0.97 weekly)",
    )
    info.add_argument(
        "--conditional-ab",
        dest="conditional_ab",
        action="store_true",
        help="Research mode: net-of-cost A/B — walk-forward the SAME "
        "alpha book against a conditional vs unconditional Σ, carrying weights forward, "
        "and compare realized net IR (not just TE-tracking).",
    )
    info.add_argument(
        "--bootstrap-skill",
        dest="bootstrap_skill",
        action="store_true",
        help="With --attribution, add a nonparametric OWN p-value (stationary "
        "block bootstrap of the active-return series under the imposed null) next to "
        "the parametric SE{IR}≈1/√Y verdict",
    )
    info.add_argument(
        "--policy-ab",
        dest="policy_ab",
        action="store_true",
        help="Research mode: net-of-cost A/B — walk-forward the myopic "
        "policy vs the aim policy on the SAME alpha book, carrying weights "
        "forward, and compare realized net IR and turnover (the promotion gate).",
    )
    info.add_argument(
        "--trade-rate",
        dest="trade_rate",
        type=float,
        default=None,
        help="Override the derived κ (only used with --policy-ab)",
    )
    _add_neutralize_factors_flag(info, note="; measure the alpha you deploy")
    add_html_flag(info)
    _add_cache_flags(info)
    _add_config_flag(info)
    info.set_defaults(func=cmd_info)

    hz = subparsers.add_parser(
        "horizon",
        help="Measure alpha decay / half-life and recommend rebalance cadence + blend — read-only",
    )
    hz.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    hz.add_argument("--source", choices=["strategy", "signal", "scanner"], default="strategy")
    hz.add_argument("--scanner", default="volume")
    hz.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    hz.add_argument("--start", type=_date, default=_now() - timedelta(days=365))
    hz.add_argument("--end", type=_date, default=_now())
    hz.add_argument("--benchmark", default="SPY")
    hz.add_argument(
        "--max-lag", dest="max_lag", type=int, default=10, help="Largest lag (periods) to measure"
    )
    hz.add_argument("--timeframe", default="1Day")
    _add_neutralize_factors_flag(hz, note="; measure the alpha you deploy")
    _add_cache_flags(hz)
    _add_config_flag(hz)
    hz.set_defaults(func=cmd_horizon)

    risk = subparsers.add_parser(
        "risk",
        help="Estimate the universe covariance Σ and summarize its risk structure — read-only",
    )
    risk.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    risk.add_argument("--as-of", dest="as_of", type=_date, default=_now(), help="Rebalance date (YYYY-MM-DD)")
    risk.add_argument(
        "--model",
        choices=["shrinkage", "sample", "factor"],
        default="shrinkage",
        help="Covariance model: shrinkage (Ledoit–Wolf), sample (raw), or factor (XFXᵀ+Δ)",
    )
    risk.add_argument("--benchmark", default="SPY", help="Benchmark for the market factor / beta")
    risk.add_argument("--timeframe", default="1Day", help="Bar timeframe for returns")
    risk.add_argument("--lookback-days", dest="lookback_days", type=int, default=365)
    risk.add_argument(
        "--conditional",
        choices=["ewma", "har"],
        default=None,
        help="Condition Σ's volatilities (EWMA or HAR-lite), holding the "
        "correlation structure fixed (Σ_t = D_t·R·D_t). Default off — see "
        "'--evaluate-conditional' before turning this on; the evidence gate's "
        "as-built notes for the evidence-gate finding that decided the default.",
    )
    risk.add_argument(
        "--conditional-lambda",
        dest="conditional_lambda",
        type=float,
        default=None,
        help="Override the EWMA decay λ (default: RiskMetrics 0.94 daily / 0.97 weekly)",
    )
    risk.add_argument(
        "--evaluate-conditional",
        dest="evaluate_conditional",
        action="store_true",
        help="Run the MZ/QLIKE evidence gate instead of the risk report: per-name and "
        "pooled forecast quality of EWMA/HAR vs the unconditional trailing baseline, "
        "by realized-vol regime — the report that decides whether --conditional is "
        "worth turning on for this universe/window.",
    )
    _add_cache_flags(risk)
    _add_config_flag(risk)
    risk.set_defaults(func=cmd_risk)

    def _add_cache_dir_flag(p) -> None:
        p.add_argument(
            "--cache-dir", dest="cache_dir", default=None, help="Bar cache directory (default: cache/bars)"
        )

    cache = subparsers.add_parser(
        "cache", help="Inspect/warm the persistent bar cache behind backtest/optimize/walkforward"
    )
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)

    c_warm = cache_sub.add_parser("warm", help="Prefetch and cache bars for a universe/window")
    _add_cache_dir_flag(c_warm)
    c_warm.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    c_warm.add_argument("--scanner", default="none", help="Universe scanner ('none' to skip)")
    c_warm.add_argument("--timeframe", default="1Day")
    c_warm.add_argument("--start", type=_date, default=_now() - timedelta(days=365))
    c_warm.add_argument("--end", type=_date, default=_now())
    c_warm.add_argument(
        "--scan-as-of",
        dest="scan_as_of",
        type=_date,
        default=None,
        help="Resolve the scanner at this date/datetime; defaults to --end",
    )

    c_status = cache_sub.add_parser("status", help="Coverage summary and a drift check — no network")
    _add_cache_dir_flag(c_status)
    c_status.add_argument("--symbols", type=_symbols, default=None, help="Filter to these symbols")
    c_status.add_argument("--timeframe", default=None, help="Filter to one timeframe")

    c_refresh = cache_sub.add_parser(
        "refresh", help="Invalidate and re-fetch symbols (corporate actions / stale data)"
    )
    _add_cache_dir_flag(c_refresh)
    c_refresh.add_argument("--symbols", type=_symbols, required=True)
    c_refresh.add_argument("--timeframe", default="1Day")
    c_refresh.add_argument(
        "--start", type=_date, default=None, help="Defaults to the symbol's previously-cached extent"
    )
    c_refresh.add_argument("--end", type=_date, default=None)

    cache.set_defaults(func=cmd_cache)

    def _add_db_flag(p) -> None:
        p.add_argument("--db", default=None, help="Trial store DB path (default: logs/trials.db)")

    trials = subparsers.add_parser(
        "trials", help="Inspect the trial store — the queryable index over the research journal"
    )
    trials_sub = trials.add_subparsers(dest="trials_command", required=True)

    t_status = trials_sub.add_parser("status", help="Row/journal-line counts and a drift check")
    _add_db_flag(t_status)
    t_status.add_argument(
        "--journal", default=None, help="Journal path (default: logs/research_journal.jsonl)"
    )

    t_rebuild = trials_sub.add_parser(
        "rebuild", help="Rebuild the index from the journal (derived — safe to delete the DB)"
    )
    _add_db_flag(t_rebuild)
    t_rebuild.add_argument(
        "--journal", default=None, help="Journal path (default: logs/research_journal.jsonl)"
    )

    def add_trial_filters(p) -> None:
        """The filters `list` and `best` share, so the two can never disagree about
        what a filter means."""
        _add_db_flag(p)
        p.add_argument("--strategy", default=None)
        p.add_argument(
            "--kind",
            default=None,
            choices=["backtest", "optimize", "walkforward", "alpha", "research", "verdict"],
        )
        p.add_argument(
            "--symbols",
            type=_symbols,
            default=None,
            help="Universe — matched on the normalized universe, so symbol order and case "
            "never change what is found; with --strategy, also prints campaign n_trials",
        )
        p.add_argument("--since", type=_date, default=None, help="Only trials recorded on/after this date")
        p.add_argument("--until", type=_date, default=None, help="Only trials recorded on/before this date")
        p.add_argument(
            "--min-sharpe",
            dest="min_sharpe",
            type=float,
            default=None,
            help="Only trials with a recorded OOS Sharpe at or above this",
        )
        p.add_argument(
            "--gates-passed",
            dest="gates_passed",
            action="store_const",
            const=True,
            default=None,
            help="Only trials recorded as promotable",
        )
        p.add_argument(
            "--accounting", type=int, default=None, help="Filter to one accounting version (default: current)"
        )
        p.add_argument(
            "--all-accounting",
            dest="all_accounting",
            action="store_true",
            help="Show every accounting version, not just current — read the ACCT column "
            "before comparing rows across versions",
        )
        p.add_argument("--json", action="store_true", help="Emit JSON instead of a table")

    def add_list_args(p) -> None:
        add_trial_filters(p)
        p.add_argument(
            "--sort",
            choices=["date", "sharpe", "dsr"],
            default="date",
            help="Sort order; unrecorded metrics sort last, never as zero",
        )
        p.add_argument("--limit", type=int, default=20)
        p.add_argument("--offset", type=int, default=0, help="Skip this many rows (paging)")

    t_list = trials_sub.add_parser(
        "list", help="List trials with filters, sorting, and paging — the campaign's memory"
    )
    add_list_args(t_list)

    # `query` predates `list` and is kept as an alias for one release so existing
    # muscle memory and scripts keep working; the docs describe `list` only.
    t_query = trials_sub.add_parser("query", help="Alias for `trials list` (kept for compatibility)")
    add_list_args(t_query)

    t_promote = trials_sub.add_parser(
        "promote",
        help="Write a portable config from a validated trial, without re-running it",
    )
    _add_db_flag(t_promote)
    t_promote.add_argument("trial_id", help="The trial id (see `trials list`)")
    t_promote.add_argument(
        "--save-config",
        dest="save_config",
        required=True,
        metavar="PATH",
        help="Where to write the config",
    )
    t_promote.add_argument(
        "--force",
        action="store_true",
        help="Promote a trial that did not clear its gates, recording that verdict in the config",
    )
    t_promote.set_defaults(func=cmd_trials)

    t_show = trials_sub.add_parser("show", help="Everything the store knows about one trial")
    _add_db_flag(t_show)
    t_show.add_argument("trial_id", help="The trial id (see `trials list`)")
    t_show.add_argument("--json", action="store_true", help="Emit JSON instead of a report")
    t_show.add_argument(
        "--trades-limit",
        dest="trades_limit",
        type=int,
        default=25,
        help="How many stored trades to print (the rest are reported as not shown)",
    )

    t_best = trials_sub.add_parser(
        "best", help="The honest leaderboard: DSR-ranked, family n_trials always shown"
    )
    add_trial_filters(t_best)
    t_best.add_argument(
        "--rank-by",
        dest="rank_by",
        choices=["dsr", "sharpe"],
        default="dsr",
        help="Ranking metric. 'dsr' (default) already discounts how many configs were tried; "
        "'sharpe' does not, and prints a caveat saying so",
    )
    t_best.add_argument(
        "--include-in-sample",
        dest="include_in_sample",
        action="store_true",
        help="Also rank in-sample rows (optimize/alpha). They are search candidates, not "
        "track records — an 'optimize' row is best-of-N by construction, so its rank "
        "measures selection rather than skill",
    )
    t_best.add_argument("--limit", type=int, default=10)
    t_best.add_argument("--offset", type=int, default=0, help=argparse.SUPPRESS)

    trials.set_defaults(func=cmd_trials)

    mcp = subparsers.add_parser(
        "mcp", help="Serve TradeFlow over MCP for an agent (opt-in; needs the 'mcp' extra)"
    )
    mcp.set_defaults(func=cmd_mcp)

    res = subparsers.add_parser(
        "research", help="Autonomous research loop -> shortlist of validated configs (needs 'ai' extra)"
    )
    add_common(res, with_dates=True)
    res.add_argument("--goal", default="Improve out-of-sample Sharpe without raising max drawdown")
    res.add_argument("--mode", choices=["anchored", "rolling"], default="anchored")
    res.add_argument("--folds", type=int, default=4)
    res.add_argument("--embargo-days", dest="embargo_days", type=int, default=None)
    res.add_argument("--holdout-days", dest="holdout_days", type=int, default=60)
    res.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    res.add_argument("--objective", default="sharpe_ratio")
    res.add_argument("--max-evals", dest="max_evals", type=int, default=25)
    res.add_argument("--max-trials", dest="max_trials", type=int, default=10)
    res.add_argument("--max-dry-rounds", dest="max_dry_rounds", type=int, default=3)
    res.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    res.add_argument("--shortlist-size", dest="shortlist_size", type=int, default=3)
    res.add_argument(
        "--allow-code-gen",
        dest="allow_code_gen",
        action="store_true",
        help="Permit agent-authored strategy code (validated in the sandbox)",
    )
    res.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama"],
        default="anthropic",
        help="LLM provider for the proposer ('ollama' runs locally, no API key)",
    )
    res.add_argument(
        "--model",
        default=None,
        help="Model id (defaults per provider, e.g. claude-opus-4-8 / gpt-4o / llama3.1)",
    )
    res.add_argument("--seed", type=int, default=42)
    res.set_defaults(func=cmd_research)

    dagent = subparsers.add_parser(
        "demo-agent",
        help="Narrate an AI research session on live data: proposal -> sandbox -> walk-forward -> verdict",
    )
    dagent.add_argument("--symbols", type=_symbols, default=["NVDA", "AAPL", "META", "AMD", "MSFT"])
    dagent.add_argument("--strategy", default="ma_crossover")
    dagent.add_argument("--start", type=_date, default=None)
    dagent.add_argument("--end", type=_date, default=None)
    dagent.add_argument(
        "--lookback-days",
        dest="lookback_days",
        type=int,
        default=1095,
        help="History to research over when --start is omitted (default: 3 years)",
    )
    dagent.add_argument("--goal", default="Improve out-of-sample Sharpe without raising max drawdown")
    dagent.add_argument("--folds", type=int, default=4)
    dagent.add_argument("--holdout-days", dest="holdout_days", type=int, default=90)
    dagent.add_argument("--max-evals", dest="max_evals", type=int, default=25)
    dagent.add_argument("--max-trials", dest="max_trials", type=int, default=6)
    dagent.add_argument("--max-dry-rounds", dest="max_dry_rounds", type=int, default=3)
    dagent.add_argument("--capital", type=float, default=100_000.0)
    dagent.add_argument(
        "--provider",
        choices=["replay", "anthropic", "openai", "ollama"],
        default="replay",
        help="'replay' (default) needs no API key and is deterministic; anything else drives a live model",
    )
    dagent.add_argument("--model", default=None, help="Model id (ignored when --provider replay)")
    dagent.add_argument("--seed", type=int, default=42)
    dagent.add_argument(
        "--no-costs",
        dest="no_costs",
        action="store_true",
        help="Validate on gross returns (diagnostic only; costs are charged by default)",
    )
    dagent.set_defaults(func=cmd_demo_agent)

    demo = subparsers.add_parser(
        "demo", help="Run the full pipeline on synthetic data (no API keys, no network)"
    )
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="also render the demo as an image to PATH (needs the viz extra: matplotlib)",
    )
    demo.set_defaults(func=cmd_demo)

    return parser


def main() -> None:
    from tradeflow.settings import SettingsError
    from tradeflow.store.bars import CacheMiss

    setup_logging()
    args = parse_cli()
    try:
        args.func(args)
    except (SettingsError, CacheMiss) as exc:
        sys.exit(str(exc))
    except KeyboardInterrupt:
        # A traceback reads as a crash; Ctrl-C is a choice. What matters more than the
        # tidiness is the second clause: a research command interrupted part-way has
        # written no config and journaled no trial, and a half-finished validation that
        # silently recorded one would corrupt the campaign's own trial count - the
        # number every deflated Sharpe deflates against.
        sys.exit("\nInterrupted - no config saved, no trial recorded.")


if __name__ == "__main__":
    main()
