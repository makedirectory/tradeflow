"""Command-line entry point.

A thin adapter that wires the layers together per command - all the real work
lives in ``src/`` (and the shared service core in ``src/services/``, so the CLI,
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
from typing import Any, Dict, List, Optional

from src.marketdata.base import MarketDataProvider
from src.services.registry import STRATEGIES
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# A reasonable default candidate list for the scanner to filter.
DEFAULT_UNIVERSE = ["NVDA", "RIVN", "NFLX", "META", "BAC", "MS", "TSLA", "GS", "AMD", "AAPL"]


# ---------------------------------------------------------------------------- #
# Wiring
# ---------------------------------------------------------------------------- #
def build_data_and_broker(cache: bool = False, offline: bool = False, cache_dir: Optional[Any] = None):
    """Construct the Alpaca-backed broker and market-data client from settings.

    ``cache``/``offline``/``cache_dir`` are forwarded to
    :func:`src.services.data.build_data_client`, which owns the actual provider
    construction (and its opt-in bar-cache wrapping) - kept in one place so the
    CLI and the read-only MCP/research path never diverge on how a data client
    gets built.
    """
    from alpaca.trading.client import TradingClient

    from src.brokers.alpaca.broker import AlpacaBroker
    from src.services.data import build_data_client
    from src.settings import load_settings

    settings = load_settings()
    trading_client = TradingClient(
        api_key=settings.alpaca_key,
        secret_key=settings.alpaca_secret,
        paper=settings.paper_trade,
    )
    broker = AlpacaBroker(trading_client, settings.alpaca_key, settings.alpaca_secret, settings.paper_trade)
    data_client = build_data_client(cache=cache, offline=offline, cache_dir=cache_dir)
    return broker, data_client


def resolve_universe(data_client, scanner_name: Optional[str], candidates: List[str]) -> List[str]:
    """Filter ``candidates`` through the scanner, falling back to them if none flag.

    Delegates to the shared service core so the CLI and MCP server use one path.
    """
    from src.services.data import resolve_universe as _resolve

    return _resolve(data_client, scanner_name, candidates)


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
    from src.costs import ParametricCostModel

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
    :meth:`~src.store.bars.CachedMarketData.vintage_stamp`) - it warms the cache
    as a side effect, which is why callers compute it once, up front, and reuse
    the same value for both the memoization lookup and the eventual record: the
    two must use an identical dedup key or a matching prior trial would never be
    found.
    """
    from src.store.bars import CachedMarketData

    provider = getattr(data_client, "provider", None)
    if not isinstance(provider, CachedMarketData):
        return None
    return provider.vintage_stamp(universe, timeframe, start, end)


@contextlib.contextmanager
def _open_trial_store(journal_path: Optional[Any] = None):
    """A trial store against ``journal_path`` (default: the current
    ``audit.DEFAULT_TRIAL_JOURNAL``), or ``None`` on any failure to open one.

    v1 of the trial store is passive and derived (see ``src.store.trials``): a
    broken store must never break the command it's attached to, memoization
    included - every caller here treats ``None`` as "skip memoization, run
    normally," never as an error to propagate.
    """
    from src.services import audit
    from src.store.trials import TrialStore, db_path_for_journal

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
    from src.optimization.config_store import current_git_sha

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


def _load_strategy_from_config(path: str):
    """Load a saved config (``walkforward --save-config``) and construct the
    strategy directly from its params - the ``--config`` path shared by
    ``backtest``/``live``. Returns ``(strategy, strategy_name, scanner_name)``.

    Construction goes through ``Strategy.__init__`` exactly as
    ``create_with_defaults()`` does, so out-of-range/unrecognized params raise
    loudly (:meth:`Strategy._validate_parameters`) rather than silently
    trading on a config an older strategy version can no longer honor.
    """
    from src.optimization.config_store import load_config
    from src.services.registry import resolve_strategy_class

    payload = load_config(path)
    strategy_name = payload["strategy"]
    cls = resolve_strategy_class(strategy_name)
    strategy = cls(dict(payload.get("params") or {}))
    return strategy, strategy_name, payload.get("scanner")


# ---------------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------------- #
def cmd_backtest(args) -> None:
    import json

    from src.analytics.reporting import format_backtest_report, format_cached_notice, log_backtest_report
    from src.engine.backtest import ACCOUNTING_VERSION, BacktestEngine
    from src.services.sizing import build_beta_sizer

    scanner = args.scanner
    if args.config:
        strategy, strategy_name, cfg_scanner = _load_strategy_from_config(args.config)
        if cfg_scanner:
            scanner = cfg_scanner
        print(f"Loaded config {args.config} -> strategy={strategy_name!r} scanner={scanner!r}")
    else:
        strategy_name = args.strategy
        strategy = STRATEGIES[strategy_name].create_with_defaults()

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    universe = resolve_universe(data_client, scanner, args.symbols)

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
            return

    sizer = None
    if args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark, as_of=args.start)

    # Metrics are net of transaction cost by default; --gross disables the charge.
    cost_model = build_cost_model(args)
    result = BacktestEngine(strategy, data_client, sizer=sizer, cost_model=cost_model).run(
        universe, args.start, args.end, args.capital
    )

    if not args.no_journal:
        from src.analytics.metrics import returns_from_equity
        from src.analytics.performance import build_dated_equity_curve
        from src.services.audit import journal_trial

        # Persist this trial's own dated return series (daily-resampled,
        # from realized trade P&L - the same construction every persisted trial
        # kind uses) so it can later join a Reality Check family panel.
        dated_equity = build_dated_equity_curve(result.trades, args.capital)
        returns_series = returns_from_equity(dated_equity) if not dated_equity.empty else None
        journal_trial(
            "backtest",
            strategy=strategy_name,
            symbols=universe,
            start=args.start,
            end=args.end,
            params=dedup_params,
            metrics=result.metrics,
            returns=returns_series,
        )

    log_backtest_report(result.metrics, result.initial_capital, result.final_capital)
    if not args.gross and result.total_cost:
        print(
            f"Transaction cost: ${result.total_cost:,.2f} "
            f"({result.total_cost / result.initial_capital * 100:.2f}% of capital); "
            f"gross final ${result.gross_final_capital:,.2f}"
        )

    if getattr(args, "chart", None):
        from src.analytics.charts import render_backtest_chart

        try:
            path = render_backtest_chart(result, args.chart, title=f"{strategy_name} — backtest")
            print(f"Chart saved to {path}")
        except RuntimeError as exc:  # matplotlib (viz extra) not installed
            print(f"Chart skipped: {exc}")

    _maybe_print_attribution_verdict(
        data_client, strategy_name, universe, args.start, args.end, args.benchmark
    )


def cmd_scan(args) -> None:
    from src.scanners.symbol_scanner import SymbolScanner

    _, data_client = build_data_and_broker()
    flagged = SymbolScanner(data_client, args.scanner).scan(args.symbols)
    if not flagged:
        print("No symbols flagged.")
        return
    print(f"{'SYMBOL':10}SIGNAL")
    for symbol, signal in flagged:
        print(f"{symbol:10}{signal}")


def cmd_allocate(args) -> None:
    if getattr(args, "objective", "weights") == "utility":
        _allocate_utility(args)
        return

    from src.scanners.symbol_scanner import SymbolScanner
    from src.services.sizing import allocate_portfolio

    _, data_client = build_data_and_broker()
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


def _allocate_utility(args) -> None:
    """Mean-variance portfolio construction (alpha + Σ) — a read-only proposal."""
    from src.services.analysis import construct_portfolio, longshort_report

    _, data_client = build_data_and_broker()
    book = args.book.replace("-", "_")

    if args.longshort_report:
        report = longshort_report(
            data_client,
            args.strategy,
            args.symbols,
            args.as_of,
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


def cmd_optimize(args) -> None:
    from src.engine.backtest import ACCOUNTING_VERSION
    from src.optimization.optimizer import ParameterOptimizer

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    universe = resolve_universe(data_client, args.scanner, args.symbols)
    timeframe = STRATEGIES[args.strategy].create_with_defaults().config["timeframe"]
    vintage = _vintage_stamp(data_client, universe, timeframe, args.start, args.end)
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
        from src.services.audit import journal_trial

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
    from src.analytics.reporting import format_cached_notice
    from src.engine.backtest import ACCOUNTING_VERSION
    from src.optimization.config_store import build_provenance, current_git_sha, save_config
    from src.optimization.walk_forward import WalkForwardValidator

    _, data_client = build_data_and_broker(cache=args.cache, offline=args.offline, cache_dir=args.cache_dir)
    universe = resolve_universe(data_client, args.scanner, args.symbols)
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

    if not args.no_journal and result.folds:
        from src.services.audit import journal_trial

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
            dedup_params=recipe,
        )

    if getattr(args, "bootstrap_skill", False) and result.folds:
        from src.services.analysis import compute_bootstrap_skill

        report = compute_bootstrap_skill(
            result.oos_returns,
            args.strategy,
            universe,
            result.n_trials_total,
            result.oos_aggregate,
            B=args.bootstrap_b,
            block_length=args.bootstrap_block_length,
            seed=args.bootstrap_seed,
        )
        _print_bootstrap_skill(report)

    if getattr(args, "chart", None):
        from src.analytics.charts import render_walkforward_chart

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

    from src.research.agent import ResearchAgent, ResearchConfig
    from src.research.proposer import build_proposer
    from src.services.data import build_data_client

    data_client = build_data_client()
    universe = resolve_universe(data_client, args.scanner, args.symbols)
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
    from src.services.audit import journal_trial

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


def cmd_alphas(args) -> None:
    """Print the ranked alpha table (residual-return forecasts) for a universe.

    Read-only research-clock flow: scores each name as of --as-of, scales the
    cross-section into comparable annualized-return forecasts, and ranks them.
    Produces no orders and saves no config.
    """
    from src.services.analysis import compute_alphas, compute_combined_alphas

    _, data_client = build_data_and_broker()

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
    _, data_client = build_data_and_broker()

    if getattr(args, "evaluate_conditional", False):
        _print_conditional_evidence_gate(data_client, args)
        return

    from src.services.analysis import compute_risk

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

    from src.services.analysis import evaluate_conditional_risk

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
    from src.services.analysis import compute_information

    _, data_client = build_data_and_broker()

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
    from src.services.analysis import run_scaling_ab

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
    from src.services.analysis import compute_attribution

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
    from src.services.analysis import run_conditional_risk_ab

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
    from src.services.analysis import run_policy_ab

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


def _print_attribution(r, strategy: str, start, end) -> None:
    """Render a ``compute_attribution`` report: per-row mean/IR/t/share-of-variance,
    honest cumulation, and the skill-vs-luck verdict."""
    if not r.get("periods"):
        print(r.get("note", "No attribution report produced."))
        return

    print(f"\nPerformance attribution: '{strategy}' {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    print(f"  measured over {r['periods']} rebalances (horizon {r['horizon_bars']} bars)")
    print(f"  {'row':28}{'mean/yr':>9}{'IR':>8}{'t':>8}{'share ψ²':>10}")

    rows = r["rows"]

    def _line(label: str, key: str, not_skill: bool = False) -> None:
        row = rows[key]
        if not_skill:
            print(f"  {label:28}{'—':>9}{'—':>8}{'—':>8}{'(not skill)':>12}")
        else:
            print(
                f"  {label:28}{row['annualized_mean'] * 100:>8.2f}%{row['ir']:>8.2f}"
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
    from src.services.analysis import compute_attribution

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
    from src.services.analysis import compute_horizon

    _, data_client = build_data_and_broker()
    r = compute_horizon(
        data_client,
        args.strategy,
        args.symbols,
        args.start,
        args.end,
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
    from src.services.data import build_data_client
    from src.store.bars import CachedMarketData

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
        universe = resolve_universe(data_client, args.scanner, args.symbols)
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
    from src.store.trials import DEFAULT_JOURNAL_PATH, TrialStore

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

        # query — defaults to the current accounting version;
        # --all-accounting opts into a listing that spans versions.
        rows = store.query(
            strategy=args.strategy,
            kind=args.kind,
            accounting=args.accounting,
            all_accounting=args.all_accounting,
            limit=args.limit,
        )
        if not rows:
            print("No trials matched.")
        else:
            print(
                f"{'ID':14}{'KIND':12}{'STRATEGY':16}{'OOS SHARPE':>11}{'DSR':>7}{'PROMO':>7}{'ACCT':>6}  TS"
            )
            for r in rows:
                oos = r["oos_sharpe"] if r["oos_sharpe"] is not None else 0.0
                dsr = r["deflated_sharpe"] if r["deflated_sharpe"] is not None else 0.0
                promo = "-" if r["promotable"] is None else ("yes" if r["promotable"] else "no")
                print(
                    f"{r['id']:14}{r['kind']:12}{(r['strategy'] or '')[:16]:16}"
                    f"{oos:>11.3f}{dsr:>7.3f}{promo:>7}{r['accounting']:>6}  {(r['ts'] or '')[:19]}"
                )

        if args.strategy and args.symbols:
            from src.engine.backtest import ACCOUNTING_VERSION

            accounting = args.accounting if args.accounting is not None else ACCOUNTING_VERSION
            n = store.family_count(args.strategy, args.symbols, accounting)
            print(
                f"\nCampaign n_trials for '{args.strategy}' over {', '.join(args.symbols)} "
                f"(accounting v{accounting}): {n}"
            )


def cmd_mcp(args) -> None:
    """Serve TradeFlow over MCP (stdio). Opt-in; requires the ``mcp`` extra.

    Live trading is intentionally not exposed: the server builds
    only a data client, so it cannot place orders.
    """
    try:
        from src.mcp.server import serve
    except ImportError:
        sys.exit("The MCP server needs the 'mcp' extra. Install it:\n    uv sync --extra mcp")
    serve()


def cmd_live(args) -> None:
    from src.engine.live import LiveEngine
    from src.execution.live_trader import LiveTrader
    from src.services.sizing import build_beta_sizer, build_portfolio_weight_sizer

    scanner = args.scanner
    if args.config:
        strategy, strategy_name, cfg_scanner = _load_strategy_from_config(args.config)
        if cfg_scanner:
            scanner = cfg_scanner
        logger.info("Loaded config %s -> strategy=%r scanner=%r", args.config, strategy_name, scanner)
    else:
        strategy = STRATEGIES[args.strategy].create_with_defaults()

    broker, data_client = build_data_and_broker()
    universe = resolve_universe(data_client, scanner, args.symbols)

    sizer = None
    if args.portfolio:
        account = broker.get_account()
        equity = account.equity if account else 100_000.0
        sizer = build_portfolio_weight_sizer(
            data_client, equity, universe, "1Day", args.max_positions, args.max_weight
        )
        if sizer is not None:
            universe = sizer.symbols  # trade only the funded names
    elif args.beta_sizing:
        sizer = build_beta_sizer(data_client, strategy, universe, args.benchmark)

    engine = LiveEngine(strategy, data_client, LiveTrader(broker, strategy, sizer=sizer))
    try:
        asyncio.run(engine.start(universe))
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down live engine.")


def cmd_demo(args) -> None:
    """Run the whole pipeline on synthetic data - no Alpaca keys, no network.

    The point isn't the numbers; it's the *shape* of the workflow. We backtest
    every registered strategy (in-sample, where everything looks plausible), then
    walk-forward one of them out-of-sample and let the promotion gates deliver the
    verdict. The data is a seeded random walk with no real edge - so a healthy run
    ends in "NOT promotable", which is exactly the honesty the engine exists for.
    """
    from src.engine.backtest import BacktestEngine
    from src.marketdata.client import MarketDataClient
    from src.marketdata.synthetic import SyntheticMarketData
    from src.marketdata.timeframe import Timeframe
    from src.optimization.walk_forward import WalkForwardValidator

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
    print(
        "\n   No edge in a random walk → the gates refuse to promote it. That refusal\n"
        "   is the product. Point TradeFlow at real data with `make backtest` (add\n"
        "   your Alpaca paper keys to .env first — see .env.example).\n"
    )

    if getattr(args, "chart", None):
        if chart_result is None:
            print("   Chart skipped: no backtest result was captured.")
        else:
            from src.analytics.charts import render_demo_summary

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
    from src.costs import ParametricCostModel
    from src.research.agent import ResearchAgent, ResearchConfig
    from src.research.demo_proposals import DEMO_PROPOSALS
    from src.research.proposer import FixedProposer, build_proposer
    from src.services.data import build_data_client

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
    from src.risk.exposures import FACTOR_NAMES

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


def _add_cache_flags(parser) -> None:
    """--cache/--offline/--cache-dir, shared by backtest/optimize/walkforward -
    the same selectivity as :func:`_add_cost_flags` (not live/research/scan/...).

    ``--cache`` wraps the data client in the persistent bar cache
    (``src.store.bars.CachedMarketData``): a repeated request for the same
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
    return datetime.strptime(value, "%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TradeFlow — a broker-agnostic trading engine that's refreshingly "
        "honest about how hard making money actually is"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p, *, with_dates: bool) -> None:
        p.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
        p.add_argument("--scanner", default="volume", help="Universe scanner ('none' to skip)")
        p.add_argument(
            "--symbols", type=_symbols, default=DEFAULT_UNIVERSE, help="Comma-separated candidate symbols"
        )
        if with_dates:
            p.add_argument("--start", type=_date, default=datetime.now() - timedelta(days=30))
            p.add_argument("--end", type=_date, default=datetime.now())
            p.add_argument("--capital", type=float, default=100_000.0)

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
        p.add_argument(
            "--config",
            metavar="PATH",
            default=None,
            help="Load strategy/scanner/params from a saved config (e.g. `walkforward --save-config`); "
            "takes precedence over --strategy/--scanner when given",
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
    _add_cache_flags(bt)
    bt.add_argument(
        "--chart",
        metavar="PATH",
        default=None,
        help="render the equity curve + metrics to an image (needs the viz extra: matplotlib)",
    )
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
    live.add_argument("--max-positions", dest="max_positions", type=int, default=5)
    live.add_argument("--max-weight", dest="max_weight", type=float, default=0.25)
    live.set_defaults(func=cmd_live)

    scan = subparsers.add_parser("scan", help="Run the universe scanner only")
    scan.add_argument("--scanner", default="volume")
    scan.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
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
    alloc.add_argument(
        "--as-of", dest="as_of", type=_date, default=datetime.now(), help="Rebalance date (utility)"
    )
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
    alloc.set_defaults(func=cmd_allocate)

    opt = subparsers.add_parser(
        "optimize", help="Tune strategy parameters via backtest (in-sample — trust nothing yet)"
    )
    add_common(opt, with_dates=True)
    opt.add_argument("--method", choices=["grid", "random", "bayesian"], default="grid")
    opt.add_argument("--objective", default="sharpe_ratio")
    opt.add_argument("--max-evals", dest="max_evals", type=int, default=50)
    opt.add_argument("--output", default="optimization_results.csv")
    _add_cost_flags(opt)
    _add_cache_flags(opt)
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
        help="Perturb chosen params +-10% and re-test",
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
    add_no_journal(wf)
    add_force(wf)
    wf.set_defaults(func=cmd_walkforward)

    alphas = subparsers.add_parser(
        "alphas",
        help="Rank a universe by continuous alpha (residual-return forecast) — read-only",
    )
    alphas.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    alphas.add_argument("--scanner", default="volume", help="Scanner used as the --source score metric")
    alphas.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    alphas.add_argument(
        "--as-of", dest="as_of", type=_date, default=datetime.now(), help="Rebalance date (YYYY-MM-DD)"
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
    info.add_argument("--start", type=_date, default=datetime.now() - timedelta(days=365))
    info.add_argument("--end", type=_date, default=datetime.now())
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
    info.set_defaults(func=cmd_info)

    hz = subparsers.add_parser(
        "horizon",
        help="Measure alpha decay / half-life and recommend rebalance cadence + blend — read-only",
    )
    hz.add_argument("--strategy", choices=STRATEGIES, default="volume_spike")
    hz.add_argument("--source", choices=["strategy", "signal", "scanner"], default="strategy")
    hz.add_argument("--scanner", default="volume")
    hz.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    hz.add_argument("--start", type=_date, default=datetime.now() - timedelta(days=365))
    hz.add_argument("--end", type=_date, default=datetime.now())
    hz.add_argument("--benchmark", default="SPY")
    hz.add_argument(
        "--max-lag", dest="max_lag", type=int, default=10, help="Largest lag (periods) to measure"
    )
    hz.add_argument("--timeframe", default="1Day")
    _add_neutralize_factors_flag(hz, note="; measure the alpha you deploy")
    hz.set_defaults(func=cmd_horizon)

    risk = subparsers.add_parser(
        "risk",
        help="Estimate the universe covariance Σ and summarize its risk structure — read-only",
    )
    risk.add_argument("--symbols", type=_symbols, default=DEFAULT_UNIVERSE)
    risk.add_argument(
        "--as-of", dest="as_of", type=_date, default=datetime.now(), help="Rebalance date (YYYY-MM-DD)"
    )
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
    c_warm.add_argument("--start", type=_date, default=datetime.now() - timedelta(days=365))
    c_warm.add_argument("--end", type=_date, default=datetime.now())

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

    t_query = trials_sub.add_parser("query", help="List recent trials, and a campaign's n_trials")
    _add_db_flag(t_query)
    t_query.add_argument("--strategy", default=None)
    t_query.add_argument(
        "--kind", default=None, choices=["backtest", "optimize", "walkforward", "alpha", "research"]
    )
    t_query.add_argument(
        "--symbols",
        type=_symbols,
        default=None,
        help="Universe — with --strategy, also prints campaign n_trials",
    )
    t_query.add_argument(
        "--accounting", type=int, default=None, help="Filter to one accounting version (default: current)"
    )
    t_query.add_argument(
        "--all-accounting",
        dest="all_accounting",
        action="store_true",
        help="Show every accounting version, not just current — read the ACCOUNTING column "
        "before comparing rows across versions",
    )
    t_query.add_argument("--limit", type=int, default=20)

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
    from src.settings import SettingsError
    from src.store.bars import CacheMiss

    setup_logging()
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (SettingsError, CacheMiss) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
