"""MCP server: a thin adapter exposing TradeFlow's deterministic core as tools, so
an agent can do the research and the engine can stay boringly deterministic.

No business logic lives here - every tool calls a function in :mod:`src.services`
and logs the call to the audit trail. The server constructs **only** a data
client (never a broker/trading client), so it is structurally incapable of
placing an order. The hard wall is the *absence* of any
order/live/account-mutation tool; it cannot be prompt-injected around because the
capability is not wired in.

Run with: ``python main.py mcp`` (needs the ``mcp`` extra).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.services import analysis, configs, glossary, registry
from src.services.audit import audit_log, new_run_id

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
    "compute_alphas",
    "combine_alphas",
    "compute_risk",
    "construct_portfolio",
    "compute_information",
    "get_metrics_glossary",
    "summarize_bars",
    "save_config",
    "load_config",
    "list_configs",
)

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


def build_server(data_client=None):
    """Construct the FastMCP server with all read/analyze/propose tools registered.

    ``data_client`` may be injected (offline tests); otherwise a data-only Alpaca
    client is built. Importing ``mcp`` is deferred to here so the package and the
    rest of the CLI never require the extra.
    """
    from mcp.server.fastmcp import FastMCP

    if data_client is None:
        from src.services.data import build_data_client

        data_client = build_data_client()
    _assert_no_trading_client(data_client)

    server = FastMCP(SERVER_NAME)
    dc = data_client

    def _logged(tool: str, inputs: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        run_id = result.get("run_id") if isinstance(result, dict) else None
        audit_log(tool, inputs, run_id=run_id or new_run_id(), result_summary=_summary(result))
        return result

    # ---------------- Discovery (safe) ---------------- #
    @server.tool()
    def list_strategies() -> List[Dict[str, str]]:
        """List available trading strategies (name, description, timeframe)."""
        return _logged("list_strategies", {}, registry.list_strategies())

    @server.tool()
    def list_scanners() -> List[Dict[str, str]]:
        """List available universe scanners (name, description)."""
        return _logged("list_scanners", {}, registry.list_scanners())

    @server.tool()
    def get_param_ranges(kind: str, name: str) -> Dict[str, Any]:
        """Get the tunable PARAM_RANGES (bounds + defaults) for a strategy or scanner.

        kind: "strategy" or "scanner". This is your map of what can be optimized.
        """
        return _logged(
            "get_param_ranges", {"kind": kind, "name": name}, registry.get_param_ranges(kind, name)
        )

    # ---------------- Read / analyze (safe) ---------------- #
    @server.tool()
    def run_scan(scanner: str, symbols: List[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a universe scanner over `symbols`; return the flagged symbols + signals."""
        inputs = {"scanner": scanner, "symbols": symbols, "config": config}
        return _logged("run_scan", inputs, analysis.run_scan(dc, scanner, symbols, config))

    @server.tool()
    def run_backtest(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        capital: float = 100_000.0,
        config: Optional[Dict[str, Any]] = None,
        beta_sizing: bool = False,
        benchmark: str = "SPY",
    ) -> Dict[str, Any]:
        """Backtest `strategy` on `symbols` over [start, end] (YYYY-MM-DD).

        Returns the full  metrics dict, trade count, and a path to the
        trades CSV (trades are not inlined). `config` overrides default params.
        """
        inputs = {
            "strategy": strategy,
            "symbols": symbols,
            "start": start,
            "end": end,
            "capital": capital,
            "config": config,
            "beta_sizing": beta_sizing,
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
        )
        return _logged("run_backtest", inputs, result)

    @server.tool()
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
    ) -> Dict[str, Any]:
        """Search a strategy's parameters IN-SAMPLE (grid|random|bayesian).

        Returns best_params, best_score, and the top-10 configs (full grid -> CSV).
        IMPORTANT: in-sample results from picking the best of many configs are
        inflated and are NOT evidence of edge. Always validate with
        run_walk_forward before trusting them.
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
        )
        return _logged("run_optimization", inputs, result)

    @server.tool()
    def run_walk_forward(
        strategy: str,
        symbols: List[str],
        start: str,
        end: str,
        mode: str = "anchored",
        n_folds: int = 4,
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
    ) -> Dict[str, Any]:
        """Honest out-of-sample evaluation across folds - your advancement criterion.

        Optimizes in-sample per fold, scores out-of-sample, and returns the OOS
        aggregate, walk-forward efficiency, IS->OOS degradation, per-fold summary,
        the holdout score (if holdout_days>0), the Deflated Sharpe (corrected for
        all configs tried), and the PROMOTION-GATE verdict (pass/fail per gate +
        overall "promotable"). A config advances only if it is promotable - never
        on in-sample Sharpe. include_pbo is expensive; leave it off unless needed.
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
        )
        return _logged("run_walk_forward", inputs, result)

    @server.tool()
    def compute_alphas(
        strategy: str,
        symbols: List[str],
        as_of: str,
        source: str = "strategy",
        scanner: str = "volume",
        ic: float = 0.03,
        benchmark: str = "SPY",
        neutralize: bool = False,
        lookback_days: int = 180,
    ) -> Dict[str, Any]:
        """Rank `symbols` by continuous alpha (residual-return forecast) as of a date.

        Turns each name's view into a comparable, annualised residual-return
        forecast via alpha = sigma * IC * z (cross-sectional z-score scaled by
        residual vol and an ASSUMED IC). source: "strategy" uses the strategy's
        continuous conviction; "signal" uses its BUY/SELL/HOLD as +1/-1/0; "scanner"
        uses the scanner's continuous strength. Read-only: forecasts only, never
        reads a forward return, places no orders. The absolute scale is only as good
        as the assumed IC; relative ranking is not.
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
        )
        return _logged("compute_alphas", inputs, result)

    @server.tool()
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
        signal correlation matrix over a trailing window of realised residual returns,
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

    @server.tool()
    def compute_risk(
        symbols: List[str],
        as_of: str,
        model: str = "shrinkage",
        timeframe: str = "1Day",
        lookback_days: int = 365,
    ) -> Dict[str, Any]:
        """Estimate the universe's covariance Σ and summarise its risk structure.

        Returns the shrinkage intensity δ, condition number, mean correlation, the
        equal-weight portfolio volatility, and the top risk contributors as of a date
        (model: "shrinkage" = Ledoit–Wolf, the well-conditioned default; "sample" =
        raw). Read-only: Σ sizes conviction, it never places an order. Risk is not
        additive — correlated names are one bet, which this matrix is what captures.
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

    @server.tool()
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
        """Construct the mean-variance optimal portfolio from alphas (005) and Σ (006).

        Maximises αᵀw − λ·wᵀΣw over long-only, box-bounded, budgeted (optionally
        cardinality-capped) weights, calibrating λ to `target_te`. Returns the proposed
        weights plus the Fundamental-Law report: IR* (best achievable IR), predicted
        tracking error and IR, the transfer coefficient (how much of IR* survives the
        constraints), and turnover. A PROPOSAL — read-only, places no orders; a human
        promotes the config.
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

    @server.tool()
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
        """Measure a strategy's IC, breadth, and predicted-vs-realized IR (spec 009).

        Pairs the alpha known at each rebalance with the subsequent realised residual
        return (strict forward alignment) to measure the information coefficient
        (Pearson + rank) and its t-stat, the effective breadth (deflated by ρ̄), and the
        predicted vs realised information ratio — with guardrails (IR standard-error
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

    @server.tool()
    def get_metrics_glossary() -> Dict[str, Any]:
        """Definitions + pitfalls of every reported metric, plus global caveats.

        Read this before interpreting results - it spells out the Deflated Sharpe /
        multiple-testing trap and the closed-trade equity-curve caveat so you don't
        over-trust in-sample Sharpe.
        """
        return _logged("get_metrics_glossary", {}, glossary.metrics_glossary())

    @server.tool()
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
    @server.tool()
    def save_config(
        name: str,
        strategy: str,
        params: Dict[str, Any],
        scanner: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save a candidate config to configs/<name>.json (with provenance).

        This writes a file for a HUMAN to review and promote. It does NOT affect
        any running process and cannot enable live trading.
        """
        inputs = {"name": name, "strategy": strategy, "params": params, "scanner": scanner}
        result = configs.save_config(
            name, strategy=strategy, params=params, scanner=scanner, provenance=provenance
        )
        return _logged("save_config", inputs, result)

    @server.tool()
    def load_config(name: str) -> Dict[str, Any]:
        """Load a previously saved candidate config by name."""
        return _logged("load_config", {"name": name}, configs.load_config(name))

    @server.tool()
    def list_configs() -> List[Dict[str, Any]]:
        """List saved candidate configs with a compact summary of each."""
        return _logged("list_configs", {}, configs.list_configs())

    return server


def serve() -> None:
    """Build the server and run it over stdio (what Claude Desktop/Code launch)."""
    logger.info("Starting TradeFlow MCP server (stdio). Live trading is NOT exposed.")
    build_server().run()


def _assert_no_trading_client(data_client) -> None:
    """Guardrail: the MCP process must hold only a data client."""
    from src.marketdata.client import MarketDataClient

    if not isinstance(data_client, MarketDataClient):
        raise RuntimeError("MCP server requires a MarketDataClient (data only); refusing to start.")
    for attr in ("broker", "trading_client", "_broker"):
        if getattr(data_client, attr, None) is not None:
            raise RuntimeError(f"MCP data client unexpectedly exposes {attr!r}; refusing to start.")


def _summary(result: Any) -> Dict[str, Any]:
    """A tiny, log-friendly digest of a tool result (no large arrays)."""
    if not isinstance(result, dict):
        return {"type": type(result).__name__, "len": len(result) if hasattr(result, "__len__") else None}
    keep = ("run_id", "best_score", "total_trades", "n_trials", "n_trials_total", "flagged_count")
    digest = {k: result[k] for k in keep if k in result}
    if "gate_report" in result and isinstance(result["gate_report"], dict):
        digest["promotable"] = result["gate_report"].get("promotable")
    return digest
