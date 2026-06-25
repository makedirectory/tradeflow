"""Reusable chart rendering for backtest and walk-forward results.

These functions operate on the engine's own result objects — a
:class:`~src.engine.backtest.BacktestResult` or a
:class:`~src.optimization.walk_forward.WalkForwardResult` — so *any* strategy or
run can be rendered, not just the demo. The CLI exposes them via a ``--chart``
argument on the ``backtest`` and ``walkforward`` commands; the demo composes the
same panels into one summary image.

Matplotlib is an optional dependency (the ``viz`` extra), imported lazily so the
base install and the text-only commands never pay for it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Palette — kept here so the look is consistent and easy to retune.
_INK = "#1b1f24"
_MUTED = "#6b7280"
_ACCENT = "#2563eb"
_PASS = "#16a34a"
_FAIL = "#dc2626"
_GRID = "#e5e7eb"


def _mpl():
    """Import matplotlib lazily; raise an actionable error if it's missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        plt.rcParams.update({"font.size": 11, "axes.edgecolor": _GRID, "text.color": _INK})
        return plt, GridSpec
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Charting needs matplotlib. Install the viz extra: `uv sync --extra viz`."
        ) from exc


def _fmt(value: Any) -> str:
    """Compact, readable formatting for values of unknown scale."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == 0:
        return "0"
    if abs(v) < 0.01 or abs(v) >= 1e4:
        return f"{v:.1e}"
    return f"{v:.2f}"


def _metrics_subtitle(metrics: Dict[str, float]) -> str:
    return (
        f"Sharpe {metrics.get('sharpe_ratio', 0):.2f}   ·   "
        f"return {metrics.get('total_return', 0):.1f}%   ·   "
        f"{int(metrics.get('total_trades', 0))} trades   ·   "
        f"max DD {metrics.get('max_drawdown', 0):.1f}%"
    )


# --------------------------------------------------------------------------- #
# Composable panels (draw onto a provided Axes)
# --------------------------------------------------------------------------- #
def _equity_panel(ax, equity_curve: List[float], *, title: str, subtitle: Optional[str]) -> None:
    n = len(equity_curve)
    x = list(range(n))
    ax.set_facecolor("white")
    ax.plot(x, equity_curve, color=_ACCENT, linewidth=2.0)
    if n:
        ax.fill_between(x, equity_curve, min(equity_curve), color=_ACCENT, alpha=0.08)
    ax.margins(x=0.01)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("Backtest timeline (days)", color=_MUTED)
    ax.set_ylabel("Account equity ($)", color=_MUTED)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"${v:,.0f}")
    ax.tick_params(colors=_MUTED)
    if subtitle:
        ax.annotate(
            subtitle, xy=(0.015, 0.04), xycoords="axes fraction",
            fontsize=10.5, color=_MUTED, style="italic",
        )


def _verdict_panel(
    ax,
    *,
    promotable: bool,
    checks: Dict[str, Dict[str, Any]],
    oos_sharpe: float,
    efficiency: float,
    oos_trades: int,
    title: str = "WALK-FORWARD VERDICT",
) -> None:
    ax.axis("off")

    def stat_row(y, label, value):
        ax.text(0.0, y, label, transform=ax.transAxes, va="top", fontsize=10, color=_INK)
        ax.text(
            1.0, y, value, transform=ax.transAxes, va="top", ha="right",
            fontsize=10, family="monospace", color=_INK,
        )

    ax.text(0.0, 0.99, title, transform=ax.transAxes, va="top",
            fontsize=10, color=_MUTED, fontweight="bold")
    ax.text(
        0.0, 0.88, "PROMOTABLE" if promotable else "NOT PROMOTABLE",
        transform=ax.transAxes, va="top", fontsize=22, fontweight="bold",
        color=_PASS if promotable else _FAIL,
    )
    stat_row(0.74, "OOS Sharpe (median)", _fmt(oos_sharpe))
    stat_row(0.67, "Efficiency (OOS / IS)", _fmt(efficiency))
    stat_row(0.60, "OOS trades", str(oos_trades))
    ax.text(0.0, 0.49, "Promotion gates", transform=ax.transAxes, va="top",
            fontsize=10.5, color=_MUTED, fontweight="bold")

    top, bottom = 0.41, 0.02
    rows = list(checks.items())
    step = (top - bottom) / max(len(rows), 1)
    for i, (name, check) in enumerate(rows):
        y = top - i * step
        passed = bool(check.get("passed"))
        ax.text(0.0, y, "PASS" if passed else "FAIL", transform=ax.transAxes, va="top",
                fontsize=9.5, fontweight="bold", family="monospace",
                color=_PASS if passed else _FAIL)
        ax.text(0.15, y, name, transform=ax.transAxes, va="top", fontsize=9.5, color=_INK)
        ax.text(1.0, y, f"{_fmt(check.get('value'))} / {_fmt(check.get('threshold'))}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                family="monospace", color=_MUTED)


def _save(fig, out_path: str) -> str:
    fig.savefig(out_path, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Public renderers (operate on engine result objects — reusable for any run)
# --------------------------------------------------------------------------- #
def render_backtest_chart(
    result, out_path: str, *, title: Optional[str] = None, subtitle: Optional[str] = None
) -> str:
    """Render any ``BacktestResult`` as an equity curve + headline metrics."""
    plt, _ = _mpl()
    fig = plt.figure(figsize=(10, 4.8), dpi=120, facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    _equity_panel(
        ax, result.equity_curve,
        title=title or "Backtest equity",
        subtitle=subtitle or _metrics_subtitle(result.metrics),
    )
    fig.subplots_adjust(top=0.9, bottom=0.13, left=0.1, right=0.97)
    return _save(fig, out_path)


def render_walkforward_chart(
    result, out_path: str, *, title: Optional[str] = None, subtitle: Optional[str] = None
) -> str:
    """Render any ``WalkForwardResult`` as the verdict + promotion-gate scorecard."""
    plt, _ = _mpl()
    report = result.gate_report()
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120, facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    _verdict_panel(
        ax,
        promotable=report["promotable"],
        checks=report["checks"],
        oos_sharpe=result.median_oos("sharpe_ratio"),
        efficiency=result.median_efficiency(),
        oos_trades=result.total_oos_trades(),
    )
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", x=0.05, ha="left")
    if subtitle:
        fig.text(0.05, 0.9, subtitle, fontsize=10.5, color=_MUTED, ha="left")
    fig.subplots_adjust(top=0.82, bottom=0.05, left=0.05, right=0.95)
    return _save(fig, out_path)


def render_demo_summary(
    backtest_result, wf_result, out_path: str, *, strategy: str = "strategy",
    seed: Optional[int] = None,
) -> str:
    """Compose the in-sample equity curve and the walk-forward verdict in one image.

    Reuses the same panels as the standalone renderers; the demo is just one more
    caller, not special-cased plotting code.
    """
    plt, GridSpec = _mpl()
    report = wf_result.gate_report()
    equity = backtest_result.equity_curve
    ret_pct = (equity[-1] / equity[0] - 1.0) * 100.0 if equity and equity[0] else 0.0

    fig = plt.figure(figsize=(12, 5.2), dpi=120, facecolor="white")
    gs = GridSpec(1, 2, width_ratios=[1.55, 1.0], wspace=0.18, figure=fig)

    _equity_panel(
        fig.add_subplot(gs[0, 0]), equity,
        title=f"In-sample equity — {strategy} on synthetic data",
        subtitle=f"in-sample return {ret_pct:+.1f}%  ·  looks tradeable",
    )
    _verdict_panel(
        fig.add_subplot(gs[0, 1]),
        promotable=report["promotable"],
        checks=report["checks"],
        oos_sharpe=wf_result.median_oos("sharpe_ratio"),
        efficiency=wf_result.median_efficiency(),
        oos_trades=wf_result.total_oos_trades(),
    )

    sub = "synthetic random walk — no real edge, so the gates refuse to promote it"
    if seed is not None:
        sub += f"  (seed {seed})"
    fig.suptitle(
        "TradeFlow demo: honest out-of-sample evaluation",
        fontsize=15, fontweight="bold", x=0.07, ha="left", y=0.99,
    )
    fig.text(0.07, 0.925, sub, fontsize=10.5, color=_MUTED, ha="left")
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.07, right=0.975)
    return _save(fig, out_path)
