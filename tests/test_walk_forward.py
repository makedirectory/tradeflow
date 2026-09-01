"""Walk-forward validation tests.

Cover the properties that make a walk-forward implementation honest: exact,
deterministic fold geometry; a provably-disjoint holdout; leakage-safe OOS trade
filtering; reproducibility under a fixed seed; and the config persistence layer.
"""

from datetime import datetime
from typing import Any, ClassVar, Dict

import pandas as pd

from tests.fakes import FakeMarketData
from tradeflow.marketdata.client import MarketDataClient
from tradeflow.optimization import config_store
from tradeflow.optimization.walk_forward import WalkForwardValidator, _filter_trades_from
from tradeflow.strategies import signals
from tradeflow.strategies.base import Strategy
from tradeflow.utils.timeutils import NEW_YORK

SYMBOLS = ["AAA", "BBB"]
START, END = datetime(2024, 1, 2), datetime(2025, 6, 1)


class PeriodicStrategy(Strategy):
    """Buys every ``buy_every`` bars - frequent, tunable, deterministic trades."""

    TIMEFRAME: ClassVar[str] = "1Day"
    PARAM_RANGES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "buy_every": {"type": "int", "min": 3, "max": 9, "step": 2, "default": 5},
    }

    def __init__(self, config: Dict[str, Any]):
        config["timeframe"] = self.TIMEFRAME
        config.setdefault("risk_per_trade", 0.02)
        config.setdefault("stop_loss", 0.05)
        config.setdefault("take_profit", 0.05)
        config.setdefault(
            "position_limits", {"max_positions": 1, "max_position_size": 5000.0, "max_total_risk": 0.1}
        )
        super().__init__(config)

    def calculate_required_lookback(self) -> int:
        return 2

    def initialize(self) -> None:
        pass

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        # This double scripts signals directly (below); the score is unused.
        return pd.Series(0.0, index=data.index)

    def generate_signals(self, data: pd.DataFrame) -> Dict[Any, str]:
        # Open at the start of each cycle and close midway, so trades round-trip
        # frequently across the window (not one position held to the end).
        every = self.config["buy_every"]
        out: Dict[Any, str] = {}
        for i, ts in enumerate(data.index):
            phase = i % every
            if phase == 0:
                out[ts] = signals.BUY
            elif phase == every // 2:
                out[ts] = signals.CLOSE_BUY
            else:
                out[ts] = signals.HOLD
        return out


def _validator():
    return WalkForwardValidator(
        PeriodicStrategy,
        MarketDataClient(FakeMarketData(SYMBOLS, n=600, freq="1D")),
        initial_capital=100_000,
        seed=42,
    )


# --- fold geometry ----------------------------------------------------------
def test_folds_are_deterministic_and_ordered():
    folds, holdout = _validator().build_folds(
        START, END, mode="anchored", n_folds=4, embargo_days=2, holdout_days=30
    )
    assert len(folds) >= 1
    for fold in folds:
        assert fold.is_start < fold.is_end < fold.oos_start < fold.oos_end
        # Embargo gap is respected between IS end and OOS start.
        assert (fold.oos_start - fold.is_end).days == 2
    # OOS windows advance in time.
    for a, b in zip(folds, folds[1:]):
        assert b.oos_start > a.oos_start


def test_holdout_is_disjoint_from_every_fold():
    folds, holdout = _validator().build_folds(
        START, END, mode="anchored", n_folds=4, embargo_days=2, holdout_days=45
    )
    assert holdout is not None
    holdout_start, holdout_end = holdout
    assert holdout_end == END
    # The holdout starts at/after the last OOS window ends - sacred and untouched.
    for fold in folds:
        assert fold.oos_end <= holdout_start


def test_anchored_is_expands_rolling_is_fixed():
    v = _validator()
    anchored, _ = v.build_folds(START, END, mode="anchored", n_folds=4, embargo_days=2)
    rolling, _ = v.build_folds(START, END, mode="rolling", n_folds=4, embargo_days=2)
    # Anchored IS always begins at the global start; rolling IS slides forward.
    assert all(f.is_start == START for f in anchored)
    assert rolling[-1].is_start > rolling[0].is_start


# --- leakage-safe OOS filtering ---------------------------------------------
def test_filter_trades_drops_entries_before_oos_start():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz=NEW_YORK)
    trades = pd.DataFrame({"entry_time": idx, "exit_time": idx, "pnl": [1, 2, 3, 4, 5]})
    kept = _filter_trades_from(trades, datetime(2024, 1, 3))
    assert (pd.to_datetime(kept["entry_time"]).dt.tz_localize(None) >= datetime(2024, 1, 3)).all()
    assert len(kept) == 3  # 01-03, 01-04, 01-05


# --- end-to-end run ---------------------------------------------------------
def test_run_produces_folds_and_no_oos_leakage():
    result = _validator().run(
        SYMBOLS,
        START,
        END,
        mode="anchored",
        n_folds=3,
        embargo_days=2,
        holdout_days=30,
        method="grid",
        objective="sharpe_ratio",
    )
    assert result.folds
    assert result.n_trials_total == sum(fr.n_trials for fr in result.folds)
    assert "sharpe_ratio" in result.oos_aggregate
    assert result.holdout is not None  # holdout was scored once

    report = result.gate_report()
    assert "promotable" in report and isinstance(report["promotable"], bool)
    assert "oos_sharpe" in report["checks"]


def test_run_is_reproducible_under_fixed_seed():
    a = _validator().run(SYMBOLS, START, END, n_folds=3, embargo_days=2, method="grid")
    b = _validator().run(SYMBOLS, START, END, n_folds=3, embargo_days=2, method="grid")
    assert [fr.is_best_params for fr in a.folds] == [fr.is_best_params for fr in b.folds]
    assert [fr.oos_metrics["sharpe_ratio"] for fr in a.folds] == [
        fr.oos_metrics["sharpe_ratio"] for fr in b.folds
    ]


def test_oos_trades_are_counted():
    result = _validator().run(SYMBOLS, START, END, n_folds=3, embargo_days=2, method="grid")
    # The PeriodicStrategy trades frequently, so at least one fold sees OOS trades.
    assert result.total_oos_trades() > 0


# --- config persistence -----------------------------------------------------
def test_save_and_load_config_round_trip(tmp_path):
    provenance = config_store.build_provenance(
        method="grid",
        objective="sharpe_ratio",
        windows={"start": START, "end": END},
        oos_metrics={"sharpe_ratio": 1.23},
        n_trials=12,
        seed=42,
    )
    path = config_store.save_config(
        tmp_path / "candidate.json",
        strategy="periodic",
        scanner="volume",
        params={"buy_every": 5},
        provenance=provenance,
    )
    loaded = config_store.load_config(path)
    assert loaded["strategy"] == "periodic"
    assert loaded["params"] == {"buy_every": 5}
    assert loaded["provenance"]["objective"] == "sharpe_ratio"
    assert loaded["provenance"]["n_trials"] == 12
    assert loaded["provenance"]["windows"]["start"] == START.isoformat()


def test_cost_model_reaches_both_in_sample_and_out_of_sample_backtests():
    """A validator built with a cost model charges it on every simulated fill.

    Threading it only into the OOS leg would let the optimizer pick a config on
    gross returns and then score it net - flattering walk-forward efficiency for
    a purely mechanical reason.
    """
    from tradeflow.costs import ParametricCostModel

    client = MarketDataClient(FakeMarketData(["AAA", "BBB"], n=600, freq="1D"))
    kwargs = dict(
        symbols=["AAA", "BBB"],
        start=datetime(2024, 1, 2),
        end=datetime(2025, 6, 1),
        n_folds=2,
        holdout_days=30,
        method="grid",
        max_evals=4,
    )

    gross = WalkForwardValidator(PeriodicStrategy, client, seed=42).run(**kwargs)
    net = WalkForwardValidator(PeriodicStrategy, client, seed=42, cost_model=ParametricCostModel()).run(
        **kwargs
    )

    # Same seed and folds, so any divergence is the cost model doing its job.
    assert net.folds and gross.folds
    is_gross = [fr.is_metrics.get("total_return", 0.0) for fr in gross.folds]
    is_net = [fr.is_metrics.get("total_return", 0.0) for fr in net.folds]
    assert is_net != is_gross, "costs did not reach the in-sample optimization"
    assert all(n <= g + 1e-9 for n, g in zip(is_net, is_gross)), "costs must not improve returns"


def test_provenance_stamps_the_accounting_model(tmp_path):
    """Saved metrics record which engine accounting produced them."""
    from tradeflow.engine.backtest import ACCOUNTING_VERSION

    provenance = config_store.build_provenance(
        method="grid", objective="sharpe_ratio", windows={}, oos_metrics={"sharpe_ratio": 1.0}
    )
    assert provenance.accounting == ACCOUNTING_VERSION

    path = config_store.save_config(
        tmp_path / "c.json", strategy="periodic", params={"buy_every": 3}, provenance=provenance
    )
    loaded = config_store.load_config(path)
    assert loaded["provenance"]["accounting"] == ACCOUNTING_VERSION
    assert config_store.is_current_accounting(loaded)


def test_config_predating_the_stamp_is_flagged_not_silently_reused(tmp_path, caplog):
    """A record with no accounting field predates the field, and saying so is the whole point."""
    import json

    legacy = {
        "strategy": "periodic",
        "params": {"buy_every": 3},
        # Written before the field existed - no "accounting" key.
        "provenance": {"method": "grid", "oos_metrics": {"sharpe_ratio": 2.0}},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))

    with caplog.at_level("WARNING"):
        loaded = config_store.load_config(path)
    assert not config_store.is_current_accounting(loaded)
    assert "not comparable" in caplog.text.lower()
    # The params are still perfectly usable; only the recorded metrics are stale.
    assert loaded["params"]["buy_every"] == 3


def test_git_sha_marks_a_dirty_working_tree(monkeypatch):
    """A dirty tree must not reuse a clean tree's identifier.

    Regression: the identifier came from ``rev-parse HEAD`` alone, so uncommitted
    edits kept the same SHA. Callers use it to decide whether a stored trial may be
    served instead of re-run, so an edited-but-uncommitted strategy would match its
    own pre-edit result and be served as though the change had been evaluated.
    """
    import subprocess

    def fake_run(cmd, **kwargs):
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M tradeflow/strategies/base.py\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert config_store.current_git_sha() == "abc1234-dirty"


def test_git_sha_is_bare_when_the_tree_is_clean(monkeypatch):
    """The suffix must appear only when there is something uncommitted - otherwise
    every ordinary run would miss the cache it is meant to protect."""
    import subprocess

    def fake_run(cmd, **kwargs):
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert config_store.current_git_sha() == "abc1234"


def test_git_sha_is_none_when_git_is_unavailable(monkeypatch):
    """Still best-effort: no git means no identifier, not a crash."""
    import subprocess

    def fake_run(cmd, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert config_store.current_git_sha() is None


# --- promotion prerequisites --------------------------------------------------
def test_an_unevaluated_prerequisite_is_never_a_pass():
    """The failure this whole block exists to avoid.

    A cost curve nobody ran and a family too thin to test are both *unknown*. Rendering
    unknown as green is how a candidate gets promoted on evidence that was never
    gathered.
    """
    from tradeflow.analytics.performance import promotion_prerequisites

    prereq = promotion_prerequisites()

    assert prereq["ready"] is None
    assert prereq["evaluated"] == 0
    assert all(not check["evaluated"] and not check["passed"] for check in prereq["checks"].values())


def test_a_family_of_two_is_reported_as_too_thin_rather_than_significant():
    """Observed in a real run: K=2, p=0.002 - a striking p-value over a family so small
    the test is arithmetic rather than evidence. It must not gate on that."""
    from tradeflow.analytics.performance import promotion_prerequisites

    prereq = promotion_prerequisites(
        bootstrap={"family": {"available": True, "family_p": 0.002, "n_used": 2}}
    )
    family = prereq["checks"]["family_bootstrap"]

    assert not family["evaluated"]
    assert not family["passed"]  # a striking p-value still does not pass on K=2
    assert "10 usable" in family["note"] and "2 available" in family["note"]


def test_a_family_large_enough_is_gated_on_its_p_value():
    """And once there is enough family, both directions bite."""
    from tradeflow.analytics.performance import promotion_prerequisites

    strong = promotion_prerequisites(
        bootstrap={"family": {"available": True, "family_p": 0.002, "n_used": 12}}
    )
    weak = promotion_prerequisites(bootstrap={"family": {"available": True, "family_p": 0.30, "n_used": 12}})

    assert strong["checks"]["family_bootstrap"]["passed"]
    assert not weak["checks"]["family_bootstrap"]["passed"]
    assert weak["ready"] is False


def test_cost_stress_must_clear_three_times_its_assumed_cost():
    from tradeflow.analytics.performance import promotion_prerequisites

    assert promotion_prerequisites(cost_stress={"survives_to_multiple": 5.0})["ready"] is True
    assert promotion_prerequisites(cost_stress={"survives_to_multiple": 1.0})["ready"] is False


def test_ready_never_implies_the_unevaluated_checks_would_have_passed():
    """The exact shape of a real run: cost stress clears at 5x, family is too thin.

    `ready` is True because every *evaluated* check passed, and the count has to travel
    with it or it reads as a full clearance.
    """
    from tradeflow.analytics.performance import promotion_prerequisites

    prereq = promotion_prerequisites(
        cost_stress={"survives_to_multiple": 5.0},
        bootstrap={"family": {"available": True, "family_p": 0.002, "n_used": 2}},
    )

    assert prereq["ready"] is True
    assert (prereq["evaluated"], prereq["total"]) == (1, 2)


def test_prerequisites_are_not_the_promotion_gates():
    """`promotable` stays statistical and keeps meaning for every trial already
    recorded; nothing here touches it."""
    from tradeflow.analytics.performance import DEFAULT_PREREQUISITES
    from tradeflow.optimization.walk_forward import DEFAULT_GATES

    assert not set(DEFAULT_PREREQUISITES) & set(DEFAULT_GATES)
