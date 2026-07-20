"""The autonomous research loop and its non-negotiable guardrails - an AI that
hunts for edge, wrapped in enough skepticism to stop it from gleefully reporting
the 200 noise configs it "discovered."

```
goal -> propose (Proposer) -> hygiene gate -> validate OUT-OF-SAMPLE (walk-forward)
     -> keep iff "promotable" AND beats incumbent OOS (drawdown-guarded)
     -> loop until trial budget / token budget / K dry rounds
     -> score the shortlist ONCE on the sacred holdout -> save configs for a human
```

Guardrails enforced here in code (not by prompt):
1. **OOS-only fitness** - selection uses the walk-forward OOS aggregate; in-sample
   metrics are recorded but never the criterion.
2. **Multiple-testing correction** - ``n_trials`` accumulates across the whole
   session and feeds the Deflated Sharpe (more attempts => higher bar).
3. **Sacred holdout** - reserved up front, never passed to any search, scored
   once at the end on the final shortlist.
4. **Budgets + dryness stop** - hard caps on trials and tokens, and a
   "K rounds with no OOS improvement" stop, so the loop can't chase noise forever.
5. **Human-in-the-loop** - output is config files + a journal; nothing live, no
   ``PAPER_TRADE`` toggle, no order capability reachable.
6. **Full audit** - every proposal, trial, and decision is journaled and replayable.
7. **Sandbox + hygiene** - generated code and configs pass :mod:`src.research.sandbox`
   (<=5 params, rationale required, contract-valid) before evaluation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Type

from src.engine.backtest import BacktestError
from src.marketdata.client import MarketDataClient
from src.optimization import config_store
from src.optimization.walk_forward import WalkForwardValidator
from src.research.proposer import Proposal, ProposalContext, Proposer
from src.research.sandbox import load_strategy_from_code, validate_hygiene
from src.services.audit import audit_log, new_run_id
from src.services.registry import resolve_strategy_class
from src.strategies.base import Strategy

logger = logging.getLogger(__name__)

#: Default research-journal location (append-only JSONL).
DEFAULT_JOURNAL = "logs/research_journal.jsonl"


@dataclass
class ResearchConfig:
    """Budgets and validation settings for one research session."""

    goal: str = ""
    # Walk-forward / validation settings (reused per trial).
    mode: str = "anchored"
    n_folds: int = 4
    embargo_days: Optional[int] = None
    holdout_days: int = 60
    method: str = "grid"
    objective: str = "sharpe_ratio"
    max_evals: int = 25
    capital: float = 100_000.0
    # Loop budgets / stopping (guardrail).
    max_trials: int = 10
    max_dry_rounds: int = 3
    max_tokens: Optional[int] = None
    shortlist_size: int = 3
    # Decision rule.
    drawdown_guard_tolerance: float = 0.25  # OOS max_dd may worsen at most this fraction
    allow_code_gen: bool = False
    gates: Optional[Dict[str, float]] = None


@dataclass
class Candidate:
    """A config that cleared the gates and entered the shortlist."""

    id: str
    lineage: str
    hypothesis: str
    kind: str
    strategy: str
    params: Dict[str, Any]
    oos_metrics: Dict[str, float]
    gate_report: Dict[str, Any]
    n_trials_at_selection: int
    holdout_metrics: Optional[Dict[str, float]] = None
    saved_path: Optional[str] = None
    strategy_cls: Optional[Type[Strategy]] = field(default=None, repr=False)


@dataclass
class ResearchResult:
    shortlist: List[Candidate]
    n_trials_total: int
    rounds: int
    stopped_reason: str
    journal_path: str
    holdout_window: Optional[Dict[str, str]] = None
    saved_configs: List[str] = field(default_factory=list)


class ResearchAgent:
    """Drives the bounded research loop over the shared service core."""

    def __init__(
        self,
        strategy_name: str,
        data_client: MarketDataClient,
        proposer: Proposer,
        config: ResearchConfig,
        *,
        seed: int = 42,
        journal_path: str = DEFAULT_JOURNAL,
        observer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.strategy_name = strategy_name
        self.data_client = data_client
        self.proposer = proposer
        self.config = config
        self.seed = seed
        self.journal_path = journal_path
        #: Optional live callback, invoked with every journaled ``(event, payload)``.
        #: Purely observational - it cannot influence the loop or its decisions.
        self.observer = observer
        self.session_id = new_run_id()

    def run(self, symbols: List[str], start: datetime, end: datetime) -> ResearchResult:
        cfg = self.config
        # Reserve the sacred holdout up front; the search never sees it (guardrail).
        research_end = end - timedelta(days=cfg.holdout_days)
        holdout_window = {"start": research_end.isoformat(), "end": end.isoformat()}
        if research_end <= start:
            raise ValueError("holdout_days leaves no research window; widen the date range")

        self._journal(
            "session_start",
            {
                "session_id": self.session_id,
                "goal": cfg.goal,
                "strategy": self.strategy_name,
                "symbols": symbols,
                "research_window": {"start": start.isoformat(), "end": research_end.isoformat()},
                "holdout_window": holdout_window,
                "budgets": {
                    "max_trials": cfg.max_trials,
                    "max_dry_rounds": cfg.max_dry_rounds,
                    "max_tokens": cfg.max_tokens,
                },
                "seed": self.seed,
            },
        )

        param_ranges = resolve_strategy_class(self.strategy_name).PARAM_RANGES
        shortlist: List[Candidate] = []
        incumbent: Optional[Candidate] = None
        history: List[Dict[str, Any]] = []
        n_trials_cumulative = 0
        tokens_used = 0
        dry = 0
        rounds = 0
        stopped = "max_trials"

        while rounds < cfg.max_trials:
            if dry >= cfg.max_dry_rounds:
                stopped = "dry_rounds"
                break
            if cfg.max_tokens is not None and tokens_used >= cfg.max_tokens:
                stopped = "token_budget"
                break

            context = ProposalContext(
                goal=cfg.goal,
                strategy=self.strategy_name,
                param_ranges=param_ranges,
                history=history[-10:],
                incumbent=self._incumbent_summary(incumbent),
                round_index=rounds,
            )
            proposal = self.proposer.propose(context)
            if proposal is None:
                stopped = "proposer_exhausted"
                break
            rounds += 1
            tokens_used += proposal.tokens_used

            verdict = self._evaluate(proposal, symbols, start, research_end, n_trials_cumulative)
            if verdict is None:
                dry += 1
                history.append({"round": rounds, "rejected": True})
                continue

            result, cls, full_params = verdict
            n_trials_cumulative = result.n_trials_total
            # In-sample median is recorded for contrast only - selection is OOS-only
            # (guardrail 1). It is what makes the IS -> OOS collapse legible.
            is_sharpe = float(
                median([fr.is_metrics.get("sharpe_ratio", 0.0) for fr in result.folds] or [0.0])
            )
            oos_sharpe = result.median_oos("sharpe_ratio")
            oos_dd = result.oos_aggregate.get("max_drawdown", 0.0)
            gate_report = result.gate_report(cfg.gates)
            promotable = gate_report["promotable"]
            advanced = promotable and self._beats_incumbent(oos_sharpe, oos_dd, incumbent)

            self._journal(
                "trial",
                {
                    "session_id": self.session_id,
                    "round": rounds,
                    "kind": proposal.kind,
                    "hypothesis": proposal.hypothesis,
                    "params": full_params,
                    "is_sharpe": is_sharpe,
                    "oos_sharpe": oos_sharpe,
                    "efficiency": result.median_efficiency(),
                    "oos_max_drawdown": oos_dd,
                    "oos_aggregate": result.oos_aggregate,
                    "gate_report": gate_report,
                    "promotable": promotable,
                    "advanced": advanced,
                    "n_trials_cumulative": n_trials_cumulative,
                    "tokens_used": tokens_used,
                },
            )
            history.append(
                {
                    "round": rounds,
                    "params": full_params,
                    "oos_sharpe": oos_sharpe,
                    "promotable": promotable,
                    "advanced": advanced,
                }
            )

            if advanced:
                candidate = Candidate(
                    id=new_run_id(),
                    lineage=self._lineage(proposal, incumbent),
                    hypothesis=proposal.hypothesis,
                    kind=proposal.kind,
                    strategy=self.strategy_name,
                    params=full_params,
                    oos_metrics=result.oos_aggregate,
                    gate_report=gate_report,
                    n_trials_at_selection=n_trials_cumulative,
                    strategy_cls=cls,
                )
                shortlist.append(candidate)
                shortlist.sort(key=lambda c: c.oos_metrics.get("sharpe_ratio", 0.0), reverse=True)
                del shortlist[cfg.shortlist_size :]
                incumbent = shortlist[0]
                dry = 0
            else:
                dry += 1

        # Final exam: score the shortlist ONCE on the sacred holdout (guardrail).
        saved = self._finalize(shortlist, symbols, research_end, end, n_trials_cumulative)

        self._journal(
            "session_end",
            {
                "session_id": self.session_id,
                "stopped_reason": stopped,
                "rounds": rounds,
                "n_trials_total": n_trials_cumulative,
                "shortlist": [c.id for c in shortlist],
                "saved_configs": saved,
            },
        )
        return ResearchResult(
            shortlist=shortlist,
            n_trials_total=n_trials_cumulative,
            rounds=rounds,
            stopped_reason=stopped,
            journal_path=self.journal_path,
            holdout_window=holdout_window,
            saved_configs=saved,
        )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def _evaluate(self, proposal: Proposal, symbols, start, research_end, n_trials_offset):
        """Hygiene-gate and validate one proposal OOS; return (result, cls, params) or None."""
        cfg = self.config
        if proposal.kind == "code":
            if not cfg.allow_code_gen:
                self._journal("reject", {"reason": "code-gen disabled", "hypothesis": proposal.hypothesis})
                return None
            try:
                cls: Type[Strategy] = load_strategy_from_code(proposal.code)
            except Exception as exc:  # noqa: BLE001 - rejection, not crash
                self._journal(
                    "reject", {"reason": f"sandbox: {exc}", "hypothesis": proposal.hypothesis}
                )
                return None
            ok, reason = validate_hygiene(proposal, cls)
        else:
            cls = resolve_strategy_class(proposal.strategy or self.strategy_name)
            ok, reason = validate_hygiene(proposal, cls)

        if not ok:
            self._journal(
                "reject", {"reason": reason, "hypothesis": proposal.hypothesis, "params": proposal.params}
            )
            return None

        try:
            return self._validate(proposal, cls, symbols, start, research_end, n_trials_offset)
        except BacktestError as exc:
            # Unrunnable is not the same as unprofitable. Journal it as its own
            # rejection so a broken proposal can never be scored as "no edge".
            self._journal(
                "reject",
                {"reason": f"unrunnable: {exc}", "hypothesis": proposal.hypothesis, "kind": proposal.kind},
            )
            return None

    def _validate(self, proposal: Proposal, cls, symbols, start, research_end, n_trials_offset):
        """Run the walk-forward validation for an already hygiene-cleared proposal."""
        cfg = self.config
        validator = WalkForwardValidator(cls, self.data_client, cfg.capital, self.seed, cfg.gates)
        if proposal.kind == "code":
            # A new mechanism: let the optimizer search its PARAM_RANGES OOS.
            result = validator.run(
                symbols,
                start,
                research_end,
                mode=cfg.mode,
                n_folds=cfg.n_folds,
                embargo_days=cfg.embargo_days,
                holdout_days=0,
                method=cfg.method,
                objective=cfg.objective,
                max_evals=cfg.max_evals,
                n_trials_offset=n_trials_offset,
            )
            full_params = result.folds[-1].is_best_params if result.folds else {}
        else:
            full_params = self._full_params(cls, proposal.params)
            result = validator.evaluate_config(
                symbols,
                start,
                research_end,
                full_params,
                mode=cfg.mode,
                n_folds=cfg.n_folds,
                embargo_days=cfg.embargo_days,
                objective=cfg.objective,
                n_trials_offset=n_trials_offset,
            )
        if not result.folds:
            self._journal("reject", {"reason": "no folds evaluable", "params": full_params})
            return None
        return result, cls, full_params

    def _finalize(self, shortlist, symbols, holdout_start, holdout_end, n_trials_total) -> List[str]:
        """Score each shortlisted config once on the holdout and persist it."""
        saved: List[str] = []
        for candidate in shortlist:
            cls = candidate.strategy_cls or resolve_strategy_class(candidate.strategy)
            validator = WalkForwardValidator(
                cls,
                self.data_client,
                self.config.capital,
                self.seed,
                self.config.gates,
            )
            candidate.holdout_metrics = validator.score_window(
                symbols,
                holdout_start,
                holdout_end,
                candidate.params,
                embargo_days=self.config.embargo_days,
                n_trials=n_trials_total,
            )
            provenance = config_store.build_provenance(
                method=self.config.method,
                objective=self.config.objective,
                windows={
                    "holdout_start": holdout_start.isoformat(),
                    "holdout_end": holdout_end.isoformat(),
                    "mode": self.config.mode,
                    "n_folds": self.config.n_folds,
                },
                oos_metrics=candidate.oos_metrics,
                n_trials=n_trials_total,
                seed=self.seed,
                notes=f"Research session {self.session_id}. Hypothesis: {candidate.hypothesis}",
            )
            path = config_store.save_config(
                f"research_{self.session_id}_{candidate.id}.json",
                strategy=candidate.strategy,
                params=candidate.params,
                provenance=provenance,
            )
            candidate.saved_path = str(path)
            saved.append(str(path))
            self._journal(
                "holdout_score",
                {
                    "session_id": self.session_id,
                    "candidate": candidate.id,
                    "holdout_metrics": candidate.holdout_metrics,
                    "saved": str(path),
                },
            )
        return saved

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _full_params(self, cls: Type[Strategy], overrides: Dict[str, Any]) -> Dict[str, Any]:
        params = {name: spec["default"] for name, spec in cls.PARAM_RANGES.items() if "default" in spec}
        params.update(overrides)
        return params

    def _beats_incumbent(self, oos_sharpe: float, oos_dd: float, incumbent: Optional[Candidate]) -> bool:
        if incumbent is None:
            return True
        better_sharpe = oos_sharpe > incumbent.oos_metrics.get("sharpe_ratio", float("-inf"))
        incumbent_dd = incumbent.oos_metrics.get("max_drawdown", float("inf"))
        dd_ok = incumbent_dd == 0.0 or oos_dd <= incumbent_dd * (1 + self.config.drawdown_guard_tolerance)
        return better_sharpe and dd_ok

    @staticmethod
    def _incumbent_summary(incumbent: Optional[Candidate]) -> Optional[Dict[str, Any]]:
        if incumbent is None:
            return None
        return {"params": incumbent.params, "oos_sharpe": incumbent.oos_metrics.get("sharpe_ratio", 0.0)}

    @staticmethod
    def _lineage(proposal: Proposal, incumbent: Optional[Candidate]) -> str:
        if proposal.parent_id:
            return f"{proposal.parent_id}->v"
        return incumbent.lineage + ".m" if incumbent else "root"

    def _journal(self, event: str, payload: Dict[str, Any]) -> None:
        audit_log(f"research:{event}", payload, path=self.journal_path)
        if self.observer is not None:
            # An observer must never be able to break the research loop.
            try:
                self.observer(event, payload)
            except Exception:  # noqa: BLE001 - narration is not load-bearing
                logger.debug("research observer raised; continuing", exc_info=True)
