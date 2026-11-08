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
7. **Sandbox + hygiene** - generated code and configs pass :mod:`quantred.research.sandbox`
   (<=5 params, rationale required, contract-valid) before evaluation.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Type

from quantred.costs.base import CostModel
from quantred.engine.backtest import ACCOUNTING_VERSION, BacktestError
from quantred.marketdata.client import MarketDataClient
from quantred.optimization import config_store
from quantred.optimization.walk_forward import WalkForwardValidator
from quantred.research.proposer import Proposal, ProposalContext, Proposer
from quantred.research.sandbox import load_strategy_from_code, validate_hygiene
from quantred.services import audit
from quantred.services.audit import audit_log, new_run_id
from quantred.services.registry import resolve_strategy_class
from quantred.store.trials import TrialStore, db_path_for_journal
from quantred.strategies.base import Strategy
from quantred.utils.errors import RunRefused
from quantred.utils.timeutils import shift_days

logger = logging.getLogger(__name__)


def draft_code_hash(code: str) -> str:
    """The digest a generated strategy is identified by (one definition, in the service)."""
    from quantred.services.analysis import draft_code_hash as _hash

    return _hash(code)


def prior_holdout_looks(journal_path: str, holdout_window: Dict[str, str], symbols: List[str]) -> int:
    """How many earlier sessions in this journal scored this same holdout over this universe.

    Each look at a holdout is information the next search can overfit to, so a repeat is
    worth saying out loud. Counted from the journal (the only record that carries both
    the window and the scoring), and best-effort: an unreadable journal is zero looks
    found, never a reason to stop a session.
    """
    wanted = sorted(s.upper() for s in symbols)
    windows: Dict[str, bool] = {}
    scored: set = set()
    try:
        with open(journal_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                inputs = record.get("inputs") or {}
                sid = inputs.get("session_id")
                if record.get("tool") == "research:session_start":
                    same = (
                        inputs.get("holdout_window") == holdout_window
                        and sorted(str(s).upper() for s in inputs.get("symbols") or []) == wanted
                    )
                    windows[sid] = same
                elif record.get("tool") == "research:holdout_score":
                    scored.add(sid)
    except OSError:
        return 0
    return sum(1 for sid in scored if windows.get(sid))


#: Default research-journal location (append-only JSONL). The same file CLI
#: ``backtest``/``optimize`` trials append to, so a trial store indexes one journal.
#: Lazy for the same reason the constant it names is: resolving the path creates the
#: directory, and doing that at import time makes a read-only state root fail before
#: any command has run.
def __getattr__(name: str) -> Any:
    if name == "DEFAULT_JOURNAL":
        return str(audit.default_trial_journal())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    #: Advisory only: annotate each shortlisted candidate with a
    #: bootstrap-skill report (own + family p) after holdout scoring. Off by
    #: default - it never influences keep/discard (guardrail 1 stays untouched).
    bootstrap_skill: bool = False
    #: Charged on every simulated fill. Left ``None`` the loop selects on gross
    #: returns, which favors turnover the strategy could not afford live.
    cost_model: Optional[CostModel] = None
    #: How that model is spelled in a trial's identity (``{"gross": True}`` or the three
    #: cost parameters) - the same shape every other command folds into its dedup key.
    #: Left ``None`` it is derived from ``cost_model``, so a caller that builds the model
    #: from flags passes the key it built the model from rather than have it re-derived.
    cost_key: Optional[Dict[str, Any]] = None

    def cost_identity(self) -> Dict[str, Any]:
        """The cost assumptions as they enter a round's recorded identity."""
        if self.cost_key is not None:
            return dict(self.cost_key)
        model = self.cost_model
        if model is None:
            return {"gross": True}
        if hasattr(model, "commission_rate"):
            return {
                "gross": False,
                "commission_bps": round(model.commission_rate * 1e4, 9),
                "impact_eta": model.impact_eta,
                "borrow_bps": round(model.annual_borrow_rate * 1e4, 9),
            }
        return {"gross": False, "model": type(model).__name__}


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
    #: Median-across-folds OOS Sharpe - the measure the promotion gate applies, and so the
    #: one candidates are compared and ranked on. ``oos_metrics`` is the stitched
    #: aggregate, a different number; mixing the two compared a median with an aggregate.
    oos_sharpe: Optional[float] = None
    #: Source of the generated strategy, for a ``kind == "code"`` candidate. The config
    #: saved from it names a ``draft:`` identity and cannot be loaded as a strategy; this
    #: is what a human reviews before adding it to a strategy pack.
    code: Optional[str] = None
    holdout_metrics: Optional[Dict[str, float]] = None
    saved_path: Optional[str] = None
    strategy_cls: Optional[Type[Strategy]] = field(default=None, repr=False)
    #: This candidate's dated OOS return series - kept so
    #: ``bootstrap_skill`` can be computed once the holdout score is in, without
    #: re-running the walk-forward.
    oos_returns: Optional[Any] = field(default=None, repr=False)
    #: The bootstrap-skill report, when ``ResearchConfig.bootstrap_skill``
    #: is on - advisory only, computed after holdout scoring; never influences
    #: ``_beats_incumbent`` or the keep/discard decision (guardrail 1: OOS-only,
    #: gate-based fitness stays exactly as specced).
    bootstrap_skill: Optional[Dict[str, Any]] = None


@dataclass
class ResearchResult:
    shortlist: List[Candidate]
    n_trials_total: int
    rounds: int
    stopped_reason: str
    journal_path: str
    holdout_window: Optional[Dict[str, str]] = None
    saved_configs: List[str] = field(default_factory=list)
    #: Rounds that reached a walk-forward (each is one journaled trial) and rounds
    #: rejected before one ran. ``rounds`` is both: the number of proposals made.
    n_evaluated: int = 0
    n_rejected: int = 0
    #: Whether the holdout was scored at all - it is only scored for a shortlist.
    holdout_scored: bool = False
    #: Earlier sessions in this journal that scored this same holdout window over this
    #: same universe; each look at a holdout erodes it.
    prior_holdout_looks: int = 0


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
        # ``None``, not the constant: a default argument is evaluated when the class
        # body runs, which is import time, and resolving this path creates a directory.
        journal_path: Optional[str] = None,
        observer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        trial_store: Optional[TrialStore] = None,
    ):
        self.strategy_name = strategy_name
        self.data_client = data_client
        self.proposer = proposer
        self.config = config
        self.seed = seed
        journal_path = journal_path or str(audit.default_trial_journal())
        self.journal_path = journal_path
        #: Optional live callback, invoked with every journaled ``(event, payload)``.
        #: Purely observational - it cannot influence the loop or its decisions.
        self.observer = observer
        self.session_id = new_run_id()
        #: The round being decided and the reason the last rejection gave; the loop reads
        #: the second to tell the proposer why its last idea was refused.
        self._round = 0
        self._last_reject: Optional[str] = None
        #: Counts as of the last failure, for a caller that has only the exception.
        self.progress: Dict[str, int] = {}
        #: The trial store this session dual-writes into and dedups
        #: against - defaults to the store alongside this session's journal, so
        #: redirecting ``journal_path`` (as tests do) isolates the store too.
        #: Best-effort like every other trial-store touchpoint: the store is
        #: derived, never authoritative, so a store-open failure (read-only
        #: filesystem, disk full, sandboxed CI) degrades to no dedup/recording
        #: rather than aborting the whole research session before it starts.
        if trial_store is not None:
            self.trial_store: Optional[TrialStore] = trial_store
        else:
            try:
                self.trial_store = TrialStore(db_path_for_journal(journal_path), journal_path=journal_path)
            except Exception:  # noqa: BLE001 - never let store setup block research
                logger.warning("Trial store unavailable; continuing without dedup/recording", exc_info=True)
                self.trial_store = None

    def run(self, symbols: List[str], start: datetime, end: datetime) -> ResearchResult:
        cfg = self.config
        # Refused before anything is journaled: these are inputs that cannot describe a
        # session at all, and a session_start for one would be a claim nobody measured.
        if cfg.holdout_days < 1:
            raise RunRefused(
                "holdout_days must be at least 1: the holdout is what the shortlist is scored on "
                "once the search is over, and an empty one scores nothing."
            )
        if cfg.max_trials < 1 or cfg.max_dry_rounds < 1:
            raise RunRefused("max_trials and max_dry_rounds must each be at least 1.")
        # Reserve the sacred holdout up front; the search never sees it (guardrail).
        research_end = shift_days(end, -cfg.holdout_days)
        holdout_window = {"start": research_end.isoformat(), "end": end.isoformat()}
        if research_end <= start:
            raise RunRefused(
                f"A {cfg.holdout_days}-day holdout leaves no research window between "
                f"{start:%Y-%m-%d} and {end:%Y-%m-%d} (holdout_days leaves no research window); "
                "widen the date range or shorten the holdout."
            )

        prior_looks = prior_holdout_looks(self.journal_path, holdout_window, symbols)
        cost_identity = cfg.cost_identity()
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
                "cost": cost_identity,
                "prior_holdout_looks": prior_looks,
            },
        )

        shortlist: List[Candidate] = []
        saved: List[str] = []
        state = {"rounds": 0, "evaluated": 0, "rejected": 0, "trials": 0}
        stopped = "max_trials"
        try:
            stopped = self._loop(symbols, start, research_end, shortlist, state)
            # Final exam: score the shortlist ONCE on the sacred holdout (guardrail).
            saved = self._finalize(shortlist, symbols, research_end, end, state["trials"])
        except Exception as exc:
            # A session that was started is a session that ends: without this line the
            # journal shows rounds that count toward the campaign and no sign of why
            # they stopped.
            category = getattr(exc, "category", None)
            reason = f"provider_error:{category}" if category else f"error:{type(exc).__name__}"
            self._journal(
                "session_end",
                {
                    "session_id": self.session_id,
                    "stopped_reason": reason,
                    "rounds": state["rounds"],
                    "n_evaluated": state["evaluated"],
                    "n_rejected": state["rejected"],
                    "n_trials_total": state["trials"],
                    "shortlist": [],
                    "saved_configs": [],
                    "holdout_scored": False,
                },
            )
            self.progress = dict(state)
            raise

        self._journal(
            "session_end",
            {
                "session_id": self.session_id,
                "stopped_reason": stopped,
                "rounds": state["rounds"],
                "n_evaluated": state["evaluated"],
                "n_rejected": state["rejected"],
                "n_trials_total": state["trials"],
                "shortlist": [c.id for c in shortlist],
                "saved_configs": saved,
                "holdout_scored": bool(shortlist),
            },
        )
        return ResearchResult(
            shortlist=shortlist,
            n_trials_total=state["trials"],
            rounds=state["rounds"],
            stopped_reason=stopped,
            journal_path=self.journal_path,
            holdout_window=holdout_window,
            saved_configs=saved,
            n_evaluated=state["evaluated"],
            n_rejected=state["rejected"],
            holdout_scored=bool(shortlist),
            prior_holdout_looks=prior_looks,
        )

    def _loop(self, symbols, start, research_end, shortlist: List[Candidate], state: Dict[str, int]) -> str:
        """The propose -> evaluate -> keep loop; returns why it stopped."""
        cfg = self.config
        param_ranges = resolve_strategy_class(self.strategy_name).PARAM_RANGES
        incumbent: Optional[Candidate] = None
        history: List[Dict[str, Any]] = []
        tokens_used = 0
        dry = 0
        stopped = "max_trials"

        # The budget counts evaluated trials - rounds that reached a walk-forward and were
        # journaled as one - not proposals. A rejected proposal costs no walk-forward and
        # adds nothing to the multiple-testing count, so it does not spend a trial; the
        # dry-round stop is what bounds a proposer that keeps being rejected.
        while state["evaluated"] < cfg.max_trials:
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
                round_index=state["rounds"],
            )
            proposal = self.proposer.propose(context)
            if proposal is None:
                stopped = "proposer_exhausted"
                break
            state["rounds"] += 1
            rounds = state["rounds"]
            self._round = rounds
            tokens_used += proposal.tokens_used

            n_trials_offset = state["trials"]
            verdict = self._evaluate(proposal, symbols, start, research_end, n_trials_offset)
            if verdict is None:
                dry += 1
                state["rejected"] += 1
                history.append({"round": rounds, "rejected": True, "reason": self._last_reject})
                continue

            state["evaluated"] += 1
            result, cls, full_params = verdict
            state["trials"] = result.n_trials_total
            n_trials_cumulative = state["trials"]
            # In-sample median is recorded for contrast only - selection is OOS-only
            # (guardrail 1). It is what makes the IS -> OOS collapse legible.
            is_sharpe = float(
                median([fr.is_metrics.get("sharpe_ratio", 0.0) for fr in result.folds] or [0.0])
            )
            # Median across folds: what the promotion gate applies, so what a round is
            # compared and ranked on. The stitched aggregate stays in `oos_aggregate`.
            oos_sharpe = result.median_oos("sharpe_ratio")
            oos_dd = result.oos_aggregate.get("max_drawdown", 0.0)
            gate_report = result.gate_report(cfg.gates)
            promotable = gate_report["promotable"]
            advanced = promotable and self._beats_incumbent(oos_sharpe, oos_dd, incumbent)

            # Generated code is a different mechanism from the base strategy it was
            # proposed beside, and recording it under the base's name made
            # `--strategy <base>` mix the two (and fold their trials into one family).
            # The draft walk-forward already records `draft:<Class>:<hash>`; this matches it.
            row_strategy = self._row_strategy(proposal, cls)
            # The cost assumptions are part of what was measured, so part of the identity
            # a repeat is looked up under - a gross round must not answer a net request.
            identity_params = {**full_params, "_cost": cfg.cost_identity()}
            trial_payload = {
                "session_id": self.session_id,
                # Written into the journal line, which the store's rebuild reads: without
                # it the rebuilt row falls back to the session's strategy and the store
                # and its journal disagree about which strategy a round ran.
                "strategy": row_strategy,
                "round": rounds,
                "kind": proposal.kind,
                "hypothesis": proposal.hypothesis,
                "params": identity_params,
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
            }
            if proposal.kind == "code":
                # The only durable copy of the mechanism: the strategy identity above is
                # a hash, and a hash cannot be reviewed or added to a strategy pack.
                trial_payload["code"] = proposal.code
                trial_payload["code_hash"] = draft_code_hash(proposal.code)
            # Persist this round's own dated OOS return series in the
            # journal too (not just the trial store) - rebuild() must be able to
            # restore it from the journal alone, the sole source of truth.
            if result.oos_returns is not None and len(result.oos_returns):
                trial_payload["oos_returns"] = {
                    "dates": [d.isoformat() for d in result.oos_returns.index],
                    "values": [float(v) for v in result.oos_returns.to_numpy()],
                }
            trial_run_id = self._journal("trial", trial_payload)
            self._record_trial_row(
                trial_run_id,
                symbols,
                start,
                research_end,
                identity_params,
                result,
                # Same resolution _evaluate's dedup check used - a proposal naming
                # a different registered strategy must be recorded (and later
                # deduped) under that strategy, not the session's default.
                strategy=row_strategy,
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                oos_dd=oos_dd,
                promotable=promotable,
                n_trials_offset=n_trials_offset,
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
                    strategy=row_strategy,
                    params=full_params,
                    oos_metrics=result.oos_aggregate,
                    gate_report=gate_report,
                    n_trials_at_selection=n_trials_cumulative,
                    oos_sharpe=oos_sharpe,
                    code=proposal.code if proposal.kind == "code" else None,
                    strategy_cls=cls,
                    oos_returns=result.oos_returns,
                )
                shortlist.append(candidate)
                shortlist.sort(key=lambda c: c.oos_sharpe, reverse=True)
                del shortlist[cfg.shortlist_size :]
                incumbent = shortlist[0]
                dry = 0
            else:
                dry += 1
        return stopped

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def _reject(self, reason: str, proposal: Proposal, **extra: Any) -> None:
        """Journal a rejected round, with the round it was and the reason it was refused."""
        self._last_reject = reason
        payload: Dict[str, Any] = {
            "session_id": self.session_id,
            "round": self._round,
            "reason": reason,
            "hypothesis": proposal.hypothesis,
            "kind": proposal.kind,
        }
        if proposal.kind == "code" and isinstance(proposal.code, str) and proposal.code:
            payload["code_hash"] = draft_code_hash(proposal.code)
        payload.update(extra)
        self._journal("reject", payload)

    def _evaluate(self, proposal: Proposal, symbols, start, research_end, n_trials_offset):
        """Hygiene-gate and validate one proposal OOS; return (result, cls, params) or None."""
        cfg = self.config
        if proposal.malformed:
            self._reject(proposal.malformed, proposal)
            return None
        if proposal.kind == "code":
            if not cfg.allow_code_gen:
                self._reject("code-gen disabled", proposal)
                return None
            ok, reason = validate_hygiene(proposal)
            if not ok:
                self._reject(reason, proposal)
                return None
            try:
                cls: Type[Strategy] = load_strategy_from_code(proposal.code)
            except Exception as exc:  # noqa: BLE001 - rejection, not crash
                self._reject(f"sandbox: {exc}", proposal)
                return None
            ok, reason = validate_hygiene(proposal, cls)
        else:
            try:
                cls = resolve_strategy_class(proposal.strategy or self.strategy_name)
            except ValueError as exc:
                self._reject(str(exc), proposal)
                return None
            ok, reason = validate_hygiene(proposal, cls)

        if not ok:
            self._reject(reason, proposal, params=(proposal.params))
            return None

        # Dedup: a repeat of an exact prior config for this strategy/universe/window/cost
        # is the same lottery ticket checked twice - reject it before it burns a
        # walk-forward. A generated strategy is identified by its code hash, so a repeat
        # is the same source over the same window (its params are the *result* of a
        # search and cannot be compared before it runs).
        if self.trial_store is not None:
            cost = {"_cost": cfg.cost_identity()}
            if proposal.kind == "code":
                repeat = self.trial_store.seen_strategy(
                    strategy=self._row_strategy(proposal, cls),
                    symbols=symbols,
                    window_start=start,
                    window_end=research_end,
                    accounting=ACCOUNTING_VERSION,
                    params_include=cost,
                )
                full_params: Dict[str, Any] = {}
            else:
                full_params = self._full_params(cls, proposal.params)
                repeat = self.trial_store.seen(
                    strategy=proposal.strategy or self.strategy_name,
                    params={**full_params, **cost},
                    symbols=symbols,
                    window_start=start,
                    window_end=research_end,
                    accounting=ACCOUNTING_VERSION,
                )
            if repeat:
                self._reject(
                    "duplicate: identical config already tried this campaign",
                    proposal,
                    params=(full_params) if full_params else None,
                )
                return None

        try:
            return self._validate(proposal, cls, symbols, start, research_end, n_trials_offset)
        except BacktestError as exc:
            # Unrunnable is not the same as unprofitable. Journal it as its own
            # rejection so a broken proposal can never be scored as "no edge".
            self._reject(f"unrunnable: {exc}", proposal)
            return None

    def _validate(self, proposal: Proposal, cls, symbols, start, research_end, n_trials_offset):
        """Run the walk-forward validation for an already hygiene-cleared proposal."""
        cfg = self.config
        validator = WalkForwardValidator(
            cls, self.data_client, cfg.capital, self.seed, cfg.gates, cfg.cost_model
        )
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
            self._reject(
                "no folds evaluable (no fold produced a valid in-sample config or out-of-sample result)",
                proposal,
                params=(full_params),
            )
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
                self.config.cost_model,
            )
            candidate.holdout_metrics = validator.score_window(
                symbols,
                holdout_start,
                holdout_end,
                candidate.params,
                embargo_days=self.config.embargo_days,
                n_trials=n_trials_total,
            )
            cost_identity = self.config.cost_identity()
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
            # Generated source never executes when a config is loaded. The config names a
            # draft identity no registry resolves; this is what a human reviews before
            # adding the class to a strategy pack.
            generated = (
                {
                    "code": candidate.code,
                    "code_hash": draft_code_hash(candidate.code),
                    "class_name": cls.__name__,
                }
                if candidate.code is not None
                else None
            )
            path = config_store.save_config(
                f"research_{self.session_id}_{candidate.id}.json",
                strategy=candidate.strategy,
                params=candidate.params,
                symbols=symbols,
                # The agent searches parameters and never the book, so what it validated
                # is the class's declared limits - stated rather than inherited, because
                # a config that omits them is not unspecified, it silently resolves to
                # the same defaults at load and nothing records that anyone chose them.
                position_limits=(candidate.strategy_cls or resolve_strategy_class(candidate.strategy))(
                    dict(candidate.params)
                ).position_limits(),
                cost=cost_identity,
                provenance=provenance,
                generated_code=generated,
            )
            candidate.saved_path = str(path)
            saved.append(str(path))
            if self.config.bootstrap_skill:
                candidate.bootstrap_skill = self._bootstrap_skill_for(candidate, symbols, n_trials_total)
            self._journal(
                "holdout_score",
                {
                    "session_id": self.session_id,
                    "candidate": candidate.id,
                    "holdout_metrics": candidate.holdout_metrics,
                    "saved": str(path),
                    "bootstrap_skill": candidate.bootstrap_skill,
                },
            )
        return saved

    def _bootstrap_skill_for(
        self, candidate: Candidate, symbols, n_trials_total: int
    ) -> Optional[Dict[str, Any]]:
        """Advisory-only: own p from this candidate's OOS return
        series, next to the family p from every trial this session (and prior
        sessions) has recorded for its strategy/universe. Never influences
        keep/discard - a best-effort annotation on the final shortlist only."""
        try:
            from quantred.services.analysis import compute_bootstrap_skill

            return compute_bootstrap_skill(
                candidate.oos_returns,
                candidate.strategy,
                symbols,
                n_trials_total,
                candidate.oos_metrics,
                journal_path=self.journal_path,
            )
        except Exception:  # noqa: BLE001 - advisory only, must never break finalize
            logger.warning("Bootstrap-skill annotation failed for a shortlisted candidate", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _full_params(self, cls: Type[Strategy], overrides: Dict[str, Any]) -> Dict[str, Any]:
        params = {name: spec["default"] for name, spec in cls.PARAM_RANGES.items() if "default" in spec}
        params.update(overrides)
        for name, value in params.items():
            # An integral float for an int parameter (10.0) is the same number; keep the
            # declared type so it is recorded, deduped and saved as 10, not 10.0.
            spec = cls.PARAM_RANGES.get(name) or {}
            if spec.get("type") == "int" and isinstance(value, float) and value.is_integer():
                params[name] = int(value)
        return params

    def _beats_incumbent(self, oos_sharpe: float, oos_dd: float, incumbent: Optional[Candidate]) -> bool:
        if incumbent is None:
            return True
        # Both sides are the median across folds - the gate's own measure. The incumbent's
        # `oos_metrics` is the stitched aggregate, which is a different number.
        incumbent_sharpe = incumbent.oos_sharpe if incumbent.oos_sharpe is not None else float("-inf")
        better_sharpe = oos_sharpe > incumbent_sharpe
        incumbent_dd = incumbent.oos_metrics.get("max_drawdown", float("inf"))
        dd_ok = incumbent_dd == 0.0 or oos_dd <= incumbent_dd * (1 + self.config.drawdown_guard_tolerance)
        return better_sharpe and dd_ok

    @staticmethod
    def _incumbent_summary(incumbent: Optional[Candidate]) -> Optional[Dict[str, Any]]:
        if incumbent is None:
            return None
        return {"params": incumbent.params, "oos_sharpe": incumbent.oos_sharpe}

    @staticmethod
    def _lineage(proposal: Proposal, incumbent: Optional[Candidate]) -> str:
        if proposal.parent_id:
            return f"{proposal.parent_id}->v"
        return incumbent.lineage + ".m" if incumbent else "root"

    def _journal(self, event: str, payload: Dict[str, Any]) -> str:
        run_id = audit_log(f"research:{event}", payload, path=self.journal_path)
        if self.observer is not None:
            # An observer must never be able to break the research loop.
            try:
                self.observer(event, payload)
            except Exception:  # noqa: BLE001 - narration is not load-bearing
                logger.debug("research observer raised; continuing", exc_info=True)
        return run_id

    def _row_strategy(self, proposal: Proposal, cls) -> str:
        """The strategy identity a round is recorded under."""
        if proposal.kind == "code":
            return f"draft:{cls.__name__}:{draft_code_hash(proposal.code)}"
        return proposal.strategy or self.strategy_name

    def _record_trial_row(
        self,
        run_id: str,
        symbols,
        start,
        research_end,
        full_params: Dict[str, Any],
        result,
        *,
        strategy: str,
        is_sharpe: float,
        oos_sharpe: float,
        oos_dd: float,
        promotable: bool,
        n_trials_offset: int,
    ) -> None:
        """Dual-write this round's trial into the trial store, keyed
        on the same ``run_id`` the journal line just got - so a later rebuild
        replaying that line is a no-op, not a duplicate.

        Best-effort: the store is derived, never authoritative, so a write
        failure here must never break the research loop.
        """
        try:
            self.trial_store.record(
                id=run_id,
                kind="research",
                strategy=strategy,
                symbols=symbols,
                params=full_params,
                accounting=ACCOUNTING_VERSION,
                session_id=self.session_id,
                window_start=start,
                window_end=research_end,
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                oos_profit_factor=result.oos_aggregate.get("profit_factor"),
                oos_max_dd=oos_dd,
                deflated_sharpe=result.oos_aggregate.get("deflated_sharpe_ratio"),
                efficiency=result.median_efficiency(),
                oos_trades=result.oos_aggregate.get("total_trades"),
                promotable=promotable,
                n_trials_in_session=result.n_trials_total - n_trials_offset,
                git_sha=config_store.current_git_sha(),
                metrics_full=result.oos_aggregate,
            )
            if result.oos_returns is not None and len(result.oos_returns):
                self.trial_store.record_returns(
                    run_id,
                    [d.isoformat() for d in result.oos_returns.index],
                    [float(v) for v in result.oos_returns.to_numpy()],
                )
        except Exception:  # noqa: BLE001 - the journal append above already succeeded
            logger.warning("Trial store dual-write failed for a research trial", exc_info=True)
