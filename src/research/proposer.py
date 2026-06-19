"""Proposers: where the *creativity* enters the loop (Spec 004 §1).

A :class:`Proposer` turns a goal + the history so far into the next
:class:`Proposal` - either a parameter config to try ("tune") or new strategy
*code* to author ("code"). The agent loop and all guardrails are proposer-agnostic,
so the real :class:`AnthropicProposer` and a deterministic test double share one
contract.

Each proposal must carry a one-paragraph **hypothesis** (the economic/behavioural
rationale). No rationale -> the agent rejects it unevaluated (research hygiene,
Spec 004 §4.8).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Proposal:
    """One experiment the agent should try next."""

    hypothesis: str                       # WHY this should work (required by hygiene)
    kind: str = "tune"                    # "tune" (config) | "code" (new subclass)
    strategy: Optional[str] = None        # registered strategy to tune (kind == "tune")
    params: Dict[str, Any] = field(default_factory=dict)  # config overrides (kind == "tune")
    code: Optional[str] = None            # source of a new Strategy subclass (kind == "code")
    tuned_params: List[str] = field(default_factory=list)  # which params this varies (hygiene cap)
    parent_id: Optional[str] = None       # lineage: the candidate this mutates, if any
    tokens_used: int = 0                  # reported LLM token cost, for the budget


@dataclass
class ProposalContext:
    """What the proposer sees each round (never includes holdout data)."""

    goal: str
    strategy: str
    param_ranges: Dict[str, Any]
    history: List[Dict[str, Any]]         # compact summary of prior trials + OOS scores
    incumbent: Optional[Dict[str, Any]]   # current best candidate, if any
    round_index: int


class Proposer(ABC):
    """Turns the goal + history into the next proposal."""

    @abstractmethod
    def propose(self, context: ProposalContext) -> Optional[Proposal]:
        """Return the next proposal, or ``None`` to signal "nothing more to try"."""


class FixedProposer(Proposer):
    """Deterministic proposer that replays a fixed list - for tests and replays."""

    def __init__(self, proposals: List[Proposal]):
        self._proposals = list(proposals)

    def propose(self, context: ProposalContext) -> Optional[Proposal]:
        if context.round_index >= len(self._proposals):
            return None
        return self._proposals[context.round_index]


class AnthropicProposer(Proposer):
    """LLM-backed proposer using the Anthropic SDK (opt-in ``ai`` extra).

    Asks the model for a JSON proposal: a hypothesis plus param overrides within
    the supplied ``PARAM_RANGES``. Model ids are pinned and logged (guardrail §6);
    consult the ``claude-api`` skill for current ids / tool-use details before
    changing them.
    """

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1024,
                 client: Any = None):
        self.model = model
        self.max_tokens = max_tokens
        self._client = client  # injectable for testing; else built lazily

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed when actually proposing via the API
            self._client = anthropic.Anthropic()
        return self._client

    def propose(self, context: ProposalContext) -> Optional[Proposal]:
        import json

        client = self._ensure_client()
        prompt = self._build_prompt(context)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        usage = getattr(response, "usage", None)
        tokens = (getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)) if usage else 0
        try:
            payload = json.loads(_extract_json(text))
        except (ValueError, KeyError):
            return None
        return Proposal(
            hypothesis=payload.get("hypothesis", ""),
            kind="tune",
            strategy=context.strategy,
            params=payload.get("params", {}),
            tuned_params=list(payload.get("params", {})),
            tokens_used=tokens,
        )

    @staticmethod
    def _build_prompt(context: ProposalContext) -> str:
        import json

        return (
            "You are a quantitative research assistant proposing the next experiment.\n"
            f"GOAL: {context.goal}\n"
            f"STRATEGY: {context.strategy}\n"
            f"TUNABLE PARAM_RANGES (stay within bounds):\n{json.dumps(context.param_ranges, default=str)}\n"
            f"HISTORY (prior out-of-sample results):\n{json.dumps(context.history, default=str)}\n\n"
            "Rules: change at most 5 parameters; give a one-paragraph hypothesis of WHY it should "
            "improve OUT-OF-SAMPLE performance (in-sample gains don't count). "
            'Respond ONLY with JSON: {"hypothesis": "...", "params": {"name": value, ...}}.'
        )


def _extract_json(text: str) -> str:
    """Pull the first {...} block out of an LLM response."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in response")
    return text[start:end + 1]
