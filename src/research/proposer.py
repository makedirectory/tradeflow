"""Proposers: where the *creativity* enters the research loop.

A :class:`Proposer` turns a goal + the history so far into the next
:class:`Proposal` - either a parameter config to try ("tune") or new strategy
*code* to author ("code"). The agent loop and all guardrails are proposer-agnostic,
so the LLM-backed :class:`LLMProposer` and a deterministic test double share one
contract.

Each proposal must carry a one-paragraph **hypothesis** (the economic/behavioural
rationale). No rationale -> the agent rejects it unevaluated (a research-hygiene
rule; see the research-agent guide in the engineering docs).

:class:`LLMProposer` is provider-agnostic - it drives any
:class:`~src.research.llm.LLMClient` (Claude, GPT, or a local Ollama model), so
the same loop runs against a hosted frontier model or one on your laptop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.research.llm import LLMClient, build_llm_client


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


class LLMProposer(Proposer):
    """Proposer backed by any :class:`~src.research.llm.LLMClient`.

    Asks the model for a JSON proposal: a hypothesis plus param overrides within
    the supplied ``PARAM_RANGES``. The model id is pinned on the client and logged
    by the agent's audit journal, so every proposal is replayable.
    """

    def __init__(self, client: LLMClient, max_tokens: int = 1024):
        self.client = client
        self.max_tokens = max_tokens

    def propose(self, context: ProposalContext) -> Optional[Proposal]:
        import json

        response = self.client.complete(_SYSTEM_PROMPT, _build_user_prompt(context), self.max_tokens)
        try:
            payload = json.loads(_extract_json(response.text))
        except (ValueError, KeyError):
            return None
        params = payload.get("params", {})
        return Proposal(
            hypothesis=payload.get("hypothesis", ""),
            kind="tune",
            strategy=context.strategy,
            params=params,
            tuned_params=list(params),
            tokens_used=response.tokens,
        )


def build_proposer(
    provider: str = "anthropic",
    model: Optional[str] = None,
    *,
    client: Optional[LLMClient] = None,
    max_tokens: int = 1024,
    **kwargs,
) -> LLMProposer:
    """Build an :class:`LLMProposer` for a provider (anthropic | openai | ollama)."""
    llm = client or build_llm_client(provider, model, **kwargs)
    return LLMProposer(llm, max_tokens=max_tokens)


_SYSTEM_PROMPT = (
    "You are a quantitative research assistant proposing the next backtest experiment. "
    "Change at most 5 parameters. Give a one-paragraph hypothesis of WHY the change should "
    "improve OUT-OF-SAMPLE performance - in-sample gains do not count. "
    'Respond ONLY with JSON: {"hypothesis": "...", "params": {"name": value, ...}}.'
)


def _build_user_prompt(context: ProposalContext) -> str:
    import json

    return (
        f"GOAL: {context.goal}\n"
        f"STRATEGY: {context.strategy}\n"
        f"TUNABLE PARAM_RANGES (stay within bounds):\n{json.dumps(context.param_ranges, default=str)}\n"
        f"HISTORY (prior out-of-sample results):\n{json.dumps(context.history, default=str)}\n"
    )


def _extract_json(text: str) -> str:
    """Pull the first {...} block out of an LLM response."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in response")
    return text[start:end + 1]
