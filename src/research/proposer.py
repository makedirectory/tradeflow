"""Proposers: where the *creativity* enters the research loop.

A :class:`Proposer` turns a goal + the history so far into the next
:class:`Proposal` - either a parameter config to try ("tune") or new strategy
*code* to author ("code"). The agent loop and all guardrails are proposer-agnostic,
so the LLM-backed :class:`LLMProposer` and a deterministic test double share one
contract.

Each proposal must carry a one-paragraph **hypothesis** (the economic/behavioral
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

    hypothesis: str  # WHY this should work (required by hygiene)
    kind: str = "tune"  # "tune" (config) | "code" (new subclass)
    strategy: Optional[str] = None  # registered strategy to tune (kind == "tune")
    params: Dict[str, Any] = field(default_factory=dict)  # config overrides (kind == "tune")
    code: Optional[str] = None  # source of a new Strategy subclass (kind == "code")
    tuned_params: List[str] = field(default_factory=list)  # which params this varies (hygiene cap)
    parent_id: Optional[str] = None  # lineage: the candidate this mutates, if any
    tokens_used: int = 0  # reported LLM token cost, for the budget


@dataclass
class ProposalContext:
    """What the proposer sees each round (never includes holdout data)."""

    goal: str
    strategy: str
    param_ranges: Dict[str, Any]
    history: List[Dict[str, Any]]  # compact summary of prior trials + OOS scores
    incumbent: Optional[Dict[str, Any]]  # current best candidate, if any
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

    Asks the model for a JSON proposal: a hypothesis plus either param overrides
    within the supplied ``PARAM_RANGES`` (``kind="tune"``) or, when
    ``allow_code_gen`` is set, the source of a brand-new ``Strategy`` subclass
    (``kind="code"``). The model id is pinned on the client and logged by the
    agent's audit journal, so every proposal is replayable.

    Code proposals are *untrusted text* until :mod:`src.research.sandbox` has
    validated them - this class only transports them.
    """

    def __init__(self, client: LLMClient, max_tokens: int = 1024, *, allow_code_gen: bool = False):
        self.client = client
        self.max_tokens = max_tokens
        self.allow_code_gen = allow_code_gen

    def propose(self, context: ProposalContext) -> Optional[Proposal]:
        import json

        system = _CODEGEN_SYSTEM_PROMPT if self.allow_code_gen else _SYSTEM_PROMPT
        response = self.client.complete(system, _build_user_prompt(context), self.max_tokens)
        try:
            payload = json.loads(_extract_json(response.text))
        except (ValueError, KeyError):
            return None

        if payload.get("kind") == "code":
            return Proposal(
                hypothesis=payload.get("hypothesis", ""),
                kind="code",
                strategy=context.strategy,
                code=payload.get("code", ""),
                tokens_used=response.tokens,
            )

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
    allow_code_gen: bool = False,
    **kwargs,
) -> LLMProposer:
    """Build an :class:`LLMProposer` for a provider (anthropic | openai | ollama)."""
    llm = client or build_llm_client(provider, model, **kwargs)
    return LLMProposer(llm, max_tokens=max_tokens, allow_code_gen=allow_code_gen)


_SYSTEM_PROMPT = (
    "You are a quantitative research assistant proposing the next backtest experiment. "
    "Change at most 5 parameters. Give a one-paragraph hypothesis of WHY the change should "
    "improve OUT-OF-SAMPLE performance - in-sample gains do not count. "
    'Respond ONLY with JSON: {"hypothesis": "...", "params": {"name": value, ...}}.'
)

#: System prompt for sessions run with ``--allow-code-gen``. The model may either
#: tune the incumbent or author a new mechanism; the sandbox enforces the contract
#: described here, so a violation is a rejected proposal, not a broken run.
_CODEGEN_SYSTEM_PROMPT = (
    "You are a quantitative research assistant proposing the next backtest experiment. "
    "You may respond with EITHER of two proposal kinds.\n\n"
    '1. Tune the existing strategy: {"kind": "tune", "hypothesis": "...", '
    '"params": {"name": value, ...}} - at most 5 parameters, each within PARAM_RANGES.\n\n'
    '2. Author a NEW strategy mechanism: {"kind": "code", "hypothesis": "...", "code": "<python source>"}.\n\n'
    "Propose 'code' when the history suggests the parameter space is exhausted and a different "
    "mechanism is needed; propose 'tune' when the incumbent looks refinable.\n\n"
    "Generated code runs in a restricted sandbox and MUST satisfy this contract or it is rejected:\n"
    "- Define exactly one concrete subclass of `Strategy` (from src.strategies.base).\n"
    "- Imports are limited to: pandas, numpy, math, src.indicators.indicators, src.strategies.base. "
    "No os, sys, io, requests, or file access of any kind.\n"
    "- Give the class a docstring stating the hypothesis (required).\n"
    "- Declare a PARAM_RANGES ClassVar dict with AT MOST 5 SEARCHABLE params. A param is "
    "searchable when its spec has min, max, and step; a spec with only a default is fixed and "
    "does NOT count against the cap. Each spec needs type ('int'|'float'), default, description.\n"
    "- PARAM_RANGES MUST include risk_per_trade, stop_loss, and take_profit (fractional distances). "
    "The execution path reads these on every bar; omitting them is rejected. Pin them with a "
    "default only if you would rather spend the searchable budget on your mechanism.\n"
    "- Set TIMEFRAME (e.g. '1Day') and call super().__init__(config) after setting config['timeframe'].\n"
    "- Implement all four abstract hooks: initialize(), process_data(data) -> DataFrame, "
    "calculate_scores(data) -> Series, calculate_required_lookback() -> int.\n"
    "- calculate_scores returns ONE continuous conviction score per bar; the base class derives "
    "BUY/SELL/HOLD from its sign. Do not emit discrete signals yourself.\n"
    "- The class must construct from its own declared defaults.\n\n"
    "Give a one-paragraph hypothesis of WHY this should improve OUT-OF-SAMPLE performance - "
    "in-sample gains do not count. Respond ONLY with the JSON object."
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
    return text[start : end + 1]
