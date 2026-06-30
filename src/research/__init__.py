"""Autonomous research agent (opt-in behind the ``ai`` extra).

A bounded, offline research loop: propose a hypothesis -> express it as a config
(or, behind a flag, as new strategy code) -> validate it out-of-sample with
walk-forward -> keep/discard -> repeat until budget or dryness -> score the
survivors once on the sacred holdout -> hand a human a shortlist.

The agent reuses the *same* :mod:`src.services` core as the MCP server, so there
is one code path. It never touches live trading, never toggles ``PAPER_TRADE``,
and is scored exclusively on out-of-sample output (see :mod:`src.research.agent`
for the non-negotiable guardrails). The proposer is provider-agnostic - Claude,
OpenAI, or a local Ollama model (see :mod:`src.research.llm`).
"""

from src.research.agent import Candidate, ResearchAgent, ResearchConfig, ResearchResult
from src.research.llm import build_llm_client
from src.research.proposer import FixedProposer, LLMProposer, Proposal, Proposer, build_proposer

__all__ = [
    "ResearchAgent",
    "ResearchConfig",
    "ResearchResult",
    "Candidate",
    "Proposer",
    "Proposal",
    "FixedProposer",
    "LLMProposer",
    "build_proposer",
    "build_llm_client",
]
