---
sidebar_position: 17
title: Research agent
---

# Research agent

`src/research/` closes the loop: an **offline, self-pacing** agent that, given a
goal, runs the hypothesis → validate → keep/discard cycle on its own and hands a
human a shortlist of vetted, provenance-stamped candidate configs. It reuses the
same `src/services/` core as the MCP server, so there is one code path.

The agent's genuine edge over the GP optimizer is *creativity* — hypotheses and
(behind a flag) new strategy code. The deterministic pipeline supplies the
skepticism: walk-forward scores out-of-sample, the promotion gates accept or
reject, and the sacred holdout is the final exam.

## The loop (`agent.py`)

```
goal ─▶ propose (Proposer) ─▶ hygiene gate ─▶ validate OUT-OF-SAMPLE (walk-forward)
     ─▶ keep iff "promotable" AND beats incumbent OOS (drawdown-guarded)
     ─▶ loop until trial budget / token budget / K dry rounds
     ─▶ score the shortlist ONCE on the sacred holdout ─▶ save configs for a human
```

## Non-negotiable guardrails (enforced in code, not by prompt)

1. **OOS-only fitness.** Selection uses the walk-forward OOS aggregate; in-sample
   metrics are recorded but never the criterion.
2. **Multiple-testing correction.** `n_trials` accumulates across the whole
   session (via `n_trials_offset`) and feeds the deflated Sharpe — the more the
   agent tries, the higher the bar each new config must clear. The count is
   **per session**: it resets when the process exits, so trials from earlier
   sessions do not raise the bar for this one. Read a deflated Sharpe as a lower
   bound on the deflation warranted — see
   [the scope limit](walk-forward.md#known-limit-n_trials-counts-a-run-not-a-campaign).
3. **Sacred holdout.** Reserved up front, never passed to any search, scored once
   at the end on the final shortlist.
4. **Budgets + dryness stop.** Hard caps on trials and tokens, plus a
   "K rounds with no OOS improvement" stop, so the loop can't chase noise forever.
5. **Human-in-the-loop.** Output is config files + a journal; nothing is promoted
   to live, no `PAPER_TRADE` toggle, no order capability reachable.
6. **Full audit.** Every proposal, trial, and decision is journaled to
   `logs/research_journal.jsonl` and is replayable.
7. **Sandbox + hygiene.** Generated code and configs pass `src/research/sandbox.py`
   before evaluation.

## Research hygiene (`sandbox.py`)

Cheap, load-bearing rules that stop the loop manufacturing overfit garbage at
scale:

- **Hypothesis before code.** Every proposal must carry a rationale; no rationale
  ⇒ rejected unevaluated.
- **≤ 5 tunable parameters** per generated strategy — more knobs = more overfit
  surface.
- **Contract validation** of agent-authored code: it must define a concrete
  `Strategy` subclass that implements the abstract hooks, declares a valid
  `PARAM_RANGES`, carries a docstring, and actually constructs. Generated code runs
  with a restricted global namespace (no `os`/`sys`/network imports, limited
  builtins) and is a *proposal artifact*, never auto-merged. Full OS-level process
  isolation is a documented future enhancement; `load_strategy_from_code` is the
  single choke point where it would be enforced.

## Provider-agnostic proposer (`proposer.py`, `llm.py`)

The loop and guardrails are proposer-agnostic. `LLMProposer` drives any
`LLMClient`:

- `AnthropicClient` — Claude via the `anthropic` SDK (`ai` extra).
- `OpenAIClient` — GPT via the `openai` SDK (`openai` extra).
- `OllamaClient` — any local model over the Ollama HTTP API (no dependency —
  standard library only).

`build_proposer(provider, model)` / `build_llm_client(provider, model)` pick the
backend. Credentials resolve through the shared settings chain — environment /
`.env` — via `src.settings.get_credential`
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`; `OLLAMA_BASE_URL` for the local server).
`FixedProposer` replays a fixed list for deterministic offline tests.

A "tune" proposal fixes specific parameter values and is validated as a
fixed-config walk-forward (`WalkForwardValidator.evaluate_config`), counting as one
trial toward the session's multiple-testing total. A "code" proposal (behind
`--allow-code-gen`) loads a new strategy class through the sandbox and runs a full
per-fold optimization over its `PARAM_RANGES`.
