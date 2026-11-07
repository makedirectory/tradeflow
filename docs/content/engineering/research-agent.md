---
sidebar_position: 17
title: Research agent
---

# Research agent

`tradeflow/research/` closes the loop: an **offline, self-pacing** agent that, given a
goal, runs the hypothesis → validate → keep/discard cycle on its own and hands a
human a shortlist of vetted, provenance-stamped candidate configs. It reuses the
same `tradeflow/services/` core as the MCP server, so there is one code path.

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
   sessions do not raise the bar for this one at gate time. Every round is also
   dual-written into the [trial store](walk-forward.md#the-trial-store), so the
   campaign-wide count is queryable by hand (`trials list`) even though it
   isn't wired into the gate itself — see
   [the open item](walk-forward.md#n_trials-still-counts-a-run-at-gate-time-not-a-campaign).
   Read a deflated Sharpe as a lower bound on the deflation warranted.
3. **Sacred holdout.** Reserved up front, never passed to any search, scored once
   at the end on the final shortlist.
4. **Budgets + dryness stop.** Hard caps on trials and tokens, plus a
   "K rounds with no OOS improvement" stop, so the loop can't chase noise forever.
5. **Human-in-the-loop.** Output is config files + a journal; nothing is promoted
   to live, no `PAPER_TRADE` toggle, no order capability reachable.
6. **Full audit.** Every proposal, trial, and decision is journaled to
   `logs/research_journal.jsonl` and is replayable.
7. **Sandbox + hygiene.** Generated code and configs pass `tradeflow/research/sandbox.py`
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
`.env` — via `tradeflow.settings.get_credential`
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`; `OLLAMA_BASE_URL` for the local server).
`FixedProposer` replays a fixed list for deterministic offline tests.

### The `LLMClient` contract and provider failures

`LLMClient.complete` is concrete; a backend implements `_complete` (the bare provider
call: one request, no error translation, no retry). Overriding `complete` instead no
longer instantiates, so every provider gets the same failure policy.

- **`ModelProviderError`** (`research/provider_errors.py`) is what the loop sees. It
  carries `provider`, a `category` and a printable `detail` and **never the vendor's own
  message**, which can echo request parameters and so the key. Categories: `auth`,
  `quota` (an empty balance), `rate_limited`, `unavailable` (5xx, Anthropic's 529
  `overloaded_error`), `entitlement`, `unreachable`, `timeout`, `model_not_found`,
  `unknown`. `cmd_research` turns it into one line and a non-zero exit; the journal is
  left as it was.
- **Classification** reads structured fields first (`code` / `type` from the SDK error
  and its body: `insufficient_quota` against `rate_limit_exceeded`, Anthropic's
  `billing_error`, `rate_limit_error`, `overloaded_error`), then the HTTP status, and
  consults message wording only for a 400/429 that nothing structured explained
  (Anthropic reports an empty balance as a plain 400). It recognises errors by shape and
  imports no SDK. It is checked against the real `anthropic` and `openai` error classes,
  not just fakes.
- **Only vendor and network exceptions are translated** (`is_provider_exception`: an
  HTTP status on the exception, an SDK `API*Error`, `urllib`/socket/`http.client`
  failures). A `KeyError` reading a reply, or any other bug here, propagates with its
  traceback instead of being reported as the provider's fault. A missing SDK
  (`ImportError`) also passes through.
- **Retry policy.** The SDK clients are built with `max_retries=0` and the wrapper owns
  the policy: up to `max_retries` (default 2) retries, after 1s then 3s (or the
  provider's `Retry-After`, unless it asks for more than 30s, in which case it stops),
  for `rate_limited`, `unavailable` and `unreachable` only. An empty balance, a bad key,
  a forbidden or missing model are never re-sent, and neither is a `timeout` (waiting the
  same long time again is not a fix). The readiness probe sets `max_retries = 0`: it is
  one request, always.
- **Ollama settings.** `OLLAMA_BASE_URL` (a full `.../api/chat` URL is trimmed),
  `OLLAMA_TIMEOUT` (seconds for one response, default 120; a slow GPU needs more) and
  `OLLAMA_NUM_CTX` (context length, sent only when set), all validated in `settings.py`
  and by the `init` wizard with the same parsers. A reply that is not JSON is reported as
  `unavailable` at that endpoint.

A "tune" proposal fixes specific parameter values and is validated as a
fixed-config walk-forward (`WalkForwardValidator.evaluate_config`), counting as one
trial toward the session's multiple-testing total. A "code" proposal (behind
`--allow-code-gen`) loads a new strategy class through the sandbox and runs a full
per-fold optimization over its `PARAM_RANGES`.
