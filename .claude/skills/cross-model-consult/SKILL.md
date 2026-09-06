---
name: cross-model-consult
description: Ask Codex or Gemini for an independent second opinion on one focused question, using Model Peer. Use before committing to an architecture, schema, or migration decision; on security-sensitive work such as authn/authz, sandboxing, input handling, secrets, or crypto; when a bug has outlived two of your own hypotheses; when you are inferring the behaviour of unfamiliar code or a dependency; and when two approaches are tied and you want the tradeoff.
---

<!-- Managed by `model-peer init`. Version 0.8.1. Re-run `model-peer update` to refresh; local edits are replaced. -->

# Consult a peer

You are Claude. **Codex** and **Gemini** are available as
independent, read-only engineering peers.

```bash
model-peer ask <codex|gemini> "<focused question>"
model-peer doctor                      # which peers are available here
```

## When it is worth the wait

- before committing to an architecture, schema, or migration decision
- when a bug has outlived two of your own hypotheses
- security-sensitive work: authn/authz, sandboxing, input handling, secrets, crypto
- unfamiliar code, or a dependency whose behavior you are inferring
- an assumption you cannot cheaply verify by reading the code
- two approaches you cannot decide between — ask for the tradeoff, not the verdict

Do not consult for anything you can settle by reading the code. Every consultation
is a real model call against the developer's account and takes tens of seconds.

## How to ask

The peer starts in this working directory with read-only tools, so **name files and
symbols instead of pasting excerpts**, and ask one focused question. A peer that has
to guess at scope returns generic advice.

```bash
# good — scoped to a symbol, answerable from the repository
model-peer ask codex "In src/auth/session.ts, can refresh_token leave the old token valid if rotation fails midway?"

# bad — no scope
model-peer ask codex "review my auth code"
```

Pipe context in when the question is about something not on disk:

```bash
git diff main... | model-peer ask gemini "What breaks in production?"
```

## The answer is advisory

Evaluate it before acting. Peers do not know this project's invariants, so advice
that contradicts the rules in this repository is wrong here however sound it
sounds in general.

When a peer materially changed your decision, say which model you asked and
whether you took the advice. Never present a peer's output as your own conclusion.

Leave `--depth` at its default: each peer answers alone, and lengthening the chain
is a human's deliberate call. Above depth 1 the peer does not run anything — it
asks Model Peer for a second opinion and Model Peer performs the consultation —
but every hop still costs time and money. A model is never consulted twice in one
chain.

If you are reading this **while acting as a peer** in someone else's
consultation, these instructions do not apply to you. Answer the question you
were asked. You cannot run Model Peer, and do not need to: if another model's
opinion would change your answer, request it in your reply exactly as the prompt
you were given describes.
