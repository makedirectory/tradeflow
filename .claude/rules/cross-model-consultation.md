# Cross-model consultation

Codex CLI is available as an independent engineering reviewer. Consult it read-only:

```
codex exec --sandbox read-only "<focused question>" </dev/null
```

Both arguments matter, and neither is the default:

- **`--sandbox read-only`.** `codex exec` otherwise runs `workspace-write` with
  `approval: never`, meaning a consultation can modify files in this repository
  without asking. Asking for an opinion should not be able to change the working
  tree.
- **`</dev/null`.** Without it the CLI waits on stdin before answering.

Good reasons to consult it:

- architecture decisions
- difficult debugging
- security-sensitive changes
- unfamiliar code
- reviewing a proposed implementation
- checking assumptions
- comparing multiple approaches

Codex is **advisory**. Evaluate its response independently before acting: it has no
context on this project's invariants, so an answer that violates the two-clocks rule
or the honesty rules is wrong here regardless of how sound it looks in general. Say
what it recommended and whether you took the advice, rather than presenting its
output as a conclusion.

Do not ask Codex to invoke Claude Code. This prevents recursive agent-to-agent loops.
