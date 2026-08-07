# Cross-model consultation

Codex CLI may be consulted as an independent engineering reviewer:

```
codex exec "<focused question>"
```

**Check it is installed first** (`command -v codex`). If it is not, say so plainly
rather than reporting a `command not found` as though it were a consultation. As of
the last check it was not installed on this machine.

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
or the honesty rules is wrong here regardless of how sound it looks in general.

Do not ask Codex to invoke Claude Code. This prevents recursive agent-to-agent loops.
