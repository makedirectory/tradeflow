---
paths:
  - "tradeflow/cli.py"
  - "tradeflow/mcp/**/*.py"
  - "tradeflow/services/**/*.py"
  - "tradeflow/settings.py"
---

# Surfaces: CLI, MCP, and the installed package

- **Logic lives in `services/`.** The CLI and the MCP server are transports: parse,
  call one service function, render. If a surface needs logic, move the logic first.
- **Messages differ by how the software was reached.** An installed copy has no
  repository, so any instruction assuming one — `.env.example`, a `make` target,
  `uv sync` — is a dead end, and worse than no instruction because it sends the
  reader looking for a file that was never there. This has been fixed in four
  separate places; check for a fifth before adding a message.
- **State resolves through `settings.state_root()`**, never a relative path. The
  multiple-testing correction rests on one journal; a campaign split across two roots
  deflates against half its evidence and nothing errors.
- **The MCP read-only wall is structural.** The server builds only a data client, so
  trading is a capability it does not have rather than a rule it was told. No new
  tool may make a trading path reachable.
- **Anything forming a run's identity has exactly one definition.** The dedup key
  decides whether a trial is found again, how many trials a family has deflated
  against, and whether the CLI and MCP see the same history. Two surfaces computing it
  separately is how a trial recorded over one stopped being found by the other. Add to
  the shared helper; never fold an assumption into one caller.
- **MCP descriptions are an interface, not documentation.** An agent cannot notice a
  stale description; it acts on one, and every action costs a journaled trial. State
  journaling, memoization, and evidence-gated defaults; pull metric definitions from
  the glossary rather than restating them.

The places this codebase keeps the same idea twice are listed in
[parity points](parity-points.md). Read it before changing anything on it.
