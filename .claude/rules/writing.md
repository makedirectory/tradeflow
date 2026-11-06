---
paths:
  - "tradeflow/**/*.py"
  - "tests/**/*.py"
  - "docs/content/**/*.md"
  - "*.md"
---

# Writing rules for code and docs

- **Self-contained.** Nothing committed references `specs/` — no link into it, and no
  planning spec by number or section in code, docstrings, tests, docs or commit
  messages. `specs/` is gitignored and not in the repository, so a reader cannot follow
  the pointer. Link to the engineering wiki instead, or describe the behavior on its
  own terms. `make check-links` refuses a tracked link into `specs/`. Implementation
  notes and decision records for the work belong *in* `specs/`, which stays ignored.
- **No external-source citations.** Explain the math and the reasoning directly; do
  not name the book, paper, author, or equation number a technique came from. Cite
  the concept (`the IC-uncertainty level shrink`), not its provenance. Identifiers in
  output follow the same rule — use a descriptive name.
- **Comments explain *why*, not *what*.** The reason a threshold is deliberately
  loose is worth a comment; a restatement of the comparison is not.
- **Absent is not zero.** A metric never recorded is not a metric of zero. This
  governs sorting and filtering as much as rendering: unrecorded values sort last,
  and never satisfy a minimum-value filter.
