---
paths:
  - "specs/**/*.md"
---

# Working with specs

`specs/` is gitignored and local-only — it never passes through CI, so the folder
itself is the only signal and has to be trustworthy.

- **A spec moves to `complete/` only when its §Testing items exist and pass.** Not
  when the design feels finished.
- **Moving one means four things**: update its `Status:` header, update all three
  indexes (`specs/README.md`, `complete/README.md`, `planning/README.md`), and run
  `make check-links` — a move breaks every sibling-relative link in both directions.
- **Record deviations under *Implementation notes*.** Anything that shipped
  differently from the design, and anything deferred, so the next reader learns it
  from the spec rather than from the code.
- **A spec without its data or use case is fiction.** Trigger-gated items stay
  one-liners in the backlog until the trigger fires.
- **If a spec only partly landed, split it** rather than moving it whole.
