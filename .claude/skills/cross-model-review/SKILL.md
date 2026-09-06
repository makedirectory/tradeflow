---
name: cross-model-review
description: Run an independent cross-model review of the current Git diff with Model Peer. Every installed model reviews the same changes without seeing the others' conclusions, then a synthesizer reconciles them. Use before opening a pull request, after any change to security-sensitive code, and when you want more than your own read on a change you just wrote.
---

<!-- Managed by `model-peer init`. Version 0.8.1. Re-run `model-peer update` to refresh; local edits are replaced. -->

# Cross-model review

You are Claude. **Codex** and **Gemini** review alongside
you, independently.

```bash
model-peer review ["focus instructions"]
```

Every installed model receives the same Git status and patch — tracked changes and
new untracked files alike — and reviews without seeing the others' conclusions.
Only then does a synthesizer reconcile the findings. Reviewers are leaves by
default: none of them consults another model, which is what makes agreement
between them real signal rather than an echo.

Because they are independent, they also run at the same time.

Run it before opening a pull request, and again after any change to
security-sensitive code.

```bash
model-peer review "Focus on authorization and tenant isolation"
model-peer review --models codex,gemini   # exclude yourself
model-peer review --timeout 300   # bound each reviewer, not the whole panel
```

It takes minutes. Let it finish, and do not be alarmed by how it looks while it
runs:

- Progress from several models arrives on stderr at once, interleaved, in a
  different order each run. That is normal.
- **Nothing appears on stdout until every reviewer has finished.** The reviews are
  then printed together, in the order the models were requested. A long silence is
  the panel working, not a hang.

A reviewer that times out, fails, or returns nothing is dropped and named;
synthesis needs two survivors and says so when the panel was incomplete.

## What to do with the report

Report the synthesized findings as they are, grouped by severity. Then, for each
one, say whether you agree and why. Reviewers do not know this project's
invariants, so a finding that contradicts the rules in this repository is wrong
here however sound it sounds in general.

Do not apply fixes unless you are asked to.

If you are reading this **while acting as a peer** in someone else's
consultation, these instructions do not apply to you. Answer the question you
were asked, and do not start a review of your own.
