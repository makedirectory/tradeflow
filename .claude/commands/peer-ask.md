---
description: Ask another vendor's model for an independent second opinion
argument-hint: <focused question>
allowed-tools: Bash(model-peer:*)
---

<!-- Managed by `model-peer init`. Version 0.8.1. Re-run `model-peer update` to refresh. -->

Ask an independent peer the question in `$ARGUMENTS` with
`model-peer ask <model> "<question>"`. Run `model-peer doctor` first if you are
unsure which peers are installed, and pick one that is not yourself.

The peer runs read-only in this working directory, so name files and symbols in
the question rather than pasting excerpts, and ask one focused thing.

If `$ARGUMENTS` is empty, ask me what I want a second opinion on rather than
guessing.

Then report the peer's answer, say which model you asked, and state plainly
whether you agree with it and why. Peers do not know this project's invariants;
project rules win. Do not act on the advice without telling me.
