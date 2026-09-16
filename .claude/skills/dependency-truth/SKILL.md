---
name: dependency-truth
description: Read a third-party library's installed source instead of recalling its behaviour, and check the resolved environment rather than the declaration that describes it. Use whenever a correctness or security decision turns on how alpaca-py, pandas, numpy, scikit-learn, ortools, mcp or anthropic actually behaves, when a lockfile or config claims something you have not seen run, and before asserting what a version ships.
---

# Read the dependency, don't recall it

When behavior depends on how a third-party library actually works, go and read it. A
remembered API compiles, passes, and reports a confident wrong answer — which is the
expensive kind.

This matters most for:

- `alpaca-py` order and data-client behavior, since it sits on the trade clock's order
  path.
- pandas/numpy edge cases — NaN propagation, index alignment, dtype coercion — in
  indicators and metrics.
- The `scikit-learn` GP surrogate in the Bayesian optimizer, `ortools` solver semantics,
  and the `mcp` / `anthropic` / `openai` client contracts.

## How

```bash
uv pip show <pkg>                                      # resolved version + location
python -c "import x, inspect; print(inspect.getsource(x.fn))"
rg "<symbol>" .venv/lib/python*/site-packages/<pkg>    # read the installed source
```

If a correctness or security decision hinges on what you find, cite the dependency
file, symbol and **version** in the PR.

## The environment, not the declaration

The same habit generalises past libraries, and this is where it has actually paid:

- **Run the thing.** Installing the published package into a clean venv
  (`make release-check`) found four onboarding dead ends that passed every test.
- **A lockfile agreeing with itself proves nothing.** An MCP dependency every test and
  lockfile agreed was fine shipped broken. Assert the version that is *installed*.
- **Confirm the environment resolves** — that an action tag exists before pushing it,
  that a URL serves before linking it, that a file the instructions name is actually in
  the distribution.

Then **say what you verified and what you did not**. A gate passing is not evidence that
the artifact you intended exists; only looking at it is.
