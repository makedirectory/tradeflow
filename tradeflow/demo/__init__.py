"""What `tradeflow demo` runs on, and what a bare install falls back to.

Not examples to copy — that is ``example-signals``, which
``tradeflow init --example-pack`` hands you as a repository you own. These exist for
three narrower jobs:

* ``tradeflow demo`` needs a strategy to put through the pipeline, offline and with no
  keys, and it cannot depend on anything the user has not installed yet.
* ``--strategy`` and ``--scanner`` need a default that resolves on a bare install, or
  every command fails at lookup rather than saying something useful.
* There should be one smallest complete strategy in the tree to read.

They are registered through the same two entry-point groups a private pack uses, and
declared in this project's own ``pyproject.toml``. That is deliberate: the discovery
path the whole private-strategy feature rests on is then exercised by every install
rather than only by tests. The registry names them directly as well, because
enumeration order across distributions is undefined and these two names are reserved -
a pack cannot take them, and which class answers to them is not left to chance.

``demo_trend`` and ``demo_volume`` are named for what they are for. Neither is an edge,
and neither is where your work goes.
"""
