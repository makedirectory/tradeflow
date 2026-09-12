"""Root conftest: plugin registration only.

``pytest_plugins`` is only honoured in a *top-level* conftest. ``tests/conftest.py``
was accepted for it purely because the directory is named ``tests`` and so loads as an
initial conftest - pytest's own check (`_check_non_top_pytest_plugins`) is a hard
collection error, gated on whether configure has already run. Renaming the directory
would have turned that into a suite-wide failure, so the declaration lives here, where
it is unconditionally valid. Everything else about the suite stays in
``tests/conftest.py``.
"""

#: The leak guard is a module rather than an inline hook so that its own tests can load
#: the same one into a nested session, instead of restating the thing under test.
pytest_plugins = ["tests.state_leak_guard", "pytester"]
