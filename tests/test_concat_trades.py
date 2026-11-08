"""``_concat_trades`` keeps "no trade detail" and "no trades" apart.

The module defined it twice. The later definition (returning an empty frame) shadowed the
earlier (returning ``None``), so a result with no stored trade detail was
indistinguishable from one whose folds simply did not trade.
"""

import ast
import inspect

import pandas as pd

from quantred.optimization import walk_forward as wf


def test_the_module_defines_it_once():
    tree = ast.parse(inspect.getsource(wf))
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert names.count("_concat_trades") == 1


def test_no_detail_is_none_but_no_trades_is_an_empty_frame():
    assert wf._concat_trades([]) is None
    assert wf._concat_trades([None, None]) is None

    empty = wf._concat_trades([pd.DataFrame(), pd.DataFrame()])
    assert isinstance(empty, pd.DataFrame) and empty.empty


def test_pooled_frames_are_ordered_by_exit_time_and_skip_empty_and_missing():
    a = pd.DataFrame({"exit_time": pd.to_datetime(["2024-01-05", "2024-01-09"]), "pnl": [1.0, 2.0]})
    b = pd.DataFrame({"exit_time": pd.to_datetime(["2024-01-02"]), "pnl": [3.0]})
    out = wf._concat_trades([a, pd.DataFrame(), None, b])
    assert out["pnl"].tolist() == [3.0, 1.0, 2.0]


def test_aggregation_callers_get_a_frame_either_way():
    assert wf._pooled_trades([]).empty
    assert wf._pooled_trades([None]).empty
