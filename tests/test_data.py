"""Tests for stages 0 to 2 helpers and tools/leak_audit.py. No model is fitted; stage 2 is never imported."""
import ast
import importlib
import io
import os
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mod(name):
    return importlib.import_module(name)


def _tree(rel):
    return ast.parse((ROOT / rel).read_text())


# ---- job 1: jasmine_lots.csv first column is named
def test_jasmine_lots_index_is_named():
    hits = [n for n in ast.walk(_tree("src/02_eda.py"))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "to_csv"
            and n.args and ast.unparse(n.args[0]) == "out / 'jasmine_lots.csv'"]
    assert len(hits) == 1
    kw = {k.arg: ast.literal_eval(k.value) for k in hits[0].keywords}
    assert kw.get("index_label") == "feature"


def test_jasmine_lots_header_on_made_up_frame():
    cols = ["AREA", "tint"]
    e = pd.Series([1.0, 2.0], index=cols)
    l = pd.Series([2.0, 3.0], index=cols)
    buf = io.StringIO()
    pd.DataFrame({"early": e, "late": l, "ratio": l / e}).to_csv(buf, index_label="feature")
    assert buf.getvalue().splitlines()[0] == "feature,early,late,ratio"


# ---- job 2: manual text names the target that runs stage 0
def test_manual_text_names_get_data_target():
    m = _mod("00_get_data").MANUAL
    assert "make get-data" in m
    assert "make data" not in m
