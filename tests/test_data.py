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


# ---- stage 0: download and read-only helpers
class _FakeResp:
    headers = {"Content-Length": "0"}

    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after, self.i = list(chunks), fail_after, 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        if self.fail_after is not None and self.i >= self.fail_after:
            raise ConnectionResetError("dropped")
        if self.i >= len(self.chunks):
            return b""
        b = self.chunks[self.i]
        self.i += 1
        return b


def test_download_dropped_connection_leaves_no_partial_file(tmp_path, monkeypatch):
    g = _mod("00_get_data")
    monkeypatch.setattr(g.urllib.request, "urlopen", lambda req: _FakeResp([b"aa", b"bb"], fail_after=1))
    dest = tmp_path / "rice-msc.zip"
    with pytest.raises(ConnectionResetError):
        g.download("rice-msc.zip", dest)
    assert list(tmp_path.iterdir()) == []


def test_download_success_publishes_complete_file(tmp_path, monkeypatch):
    g = _mod("00_get_data")
    monkeypatch.setattr(g.urllib.request, "urlopen", lambda req: _FakeResp([b"aa", b"bb"]))
    dest = tmp_path / "rice-msc.zip"
    g.download("rice-msc.zip", dest)
    assert dest.read_bytes() == b"aabb"
    assert [p.name for p in tmp_path.iterdir()] == ["rice-msc.zip"]


def test_make_read_only_handles_apostrophe_in_path(tmp_path):
    g = _mod("00_get_data")
    d = tmp_path / "o'brien"
    d.mkdir()
    (d / "f.txt").write_text("x")
    try:
        g.make_read_only(d)
        assert not os.access(d / "f.txt", os.W_OK) or os.geteuid() == 0
    finally:
        os.system(f"chmod -R u+w {str(tmp_path)!r}")


# ---- job 3: stage 2 must not load test labels (read from source; stage 2 is never imported)
def test_eda_never_reads_splits_csv():
    src = (ROOT / "src/02_eda.py").read_text()
    assert "SPLITS" not in src and "splits.csv" not in src
    assert "load_split" not in src
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]
    assert "pd.read_csv" not in calls or all("SPLITS" not in c for c in calls)


def test_class_balance_expression_on_train_val_frame():
    g = pd.DataFrame({"class": ["A", "A", "B", "B", "B"], "split": ["train", "val", "train", "train", "val"]})
    cnt = g.groupby(["class", "split"]).size().unstack()
    assert list(cnt.columns) == ["train", "val"]
    assert cnt.loc["B", "train"] == 2
    src = (ROOT / "src/02_eda.py").read_text()
    assert 'cnt = g.groupby(["class", "split"]).size().unstack()' in src
