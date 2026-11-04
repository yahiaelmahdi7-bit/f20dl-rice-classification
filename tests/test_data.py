"""Tests for stages 0 to 2 helpers and tools/leak_audit.py. No model is fitted; stage 2 is never imported."""
import ast
import importlib
import io
import os
import pathlib

import numpy as np
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
def _code(rel):
    """Source after the prompt-record header, whose comments quote the rules."""
    return (ROOT / rel).read_text().split("PROMPT RECORD END ===", 1)[1]


def test_eda_never_reads_splits_csv():
    src = _code("src/02_eda.py")
    assert "SPLITS" not in src and "splits.csv" not in src
    assert "load_split" not in src
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]
    assert "pd.read_csv" not in calls or all("SPLITS" not in c for c in calls)


def test_class_balance_expression_on_train_val_frame():
    g = pd.DataFrame({"class": ["A", "A", "B", "B", "B"], "split": ["train", "val", "train", "train", "val"]})
    cnt = g.groupby(["class", "split"]).size().unstack()
    assert list(cnt.columns) == ["train", "val"]
    assert cnt.loc["B", "train"] == 2
    src = _code("src/02_eda.py")
    assert 'cnt = g.groupby(["class", "split"]).size().unstack()' in src


# ---- stage 1: safe writes and the subset guard
def _g1():
    return _mod("01_build_grains")


def _fake_imap(fail_at=None):
    def imap(f, paths):
        for i, p in enumerate(paths):
            if fail_at is not None and i == fail_at:
                raise RuntimeError("crash mid-cache")
            yield np.full((128, 128, 3), i + 1, dtype=np.uint8)
    return imap


def _table(v):
    return pd.DataFrame({"grain_id": ["a-1", "a-2", "a-3"], "v": [v] * 3})


def test_write_derived_success_publishes_both_and_leaves_no_tmp(tmp_path):
    m = _g1()
    pq, mm = tmp_path / "grains.parquet", tmp_path / "images128.u8"
    m.write_derived(_table(1), ["x", "y", "z"], pq, mm, imap=_fake_imap())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["grains.parquet", "images128.u8"]
    assert pd.read_parquet(pq).equals(_table(1))
    arr = np.fromfile(mm, dtype=np.uint8).reshape(3, 128, 128, 3)
    assert [int(arr[i, 0, 0, 0]) for i in range(3)] == [1, 2, 3]


def test_write_derived_crash_in_cache_keeps_old_files_byte_identical(tmp_path):
    m = _g1()
    pq, mm = tmp_path / "grains.parquet", tmp_path / "images128.u8"
    m.write_derived(_table(1), ["x", "y", "z"], pq, mm, imap=_fake_imap())
    old_pq, old_mm = pq.read_bytes(), mm.read_bytes()
    with pytest.raises(RuntimeError):
        m.write_derived(_table(2), ["x", "y", "z"], pq, mm, imap=_fake_imap(fail_at=1))
    assert pq.read_bytes() == old_pq and mm.read_bytes() == old_mm
    assert sorted(p.name for p in tmp_path.iterdir()) == ["grains.parquet", "images128.u8"]


def test_write_derived_crash_in_parquet_keeps_old_files(tmp_path):
    m = _g1()
    pq, mm = tmp_path / "grains.parquet", tmp_path / "images128.u8"
    m.write_derived(_table(1), ["x", "y", "z"], pq, mm, imap=_fake_imap())
    old_pq, old_mm = pq.read_bytes(), mm.read_bytes()

    class Bad:
        def to_parquet(self, *a, **k):
            raise OSError("disk full")
    with pytest.raises(OSError):
        m.write_derived(Bad(), ["x", "y", "z"], pq, mm, imap=_fake_imap())
    assert pq.read_bytes() == old_pq and mm.read_bytes() == old_mm
    assert len(list(tmp_path.iterdir())) == 2


def test_atomic_parquet_crash_keeps_old_file(tmp_path, monkeypatch):
    m = _g1()
    p = tmp_path / "measures.parquet"
    m.atomic_parquet(_table(1), p)
    old = p.read_bytes()

    def boom(self, path, *a, **k):
        open(path, "wb").write(b"half")
        raise OSError("crash")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        m.atomic_parquet(_table(2), p)
    assert p.read_bytes() == old
    assert [q.name for q in tmp_path.iterdir()] == ["measures.parquet"]


def test_main_publishes_through_the_safe_helpers():
    src = _code("src/01_build_grains.py")
    main = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {ast.unparse(n.func) for n in ast.walk(main) if isinstance(n, ast.Call)}
    assert {"write_derived", "require_smoke_for_subset"} <= called
    assert not any(c.endswith(".to_parquet") or c == "np.memmap" for c in called)
    assert "atomic_parquet(m, cache)" in src


def test_subset_outside_smoke_is_refused_before_any_write():
    m = _g1()
    with pytest.raises(SystemExit):
        m.require_smoke_for_subset(False, False)
    m.require_smoke_for_subset(False, True)
    m.require_smoke_for_subset(True, False)
    main = next(n for n in ast.parse(_code("src/01_build_grains.py")).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    first_write = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(("mkdir", "to_csv")))
    guard = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == "require_smoke_for_subset")
    assert guard < first_write
