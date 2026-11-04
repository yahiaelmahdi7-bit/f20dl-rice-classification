"""Tests for stages 0 to 2 helpers and tools/leak_audit.py. No model is fitted; stage 2 is never imported."""
import ast
import importlib
import io
import os
import pathlib
import subprocess
import types

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


# ---- job 3: stage 2 reads splits.csv for per-split class counts only, the one use its prompt allows
def _code(rel):
    """Source after the prompt-record header, whose comments quote the rules."""
    return (ROOT / rel).read_text().split("PROMPT RECORD END ===", 1)[1]


def test_eda_uses_splits_for_class_counts_only():
    src = _code("src/02_eda.py")
    lines = src.splitlines()
    used = {lines[n.lineno - 1].strip() for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name) and n.id == "sp"}
    allowed = ("sp = pd.read_csv(C.SPLITS)", 'cnt = sp.groupby(["class", "split"]).size().unstack()', 'ax.set(title=f"Grains per variety and split (n={len(sp)})"')
    assert all(u.startswith(allowed) for u in used), used
    assert src.count("SPLITS") == 1 and "load_split" not in src


# ---- job 4: stage 3 runs before stage 2, so the dictionary's selected tags are right on a clean run
def test_make_runs_features_before_eda():
    mk = (ROOT / "Makefile").read_text()
    data = next(l for l in mk.splitlines() if l.startswith("data:")).split("##")[0].split()
    assert data.index("features") < data.index("eda")
    recipe = mk[mk.index("\nsmoke:"):]
    assert recipe.index("src/03_features.py") < recipe.index("src/02_eda.py")


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


# ---- tools/leak_audit.py on tiny made-up data (the tool is imported, never run on real data)
def _audit_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("leak_audit", ROOT / "tools" / "leak_audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tiny(tmp_path, n=20):
    import common as C
    rng = np.random.default_rng(1)
    split = ["train"] * 12 + ["val"] * 4 + ["test"] * 4
    g = pd.DataFrame(rng.normal(size=(n, len(C.ALL106))) * 10 + 50, columns=C.ALL106)
    g.insert(0, "split", split)
    g.insert(0, "grain_id", [f"c-{i}" for i in range(n)])
    paths = []
    for i in range(n):
        p = tmp_path / f"{i}.jpg"
        p.write_bytes(f"jpg-{i}".encode())
        paths.append(str(p))
    g.insert(2, "path", paths)
    g.insert(3, "row", np.arange(n))
    sp = pd.DataFrame({"grain_id": g.grain_id, "msc_row": np.arange(n), "split": split})
    mm = rng.integers(0, 256, size=(n, 128, 128, 3), dtype=np.uint8)
    return sp, g, mm


def test_leak_audit_clean_data_reports_no_overlap(tmp_path):
    L = _audit_mod()
    sp, g, mm = _tiny(tmp_path)
    R = L.audit(sp, g, mm)
    assert R["counts"] == {"train": 12, "val": 4, "test": 4}
    assert R["dup_grain_id"] == 0 and R["dup_msc_row"] == 0
    assert R["id_overlap"] == {"train&val": 0, "train&test": 0, "val&test": 0}
    assert R["parquet_vs_csv_split_mismatch"] == 0 and R["parquet_rows"] == 20
    for k in ["table_exact", "table_6dp", "table_3dp", "shape16_6dp", "colour90_6dp", "jpg_bytes",
              "pixels128", "pixels_any_rotation_flip", "thumb16_any_rotation_flip"]:
        assert R[k]["groups_across_splits"] == 0 and R[k]["grains_across_splits"] == 0, k
    nn = R["nearest_train_neighbour_distinct91"]
    assert nn["train_to_train"]["n"] == 12 and nn["val_to_train"]["n"] == 4 and nn["test_to_train"]["n"] == 4
    assert nn["val_to_train"]["under_0_5"] == 0


def test_leak_audit_detects_planted_overlaps(tmp_path):
    import common as C
    L = _audit_mod()
    sp, g, mm = _tiny(tmp_path)
    g.loc[18, C.ALL106] = g.loc[0, C.ALL106].to_numpy()          # test grain 18 repeats train grain 0's table row
    mm[13] = mm[1]                                                # val grain 13 has train grain 1's pixels
    mm[19] = np.rot90(mm[2], 1)                                   # test grain 19 is train grain 2 rotated a quarter turn
    (tmp_path / "14.jpg").write_bytes((tmp_path / "3.jpg").read_bytes())   # val 14 is train 3's file
    sp.loc[16, "grain_id"] = "c-0"                                # id appears in train and test
    R = L.audit(sp, g, mm)
    assert R["id_overlap"]["train&test"] == 1 and R["dup_grain_id"] == 1
    for k in ["table_exact", "table_6dp", "table_3dp", "shape16_6dp", "colour90_6dp"]:
        assert R[k]["groups_across_splits"] == 1 and R[k]["by_split"] == {"test": 1, "train": 1}, k
    assert R["pixels128"]["groups_across_splits"] == 1 and R["pixels128"]["by_split"] == {"train": 1, "val": 1}
    assert R["pixels_any_rotation_flip"]["groups_across_splits"] == 2      # the exact copy and the rotated copy
    assert R["thumb16_any_rotation_flip"]["groups_across_splits"] == 2
    assert R["jpg_bytes"]["groups_across_splits"] == 1
    assert R["nearest_train_neighbour_distinct91"]["test_to_train"]["min"] == 0.0


def test_leak_audit_flatten_names_and_values(tmp_path):
    L = _audit_mod()
    sp, g, mm = _tiny(tmp_path)
    rows = dict(L.flatten(L.audit(sp, g, mm)))
    for name in ["counts.train", "dup_grain_id", "id_overlap.train&val", "parquet_rows", "table_exact.repeat_groups",
                 "jpg_bytes.groups_across_splits", "nearest_train_neighbour_distinct91.val_to_train.p0.1",
                 "nearest_train_neighbour_distinct91.test_to_train.under_0_5"]:
        assert name in rows, name
    assert rows["counts.test"] == 4 and rows["parquet_rows"] == 20
    assert all(not isinstance(v, (np.generic, dict)) for v in rows.values())
    assert L.flatten({"a": {"b": {}, "c": np.int64(3)}, "d": 1.5}) == [("a.c", 3), ("d", 1.5)]


def test_leak_audit_csv_columns_and_atomic_write(tmp_path):
    L = _audit_mod()
    out = tmp_path / "leak_audit.csv"
    L.write_rows([("a.b", 1), ("c", 2.5)], out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["check", "value"] and df.check.tolist() == ["a.b", "c"]
    assert [p.name for p in tmp_path.iterdir()] == ["leak_audit.csv"]


def test_leak_audit_source_rules():
    src = (ROOT / "tools" / "leak_audit.py").read_text()
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert ast.unparse(main.body[0]) == "assert not C.SMOKE"
    assert not main.args.args
    assert "sys.argv" not in src
    reads = [ast.unparse(n) for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) in ("pd.read_csv", "pd.read_parquet")]
    assert len(reads) == 2 and all("class" not in r and "'y'" not in r and '"y"' not in r for r in reads)
    assert "usecols=['grain_id', 'msc_row', 'split']" in reads[0]
    assert "'path', 'row', *C.ALL106" in reads[1]
    assert "quality" in ast.unparse(main) and "leak_audit.csv" in ast.unparse(main)


# ---- claims the first tests left unguarded: each of these passed when its behaviour was broken
def test_make_read_only_raises_when_chmod_fails(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _mod("00_get_data").make_read_only(tmp_path / "missing")


def test_write_derived_renames_memmap_before_parquet(tmp_path, monkeypatch):
    m, order, real = _g1(), [], os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append(pathlib.Path(b).name), real(a, b))[1])
    m.write_derived(_table(1), ["x", "y", "z"], tmp_path / "grains.parquet", tmp_path / "images128.u8", imap=_fake_imap())
    assert order == ["images128.u8", "grains.parquet"]


def test_leak_audit_crash_mid_write_keeps_old_csv(tmp_path, monkeypatch):
    L, out = _audit_mod(), tmp_path / "leak_audit.csv"
    out.write_text("old\n")

    def boom(self, target, **k):
        pathlib.Path(target).write_text("half")
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        L.write_rows([("a", 1)], out)
    assert out.read_text() == "old\n" and [p.name for p in tmp_path.iterdir()] == ["leak_audit.csv"]


# ---- R2 review, Fix 4: the real lines of stages 2 and 4 run on made-up frames; importing either would run the stage
def _block(rel, start, end):
    """Source text from one marker up to the next."""
    src = _code(rel)
    i = src.index(start)
    return src[i:src.index(end, i)]


def _fstrings(rel, marker, **names):
    """Every f-string in a stage whose text holds marker, evaluated with names."""
    found = [n for n in ast.walk(ast.parse(_code(rel))) if isinstance(n, ast.JoinedStr) and marker in ast.unparse(n)]
    return [eval(compile(ast.Expression(n), rel, "eval"), dict(names)) for n in found]


def _eda_ns(tmp_path, g, classes=()):
    """Names a stage 2 block needs, with C reduced to the columns of g and a log that collects its rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = {}
    C = types.SimpleNamespace(ALL106=[c for c in g.columns if c not in ("class", "grain_id")], CLASSES=list(classes), wrote=lambda p: None,
                              read_images=lambda s: [np.zeros((4, 4, 3), np.uint8)] * len(s), savefig=lambda fig, name: plt.close(fig))
    ns = dict(np=np, pd=pd, plt=plt, C=C, g=g, q=tmp_path, n=len(g), ENT_FORMULA="formula", add_insight=lambda *a, **k: None,
              append_log=lambda rows, checks: log.update({r["check"]: r for r in rows}))
    exec(_block("src/02_eda.py", "def flagged_rows(", "# validity"), ns)
    return ns, log


def test_insight_n_defaults_to_all_grains_and_takes_the_rows_used():
    ns = {"ins": [], "n": 999}
    exec(_block("src/02_eda.py", "def add_insight(", "def append_log("), ns)
    ns["add_insight"]("a", "all grains", 1)
    ns["add_insight"]("b", "a sample", 2, "fig", n_used=600)
    assert [(r["id"], r["n"], r["figure"]) for r in ns["ins"]] == [("a", 999, ""), ("b", 600, "fig")]


def test_each_insight_on_a_subset_states_its_own_rows():
    calls = {n.args[0].value: next((k.value for k in n.keywords if k.arg == "n_used"), None)
             for n in ast.walk(ast.parse(_code("src/02_eda.py")))
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "add_insight"}
    fo = pd.DataFrame({"lag1_autocorr": [0.2, -0.9, 0.5], "n": [10, 20, 30]})    # largest by size is negative, on the n=20 row
    names = dict(fo=fo, rs=range(6), sub=range(5), J=range(4))
    got = {i: eval(compile(ast.Expression(e), "n_used", "eval"), dict(names)) for i, e in calls.items() if e is not None}
    assert got == {"max_lag1_autocorr": 20, "resize_area_ratio": 6, "corr_clusters": 5, "jasmine_late_area_ratio": 4}


def test_validity_log_counts_rows_not_cells(tmp_path):
    g = pd.DataFrame({"AREA": [-1.0, -2.0, 3.0, 4.0], "PERIMETER": [-1.0, 2.0, 3.0, 4.0], "EXTENT": [0.5, 0.25, 1.5, 0.75],
                      "MAJOR_AXIS": [np.inf, 1.0, 2.0, 3.0], "MINOR_AXIS": [np.inf, np.inf, 2.0, 3.0], "ECCENTRICITY": [0.7] * 4})
    ns, log = _eda_ns(tmp_path, g)
    exec(_block("src/02_eda.py", "# validity", "# outliers: robust z"), ns)
    rng, inf, flat = log["validity_out_of_range"], log["validity_infinite"], log["validity_near_constant"]
    assert (rng["rows_affected"], inf["rows_affected"], flat["rows_affected"]) == (3, 2, 0)    # 4 and 3 cells; a column is no row
    assert "(4 cells)" in rng["reason"] and "(3 cells)" in inf["reason"] and "cells" not in flat["reason"]
    assert "formula" in rng["reason"] and flat["reason"].startswith("1 columns flagged: ECCENTRICITY")


def test_outlier_log_counts_rows_not_cells(tmp_path):
    t = np.tile(np.linspace(-1, 1, 20), 2)
    g = pd.DataFrame({"a": t, "b": t, "c": t, "class": ["X"] * 20 + ["Y"] * 20, "grain_id": [f"g{i}" for i in range(40)]})
    g.loc[0, ["a", "b"]] = 100.0     # one row, two cells
    g.loc[25, "a"] = 100.0           # second class, one cell
    ns, log = _eda_ns(tmp_path, g, classes=("X", "Y"))
    exec(_block("src/02_eda.py", "# outliers: robust z", "# trends against file number"), ns)
    r = log["outliers_robust_z6"]
    assert r["rows_affected"] == 2 and r["reason"].startswith("3 cells with |robust z| > 6")
    assert pd.read_csv(tmp_path / "outliers.csv").n_outliers.sum() == 3      # the table keeps cells


def test_eda_titles_state_n():
    C = types.SimpleNamespace(CLASSES=list("abcde"))
    assert _fstrings("src/02_eda.py", "rolling median 200", f="AREA", n=7) == ["AREA, rolling median 200 (n=7)"]
    assert _fstrings("src/02_eda.py", "Example grains", C=C, n=70) == ["Example grains, 128 px, train+val (n=30 of 70)"]


def test_cluster_titles_state_their_own_n():
    names = dict(sidx=range(10), g=range(63))
    assert _fstrings("src/04_cluster.py", "Silhouette vs k", **names) == ["Silhouette vs k, k-means (n=10 of 63)"]
    assert _fstrings("src/04_cluster.py", "inertia vs k", **names) == ["K-means inertia vs k (n=63)"]
    assert _fstrings("src/04_cluster.py", "BIC vs k", **names) == ["GMM BIC vs k (n=63)"]


# ---- stage 0: a part-filled folder is not unpacked
def _stage0(tmp_path, monkeypatch, members):
    import zipfile
    m = _mod("00_get_data")
    raw = tmp_path / "raw"
    raw.mkdir()
    z = raw / "rice-msc.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, text in members.items(): zf.writestr(name, text)
    monkeypatch.delenv("RICE_ZIP_DIR", raising=False)
    for k, v in dict(RAW=raw, DIRS={"rice-msc.zip": "D"}, expected=lambda: {"rice-msc.zip": m.sha(z)}, make_read_only=lambda p: None).items():
        monkeypatch.setattr(m, k, v)
    (raw / "D").mkdir()
    (raw / "D" / "a.txt").write_text("aaaa")
    return m, raw


def test_part_filled_folder_is_filled_from_the_zip(tmp_path, monkeypatch, capsys):
    m, raw = _stage0(tmp_path, monkeypatch, {"D/a.txt": "aaaa", "D/s/b.txt": "bb"})
    assert m.main() == 0
    assert (raw / "D/s/b.txt").read_text() == "bb" and "already unpacked" not in capsys.readouterr().out


def test_part_written_file_is_replaced(tmp_path, monkeypatch):
    m, raw = _stage0(tmp_path, monkeypatch, {"D/a.txt": "aaaaaaaa"})
    assert m.main() == 0 and (raw / "D/a.txt").read_text() == "aaaaaaaa"


def test_part_filled_read_only_folder_stops_loudly(tmp_path, monkeypatch, capsys):
    m, raw = _stage0(tmp_path, monkeypatch, {"D/a.txt": "aaaa", "D/b.txt": "bb"})
    os.chmod(raw / "D", 0o555)
    try:
        assert m.main() == 1
    finally:
        os.chmod(raw / "D", 0o755)
    assert "cannot finish unpacking" in capsys.readouterr().out


def test_full_folder_is_left_alone(tmp_path, monkeypatch, capsys):
    m, raw = _stage0(tmp_path, monkeypatch, {"D/a.txt": "aaaa"})
    os.utime(raw / "D/a.txt", (1, 1))
    assert m.main() == 0 and "already unpacked" in capsys.readouterr().out
    assert (raw / "D/a.txt").stat().st_mtime == 1


# ---- leak audit: parquet rows the inner merge would drop are counted
def test_leak_audit_counts_parquet_rows_not_in_csv(tmp_path):
    L = _audit_mod()
    sp, g, mm = _tiny(tmp_path)
    assert L.audit(sp, g, mm)["parquet_rows_not_in_csv"] == 0
    R = L.audit(sp.iloc[2:], g, mm)
    assert R["parquet_rows_not_in_csv"] == 2 and R["parquet_vs_csv_split_mismatch"] == 0


# ---- stage 9: no unused size list that holds EXTENT
def test_stage9_has_no_size_cols_constant():
    assert "SIZE_COLS" not in _code("src/09_robustness.py")
