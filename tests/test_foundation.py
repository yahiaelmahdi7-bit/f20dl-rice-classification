"""Model-free tests for Foundation fixes and for shared lines the 2,000-grain slice cannot reach."""
import ast, re, runpy, subprocess, sys
import numpy as np
import pandas as pd
import pytest
import common as C
from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import stamp_prompts as S  # noqa: E402

MAKEFILE = (ROOT / "Makefile").read_text()
STUB = """#!/bin/sh
case "$1" in
  tools/*) [ -z "$TOOL_FAIL" ] || exit 1; printf '/r/a b (1).jpg\\n/r/c d (2).jpg\\n' ;;
  *) [ $# -eq 2 ] || exit 2; echo "$2" >> got.txt; [ -z "$PREDICT_FAIL" ] ;;
esac
"""


def smoke_predict(tmp_path, **env):
    """Run the Makefile's own smoke predict line under sh, against a stub Python."""
    line = next(l for l in MAKEFILE.splitlines() if l.startswith("\t") and "smoke_images.py" in l)
    (tmp_path / "py").write_text(STUB)
    (tmp_path / "py").chmod(0o755)
    cmd = line.strip().replace("$(SM)", "RICE_SMOKE=1").replace("$(PY)", "./py").replace("$$", "$")
    return subprocess.run(["sh", "-c", cmd], cwd=tmp_path, env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True)


def manifest():
    tree = ast.parse((ROOT / "src" / "report.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "MANIFEST")


def test_floor_ignores_a_tie_that_would_let_a_wrong_answer_in():
    conf, ok = np.array([0.9, 0.9, 0.95]), np.array([0, 1, 1])
    for order in ([0, 1, 2], [1, 0, 2]):
        assert C.floor_for(conf[order], ok[order], 1.0) == 0.95
    assert C.floor_for(np.array([1.0, 1.0]), np.array([0, 1]), 1.0) == 1.01    # a saturated top run that misses the target


def test_smoke_predict_passes_each_path_whole(tmp_path):
    r = smoke_predict(tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "got.txt").read_text().splitlines() == ["/r/a b (1).jpg", "/r/c d (2).jpg"]


def test_smoke_predict_fails_when_a_step_fails(tmp_path):
    assert smoke_predict(tmp_path, PREDICT_FAIL="1").returncode != 0
    assert smoke_predict(tmp_path, TOOL_FAIL="1").returncode != 0


def test_smoke_starts_with_stage_0():
    assert re.search(r"^smoke: get-data\b", MAKEFILE, re.M)


def test_smoke_images_prints_one_path_per_line(monkeypatch, capsys):
    g = pd.DataFrame({"y": [0, 1, 2], "path": ["a b (1).jpg", "c d (2).jpg", "e f (3).jpg"]})
    monkeypatch.setattr(C, "load_split", lambda name: g)
    monkeypatch.setattr("sys.argv", ["smoke_images.py", "3"])
    runpy.run_path(str(ROOT / "tools" / "smoke_images.py"))
    assert capsys.readouterr().out.splitlines() == list(g.path)


def test_each_stage_stamps_its_own_prompt():
    for p in sorted((ROOT / "src").glob("0*.py")):
        assert S.prompt_for(p).name[:2] == p.name[:2] and S.prompt_for(p).name != "00-context.md"


def test_report_writes_inside_results():
    assert "C.RES.parent" not in (ROOT / "src" / "report.py").read_text()
    assert {"results/SUMMARY.csv", "results/MANIFEST.csv"} <= set(C.ARTEFACTS["files"])


def test_manifest_points_at_real_artefacts():
    files = {k.replace("{sp}", s) for k in C.ARTEFACTS["files"] for s in ("val", "test")}
    for row in manifest():
        out = row[2]
        if out.startswith("src/"):
            assert (ROOT / out).exists()
        elif out.endswith("/"):
            assert any(f.startswith("results/" + out) for f in files), out
        else:
            assert "results/" + out in files, out
    assert [r[2] for r in manifest() if "dictionary" in r[0]] == ["quality/"]


# ---- lines the slice cannot reach -----------------------------------------------------
def test_read_images_gates_test_rows_in_full_mode(gate_env):
    with pytest.raises(PermissionError):
        C.read_images(pd.DataFrame({"split": ["test"], "row": [0]}))


def test_grain_mask_on_a_blank_image_is_none():
    assert C.grain_mask(np.zeros((20, 20, 3), np.uint8)) is None


def test_wilson_with_no_grains_is_nan():
    assert np.isnan(C.wilson(0, 0)).all()


def test_selected_features_come_from_stage_3(tmp_path, monkeypatch):
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "selected.json").write_text('{"selected": ["AREA", "meanRR"]}')
    monkeypatch.setattr(C, "RES", tmp_path)
    assert C.features("selected") == ["AREA", "meanRR"]
