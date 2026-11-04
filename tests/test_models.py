"""Models team fixes to stages 5, 6 and predict. No model is fitted: helpers run on made-up rows, main() is read as text."""
import ast, csv, importlib, io, json, re
from pathlib import Path
import numpy as np, pandas as pd, pytest
import common as C
from conftest import ROOT

B = importlib.import_module("05_baselines")
N = importlib.import_module("06_neural")
P = importlib.import_module("predict")


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(C, "wrote", lambda *a, **k: None)


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "stage_dir", lambda name: tmp_path)
    return tmp_path


def func(mod, name):
    return next(n for n in ast.parse(Path(mod.__file__).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node, name):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and getattr(c.func, "id", getattr(c.func, "attr", None)) == name]


def seed_loops(main):
    return [n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.iter) == "a.seeds"]


# F11: every row of the epoch log parses with the contract's columns
def test_epoch_log_rows_parse(out):
    cfg = {"hidden": [256, 128], "lr": 0.001}
    N.log_epoch("t", 1, 114.8, cfg, 100, dev="cpu")
    N.log_epoch("t", 2, 1.5, cfg, 100, dev="cpu")
    raw = (out / "epoch_times.csv").read_bytes()
    assert b"\r" not in raw
    rows = list(csv.reader(io.StringIO(raw.decode())))
    cols = list(C.ARTEFACTS["files"]["results/neural/epoch_times.csv"]["columns"])
    assert rows[0] == cols and len(rows) == 3
    for r in rows[1:]:
        assert len(r) == len(cols) and json.loads(r[-1]) == cfg and r[cols.index("device")] == "cpu"
    assert rows[1][cols.index("seconds")] == "114.80"


def test_mlp_logs_each_epoch_as_cpu():
    (call,) = calls(func(N, "train_mlp"), "log_epoch")
    assert {k.arg: ast.unparse(k.value) for k in call.keywords}.get("dev") == "'cpu'"


# F1: one column order, whichever writer came first
@pytest.mark.parametrize("order", [["tag", "epoch", "train_loss", "seconds", "val_loss", "val_acc"], list(reversed(N.CURVES))])
def test_curves_keep_contract_order(out, order):
    cnn = ["tag", "epoch", "train_loss", "seconds", "val_loss", "val_acc"]
    assert set(cnn) == set(N.CURVES)
    pd.DataFrame([dict(zip(cnn, ["old", 1, 0.5, 9.0, 0.6, 0.7])), dict(zip(cnn, ["redo", 1, 0.5, 9.0, 0.6, 0.7]))]).to_csv(out / "curves.csv", index=False)
    N.save_curves([{k: {"tag": "redo", "seconds": 3.0}.get(k, 1) for k in order}])
    got = pd.read_csv(out / "curves.csv")
    assert list(got.columns) == list(N.CURVES)
    assert sorted(got.tag) == ["old", "redo"]
    assert got.set_index("tag").seconds.to_dict() == {"old": 9.0, "redo": 3.0}


# F3: each finished seed is saved inside its loop
def test_curves_saved_inside_each_seed_loop():
    main = func(N, "main")
    loops = seed_loops(main)
    assert len(loops) == 2
    assert all(len(calls(l, "save_curves")) == 1 for l in loops)
    assert len(calls(main, "save_curves")) == len(loops)


@pytest.mark.parametrize("merge,key", [(N.merge_rows, "k"), (B.merge_csv, ["k"])])
def test_crash_mid_write_keeps_last_good_file(tmp_path, monkeypatch, merge, key):
    p = tmp_path / "t.csv"
    p.write_text("k,v\na,1\n")

    def broken(self, path, **kw):
        Path(path).write_text("k,v\npart")
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_csv", broken)
    with pytest.raises(OSError):
        merge(p, pd.DataFrame({"k": ["b"], "v": [2]}), key)
    assert p.read_text() == "k,v\na,1\n"


# F9: --control --only cnn keeps the MLP row
def test_control_rows_merge_by_model(tmp_path):
    p = tmp_path / "control.csv"
    pd.DataFrame({"model": ["mlp", "cnn"], "val_accuracy": [0.2, 0.2]}).to_csv(p, index=False)
    N.merge_rows(p, pd.DataFrame({"model": ["cnn"], "val_accuracy": [0.3]}), "model")
    assert pd.read_csv(p).set_index("model").val_accuracy.to_dict() == {"mlp": 0.2, "cnn": 0.3}


def test_main_merges_control_csv():
    main = func(N, "main")
    assert any("control.csv" in ast.unparse(c) for c in calls(main, "merge_rows"))
    assert not any("control.csv" in ast.unparse(c) for c in calls(main, "to_csv"))


# F2: a shorter run reuses only this grid's own winner
def table(path, grid, win):
    t = pd.DataFrame(grid)
    t["val_macro_f1"] = [0.9 if i == win else 0.5 for i in range(len(grid))]
    t.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("name,grid", [("cnn", N.CNN_GRID), ("mlp", N.MLP_GRID)])
def test_must_tune_reuses_only_the_winner(tmp_path, name, grid):
    t = table(tmp_path / "t.csv", grid, 3)
    assert N.must_tune({name: grid[3]}, name, grid, t, [2]) is False
    assert N.must_tune({name: grid[3]}, name, grid, t, list(C.SEEDS)) is True      # a clean run always tunes
    assert N.must_tune({}, name, grid, t, [2]) is True                             # no config
    assert N.must_tune({name: grid[3]}, name, grid, tmp_path / "none.csv", [2]) is True
    assert N.must_tune({name: grid[0]}, name, grid, t, [2]) is True                # another winner
    one = table(tmp_path / "smoke.csv", grid[:1], 0)
    assert N.must_tune({name: grid[0]}, name, grid, one, [2]) is True              # smoke table
    older = table(tmp_path / "older.csv", [{**g, "lr": 0.5} for g in grid], 3)
    assert N.must_tune({name: grid[3]}, name, grid, older, [2]) is True            # older grid


def test_mlp_config_saved_before_its_seeds():
    main = func(N, "main")
    assert min(c.lineno for c in calls(main, "save_config")) < min(l.lineno for l in seed_loops(main))


# F4: a crash keeps every finished job, a full run keeps no old row
class Stub:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X), dtype=int)
    def predict_proba(self, X): return np.full((len(X), 5), 0.2)


def frame():
    cols = C.features("shape16")
    f = pd.DataFrame(np.ones((10, len(cols))), columns=cols)
    f["grain_id"], f["y"] = [f"g{i}" for i in range(10)], np.arange(10) % 5
    return f


def test_crash_keeps_finished_jobs_and_full_run_clears(tmp_path, monkeypatch):
    m1, m2 = list(B.GRIDS)[:2]

    def build(m, p):
        if m == m2: raise RuntimeError("crash")
        return Stub()
    monkeypatch.setattr(B, "build", build)
    monkeypatch.setattr(C, "save_preds", lambda *a, **k: None)
    monkeypatch.setattr(B.joblib, "dump", lambda *a, **k: None)
    monkeypatch.setattr(C, "MODELS", tmp_path)
    with pytest.raises(RuntimeError):
        B.tune_jobs([(m1, "shape16"), (m2, "shape16")], frame(), frame(), tmp_path)
    assert list(json.loads((tmp_path / "best_params.json").read_text())) == [f"{m1}:shape16"]
    assert len(pd.read_csv(tmp_path / "grid.csv")) == len(B.points(m1))
    assert pd.read_csv(tmp_path / "val_results.csv").model.tolist() == [m1]
    small = pd.DataFrame({"model": [m1], "featureset": ["shape16"], "params": ["{}"], "val_macro_f1": [0.1]})
    B.merge_csv(tmp_path / "grid.csv", small, ["model", "featureset"])
    assert pd.read_csv(tmp_path / "grid.csv").params.tolist() == ["{}"]         # a smaller grid leaves no old point
    B.clear_results(tmp_path)
    assert not any((tmp_path / n).exists() for n in ("best_params.json", "grid.csv", "val_results.csv"))


def test_full_run_calls_clear_results():
    assert calls(func(B, "main"), "clear_results")


# float32 sums stall at 2**24: 5,000 images of 64x64 pass that
def test_chan_stats_hold_at_full_sample(monkeypatch):
    monkeypatch.setattr(C, "read_images", lambda s: np.full((5000, 64, 64, 3), 230, np.uint8))
    mean, std = N.chan_stats(pd.DataFrame({"a": range(5000)}))
    assert mean == pytest.approx([230 / 255] * 3, abs=1e-4)
    assert std == pytest.approx([0.0] * 3, abs=1e-4)


# Job 2: read as text, make is never run
def test_controls_target_text():
    mk = (ROOT / "Makefile").read_text()
    assert {"overfit", "controls"} <= set(re.search(r"^\.PHONY:(.*)$", mk, re.M).group(1).split())
    pre, _, note = re.search(r"^controls:(.*)$", mk, re.M).group(1).partition("##")
    assert pre.strip() == "" and note.strip()                                   # no prerequisite can retrain
    recipe = re.search(r"^controls:.*\n((?:\t.*\n)+)", mk, re.M).group(1)
    at = [recipe.index(x) for x in ("best_config.json", "05_baselines.py --control", "06_neural.py --control")]
    assert at == sorted(at) and "--overfit-test" not in recipe
    overfit = re.search(r"^overfit:(.*)\n((?:\t.*\n)+)", mk, re.M)
    assert overfit.group(1).partition("##")[0].strip() == ""                    # the sanity check waits on nothing trained
    assert "06_neural.py --overfit-test" in overfit.group(2) and "best_config" not in overfit.group(2)
    steps = re.search(r"^all:([^#]*)##", mk, re.M).group(1).split()
    assert steps.index("overfit") < steps.index("train") < steps.index("controls") < steps.index("freeze")


# predict
def test_table_row_reads_no_labels_and_survives_gaps(tmp_path, monkeypatch):
    g = pd.DataFrame(np.ones((2, len(C.ALL106))), columns=C.ALL106)
    g.insert(0, "grain_id", ["Basmati-12", "Jasmine-3"])
    g["class"], g["y"] = "Basmati", 1
    g.to_parquet(tmp_path / "grains.parquet")
    monkeypatch.setattr(C, "DERIVED", tmp_path)
    row = P.table_row("Basmati (12).jpg")
    assert row.grain_id.tolist() == ["Basmati-12"] and not {"class", "y"} & set(row.columns)
    assert P.table_row("Basmati (99).jpg") is None
    assert P.table_row("rice.png") is None
    monkeypatch.setattr(C, "DERIVED", tmp_path / "none")
    assert P.table_row("Basmati (12).jpg") is None


@pytest.mark.parametrize("size,mode", [((100, 100), "RGB"), ((250, 250), "L")])
def test_wrong_image_exits_2(tmp_path, size, mode):
    from PIL import Image
    p = tmp_path / "x.png"
    Image.new(mode, size).save(p)
    with pytest.raises(SystemExit) as e:
        P.load_image(p)
    assert e.value.code == 2


def test_missing_file_exits_3_and_names_both_targets(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        P.need(tmp_path / "gate.joblib")
    err = capsys.readouterr().err
    assert e.value.code == 3 and "gate.joblib" in err and "make train" in err and "make fit" in err
