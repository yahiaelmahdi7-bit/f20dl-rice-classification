"""Stages 7, 8, 9 and freeze: refusals before a read, on made-up rows. No model is fitted and no real path is touched."""
import ast, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, pytest
import common as C

ROOT = Path(__file__).resolve().parents[1]


def load(rel):
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location("ev_" + p.stem.replace("0", "m", 1), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def frame(lots=("early", "late")):
    return pd.DataFrame({"grain_id": [f"g{i}" for i in range(len(lots))], "lot": list(lots), "y": 0, "path": "/nowhere/Jasmine/a.jpg"})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SMOKE", False)
    for n in ("RES", "MODELS", "DERIVED"):
        (tmp_path / n).mkdir()
        monkeypatch.setattr(C, n, tmp_path / n)
    monkeypatch.setattr(C, "FROZEN", tmp_path / "frozen.json")
    monkeypatch.setattr(C, "TEST_READS", tmp_path / "reads.csv")

    def split(name):
        if name == "test": pytest.fail("test rows were read")
        return frame()
    monkeypatch.setattr(C, "load_split", split)
    return tmp_path


def refuse(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert not C.TEST_READS.exists()
    return str(e.value)


def test_stage7_stops_before_the_read(env, monkeypatch):
    msg = refuse(load("src/07_compare.py"), ["07", "--split", "test"], monkeypatch)
    assert "cnn_seed2.pt" in msg and "mlp_prep.joblib" in msg and "hgb__all106.joblib" in msg


def test_stage7_never_drops_a_model_on_test(env, monkeypatch):
    s = load("src/07_compare.py")
    monkeypatch.setattr(s, "test_probs", lambda *a: (_ for _ in ()).throw(FileNotFoundError("gone")))
    with pytest.raises(FileNotFoundError):
        s.score_models(frame())


def test_stage7_validation_stays_tolerant(env, monkeypatch):
    s = load("src/07_compare.py")
    monkeypatch.setattr(C, "load_preds", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["07", "--split", "val"])
    with pytest.raises(AssertionError, match="no predictions"):
        s.main()


def test_stage8_stops_before_the_read(env, monkeypatch):
    msg = refuse(load("src/08_routing.py"), ["08", "--score"], monkeypatch)
    assert "routing.json" in msg and "gate_image.joblib" in msg and "cnn__image__seed2__test" in msg


def test_stage9_stops_before_the_read(env, monkeypatch):
    msg = refuse(load("src/09_robustness.py"), ["09", "--score"], monkeypatch)
    assert "full_hgb__sizefree22" in msg and "/nowhere/Jasmine" in msg


def configs(best=None, cfg=None):
    for d, n, v in [("baselines", "best_params.json", best), ("neural", "best_config.json", cfg)]:
        (C.RES / d).mkdir(exist_ok=True)
        if v is not None: (C.RES / d / n).write_text(json.dumps(v))


def test_stage9_train_stops_before_any_fit(env, monkeypatch):
    s = load("src/09_robustness.py")
    monkeypatch.setattr(s, "_load", lambda f: pytest.fail("reached training"))
    assert "best_params.json" in refuse(s, ["09", "--train"], monkeypatch)
    configs({"hgb:shape16": {"a": 1}}, {"mlp": {}})
    msg = refuse(s, ["09", "--train"], monkeypatch)
    assert "hgb:all106" in msg and "cnn" in msg


def test_untuned_sets_take_the_all106_point_and_say_so(env):
    s = load("src/09_robustness.py")
    configs({"hgb:all106": {"a": 1}, "hgb:shape16": {"a": 2}}, {"cnn": {"width": 8}})
    p, cfg = s.frozen_configs()
    assert p["sizefree7"] == {"a": 1} and p["shape16"] == {"a": 2} and cfg == {"width": 8}
    assert s.params_from("holdout_hgb:sizefree7") == "hgb:all106" and s.params_from("holdout_hgb:all106") == "" and s.params_from("holdout_cnn:seed0") == ""


def test_late_grains_never_reach_a_holdout_fit():
    s = load("src/09_robustness.py")
    tr, va = frame(("early", "late", "late")), frame(("late", "early"))
    trh, vah = s.holdout_frames(tr, va)
    assert set(trh.lot) == {"early"} and set(vah.lot) == {"early"}
    jobs = s.table_jobs(tr, trh)
    assert all(set(r.lot) == {"early"} for st, _, r in jobs if st.startswith("hgb__"))
    assert all(len(r) == 3 for st, _, r in jobs if st.startswith("full_hgb__"))
    assert len(jobs) == 6


def preds(ids, p):
    return pd.DataFrame({"grain_id": ids, "y": 0} | {f"p{i}": [p if i == 0 else (1 - p) / 4] * len(ids) for i in range(5)})


def test_control_records_and_carries_on():
    s = load("src/09_robustness.py")
    te = frame(("late", "late"))
    P = np.tile([0.9, 0.025, 0.025, 0.025, 0.025], (2, 1))
    r = s.control_row(0, P, te, preds(["g1"], 0.8))
    assert r["n"] == 1 and r["max_abs_diff"] == pytest.approx(0.1) and r["argmax_equal"]
    r = s.control_row(0, P, te, preds(["other"], 0.8))
    assert r["n"] == 0 and np.isnan(r["max_abs_diff"])


def test_class_is_the_one_name():
    src = (ROOT / "src/09_robustness.py").read_text()
    assert "dict(cls=" not in src and '**{"class": C.CLASSES[k]}' in src


def test_paired_hand_example():
    s = load("src/07_compare.py")
    a, b = np.array([1, 1, 1, 1, 0, 1], bool), np.array([0, 0, 0, 1, 1, 1], bool)
    nb, nc, p, diff = s.paired(a, b, np.arange(6)[None, :])
    assert (nb, nc) == (3, 1) and p == pytest.approx(0.625) and diff[0] == pytest.approx(2 / 6)
    assert s.holm(np.array([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_only_holdout_rows_cover_every_late_grain():
    s = load("src/09_robustness.py")
    P, y, te = np.tile([0.9, 0.025, 0.025, 0.025, 0.025], (3, 1)), np.zeros(3, int), np.array([True, False, False])
    assert [r["subset"] for r in s.late_rows("full_hgb:all106", P, y, te)] == ["late_test"]
    assert [r["n"] for r in s.late_rows("holdout_hgb:all106", P, y, te)] == [3, 1]


@pytest.fixture
def fz(env, monkeypatch):
    f = load("tools/freeze.py")
    configs({"hgb:all106": {}, "nb:all106": {}}, {"cnn": {}, "mlp": {}})
    sp = env / "splits.csv"; sp.write_text("x\n")
    rt = C.MODELS / "routing.json"; rt.write_text("{}")
    monkeypatch.setattr(f, "PARTS", [C.RES / "baselines" / "best_params.json", C.RES / "neural" / "best_config.json", rt, sp])
    return f


def controls(base=0.2, neural=("cnn", "mlp")):
    pd.DataFrame({"model": ["hgb", "nb"], "featureset": "all106", "val_accuracy": base}).to_csv(C.RES / "baselines" / "control.csv", index=False)
    pd.DataFrame({"model": list(neural), "val_accuracy": 0.2}).to_csv(C.RES / "neural" / "control.csv", index=False)


def test_freeze_passes_clean_controls(fz):
    controls()
    assert fz.main() == 0 and C.FROZEN.exists()


def test_freeze_refuses_missing_control(fz):
    assert fz.main() == 1 and not C.FROZEN.exists()


def test_freeze_refuses_failed_control(fz):
    controls(base=0.95)
    assert fz.main() == 1 and not C.FROZEN.exists()


def test_freeze_refuses_a_control_with_a_row_missing(fz):
    controls(neural=("cnn",))
    assert fz.main() == 1 and not C.FROZEN.exists()


def test_freeze_skips_controls_in_smoke(fz, monkeypatch, capsys):
    monkeypatch.setattr(C, "SMOKE", True)
    assert fz.main() == 0 and "controls not checked" in capsys.readouterr().out


def rows_for(y, conf=0.9):
    P = np.full((len(y), 5), (1 - conf) / 4)
    P[np.arange(len(y)), y] = conf
    return P


def routing(floors=(0.7, 0.95, 0.7), temps=(1.0, 1.0, 3.0)):
    seeds = {str(s): {"temperature": t, "floor": f} for s, (t, f) in enumerate(zip(temps, floors))}
    return {"hgb": {"temperature": 1.0, "floor": 0.99}, "cnn": seeds["0"], "cnn_seeds": seeds}


def test_fit_stores_every_seed_and_keeps_the_keys_predict_reads(monkeypatch):
    s = load("src/08_routing.py")
    y = np.array([0, 1, 2, 3])
    Pcs = {k: rows_for(y, c) for k, c in enumerate((0.9, 0.8, 0.7))}
    seen = []
    monkeypatch.setattr(C, "temperature", lambda P, yy: seen.append(yy) or float(P.max()))    # stands in for the optimiser
    R = s.fit_routes(rows_for(y, 0.6), Pcs, y)
    assert [R["cnn_seeds"][k]["temperature"] for k in "012"] == pytest.approx([0.9, 0.8, 0.7])
    assert R["cnn"] == R["cnn_seeds"]["0"] and R["cnn"]["floor"] <= 1.01
    assert R["hgb"]["temperature"] == pytest.approx(0.6) and "floor" in R["hgb"]
    assert len(seen) == 4 and all(v is y for v in seen)


def test_route_table_has_one_block_per_seed_on_its_own_numbers(tmp_path, monkeypatch):
    s = load("src/08_routing.py")
    monkeypatch.setattr(C, "savefig", lambda fig, name: s.plt.close(fig))
    y = np.array([0, 1, 2, 3])
    Pcs = {k: rows_for(y) for k in C.SEEDS}
    unf_t = np.array([False, False, False, True])
    s.report("test", y, rows_for(y, 0.4), Pcs, routing(), unf_t, np.zeros(4, bool), tmp_path)
    t = pd.read_csv(tmp_path / "test_routes.csv")
    assert list(t.columns) == "split seed route grains share errors accuracy".split()
    assert list(t.seed.drop_duplicates()) == [0, 1, 2] and t.groupby("seed").size().eq(8).all()
    g = t.set_index(["seed", "route"]).grains
    assert g[0, "cnn"] == 3 and g[0, "unsure"] == 0            # seed 0 answers
    assert g[1, "cnn"] == 0 and g[1, "unsure"] == 3            # seed 1 is held back by its own floor
    assert g[2, "cnn"] == 0 and g[2, "unsure"] == 3            # seed 2 is softened by its own temperature
    assert (t[t.route == "unfamiliar"].grains == 1).all()
    c = pd.read_csv(tmp_path / "test_calibration.csv").set_index("model").temperature
    assert c["cnn_seed2"] == 3.0 and c["cnn_seed0"] == 1.0


def test_fit_stops_on_a_seed_with_no_validation_predictions(env, monkeypatch):
    s = load("src/08_routing.py")
    for p in s.fit_needs()[:-1]:
        p.parent.mkdir(exist_ok=True); p.touch()
    msg = refuse(s, ["08", "--fit"], monkeypatch)
    assert "cannot fit" in msg and "cnn__image__seed2__val" in msg and "seed1" not in msg


def test_score_stops_before_the_read_on_a_routing_file_with_a_seed_missing(env, monkeypatch):
    s = load("src/08_routing.py")
    for p in s.read_needs():
        p.parent.mkdir(parents=True, exist_ok=True); p.touch()
    R = routing()
    del R["cnn_seeds"]["1"]
    (C.MODELS / "routing.json").write_text(json.dumps(R))
    assert "seeds [1]" in refuse(s, ["08", "--score"], monkeypatch)
    del R["cnn_seeds"]
    (C.MODELS / "routing.json").write_text(json.dumps(R))
    assert "seeds [0, 1, 2]" in refuse(s, ["08", "--score"], monkeypatch)


RUN = {"width": 8, "dropout": 0.1, "lr": 0.001, "epochs": 5, "patience": 2}


def test_a_saved_seed_counts_only_from_the_same_run():
    s = load("src/09_robustness.py")
    ck = {"run": RUN, "train_ids": ["a", "b"]}
    assert s.same_run(ck, dict(RUN), ["a", "b"])
    assert not s.same_run(ck, RUN | {"lr": 0.01}, ["a", "b"])
    assert not s.same_run(ck, RUN | {"epochs": 6}, ["a", "b"])
    assert not s.same_run(ck, RUN, ["a", "late"])
    assert not s.same_run({"train_ids": ["a", "b"]}, RUN, ["a", "b"])   # a file with no run record


def test_a_missing_or_torn_file_is_never_reused(tmp_path):
    import torch
    s = load("src/09_robustness.py")
    p = tmp_path / "cnn_seed0.pt"
    assert not s.reusable(p, RUN, ["a"])
    torch.save({"run": RUN, "train_ids": ["a"]}, p)
    assert s.reusable(p, RUN, ["a"]) and not s.reusable(p, RUN, ["b"])
    p.write_bytes(p.read_bytes()[:40])
    assert not s.reusable(p, RUN, ["a"])


def test_holdout_cnn_trains_only_when_it_must(tmp_path, monkeypatch):
    s = load("src/09_robustness.py")
    trained = []

    class Net:
        def state_dict(self): return {}

    def train_cnn(cfg, seed, *a, **k):      # stands in for training: no model exists here
        trained.append(cfg["lr"])
        return Net(), [], 0.5, 0.2

    N = SimpleNamespace(EP_CNN=5, PAT_CNN=2, train_cnn=train_cnn)
    monkeypatch.setattr(C, "load_cnn", lambda p: ("from file", 0.5, 0.2))
    cfg, rows, p = {"width": 8, "dropout": 0.1, "lr": 0.001}, frame(("early", "early")), tmp_path / "cnn_seed0.pt"
    assert s.holdout_cnn(N, 0, cfg, rows, rows, None, p)[0] != "from file" and trained == [0.001]    # clean run: trains and saves
    assert s.holdout_cnn(N, 0, cfg, rows, rows, None, p)[0] == "from file" and trained == [0.001]    # rerun after a crash: reused
    assert s.holdout_cnn(N, 0, cfg | {"lr": 0.01}, rows, rows, None, p)[0] != "from file" and trained == [0.001, 0.01]   # other settings
    assert s.holdout_cnn(N, 0, cfg, frame(("early",)), rows, None, p)[0] != "from file" and trained == [0.001, 0.01, 0.001]   # other rows


def rows_of(tag, lots, split):
    f = pd.DataFrame({"grain_id": [f"{tag}{i}" for i in range(len(lots))], "lot": list(lots), "y": 0, "path": "/nowhere/a.jpg", "split": split})
    return pd.concat([f, pd.DataFrame(0.0, index=f.index, columns=C.ALL106)], axis=1)


class Saved:
    """Stands in for every saved model and gate score_late loads: no model exists here."""
    def __getitem__(self, _): return self
    def predict_proba(self, X): return rows_for(np.zeros(len(X), int))
    def unfamiliar(self, X): return np.zeros(len(X), bool)


def drive_late(monkeypatch, tr, va, te):
    """Runs the real score_late on made-up frames and returns the frame it hands to grey_world."""
    s, N = load("src/09_robustness.py"), load("src/08_routing.py")
    monkeypatch.setattr(s.joblib, "load", lambda p: Saved())
    monkeypatch.setattr(C, "cnn_on_df", lambda p, df: (rows_for(np.zeros(len(df), int)), np.zeros((len(df), 2))))
    hd = s.hold_dir()
    (hd / "routing.json").write_text(json.dumps({k: {"temperature": 1.0, "floor": 0.5} for k in ("hgb", "cnn")}))
    grabbed = []
    monkeypatch.setattr(s, "grey_world", lambda first, *rest: grabbed.append(first))
    s.score_late(N, hd, C.stage_dir("robustness"), tr, va, te)
    return grabbed[0]


def test_grey_world_is_handed_the_late_test_grains_at_the_call_site(env, monkeypatch):
    tr, va, te = rows_of("tr", ("early", "late"), "train"), rows_of("va", ("late", "late"), "val"), rows_of("te", ("late", "early", "late"), "test")
    assert set(drive_late(monkeypatch, tr, va, te).grain_id) == {"te0", "te2"}


def test_grey_world_gets_the_late_stand_ins_in_smoke(env, monkeypatch):
    tr, va = rows_of("tr", ("early", "late"), "train"), rows_of("va", ("late", "early"), "val")
    assert set(drive_late(monkeypatch, tr, va, va.copy()).grain_id) == {"va0"}


def test_a_scored_grain_in_train_stops_the_tint_arms(env, monkeypatch):
    tr, va = rows_of("tr", ("early", "late"), "train"), rows_of("va", ("late",), "val")
    te = pd.concat([rows_of("te", ("late",), "test"), tr.tail(1)], ignore_index=True)
    with pytest.raises(AssertionError, match="a scored grain is in train"):
        drive_late(monkeypatch, tr, va, te)


def test_each_saved_table_record_holds_the_rows_it_was_fitted_on(env, monkeypatch):
    s = load("src/09_robustness.py")
    tr, trh = rows_of("tr", ("early", "late", "late"), "train").assign(y=[1, 2, 3]), rows_of("h", ("early", "early"), "train").assign(y=[7, 8])
    name = dict(zip(pd.concat([tr, trh]).y, pd.concat([tr, trh]).grain_id))
    fitted, saved = [], {}

    class Pipe:                                     # stands in for a model: nothing is fitted
        def fit(self, X, y):
            fitted.append([name[v] for v in y])
            return self

    monkeypatch.setattr(s.joblib, "dump", lambda rec, p: saved.update({p.stem: rec}))
    s.fit_tables(SimpleNamespace(build=lambda kind, params: Pipe()), s.table_jobs(tr, trh), dict.fromkeys(s.HOLD_FS, {}), C.MODELS)
    assert [saved[k]["train_ids"] for k in saved] == fitted and len(saved) == 6
    assert saved["hgb__all106"]["train_ids"] == ["h0", "h1"] and saved["full_hgb__sizefree7"]["train_ids"] == ["tr0", "tr1", "tr2"]
    assert saved["hgb__sizefree7"]["params_from"] == "hgb:all106" and saved["hgb__shape16"]["params_from"] == "hgb:shape16"


def test_a_saved_cnn_record_holds_the_rows_it_was_fitted_on(tmp_path):
    import torch
    s = load("src/09_robustness.py")
    fitted = []

    class Net:
        def state_dict(self): return {}

    def train_cnn(cfg, seed, trh, *a, **k):         # stands in for training: no model exists here
        fitted.append(trh.grain_id.tolist())
        return Net(), [], 0.5, 0.2

    N = SimpleNamespace(EP_CNN=5, PAT_CNN=2, train_cnn=train_cnn)
    rows, p = frame(("early", "early")), tmp_path / "cnn_seed0.pt"
    s.holdout_cnn(N, 0, {"width": 8, "dropout": 0.1, "lr": 0.001}, rows, frame(("early",)), None, p)
    assert torch.load(p, weights_only=False)["train_ids"] == fitted[0] == ["g0", "g1"]


def test_late_recall_says_which_arm_borrowed_its_point(env, monkeypatch):
    drive_late(monkeypatch, rows_of("tr", ("early",), "train"), rows_of("va", ("early",), "val"), rows_of("te", ("late",), "test"))
    r = pd.read_csv(C.RES / "robustness" / "late_recall.csv", keep_default_na=False).set_index(["model", "subset"]).params_from
    assert r["holdout_hgb:sizefree22", "late_all"] == "hgb:all106" and r["full_hgb:sizefree7", "late_test"] == "hgb:all106"
    assert r["holdout_hgb:all106", "late_all"] == "" and r["holdout_cnn:seed0", "late_all"] == ""


def test_paired_gives_what_the_inline_lines_gave():
    from scipy.stats import binomtest
    s = load("src/07_compare.py")
    rng = np.random.default_rng(0)
    cases = [(np.array([1, 1, 1, 1, 0, 1], bool), np.array([0, 0, 0, 1, 1, 1], bool)), (np.ones(6, bool), np.ones(6, bool))]
    cases += [(rng.random(40) < 0.8, rng.random(40) < 0.7) for _ in range(5)]
    for a, b_ in cases:
        idx = rng.integers(0, len(a), (50, len(a)))
        ok, ka, kb = {"a": a, "b": b_}, "a", "b"
        b = int((ok[ka] & ~ok[kb]).sum()); c = int((~ok[ka] & ok[kb]).sum())           # src/07_compare.py at 8c7c592, lines 230-232
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        diff = ok[ka][idx].mean(1) - ok[kb][idx].mean(1)
        nb, nc, np_, nd = s.paired(a, b_, idx)
        assert (nb, nc, np_) == (b, c, p) and (nd == diff).all()


STAGES = ["src/07_compare.py", "src/08_routing.py", "src/09_robustness.py"]


@pytest.mark.parametrize("stage", STAGES)
def test_an_infinite_cell_is_missing_once_the_read_returns(stage, monkeypatch, capsys):
    te = rows_of("te", ("late", "early"), "test")
    te.loc[0, "AREA"], te.loc[1, "MINOR_AXIS"] = np.inf, -np.inf
    monkeypatch.setattr(C, "load_split", lambda name: te)
    got = load(stage).read_test()
    X = got[C.ALL106]
    assert not np.isinf(X.to_numpy(np.float64)).any() and int(X.isna().sum().sum()) == 2
    assert np.isnan(got.loc[0, "AREA"]) and got.lot.tolist() == ["late", "early"]
    assert "2 infinite cells made missing" in capsys.readouterr().out


@pytest.mark.parametrize("stage", STAGES)
def test_a_stage_reads_test_in_one_place(stage):
    tree = ast.parse((ROOT / stage).read_text())
    at = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) for n in ast.walk(f)
          if isinstance(n, ast.Call) and ast.unparse(n) == "C.load_split('test')"]
    assert at == ["read_test"]


def made_preds(y, miss):
    """A prediction frame right on every grain but the first miss."""
    y = np.asarray(y)
    said = (y + (np.arange(len(y)) < miss)) % 5
    return pd.DataFrame({"grain_id": [f"g{i}" for i in range(len(y))], "y": y} | {f"p{k}": np.where(said == k, 0.9, 0.025) for k in range(5)})


@pytest.mark.parametrize("y,cnn_misses", [([0, 1, 2, 3, 4, 0], 0), ([0, 1, 2, 3, 0, 1], 2), ([0, 1, 2, 3, 4, 0], 6), ([3], 0), ([3], 1)],
                         ids=["no disagreements", "class 4 absent", "one model wrong on every grain", "one grain right", "one grain wrong"])
def test_stage7_tables_stand_on_a_thin_read(y, cnn_misses, tmp_path, monkeypatch):
    s = load("src/07_compare.py")
    monkeypatch.setattr(s, "NBOOT", 20)
    monkeypatch.setattr(C, "FIG", tmp_path / "no" / "such" / "folder")
    keys = [("hgb", "all106", 0)] + [(m, fs, k) for m, fs in (("cnn", "image"), ("mlp", "all106")) for k in C.SEEDS]
    s.tables({k: made_preds(y, cnn_misses if k[0] == "cnn" else 0) for k in keys}, "test", tmp_path)
    M, T = pd.read_csv(tmp_path / "test_metrics.csv"), pd.read_csv(tmp_path / "test_mcnemar.csv")
    assert len(M) == 7 and len(T) == 3 * len(C.PAIRS)
    assert len(pd.read_csv(tmp_path / "test_confusion.csv")) == 7 * 25 and len(pd.read_csv(tmp_path / "test_recall.csv")) == 7 * 5
    assert len(pd.read_csv(tmp_path / "test_seed_spread.csv")) == 3
    assert (T[T.disagreements == 0].p_exact == 1.0).all() and T.p_holm.between(0, 1).all()
    assert (M[M.model == "cnn"].errors == cnn_misses).all() and (tmp_path / "no/such/folder/compare_confusion_test.png").exists()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("y,conf,hgb_floor,unfamiliar,route", [
    ([0, 1, 2, 3], 0.9, 0.0, False, "hgb"), ([0, 1, 2, 3], 0.9, 0.99, True, "unfamiliar"), ([0, 1, 2, 3], 1.0, 0.0, False, "hgb"), ([2], 0.9, 0.0, False, "hgb"),
    ([0, 1, 2, 3], 0.3, 0.99, False, "unsure")], ids=["all on hgb", "all unfamiliar", "confidence exactly 1", "one grain", "all unsure"])
def test_stage8_report_stands_with_empty_routes(y, conf, hgb_floor, unfamiliar, route, tmp_path, monkeypatch):
    s = load("src/08_routing.py")
    monkeypatch.setattr(C, "FIG", tmp_path / "no" / "such" / "folder")
    y = np.array(y)
    R = routing()
    R["hgb"]["floor"] = hgb_floor
    s.report("test", y, rows_for(y, conf), {k: rows_for(y, conf) for k in C.SEEDS}, R, np.full(len(y), unfamiliar), np.zeros(len(y), bool), tmp_path)
    t = pd.read_csv(tmp_path / "test_routes.csv")
    assert len(t) == 8 * len(C.SEEDS) and len(pd.read_csv(tmp_path / "test_calibration.csv")) == 1 + len(C.SEEDS)
    routes = t[t.route.isin(["hgb", "cnn", "unsure", "unfamiliar"])]
    assert (routes[routes.route == route].grains == len(y)).all() and (routes[routes.route != route].grains == 0).all()
    assert routes[routes.route.isin(["hgb", "cnn"]) & (routes.grains == 0)].accuracy.isna().all()
    assert all((tmp_path / "no/such/folder" / f"routing_{n}_test.png").exists() for n in ("risk_coverage", "reliability"))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_read_with_no_late_grain_in_test_leaves_empty_rows(env, monkeypatch):
    tr, va, te = rows_of("tr", ("early", "late"), "train"), rows_of("va", ("late",), "val"), rows_of("te", ("early",), "test")
    assert drive_late(monkeypatch, tr, va, te).empty
    r = pd.read_csv(C.RES / "robustness" / "late_recall.csv")
    t = r[r.subset == "late_test"]
    assert len(t) and (t.n == 0).all() and t.recall.isna().all() and t.ci_low.isna().all()
    c = pd.read_csv(C.RES / "robustness" / "late_chain.csv")
    assert (c[c.subset == "late_test"].grains == 0).all() and (c[c.subset == "late_all"].grains.max() == 2)


def test_every_late_grain_on_one_route_and_a_missing_cell_stand(env, monkeypatch):
    tr, va, te = rows_of("tr", ("early", "late"), "train"), rows_of("va", ("late",), "val"), rows_of("te", ("late", "late"), "test")
    te.loc[0, "AREA"] = np.nan
    assert len(drive_late(monkeypatch, tr, va, te)) == 2
    c = pd.read_csv(C.RES / "robustness" / "late_chain.csv").set_index(["subset", "route"])
    assert c.loc[("late_test", "hgb"), "grains"] == 2 and c.loc[("late_test", "cnn"), "grains"] == 0 and c.loc[("late_test", "answered_total"), "errors"] == 0


def test_empty_inputs_give_a_stated_answer():
    s = load("src/09_robustness.py")
    assert all(np.isnan(C.wilson(0, 0)))
    assert C.floor_for(np.array([]), np.array([], bool), 0.99) == 1.01 and C.ece(np.array([]), np.array([], bool)) == 0.0
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        r = s.recall_row("m", "late_test", np.zeros((0, 5)), np.zeros(0, int))
    assert r["n"] == 0 and np.isnan(r["recall"]) and r["confidently_wrong"] == 0


def test_missing_output_folders_are_made(env, monkeypatch):
    C.save_preds("cnn", "image", 0, "test", ["g0"], [0], rows_for(np.array([0])))
    assert C.load_preds("cnn", "image", 0, "test").grain_id.tolist() == ["g0"]
    assert load("src/09_robustness.py").hold_dir().is_dir() and C.stage_dir("robustness").is_dir()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_channel_with_no_answer_fails_the_gain_check():
    s = load("src/09_robustness.py")
    area = np.exp(np.arange(1.0, 9.0))
    even = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])       # no slope against log area
    samp = pd.DataFrame({"y": [0] * 8 + [1] * 8, "AREA": np.r_[area, area]})
    G = np.column_stack([np.r_[even, even], np.r_[np.full(8, 2.0), even], np.r_[even, even]])     # class 0: green has no spread, so no answer
    a, b, *rest = s.gain_rows(samp, G)
    assert a["n"] == 8 and np.isnan(a["r_G"]) and np.isnan(a["max_abs_r"]) and not a["passes_abs_lt_0_1"]
    assert b["max_abs_r"] < 0.1 and b["passes_abs_lt_0_1"]
    assert [r["n"] for r in rest] == [0, 0, 0] and not any(r["passes_abs_lt_0_1"] for r in rest)     # a class with no rows


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_no_late_test_grain_keeps_the_gain_tables_and_stops_before_the_tint_arms(env, monkeypatch):
    s = load("src/09_robustness.py")

    def photo(_):
        a = np.zeros((250, 250, 3), np.uint8)
        a[80:170, 60:190] = (200, 170, 140)
        return a

    class NoFit:
        def __init__(self, *a, **k): pytest.fail("the tint arms went on to fit a model")
    monkeypatch.setattr(s, "read250", photo)
    monkeypatch.setattr("sklearn.ensemble.HistGradientBoostingClassifier", NoFit)
    tr = rows_of("tr", ("early",) * 5, "train").assign(y=range(5))
    va = rows_of("va", ("late",) * 5, "val").assign(y=range(5))
    out = C.stage_dir("robustness")
    s.grey_world(tr.iloc[:0], tr, va, out)
    assert len(pd.read_csv(out / "greyworld_gain_vs_area.csv")) == 5 and len(pd.read_csv(out / "mean_colour_check.csv")) == 3
    assert not (out / "greyworld_results.csv").exists()


def test_stage9_stops_before_the_read_with_no_late_rows(env, monkeypatch):
    def split(name):
        if name == "test": pytest.fail("test rows were read")
        return frame(("early",))
    monkeypatch.setattr(C, "load_split", split)
    assert "no late grains" in refuse(load("src/09_robustness.py"), ["09", "--score"], monkeypatch)
