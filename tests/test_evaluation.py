"""Stages 7, 8, 9 and freeze: refusals before a read, on made-up rows. No model is fitted and no real path is touched."""
import importlib.util, json, sys
from pathlib import Path
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
