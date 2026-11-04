"""Regression tests for the R2 fix round. Slices of stage source run on made-up data; the end-to-end ones run in a temporary copy on synthetic grains."""
import json, os, pathlib, shutil, subprocess, sys
import numpy as np
import pandas as pd
import pytest
import common as C

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _block(rel, start, end):
    src = (ROOT / rel).read_text().split("PROMPT RECORD END ===", 1)[-1]
    i = src.index(start)
    return src[i:src.index(end, i)]


# ---- fix 4: silhouette on several samples
def test_first_silhouette_draw_keeps_its_old_stream():
    ns = {"np": np, "C": C, "len": len}
    exec(_block("src/04_cluster.py", "rng = np.random", "SETS ="), ns)
    body = _block("src/04_cluster.py", "    sidx = rng.choice", "    held +=")
    old = np.random.default_rng(C.SEED)
    ns["Z"] = np.zeros((20000, 1))
    for _ in range(3):                 # the second and third set are the ones extra draws would shift
        exec("if 1:\n" + body, ns)
        assert np.array_equal(ns["sidx"], old.choice(20000, 10000, replace=False))
        assert len(ns["draws"]) == ns["N_SIL"] > 1 and ns["draws"][0] is ns["sidx"]
        assert not np.array_equal(ns["draws"][1], ns["sidx"])


def test_silhouette_row_keeps_first_draw_and_adds_spread():
    vals = iter([0.5, 0.1, 0.3])
    ns = {"np": np, "silhouette_score": lambda a, b: next(vals)}
    exec(_block("src/04_cluster.py", "def sil_draws(", "for fs in SETS:"), ns)
    r = ns["sil_draws"](np.zeros(4), np.zeros(4), [0, 1, 2])
    assert r["silhouette"] == 0.5 and r["silhouette_mean"] == pytest.approx(0.3) and r["silhouette_sd"] == pytest.approx(0.2) and r["silhouette_draws"] == 3


# ---- synthetic repo: stages 3 and 4 and the recheck, in a temporary copy, on made-up grains
def _synth():
    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(5), 80)
    X = rng.normal(size=(400, 106)) + 20
    X[:, :20] += y[:, None] * 1.5
    g = pd.DataFrame(X, columns=C.ALL106)
    i = np.arange(400)
    g.insert(0, "split", np.array(["train", "train", "train", "val", "test"])[i % 5])
    g.insert(0, "grain_id", [f"g{k}" for k in i])
    g["class"], g["y"], g["file_no"], g["row"], g["path"], g["msc_row"] = np.array(C.CLASSES)[y], y, i, i, "", i
    g["lot"] = np.where((y == 3) & (i % 80 >= 50), "late", "main")
    t = g.split == "test"
    g.loc[t, C.ALL106] = 1e6       # any stage or check that read these rows would give different numbers
    g.loc[t, "y"] = 0
    return g


def _run(repo, *args):
    env = {k: v for k, v in os.environ.items() if k != "RICE_MODELS_DIR"} | {"RICE_SMOKE": "1", "OMP_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, *args], cwd=repo, env=env, capture_output=True, text=True)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    r = tmp_path_factory.mktemp("repo")
    for d in ("src", "tools"):
        shutil.copytree(ROOT / d, r / d, ignore=shutil.ignore_patterns("__pycache__"))
    (r / "data/derived/smoke").mkdir(parents=True)
    _synth().to_parquet(r / "data/derived/smoke/grains.parquet")
    for s in ("src/03_features.py", "src/04_cluster.py"):
        p = _run(r, s)
        assert p.returncode == 0, p.stderr[-2000:]
    return r


def test_recheck_agrees_with_the_stages_on_every_row(repo):
    p = _run(repo, "tools/recheck_r2.py")
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    c = pd.read_csv(repo / "results/smoke/checks/recheck_clusters.csv")
    assert c.key.nunique() == 54 and c.passes.all()
    assert {"silhouette", "silhouette_mean", "silhouette_sd", "ARI", "NMI", "inertia", "bic", "pca_cum_var"} <= set(c.metric)
    f = pd.read_csv(repo / "results/smoke/checks/recheck_features.csv")
    assert f.passes.all() and set(f.table) == {"selected.json", "accuracy_vs_k.csv", "set_comparison.csv"}


@pytest.mark.parametrize("rel,col,only", [("cluster/cluster_metrics.csv", "ARI", "clusters"), ("features/accuracy_vs_k.csv", "hgb_val_accuracy", "features")])
def test_recheck_fails_on_a_changed_saved_value(repo, rel, col, only):
    p = repo / "results/smoke" / rel
    keep = p.read_text()
    d = pd.read_csv(p)
    d.loc[0, col] = d.loc[0, col] - 0.05
    d.to_csv(p, index=False)
    try:
        r = _run(repo, "tools/recheck_r2.py", "--only", only)
    finally:
        p.write_text(keep)
    assert r.returncode == 1
    bad = pd.read_csv(repo / f"results/smoke/checks/recheck_{only}.csv").query("~passes")
    assert len(bad) == 1 and col in bad.metric.iloc[0]


def test_recheck_tolerance_is_stated_and_applied():
    sys.path.insert(0, str(ROOT / "tools"))
    import recheck_r2 as R
    assert R.check("t", "k", "ARI", 0.5, 0.5 + 5e-7, R.TOL["ARI"])["passes"]
    assert not R.check("t", "k", "ARI", 0.5, 0.5 + 2e-6, R.TOL["ARI"])["passes"]
    assert R.check("t", "k", "bic", 1e6, 1e6 + 0.5, R.TOL["bic"])["passes"] and not R.check("t", "k", "bic", 1e6, 1e6 + 2, R.TOL["bic"])["passes"]
    assert not R.check("t", "k", "ARI", float("nan"), 0.5, R.TOL["ARI"])["passes"]


def test_recheck_draws_the_same_silhouette_samples_as_stage_4_when_sampling_bites():
    """The synthetic repo has fewer rows than the sample size, so every draw holds all rows. This one has more rows than the sample."""
    sys.path.insert(0, str(ROOT / "tools"))
    import recheck_r2 as R
    ns = {"np": np, "C": C, "len": len, "min": min, "range": range}
    exec(_block("src/04_cluster.py", "rng = np.random", "SETS ="), ns)
    ns["Z"] = np.zeros((20000, 1))
    body = _block("src/04_cluster.py", "    sidx = rng.choice", "    held +=")
    first, more = R.sil_streams()
    for _ in range(3):
        exec("if 1:\n" + body, ns)
        got = R.sil_samples(20000, first, more)
        assert len(got) == len(ns["draws"]) and all(np.array_equal(a, b) for a, b in zip(got, ns["draws"]))


# ---- fix 6: held-out cluster score
def test_heldout_fits_on_train_rows_and_predicts_validation_rows():
    seen = {"fit": [], "predict": []}

    class Stub:
        def __init__(self, *a, **k): pass
        def fit(self, Z):
            seen["fit"].append(Z[:, 0].tolist()); return self
        def predict(self, Z):
            seen["predict"].append(Z[:, 0].tolist()); return (Z[:, 0] % 2).astype(int)

    sizes = []
    def agree(a, b):
        sizes.append(len(a)); return 1.0
    ns = {"KMeans": Stub, "GaussianMixture": Stub, "ARI": agree, "NMI": agree, "NINIT_GMM": 1, "C": C}
    exec(_block("src/04_cluster.py", "def heldout_rows(", "for fs in SETS:"), ns)
    trm = np.arange(10) < 7
    r = ns["heldout_rows"]("x", np.arange(10.0)[:, None], np.arange(10) % 3, trm)
    assert len(r) == 18 and {(d["k"], d["method"]) for d in r} == {(k, m) for k in range(2, 11) for m in ("kmeans", "gmm")}
    assert all(f == list(range(7)) for f in seen["fit"]) and len(seen["fit"]) == 18
    assert all(p in (list(range(7)), [7, 8, 9]) for p in seen["predict"]) and seen["predict"].count([7, 8, 9]) == 18
    assert r[0]["n_fit"] == 7 and r[0]["n_val"] == 3 and sizes[:2] == [3, 3]


def test_heldout_table_sits_beside_the_existing_rows(repo):
    h = pd.read_csv(repo / "results/smoke/cluster/heldout.csv")
    assert len(h) == 54 and set(h.columns) == set(C.ARTEFACTS["files"]["results/cluster/heldout.csv"]["columns"])
    assert h.ARI_val.notna().all() and (h.n_fit + h.n_val == 320).all() and (h.n_fit == 240).all()
    assert len(pd.read_csv(repo / "results/smoke/cluster/cluster_metrics.csv")) == 54


# ---- fix 7: k chosen on an inner split of train
def _rule_ns():
    ns = {}
    exec(_block("src/03_features.py", "def rule_k(", "ks = ["), ns)
    return ns


def _curves():
    return pd.DataFrame({"k": [1, 3, 8, 32], "hgb_val_accuracy": [.5, .7, .9, .9], "logreg_val_accuracy": [.5, .6, .7, .8],
                         "hgb_inner_accuracy": [.5, .9, .9, .8], "logreg_inner_accuracy": [.8, .8, .7, .7]})


def test_rule_k_is_the_smallest_near_best_k_then_the_larger_model():
    k, each = _rule_ns()["rule_k"](_curves(), "{}_val_accuracy")
    assert k == 32 and each == {"hgb": 8, "logreg": 32}


def test_selection_follows_the_inner_curve_and_keeps_the_old_rule_beside_it():
    ns = _rule_ns() | {"res": _curves(), "order": list("abcdefghijklmnopqrstuvwxyz") * 2}
    exec(_block("src/03_features.py", "k_old, k_each_old =", "best = float"), ns)
    assert ns["k_sel"] == 3 and ns["k_each"] == {"hgb": 3, "logreg": 1} and ns["selected"] == list("abc")
    assert ns["k_old"] == 32 and ns["k_each_old"]["logreg"] == 32


def test_sweep_inner_columns_fit_and_score_on_train_rows_only():
    cols = [f"c{i}" for i in range(91)]
    def frame(ids):
        d = pd.DataFrame(0.0, index=range(len(ids)), columns=cols)
        d["c0"], d["y"], d["class"], d["lot"] = ids, 0, np.array(C.CLASSES)[np.arange(len(ids)) % 5], "main"
        return d
    calls = []

    class Stub:
        def fit(self, X, y):
            self.f = set(X.c0); return self
        def predict(self, X):
            calls.append((self.f, set(X.c0))); return np.zeros(len(X), dtype=int)

    ns = {"tr": frame(list(range(100))), "va": frame(list(range(1000, 1020))), "order": cols, "C": C, "pd": pd, "hgb": Stub, "lr": Stub, "print": lambda *a, **k: None}
    exec(_block("src/03_features.py", "def score(", "res = pd.DataFrame(rows)"), ns)
    a, b = set(ns["inner_fit"].c0), set(ns["inner_score"].c0)
    assert a | b == set(range(100)) and not a & b
    n = 2 * len(ns["ks"])
    assert calls.count((set(range(100)), set(range(1000, 1020)))) == n and calls.count((a, b)) == n and len(calls) == 2 * n
    assert all(k in ns["rows"][0] for k in ("hgb_inner_accuracy", "logreg_inner_accuracy", "hgb_val_accuracy", "logreg_val_accuracy"))


def test_inner_split_is_stratified_seeded_and_covers_train_once():
    lots = np.array(["late"] * 10 + ["main"] * 490 + ["main"] * 500)
    tr = pd.DataFrame({"class": ["Jasmine"] * 500 + ["Ipsala"] * 500, "lot": lots, "grain_id": range(1000)})
    a, b = C.inner_split(tr)
    a2, b2 = C.inner_split(tr)
    assert a.grain_id.tolist() == a2.grain_id.tolist() and b.grain_id.tolist() == b2.grain_id.tolist()
    assert len(b) == 200 and set(a.grain_id) | set(b.grain_id) == set(range(1000)) and not set(a.grain_id) & set(b.grain_id)
    assert (b["class"] + b["lot"]).value_counts().to_dict() == {"Ipsalamain": 100, "Jasminemain": 98, "Jasminelate": 2}


def test_stage_3_saves_both_rules_and_recheck_covers_the_inner_columns(repo):
    s = json.loads((repo / "results/smoke/features/selected.json").read_text())
    assert set(s) == set(C.ARTEFACTS["files"]["results/features/selected.json"]["columns"])
    assert s["n_inner_fit"] + s["n_inner_score"] == 240 and len(s["selected_old_rule"]) == s["k_old_rule"] and len(s["selected"]) == s["k"]
    a = pd.read_csv(repo / "results/smoke/features/accuracy_vs_k.csv")
    assert a.loc[a.k == "authors_top5", "hgb_inner_accuracy"].isna().all() and a.loc[a.k != "authors_top5", "hgb_inner_accuracy"].notna().all()
    p = _run(repo, "tools/recheck_r2.py", "--only", "features")
    assert p.returncode == 0, p.stdout[-1500:]
    f = pd.read_csv(repo / "results/smoke/checks/recheck_features.csv")
    assert {"hgb_inner_accuracy", "logreg_inner_accuracy"} <= set(f.metric)
