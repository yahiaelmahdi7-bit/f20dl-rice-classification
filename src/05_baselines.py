"""Stage 5: table baselines tuned on validation macro-F1; saves models and validation predictions."""
import argparse, itertools, json
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

GRIDS = {
    "nb": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    "gmm": {"n_components": [1, 2, 4, 8]},
    "tree": {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 5, 20]},
    "logreg": {"C": [0.01, 0.1, 1, 10, 100]},
    "hgb": {"learning_rate": [0.05, 0.1], "max_iter": [150, 300], "max_leaf_nodes": [31, 63]},
}
if C.SMOKE:
    GRIDS = {"nb": {"var_smoothing": [1e-9, 1e-5]}, "gmm": {"n_components": [1, 2]}, "tree": {"max_depth": [5], "min_samples_leaf": [5]},
             "logreg": {"C": [1]}, "hgb": {"learning_rate": [0.1], "max_iter": [30], "max_leaf_nodes": [31]}}
FSETS = ["shape16", "colour90", "all106", "selected"]
JOBS = [(m, f) for f in FSETS for m in GRIDS] + [("nb", "distinct91")]


def points(m):
    k = list(GRIDS[m])
    return [dict(zip(k, v)) for v in itertools.product(*GRIDS[m].values())]


def build(m, p):
    est = {"nb": lambda: GaussianNB(**p), "gmm": lambda: C.ClassGMM(pca=20, seed=C.SEED, **p),
           "tree": lambda: DecisionTreeClassifier(random_state=C.SEED, **p),
           "logreg": lambda: LogisticRegression(max_iter=300, **p),
           "hgb": lambda: HGB(random_state=C.SEED, early_stopping=False, **p)}[m]()
    return C.Pipeline([("imp", C.SimpleImputer(strategy="median")), ("sc", C.StandardScaler()), ("m", est)])


def merge_csv(path, new, keys):
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.set_index(keys).index.isin(new.set_index(keys).index)]
        new = pd.concat([old, new])
    new.to_csv(path, index=False)
    C.wrote(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    out = C.stage_dir("baselines")
    (C.MODELS / "baselines").mkdir(parents=True, exist_ok=True)
    tr, va = C.load_split("train"), C.load_split("val")
    jobs = [tuple(a.only.split(":"))] if a.only else JOBS
    best_path = out / "best_params.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    grid_rows, res_rows, ctl = [], [], []
    for m, fs in jobs:
        Xtr, ytr = C.xy(tr, fs)
        Xva, yva = C.xy(va, fs)
        if a.control:
            yshuf = np.random.default_rng(C.SEED).permutation(ytr)
            pipe = build(m, points(m)[0]).fit(Xtr, yshuf)
            ctl.append({"model": m, "featureset": fs, "val_accuracy": float((pipe.predict(Xva) == yva).mean())})
            print(ctl[-1], flush=True)
            continue
        top = None
        for p in points(m):
            pipe = build(m, p).fit(Xtr, ytr)
            pr = pipe.predict(Xva)
            f1 = C.macro_f1(yva, pr)
            grid_rows.append({"model": m, "featureset": fs, "params": json.dumps(p), "val_macro_f1": f1})
            if top is None or f1 > top[0]: top = (f1, p, pipe, pr)
        f1, p, pipe, pr = top
        P = pipe.predict_proba(Xva)
        C.save_preds(m, fs, 0, "val", va.grain_id, yva, P)
        obj = {"pipe": pipe, "cols": C.features(fs), "params": p}
        joblib.dump(obj, C.MODELS / "baselines" / f"{m}__{fs}.joblib")
        if (m, fs) == ("hgb", "all106"): joblib.dump(obj, C.MODELS / "hgb_all106.joblib")
        best[f"{m}:{fs}"] = p
        res_rows.append({"model": m, "featureset": fs, "val_accuracy": float((pr == yva).mean()), "val_macro_f1": f1, "params": json.dumps(p)})
        print(res_rows[-1], flush=True)
    if a.control:
        merge_csv(out / "control.csv", pd.DataFrame(ctl), ["model", "featureset"])
        return
    best_path.write_text(json.dumps(best, indent=1, sort_keys=True))
    merge_csv(out / "grid.csv", pd.DataFrame(grid_rows), ["model", "featureset", "params"])
    merge_csv(out / "val_results.csv", pd.DataFrame(res_rows), ["model", "featureset"])
    # depth-4 tree on all106 for the practical-applications discussion
    imp = C.SimpleImputer(strategy="median").fit(tr[C.ALL106])
    t4 = DecisionTreeClassifier(max_depth=4, random_state=C.SEED).fit(imp.transform(tr[C.ALL106]), tr.y)
    (out / "tree_depth4.txt").write_text(export_text(t4, feature_names=C.ALL106))
    C.wrote(out / "tree_depth4.txt")
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(t4, feature_names=C.ALL106, class_names=C.CLASSES, filled=True, fontsize=6, ax=ax)
    ax.set_title(f"Depth-4 decision tree, all106 (train n={len(tr)}); axes are not data axes")
    C.savefig(fig, "baselines_tree_depth4")


if __name__ == "__main__":
    main()
