"""Independent recheck of R2: refit every cluster row and recompute stage 3 accuracies, then compare with the saved CSVs.

Imports neither stage. Train and validation rows only, through common.load_grains and common.load_split.
Usage: python tools/recheck_r2.py [--only clusters|features]. Exit 1 if any value is outside its tolerance.
Writes results/checks/recheck_clusters.csv, recheck_features.csv and recheck_summary.csv.

Tolerances: a seeded refit should agree to machine rounding, so the limits are near that and a changed fit cannot pass.
  silhouette, ARI, NMI: 1e-6 absolute. pca_cum_var: 1e-9 absolute. inertia, bic: 1e-6 relative.
  accuracy, macro F1: one grain of the scored rows, 1/n.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C  # noqa: E402

TOL = {"silhouette": ("abs", 1e-6), "silhouette_mean": ("abs", 1e-6), "silhouette_sd": ("abs", 1e-6), "ARI": ("abs", 1e-6), "NMI": ("abs", 1e-6),
       "pca_cum_var": ("abs", 1e-9), "inertia": ("rel", 1e-6), "bic": ("rel", 1e-6)}
SETS, KS, N_SIL = ["all106", "distinct91", "sizefree22"], range(2, 11), 5
COLS = ["table", "key", "metric", "saved", "recomputed", "abs_diff", "limit", "passes"]


def check(table, key, metric, saved, got, tol):
    """One comparison row. A missing value on either side fails, because NaN is never within a limit."""
    kind, t = tol
    diff = abs(float(saved) - float(got))
    limit = t * max(abs(float(saved)), 1.0) if kind == "rel" else t
    return dict(table=table, key=key, metric=metric, saved=float(saved), recomputed=float(got), abs_diff=diff, limit=limit, passes=bool(diff <= limit))


def refit_clusters(g, saved):
    """Fresh fits for every row of cluster_metrics.csv: train-fitted scaling and PCA, clusters on train and validation rows."""
    tr, y, out = (g.split == "train").to_numpy(), g.y.to_numpy(), []
    first, more = np.random.default_rng(C.SEED), np.random.default_rng(C.SEED + 1)
    for fs in SETS:
        cols = C.features(fs)
        prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(min(20, len(cols)), random_state=C.SEED)).fit(g.loc[tr, cols])
        Z, cum = prep.transform(g[cols]), float(prep[-1].explained_variance_ratio_.sum())
        s0 = first.choice(len(Z), min(10000, len(Z)), replace=False)
        samples = [s0] + [more.choice(len(Z), len(s0), replace=False) for _ in range(N_SIL - 1)]
        for k in KS:
            for method in ("kmeans", "gmm"):
                if method == "kmeans":
                    m = KMeans(k, n_init=3 if C.SMOKE else 10, random_state=C.SEED).fit(Z)
                    lab, got = m.labels_, {"inertia": m.inertia_}
                else:
                    m = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, n_init=3 if C.SMOKE else 5, random_state=C.SEED).fit(Z)
                    lab, got = m.predict(Z), {"bic": m.bic(Z)}
                sil = [silhouette_score(Z[s], lab[s]) for s in samples]
                got |= {"silhouette": sil[0], "silhouette_mean": np.mean(sil), "silhouette_sd": np.std(sil, ddof=1), "pca_cum_var": cum,
                        "ARI": adjusted_rand_score(y, lab), "NMI": normalized_mutual_info_score(y, lab)}
                row = saved[(saved.featureset == fs) & (saved.method == method) & (saved.k == k)]
                key = f"{fs}/{method}/k={k}"
                if len(row) != 1:
                    out.append(check("cluster_metrics.csv", key, "row_count", 1, len(row), ("abs", 0))); continue
                out += [check("cluster_metrics.csv", key, n, row[n].iloc[0], v, TOL[n]) for n, v in got.items() if n in saved.columns]
                print(key, "checked", flush=True)
    return out


def model(name):
    if name == "hgb": return HistGradientBoostingClassifier(max_iter=20 if C.SMOKE else 200, random_state=C.SEED)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300))


def recompute_features(tr, va, F):
    """Stage 3 accuracies from its saved column order and saved lists."""
    reps = pd.read_csv(F / "cluster_representatives.csv").feature.tolist()
    order = reps + [c for c in pd.read_csv(F / "ranking.csv").feature if c not in reps]
    sel = json.loads((F / "selected.json").read_text())
    acc = pd.read_csv(F / "accuracy_vs_k.csv")
    top5 = "ROUNDNESS COMPACTNESS SHAPEFACTOR_3 ASPECT_RATIO ECCENTRICITY".split()
    one = ("abs", 1.0 / len(va))
    out = [check("selected.json", "selected", "is_first_k_of_saved_order", 1, float(sel["selected"] == order[:sel["k"]]), ("abs", 0))]

    def fit(name, cs):
        pred = model(name).fit(tr[cs], tr.y).predict(va[cs])
        return float((pred == va.y).mean()), float(f1_score(va.y, pred, average="macro", labels=range(len(C.CLASSES)), zero_division=0))

    for _, r in acc.iterrows():
        cs = top5 if r.k == "authors_top5" else order[:int(r.k)]
        for name in ("hgb", "logreg"):
            out.append(check("accuracy_vs_k.csv", f"k={r.k}", f"{name}_val_accuracy", r[f"{name}_val_accuracy"], fit(name, cs)[0], one))
        print(f"k={r.k} checked", flush=True)
    cmp = pd.read_csv(F / "set_comparison.csv")
    named = {"selected": sel["selected"], "authors_top5": top5}
    for _, r in cmp.iterrows():
        a, f = fit(r.model, named.get(r.featureset) or C.features(r.featureset))
        key = f"{r.featureset}/{r.model}"
        out += [check("set_comparison.csv", key, "val_accuracy", r.val_accuracy, a, one), check("set_comparison.csv", key, "val_macro_f1", r.val_macro_f1, f, one)]
        print(key, "checked", flush=True)
    return out


def summary(name, rows):
    d = pd.DataFrame(rows, columns=COLS)
    return {"check": name, "rows": len(d), "failed": int((~d.passes.astype(bool)).sum()), "max_abs_diff": d.abs_diff.max(), "passes": bool(len(d) and d.passes.all())}


def main(argv=None):
    a = argparse.ArgumentParser(); a.add_argument("--only", choices=["clusters", "features"]); a = a.parse_args(argv)
    out, done = C.stage_dir("checks"), []
    if a.only != "features":
        saved = pd.read_csv(C.RES / "cluster" / "cluster_metrics.csv")
        rows = refit_clusters(C.load_grains(), saved)
        pd.DataFrame(rows, columns=COLS).to_csv(out / "recheck_clusters.csv", index=False); C.wrote(out / "recheck_clusters.csv")
        done.append(summary("clusters", rows))
    if a.only != "clusters":
        rows = recompute_features(C.load_split("train"), C.load_split("val"), C.RES / "features")
        pd.DataFrame(rows, columns=COLS).to_csv(out / "recheck_features.csv", index=False); C.wrote(out / "recheck_features.csv")
        done.append(summary("features", rows))
    s = pd.DataFrame(done); s.to_csv(out / "recheck_summary.csv", index=False); C.wrote(out / "recheck_summary.csv")
    print(s.to_string(index=False))
    return 0 if s.passes.all() else 1


if __name__ == "__main__":
    sys.exit(main())
