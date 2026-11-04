"""Stage 3: feature ranking, correlation-cluster picks and the selected list (train and validation only)."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
import common as C

out = C.stage_dir("features")
tr, va = C.load_split("train"), C.load_split("val")
assert set(tr.split) == {"train"} and set(va.split) == {"val"}, "fit on train only, score on validation only"
cols = C.features("distinct91")


def build_order(df, seed):
    """Mutual information + ANOVA on df, correlation clusters of columns, one representative per cluster first."""
    sub = df.sample(min(len(df), 20000), random_state=seed)
    Xs = sub[cols].fillna(sub[cols].median())
    mi = mutual_info_classif(Xs, sub.y, random_state=seed)
    F, _ = f_classif(Xs, sub.y)
    rho = np.abs(spearmanr(Xs, nan_policy="omit").statistic)
    dist = np.clip((1 - rho + (1 - rho).T) / 2, 0, None)
    np.fill_diagonal(dist, 0)
    cl = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.05, criterion="distance")
    mi_s = pd.Series(mi, index=cols)
    reps = sorted([mi_s[[c for c, k in zip(cols, cl) if k == j]].idxmax() for j in sorted(set(cl))], key=lambda c: -mi_s[c])
    rank = pd.DataFrame({"feature": cols, "mutual_info": mi, "anova_F": F}).sort_values("mutual_info", ascending=False)
    return reps + [c for c in rank.feature if c not in reps], rank, reps, mi_s, cl


order, rank, reps, mi_s, cl = build_order(tr, C.SEED)
rank.to_csv(out / "ranking.csv", index=False); C.wrote(out / "ranking.csv")
pd.DataFrame({"feature": reps, "mutual_info": mi_s[reps].values}).to_csv(out / "cluster_representatives.csv", index=False)
print(f"{len(set(cl))} correlation clusters, one representative each")


def hgb():
    return HGB(max_iter=it, random_state=C.SEED)


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300))


def score(mk, cs):
    m = mk().fit(tr[cs], tr.y)
    pr = m.predict(va[cs])
    return float((pr == va.y).mean()), C.macro_f1(va.y.to_numpy(), pr)


ks = [1, 3, 8, 32, 91] if C.SMOKE else [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 91]
it = 20 if C.SMOKE else 200
rows = []
for k in ks:
    r = {"k": k}
    for nm, mk in (("hgb", hgb), ("logreg", lr)):
        r[f"{nm}_val_accuracy"] = score(mk, order[:k])[0]
    rows.append(r)
    print(r, flush=True)
res = pd.DataFrame(rows)
top5 = "ROUNDNESS COMPACTNESS SHAPEFACTOR_3 ASPECT_RATIO ECCENTRICITY".split()   # authors' top 5
k_each = {nm: int(res[res[f"{nm}_val_accuracy"] >= res[f"{nm}_val_accuracy"].max() - 0.001].k.min()) for nm in ("hgb", "logreg")}
k_sel = max(k_each.values())     # the larger k, so neither model is short of features
selected = order[:k_sel]
best = float(res.hgb_val_accuracy.max())
res["val_accuracy"] = res.hgb_val_accuracy
a5 = {nm: score(mk, top5)[0] for nm, mk in (("hgb", hgb), ("logreg", lr))}
res = pd.concat([res, pd.DataFrame([{"k": "authors_top5", "hgb_val_accuracy": a5["hgb"], "logreg_val_accuracy": a5["logreg"], "val_accuracy": a5["hgb"]}])])
res.to_csv(out / "accuracy_vs_k.csv", index=False); C.wrote(out / "accuracy_vs_k.csv")
(out / "selected.json").write_text(json.dumps({"selected": selected, "k": k_sel, "k_by_model": k_each, "best_val_accuracy": best}, indent=1))
C.wrote(out / "selected.json")
print(f"selected k={k_sel} (by model {k_each}): {selected}")

# selected columns: family and within-class correlation with log area
FAMS = ["ALLdaub4", "StdDev", "kurtosis", "entropy", "skew", "mean"]
la = np.log(tr.AREA)
info = []
for c in selected:
    fam = "shape" if c in C.ALL106[:16] else next((f for f in FAMS if c.startswith(f)), "other")
    cc = [tr.loc[tr["class"] == k, c].corr(la[tr["class"] == k]) for k in C.CLASSES]
    info.append({"feature": c, "family": fam, "median_within_class_corr_logarea": float(np.nanmedian(cc))})
pd.DataFrame(info).to_csv(out / "selected_info.csv", index=False); C.wrote(out / "selected_info.csv")

# stability: 5 bootstrap resamples of train, re-rank, compare top-k_sel with the main list
rng = np.random.default_rng(C.SEED)
st, cnt = [], {c: 0 for c in selected}
for b in range(5):
    bs = tr.iloc[rng.integers(0, len(tr), len(tr))].reset_index(drop=True)
    ob = build_order(bs, b + 1)[0][:k_sel]
    ov = len(set(ob) & set(selected))
    st.append({"kind": "resample_overlap", "name": f"bootstrap{b}", "value": ov / k_sel, "k": k_sel})
    for c in ob:
        if c in cnt: cnt[c] += 1
st += [{"kind": "selected_column_appearances_of_5", "name": c, "value": v, "k": k_sel} for c, v in cnt.items()]
pd.DataFrame(st).to_csv(out / "stability.csv", index=False); C.wrote(out / "stability.csv")

# set comparison on validation, two model families
cmp = []
for nm, cs in [("all106", C.features("all106")), ("distinct91", C.features("distinct91")), ("shape16", C.features("shape16")),
               ("colour90", C.features("colour90")), ("sizefree7", C.features("sizefree7")), ("sizefree22", C.features("sizefree22")),
               ("selected", selected), ("authors_top5", top5)]:
    for mn, mk in (("hgb", hgb), ("logreg", lr)):
        acc, f1 = score(mk, cs)
        cmp.append({"featureset": nm, "n_features": len(cs), "model": mn, "val_accuracy": acc, "val_macro_f1": f1, "n_val": len(va)})
        print(cmp[-1], flush=True)
pd.DataFrame(cmp).to_csv(out / "set_comparison.csv", index=False); C.wrote(out / "set_comparison.csv")

fig, ax = plt.subplots(figsize=(6, 3.8))
r = res[res.k != "authors_top5"]
for nm in ("hgb", "logreg"): ax.plot(r.k.astype(int), 100 * r[f"{nm}_val_accuracy"], "o-", label=nm)
ax.axhline(100 * a5["hgb"], color="grey", ls="--", label="authors' top 5 (HistGB)")
ax.set(xscale="log", title=f"Validation accuracy vs number of features (n val={len(va)})", xlabel="Features kept (count)", ylabel="Validation accuracy (%)")
ax.legend()
C.savefig(fig, "features_accuracy_vs_k")
