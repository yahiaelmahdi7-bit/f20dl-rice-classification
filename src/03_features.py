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
from sklearn.feature_selection import f_classif, mutual_info_classif
import common as C

out = C.stage_dir("features")
tr, va = C.load_split("train"), C.load_split("val")
cols = C.features("distinct91")
sub = tr.sample(min(len(tr), 20000), random_state=C.SEED)
Xs = sub[cols].fillna(sub[cols].median())
mi = mutual_info_classif(Xs, sub.y, random_state=C.SEED)
F, _ = f_classif(Xs, sub.y)
rank = pd.DataFrame({"feature": cols, "mutual_info": mi, "anova_F": F}).sort_values("mutual_info", ascending=False)
rank.to_csv(out / "ranking.csv", index=False); C.wrote(out / "ranking.csv")

rho = np.abs(spearmanr(Xs, nan_policy="omit").statistic)
dist = np.clip((1 - rho + (1 - rho).T) / 2, 0, None)
np.fill_diagonal(dist, 0)
cl = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.05, criterion="distance")
mi_s = pd.Series(mi, index=cols)
reps = [mi_s[[c for c, k in zip(cols, cl) if k == j]].idxmax() for j in sorted(set(cl))]
reps = sorted(reps, key=lambda c: -mi_s[c])
order = reps + [c for c in rank.feature if c not in reps]
pd.DataFrame({"feature": reps, "mutual_info": mi_s[reps].values}).to_csv(out / "cluster_representatives.csv", index=False)
print(f"{len(set(cl))} correlation clusters, one representative each")

ks = [1, 3, 8, 32, 91] if C.SMOKE else [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 91]
it = 20 if C.SMOKE else 200
rows = []
for k in ks:
    m = HGB(max_iter=it, random_state=C.SEED).fit(tr[order[:k]], tr.y)
    rows.append({"k": k, "val_accuracy": float((m.predict(va[order[:k]]) == va.y).mean())})
    print(rows[-1], flush=True)
res = pd.DataFrame(rows)
best = res.val_accuracy.max()
k_sel = int(res[res.val_accuracy >= best - 0.001].k.min())
selected = order[:k_sel]
top5 = "ROUNDNESS COMPACTNESS SHAPEFACTOR_3 ASPECT_RATIO ECCENTRICITY".split()   # authors' top 5
m5 = HGB(max_iter=it, random_state=C.SEED).fit(tr[top5], tr.y)
res = pd.concat([res, pd.DataFrame([{"k": "authors_top5", "val_accuracy": float((m5.predict(va[top5]) == va.y).mean())}])])
res.to_csv(out / "accuracy_vs_k.csv", index=False); C.wrote(out / "accuracy_vs_k.csv")
(out / "selected.json").write_text(json.dumps({"selected": selected, "k": k_sel, "best_val_accuracy": best}, indent=1))
C.wrote(out / "selected.json")
print(f"selected k={k_sel}: {selected}")
fig, ax = plt.subplots(figsize=(6, 3.8))
r = res[res.k != "authors_top5"]
ax.plot(r.k.astype(int), 100 * r.val_accuracy, "o-")
ax.axhline(100 * float(res[res.k == "authors_top5"].val_accuracy.iloc[0]), color="grey", ls="--", label="authors' top 5")
ax.set(xscale="log", title=f"HistGB validation accuracy vs number of features (n val={len(va)})", xlabel="Features kept (count)", ylabel="Validation accuracy (%)")
ax.legend()
C.savefig(fig, "features_accuracy_vs_k")
