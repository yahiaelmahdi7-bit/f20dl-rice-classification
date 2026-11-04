"""Stage 4: k-means and Gaussian mixture on PCA-20 of all106 (train and validation only)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import common as C

out = C.stage_dir("cluster")
g = C.load_grains()
X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(g[C.ALL106]))
Z = PCA(20, random_state=C.SEED).fit_transform(X)
y, late = g.y.values, (g.lot == "late").values
rng = np.random.default_rng(C.SEED)
sidx = rng.choice(len(Z), min(10000, len(Z)), replace=False)
rows, lab6 = [], {}
for k in range(2, 11):
    for name in ("kmeans", "gmm"):
        if name == "kmeans":
            m = KMeans(k, n_init=3 if C.SMOKE else 10, random_state=C.SEED).fit(Z); lab = m.labels_; bic = np.nan
        else:
            m = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=C.SEED).fit(Z); lab = m.predict(Z); bic = m.bic(Z)
        rows.append({"method": name, "k": k, "silhouette": silhouette_score(Z[sidx], lab[sidx]), "bic": bic,
                     "ARI": ARI(y, lab), "NMI": NMI(y, lab)})
        if k == 6: lab6[name] = lab
        print(rows[-1], flush=True)
res = pd.DataFrame(rows)
res.to_csv(out / "cluster_metrics.csv", index=False); C.wrote(out / "cluster_metrics.csv")

# late-lot check at k=6: one cluster holding >= 80% of the late lot at >= 80% purity
chk = []
for name, lab in lab6.items():
    for c in sorted(set(lab)):
        inc = lab == c
        chk.append({"method": name, "cluster": c, "size": int(inc.sum()), "share_of_late": float(late[inc].sum() / max(late.sum(), 1)),
                    "purity_late": float(late[inc].mean())})
chk = pd.DataFrame(chk)
chk["passes"] = (chk.share_of_late >= 0.8) & (chk.purity_late >= 0.8)
chk.to_csv(out / "late_lot_check.csv", index=False); C.wrote(out / "late_lot_check.csv")
print(chk.groupby("method").passes.any().to_dict(), "late-lot cluster found (>=80% share at >=80% purity)")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (lab, t) in zip(axs[:2], [(y, "true variety"), (lab6["kmeans"], "k-means k=6")]):
    ax.scatter(Z[sidx, 0], Z[sidx, 1], c=lab[sidx], s=2, cmap="tab10")
    ax.set(title=f"PCA view by {t} (n={len(sidx)})", xlabel="PC1 (standardised units)", ylabel="PC2 (standardised units)")
for name, d in res.groupby("method"): axs[2].plot(d.k, d.silhouette, "o-", label=name)
axs[2].set(title=f"Silhouette vs k (sample n={len(sidx)})", xlabel="Clusters k (count)", ylabel="Silhouette")
axs[2].legend()
C.savefig(fig, "cluster_overview")
