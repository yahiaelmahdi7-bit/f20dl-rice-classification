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
trm = (g.split == "train").values
y, late = g.y.values, (g.lot == "late").values
rng = np.random.default_rng(C.SEED)
SETS = ["all106", "distinct91", "sizefree22"]
rows, chk, prof, labs, cum = [], [], [], {}, {}
for fs in SETS:
    cs = C.features(fs)
    imp = SimpleImputer(strategy="median").fit(g.loc[trm, cs])
    sc = StandardScaler().fit(imp.transform(g.loc[trm, cs]))     # standardise on train, then PCA fitted on train
    pca = PCA(min(20, len(cs)), random_state=C.SEED).fit(sc.transform(imp.transform(g.loc[trm, cs])))
    Z = pca.transform(sc.transform(imp.transform(g[cs])))
    cum[fs] = float(pca.explained_variance_ratio_.sum())
    print(f"{fs}: PCA-{pca.n_components_} cumulative explained variance {cum[fs]:.4f}", flush=True)
    sidx = rng.choice(len(Z), min(10000, len(Z)), replace=False)
    lab6 = {}
    for k in range(2, 11):
        for name in ("kmeans", "gmm"):
            if name == "kmeans":
                m = KMeans(k, n_init=3 if C.SMOKE else 10, random_state=C.SEED).fit(Z); lab = m.labels_; bic = np.nan; inertia = float(m.inertia_)
            else:
                m = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=C.SEED).fit(Z); lab = m.predict(Z); bic = m.bic(Z); inertia = np.nan
            rows.append({"featureset": fs, "method": name, "k": k, "silhouette": silhouette_score(Z[sidx], lab[sidx]), "bic": bic, "inertia": inertia,
                         "ARI": ARI(y, lab), "NMI": NMI(y, lab), "pca_cum_var": cum[fs]})
            if k in (5, 6): labs[(fs, name, k)] = lab
            print(rows[-1], flush=True)
    for name in ("kmeans", "gmm"):
        for c in sorted(set(labs[(fs, name, 6)])):
            inc = labs[(fs, name, 6)] == c
            chk.append({"featureset": fs, "method": name, "cluster": c, "size": int(inc.sum()), "share_of_late": float(late[inc].sum() / max(late.sum(), 1)),
                        "purity_late": float(late[inc].mean())})
    for k in (5, 6):
        for name in ("kmeans", "gmm"):
            lab = labs[(fs, name, k)]
            for c in sorted(set(lab)):
                inc = lab == c
                vc = pd.Series(g["class"][inc]).value_counts()
                d = g[inc]
                prof.append({"featureset": fs, "method": name, "k": k, "cluster": c, "size": int(inc.sum()), "median_AREA": d.AREA.median(),
                             "median_ASPECT_RATIO": d.ASPECT_RATIO.median(), "median_SOLIDITY": d.SOLIDITY.median(), "median_meanRR": d.meanRR.median(),
                             "median_meanRG": d.meanRG.median(), "median_meanRB": d.meanRB.median(), "dominant_variety": vc.index[0],
                             "purity": float(vc.iloc[0] / inc.sum())})
res = pd.DataFrame(rows)
res.to_csv(out / "cluster_metrics.csv", index=False); C.wrote(out / "cluster_metrics.csv")
pd.DataFrame(prof).to_csv(out / "profiles.csv", index=False); C.wrote(out / "profiles.csv")
pd.DataFrame([{"featureset": k, "pca_components": 20, "cumulative_explained_variance": v} for k, v in cum.items()]).to_csv(out / "pca_variance.csv", index=False)

# late-lot check at k=6: one cluster holding >= 80% of the late lot at >= 80% purity
chk = pd.DataFrame(chk)
chk["passes"] = (chk.share_of_late >= 0.8) & (chk.purity_late >= 0.8)
chk.to_csv(out / "late_lot_check.csv", index=False); C.wrote(out / "late_lot_check.csv")
print(chk.groupby(["featureset", "method"]).passes.any().to_dict(), "late-lot cluster found (>=80% share at >=80% purity)")

# cluster-by-variety heatmaps, k-means at k=5 and k=6
fig, axs = plt.subplots(2, len(SETS), figsize=(4.5 * len(SETS), 7))
for r, k in enumerate((5, 6)):
    for c_, fs in enumerate(SETS):
        ct = pd.crosstab(labs[(fs, "kmeans", k)], g["class"]).reindex(columns=C.CLASSES)
        ax = axs[r, c_]
        im = ax.imshow(ct.values / ct.values.sum(1, keepdims=True), cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(5), C.CLASSES, rotation=45, fontsize=7)
        ax.set(title=f"{fs}, k-means k={k} (n={len(g)})", ylabel="Cluster", xlabel="Variety")
fig.colorbar(im, ax=axs, label="Share of cluster")
C.savefig(fig, "cluster_heatmaps")

Z = None
sil = res[res.method == "kmeans"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for fs, d in sil.groupby("featureset"):
    axs[0].plot(d.k, d.silhouette, "o-", label=fs); axs[1].plot(d.k, d.inertia, "o-", label=fs)
for fs, d in res[res.method == "gmm"].groupby("featureset"): axs[2].plot(d.k, d.bic, "o-", label=fs)
axs[0].set(title=f"Silhouette vs k, k-means (n={len(g)})", xlabel="Clusters k (count)", ylabel="Silhouette")
axs[1].set(title="K-means inertia vs k", xlabel="Clusters k (count)", ylabel="Inertia (PCA-20 units)")
axs[2].set(title="GMM BIC vs k", xlabel="Clusters k (count)", ylabel="BIC")
axs[0].legend()
C.savefig(fig, "cluster_overview")
