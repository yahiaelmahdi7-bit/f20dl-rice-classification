"""Stage 2: exploratory analysis on train and validation grains."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
import common as C

out = C.stage_dir("eda")
sp = pd.read_csv(C.SPLITS)
g = C.load_grains()
n = len(g)
col = dict(zip(C.CLASSES, plt.cm.tab10.colors))

# class balance (counts only, all splits)
cnt = sp.groupby(["class", "split"]).size().unstack()
cnt.to_csv(out / "class_balance.csv"); C.wrote(out / "class_balance.csv")
fig, ax = plt.subplots(figsize=(6, 3.5))
cnt.plot.bar(stacked=True, ax=ax)
ax.set(title=f"Grains per variety and split (n={len(sp)})", xlabel="Variety", ylabel="Grains (count)")
C.savefig(fig, "eda_class_balance")

# distributions by class
feats = ["AREA", "ASPECT_RATIO", "ECCENTRICITY", "meanRR", "meanRB", "entropyRR"]
fig, axs = plt.subplots(2, 3, figsize=(11, 6))
for ax, f in zip(axs.ravel(), feats):
    for c in C.CLASSES: ax.hist(g.loc[g["class"] == c, f].dropna(), bins=60, alpha=0.5, color=col[c], label=c)
    ax.set(title=f, xlabel=f"{f} (table units)", ylabel="Grains (count)")
axs[0, 0].legend(fontsize=7)
fig.suptitle(f"Distributions by variety (train+val, n={n})")
C.savefig(fig, "eda_distributions")

# correlation clusters on distinct91 (|Spearman| >= 0.95, average linkage)
d91 = C.features("distinct91")
sub = g.sample(min(n, 20000), random_state=C.SEED)[d91]
rho = np.abs(spearmanr(sub.fillna(sub.median()), nan_policy="omit").statistic)
dist = np.clip((1 - rho + (1 - rho).T) / 2, 0, None)
np.fill_diagonal(dist, 0)
cl = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.05, criterion="distance")
pd.DataFrame({"feature": d91, "cluster": cl}).to_csv(out / "corr_clusters.csv", index=False); C.wrote(out / "corr_clusters.csv")
o = np.argsort(cl, kind="stable")
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(rho[np.ix_(o, o)], cmap="viridis", vmin=0, vmax=1)
fig.colorbar(im, label="|Spearman rho|")
ax.set(title=f"Correlation structure, {len(set(cl))} clusters at 0.95 (n={len(sub)})", xlabel="Feature index (cluster order)", ylabel="Feature index (cluster order)")
C.savefig(fig, "eda_correlation")

# tint by class
g["tint"] = g.meanRB - g.meanRR
t = g.groupby("class").tint.agg(["mean", "std", "median", "count"])
t.to_csv(out / "tint_by_class.csv"); C.wrote(out / "tint_by_class.csv")
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.boxplot([g.loc[g["class"] == c, "tint"].dropna() for c in C.CLASSES], tick_labels=C.CLASSES, showfliers=False)
ax.set(title=f"Tint (meanRB - meanRR) by variety (n={n})", xlabel="Variety", ylabel="meanRB - meanRR (intensity levels)")
C.savefig(fig, "eda_tint")

# Jasmine break
J = g[g["class"] == "Jasmine"].sort_values("file_no")
cols = ["AREA", "MAJOR_AXIS", "MINOR_AXIS", "EXTENT", "meanRR", "meanRB", "tint"]
e, l = J[J.lot == "main"][cols].median(), J[J.lot == "late"][cols].median()
pd.DataFrame({"early": e, "late": l, "ratio": l / e}).to_csv(out / "jasmine_lots.csv"); C.wrote(out / "jasmine_lots.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, f in zip(axs, ["AREA", "tint"]):
    ax.plot(J.file_no, J[f].rolling(100, center=True, min_periods=10).median(), lw=1)
    ax.axvline(C.LATE_FROM, color="r", ls="--")
    ax.set(title=f"Jasmine {f}, rolling median 100 (n={len(J)})", xlabel="Image file number", ylabel=f"{f} (table units)")
C.savefig(fig, "eda_jasmine_break")

# entropy against log area within class
ent = [c for c in d91 if c.startswith("entropy")]
rows = [{"class": c, "feature": f, "spearman_vs_logarea": spearmanr(np.log(g.loc[g["class"] == c, "AREA"]), g.loc[g["class"] == c, f], nan_policy="omit").statistic}
        for c in C.CLASSES for f in ent]
pd.DataFrame(rows).to_csv(out / "entropy_vs_logarea.csv", index=False); C.wrote(out / "entropy_vs_logarea.csv")
fig, axs = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
for ax, c in zip(axs, C.CLASSES):
    s = g[g["class"] == c].sample(min(3000, (g["class"] == c).sum()), random_state=0)
    ax.scatter(np.log(s.AREA), s.entropyRR, s=2, alpha=0.4, color=col[c])
    ax.set(title=f"{c} (n={len(s)})", xlabel="log AREA (log pixels)")
axs[0].set_ylabel("entropyRR (bits)")
C.savefig(fig, "eda_entropy_vs_area")

# copy columns against their mean column
rows = []
for c in C.ALL106:
    if c.startswith("ALLdaub4"):
        suf = c[8:]
        m = "mean" + (suf.upper() if suf in ("a", "b") else suf)
        x, y = g[c], g[m]
        rows.append({"copy": c, "mean_column": m, "max_abs_diff": float((x - y).abs().max()), "pearson": float(x.corr(y)),
                     "identical": bool(np.allclose(x.fillna(0), y.fillna(0), atol=1e-9))})
pd.DataFrame(rows).to_csv(out / "copy_columns.csv", index=False); C.wrote(out / "copy_columns.csv")
print(pd.DataFrame(rows).identical.value_counts().to_dict(), "copy columns identical to their mean column")

# example grains
fig, axs = plt.subplots(5, 6, figsize=(9, 8))
for i, c in enumerate(C.CLASSES):
    s = g[g["class"] == c].sample(6, random_state=1)
    for ax, im, gid in zip(axs[i], C.read_images(s), s.grain_id):
        ax.imshow(im); ax.set_title(gid, fontsize=6); ax.axis("off")
fig.suptitle("Example grains, 128 px (train+val)")
C.savefig(fig, "eda_examples")
