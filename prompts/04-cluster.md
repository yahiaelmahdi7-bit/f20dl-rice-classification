# Stage 4

| Stage | Script | Meets | Content |
|---|---|---|---|
| 4 | `04_cluster.py` | R2, generative | k-means and Gaussian mixture (EM) on PCA-20 of `all106`, k from 2 to 10, train and validation only. Silhouette on a 10,000-grain sample, BIC, and ARI and NMI against true variety. PCA view. Late-lot check at k=6: does one cluster hold at least 80% of the late lot at 80% purity? |

Write `src/04_cluster.py`.

## Fix 1
Standardise on train before PCA, and report cumulative explained variance of PCA-20. Add k-means inertia. Run on `distinct91` and `sizefree22` beside `all106`, because the copied columns give mean colour double weight. At k=5 and k=6, write a cluster-by-variety heatmap, and `results/cluster/profiles.csv` with size, grain shape, mean colour and purity for each cluster.
