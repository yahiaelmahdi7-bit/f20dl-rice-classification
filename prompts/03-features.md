# Stage 3

| Stage | Script | Meets | Content |
|---|---|---|---|
| 3 | `03_features.py` | R2 | Mutual information, ANOVA, and one per correlation cluster (absolute Spearman at least 0.95, average linkage, keep the column with most mutual information). HistGB validation accuracy for k = 1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 91. `selected` is the smallest k within 0.1 points of the best. Compared with the authors' top 5: `ROUNDNESS, COMPACTNESS, SHAPEFACTOR_3, ASPECT_RATIO, ECCENTRICITY` (Cinar & Koklu 2021). Train and validation only |

Write `src/03_features.py`. It writes the `selected` list to `results/features/selected.json`, which `common.py` reads, and a chart of accuracy against number of features.

## Fix 1
Fit mutual information, ANOVA and clustering of columns on train only; score on validation only, and assert it. Repeat the k sweep with Logistic Regression beside HistGB, so `selected` does not rest on one model. Re-rank on 5 bootstrap resamples of train and write top-k overlap and how often each selected column appears to `results/features/stability.csv`. For each selected column, write its family and within-class correlation with log area. Write `results/features/set_comparison.csv`: accuracy and macro-F1 on validation for `all106`, `distinct91`, `shape16`, `colour90`, `sizefree7`, `sizefree22`, `selected` and the authors' top 5, with HistGB and Logistic Regression.
