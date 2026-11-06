# Stage 5

| Stage | Script | Meets | Content |
|---|---|---|---|
| 5 | `05_baselines.py` | R3, generative + discriminative | Table below, on `shape16`, `colour90`, `all106`, `selected`. Naive Bayes also runs on `distinct91`, because copied columns break its independence assumption. Depth-4 tree exported for the practical-applications discussion |

**Stage 5 models, tuned on validation.**

| Model | Type | Grid |
|---|---|---|
| Gaussian Naive Bayes | Generative | `var_smoothing` 1e-9, 1e-7, 1e-5 |
| Per-class Gaussian mixture on PCA-20 | Generative | 1, 2, 4, 8 components, full covariance |
| Decision Tree | Discriminative | `max_depth` 5, 10, 20, none; `min_samples_leaf` 1, 5, 20 |
| Logistic Regression | Discriminative | `C` 0.01, 0.1, 1, 10, 100 |
| HistGradientBoosting (HistGB) | Discriminative, ensemble | learning rate 0.05, 0.1; `max_iter` 150, 300; `max_leaf_nodes` 31, 63 |

Image damage, SmallCNN only, on test images, with this grid:

Write `src/05_baselines.py`. Options: `--only <model>:<featureset>` and `--control` (shuffled training labels). Each pipeline fits median imputation and scaling on train only. Select each model's grid point by validation macro-F1. Save final models to `models/baselines/`, with HistGB `all106` also at `models/hgb_all106.joblib`. Save validation predictions.

## Fix 2
The brief's R3 names accuracy, precision, recall and F1. Add `val_macro_precision` and `val_macro_recall` to the stage 5 results beside `val_accuracy` and `val_macro_f1`. Selection stays on macro-F1.

## Fix 3
Save each job as it finishes: model files first, then its `best_params.json` entry, its `grid.csv` rows and its `val_results.csv` row, merged by `model, featureset`, so a crash keeps every finished job and a row always has its model. CSV and JSON writes go to a temp file and then rename. A run without `--only` or `--control` first deletes those three files, so no row of an older grid survives it.

## Fix 4
`--control` takes its labels from `C.shuffled_labels` and calls `C.control_gate` on `control.csv` once every row is saved.

## Fix 9 (harden before go)
- Add `perceptron` and `knn` to `GRIDS`, so each runs on the four feature sets. The job count goes from 21 to 29 (7 models x 4 feature sets, plus `nb:distinct91`). Grids: perceptron `alpha` 1e-5, 1e-4, 1e-3; knn `n_neighbors` 3, 5, 11, 21; one point each in smoke (`alpha` 1e-4, `n_neighbors` 5).
- `build` makes `perceptron` from `C.SoftmaxPerceptron` and `knn` from scikit-learn's `KNeighborsClassifier`, behind the same imputer and scaler as the other models.
- The perceptron's scores are not calibrated. They keep the predicted class, which is all stage 7 uses; routing uses only `hgb:all106` and the CNN.
- Each kNN file is about 44 MB, in the git-ignored `models/baselines/`.

## Fix 10 (harden before go)
- The depth-4 tree on `all106` is also saved as model `tree4`: an imputer and the tree in `models/baselines/tree4__all106.joblib` (same `pipe`, `cols`, `params` keys as every baseline), with its validation predictions saved through `C.save_preds`. It is a fixed teaching model, not a tuned job, so it adds no row to `JOBS` (still 29) and no control row. The text and figure of the tree are unchanged.

## Fix 11 (harden before go)
- `val_results.csv` gains a last column `fit_seconds`: wall-clock seconds for fitting the chosen grid point's pipeline (imputer, scaler and model) on train. It is the cost the report quotes beside each model's score.
