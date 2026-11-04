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
