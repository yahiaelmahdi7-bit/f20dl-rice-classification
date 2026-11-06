# Stage 7

| Stage | Script | Meets | Content |
|---|---|---|---|
| 7 | `07_compare.py` | R3, R4 | Test pass: accuracy, macro-F1, per-class recall, confusion matrices, intervals, paired tests |

**Stage 7 statistics, fixed in advance.** McNemar's test only looks at grains where two models disagree. Family of three pairs: HistGB `all106` against SmallCNN (primary), HistGB `all106` against MLP, MLP against SmallCNN. Exact binomial form, seed 0 of each model, with disagreement counts b and c always shown. Holm correction, which adjusts for running several tests, across those three. Seeds 1 and 2 reported as a check. Intervals: 2,000 paired bootstrap resamples, percentile, fixed seed. Near the ceiling we report error counts, because 0.05 points is about 6 grains here.

Write `src/07_compare.py`. Option `--split val|test` (default test). With `--split val` it scores saved validation predictions for the wedge and makes no test read. With test, it loads the saved models, creates test predictions for every model, and computes the same outputs.

## Fix 2
Add macro precision and macro recall to every metrics row. Add per-class precision and F1 beside recall in `{split}_recall.csv`. File names stay, because `report.py` reads them. No new intervals.

## Fix 5
- New `results/compare/val_by_lot.csv`, validation only: recall per model, column set, seed, class and lot.

## Fix 6
- New `results/compare/val_by_angle.csv`, validation only: row counts, accuracy, macro F1 and class F1 per grain-angle band, overall and per class. Angle is measured on the image by `common.angles`, moved unchanged from stage 2. Bands are the constants `common.ANGLE_BANDS`, saved as `lo_deg` and `hi_deg`; rows with no usable mask get band `no_mask`.
- New `results/compare/val_by_angle_lot.csv`, validation only: row counts, recall and errors per grain-angle band, class and lot, every band listed even when empty. Same angle measure and fixed bands as `val_by_angle.csv`, measured once for both. Recall only, as in `val_by_lot.csv`: late is Jasmine only.

## Fix 7
- `--split test`: before the read, `preflight(read_needs())` names every missing file and stops. The list is built from `model_list()`, through `model_file`, which `test_probs` also loads from.
- On test a model that cannot load stops the stage (`score_models`); it is never dropped. Validation stays tolerant, so `make wedge` still runs with some models present.
- `paired` holds the b, c, exact p and bootstrap difference, unchanged, so it can be tested on made-up rows.

## Fix 8
- `read_test` is the one gated read. An infinite cell becomes a missing one, which every pipeline imputes, and the count is printed. Left in, the first imputer raises right after the read. Stage 5 still stops on an infinite train or validation cell, before any read.
- `tables` is everything after the predictions, split out with no change in behaviour, so a test can drive the real metrics, recall, confusion, spread and McNemar code on made-up predictions.

## Fix 9 (harden before go)
- `BASE` gains `perceptron` and `knn`, so `model_list`, `read_needs` and the freeze hash cover both on the four feature sets. Stage 7 reports accuracy, F1 and counts for them, and no probability metric.
