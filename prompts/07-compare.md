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
