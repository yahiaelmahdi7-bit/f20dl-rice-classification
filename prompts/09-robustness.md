# Stage 9

| Stage | Script | Meets | Content |
|---|---|---|---|
| 9 | `09_robustness.py` | R1, R4 | Lot holdout and tint test, details below |

**Stage 9 robustness.** Configurations are frozen first.
- Lot holdout: retrain with no late-lot Jasmine in train, HistGB on `all106`, `shape16`, `sizefree7`, `sizefree22`, and SmallCNN over 3 seeds. No late grain was seen in these runs, so recall is scored on all 3,283 late grains, with the test-only subset (about 490) beside it. Counts and Wilson intervals. Temperature, floors and familiarity cut refit on validation without the late lot, then the chain is run on the late lot.
- Tint arm A: grey-world correction per image with gains computed from grain-mask pixels only, so grain size cannot enter. Check that gains do not track area inside a class (absolute r under 0.1), and report mean-colour accuracy after correction. Frozen SmallCNN scored on corrected test images. `shape16` and `sizefree7` are the reference, since they use no colour. Correction also removes real grain colour, so this measures reliance on colour, not cause.

Write `src/09_robustness.py` with `--train` (retrains without the late lot, using frozen configurations, no test read) and `--score` (the single test read). Also score an unchanged-image control, which must match stage 7 predictions exactly. Each table model on the late lot also reports mean confidence and count of confidently wrong grains.

## Fix 2
The mean-colour model is fitted on table means and scored on image-computed means, so its raw and corrected scores are not like for like. Fit it on `mask_means` from a stratified 5,000-grain train sample, fixed seed, and score on the same measure. Write `results/robustness/mean_colour_check.csv` comparing table means with `mask_means` on that sample: `channel, n, pearson, median_abs_diff`.

## Fix 3
A model must never be scored on grains it trained on. `full_hgb` and `full_cnn` trained on train, which holds 2,298 of the 3,283 late grains, so score them on the late test subset only and write no `late_all` row for them. Hold-out models never saw a late grain and keep both rows. The tint arms use a model fitted on train and the frozen CNNs, so score them on the late test subset only. In smoke mode the late test subset is the validation stand-in, so pick it by `grain_id`, not by the split label. Stop if a scored grain is in train. Sample per class without `groupby().apply`, because pandas 3 drops the `y` column there and `samp.y` fails.

## Fix 4
- `--train`: `frozen_configs()` runs first and stops, naming any missing `best_params.json`, `best_config.json` or needed key. There are no default settings. `sizefree7` and `sizefree22` take the `hgb:all106` point by design.
- Saved hold-out models carry `train_ids`, and joblib models `params_from`. `late_recall.csv` gains `params_from`, so the borrowed point is disclosed in a committed file.
- `--score`: `preflight` stops before the read on a missing file or image folder. The unchanged-image control records the largest difference on grains both stages hold and never stops the stage.
- `greyworld_gain_vs_area.csv` column `cls` is now `class`.

## Fix 5
`--train`: routing and gates run after the CNN loop, so a crash there used to cost all three seeds again. A hold-out CNN file now also holds `run`, every setting that shaped its weights. A seed whose file holds the same `run` and the same `train_ids` is reused, not retrained, and seed 0's embeddings are rebuilt from the saved weights. A missing file, a torn file, or a file with a different `run` or `train_ids` is retrained and overwritten. A clean run with no files behaves as before.

## Fix 6
`score_late` is everything after the test read and `fit_tables` is the table-model fit and save, each split out with no change in behaviour, so a test can drive the real `grey_world` call and check that a saved record holds the rows its model was fitted on, on made-up rows with no model fitted.

## Fix 7
- `read_test` is the one gated read. An infinite cell becomes a missing one, as in stage 7, and the count is printed.
- `--score` stops before the read if train and validation hold no late grains: the first hold-out model would raise on no rows, after the read.
- `grey_world` writes parts 1 and 1b, then stops with a printed line if no late grain reached test. `greyworld_results.csv` stays absent, so the report and the go check notice.
- `gain_rows` takes the largest correlation with `np.max`, so a channel with no answer fails `passes_abs_lt_0_1`. Python's `max` could drop the nan and pass it.
