# Stage 8

| Stage | Script | Meets | Content |
|---|---|---|---|
| 8 | `08_routing.py` | Generative + discriminative | `--fit` on validation, `--score` on test. Details below |

**Stage 8 routing.** All fitted on validation, then scored on test.
- Temperature scaling: one number per model that softens or sharpens its probabilities until they match how often it is right.
- Floors: the lowest confidence where validation accuracy above it reaches the target, 99.95% for the first model and 99% for the second.
- Familiarity check: per-class Gaussian mixture on PCA-20, cut at the 1st percentile of validation log-likelihood. Table version on `all106`; image version on the SmallCNN's final 256-wide layer (or its tuned width).
- Chain: HistGB `all106` answers if above its floor; otherwise SmallCNN answers if above its floor; otherwise `unsure`. The `unfamiliar` flag is separate and overrides the answer, and fires if either version trips.
- Reports: calibration error, risk-coverage curve, error rate at a fixed 95% coverage (the same yardstick for every model, since floors differ), errors and unsure counts by route, over 3 seeds. This can separate models tied on accuracy.

Write `src/08_routing.py` with `--fit` and `--score`. Saves `models/routing.json`, `models/gate_table.joblib`, `models/gate_image.joblib`. Routing is plain local statistics inside this code, with no external service. Outputs include a risk-coverage figure and a reliability figure.

## Fix 2
- `--score`: before the read, `preflight(read_needs())` names every missing file and stops.

## Fix 3
- `--fit` stores a temperature and a floor per CNN seed in `models/routing.json` under `cnn_seeds`, fitted on validation only. `cnn` stays as seed 0, the primary answer and the entry `predict.py` reads. `--fit` stops if any seed has no validation predictions.
- `--score` stops before the read if `routing.json` has no entry for a seed. Each seed's predictions are checked against the test rows by `grain_id`.
- The route table carries a `seed` column, one block per seed, seed 0 first. Stated limit: seed 0's image familiarity gate serves all three seeds, so `cnn_seed1.pt` and `cnn_seed2.pt` are not new read-time needs.

## Fix 4
`read_test` is the one gated read. An infinite cell becomes a missing one, as in stage 7, so the table gate's imputer does not raise after the read, and the count is printed.

## Fix 5 (row 9 reading, before any test read)
This replaces the floor line above for the second model; the original line stays as the record. `fit_routes`: each CNN seed's temperature still comes from every validation row, but its floor comes only from the validation grains hgb declines, meaning calibrated hgb confidence under the hgb floor. Reason: the CNN answers only those grains, and the first fit, on all grains, refused none of them on any seed. `floor_for` is unchanged, so with no floor reaching the target it returns 1.01 and the CNN answers nothing. Unfamiliar grains stay in the fit: on validation the floors came out the same with and without them, and the fit then needs no gate.
