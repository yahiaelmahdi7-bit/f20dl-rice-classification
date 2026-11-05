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
