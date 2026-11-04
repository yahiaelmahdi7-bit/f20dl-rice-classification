# Stage 6

| Stage | Script | Meets | Content |
|---|---|---|---|
| 6 | `06_neural.py` | R4 | MLP and SmallCNN, spec below |

**Stage 6 spec.** Tuned on validation, then final models trained with seeds 0, 1, 2.

| | MLP | SmallCNN |
|---|---|---|
| Input | `all106`, standardised | 128px RGB, scaled by train channel mean and spread |
| Body | Hidden layers, ReLU, dropout | 4 blocks of 3x3 conv, BatchNorm, ReLU, 2x2 max-pool; global average pool; dropout; linear to 5 |
| Optimiser | AdamW, weight decay 1e-4, cosine schedule | Same |
| Batch | 512 | 128 |
| Epochs | Up to 100, stop after 10 without validation-loss gain | Up to 20, stop after 4 |
| Augmentation | None | Horizontal and vertical flips only. No zoom or crop (size is a signal to study); no colour jitter (keeps the tint test clean) |
| Grid | Hidden [100], [50,50] (the published shapes), [256,128]; lr 1e-3, 3e-4; dropout 0, 0.2 | Base width 16, 32 doubling per block; lr 1e-3, 3e-4; dropout 0, 0.3; 8 epochs on a 10,000-image train subset |

Images: resize the whole 250px crop uniformly to 128px, never crop to the grain. Cached once as a `uint8` `np.memmap` (about 3.7 GB) in `data/derived/`, converted per batch. Minimum 8 GB RAM.

**Compute budget.** Estimate until Step 3 measures one epoch.

| Job | Full-epoch equivalents |
|---|---|
| SmallCNN tuning, 8 configs | About 12 |
| SmallCNN final, 3 seeds | Up to 60 |
| SmallCNN lot holdout, 3 seeds | Up to 60 |
| Total | Up to about 132, so 2-4.5 hours at an assumed 60-120 s per epoch |

Table models and the MLP add under an hour. Training runs on one machine. Predictions in `results/preds/` and final weights in `models/` are committed, so teammates rebuild tables and figures with `make report` and no GPU.

Checks: A system that knows its limits: confidence routing handles doubt between known varieties; a generative familiarity check handles grains unlike anything seen. Stage 6. Check: shuffled-label control between 18% and 22%; 3 seeds complete.

Write `src/06_neural.py`. Options: `--only mlp|cnn`, `--seeds 0 1 2`, `--overfit-test`, `--control`, `--time-epoch`. Print and log seconds per epoch to `results/neural/epoch_times.csv`. Save tuning results, per-epoch curves, final weights to `models/`, validation predictions, and the final-layer embedding for train and validation (for stage 8). Log progress to stdout with flush, so a detached run can be watched.

## Fix 2
`--overfit-test` must match the plan: 200 grains, up to 30 epochs, pass at 99% train accuracy or above. Stop at the first epoch that reaches it and print that epoch. Smoke mode keeps its short run but prints `smoke path, no verdict` instead of PASS or FAIL.

## Fix 3
- `epoch_times.csv` is written with `csv.writer`, so every row parses to the contract's columns. MLP epochs are logged too, with device `cpu`.
- `curves.csv` always has the contract's column order, and each finished seed's curves are saved as it finishes.
- `control.csv` is merged by `model`, and `curves.csv` by `tag`. CSV and JSON writes go to a temp file and then rename.
- `best_config.json` is written straight after tuning. A run of fewer seeds than `C.SEEDS` keeps the saved config and skips tuning, but only if the tuning table beside it holds this exact grid and names that config as its winner. Any other run tunes as before.
- `chan_stats` sums in float64, because float32 sums drift on a 5,000-image sample.
