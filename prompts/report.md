# Report outputs

Write `src/report.py`, run by `make report`, reading only saved CSVs and predictions, with no GPU and no test gate. It writes `results/SUMMARY.csv` (every model and feature set: accuracy, macro-F1, interval, seed spread), `results/MANIFEST.csv` (requirement, stage, output file, covering R2, R3, R4, generative task, discriminative task), and rebuilds comparison figures. Numbers only, no prose conclusions.

## Fix 2
`SUMMARY.csv` gains `macro_precision` and `macro_recall`, read from the stage 5 and 7 outputs.

## Fix 3: `SUMMARY.csv` and `MANIFEST.csv` are written to `C.RES` in every mode. In `MANIFEST.csv`, the dictionary, validity and outliers row points at `quality/`, and the two shuffled-label rows point at `baselines/control.csv` and `neural/control.csv`.

## Fix 4 (harden before go)
- `report.py` adds two figures from saved CSVs only. `learning_curve_figure(curves)` plots train and validation loss per epoch for every `tag` in `results/neural/curves.csv` (`report_learning_curves.png`). `late_recall_figure(late)` plots recall with its interval per model for the `late_all` and `late_test` subsets of `results/robustness/late_recall.csv` (`report_late_lot_recall.png`). A model with only one of the two subsets draws only that point, an empty cell stays empty, and a missing or empty file prints a skip line and draws nothing. `main` calls both after `figure(S)`.

## Fix 5 (harden before go, D8, D9)
- `report.py` writes `results/DISCLOSURES.csv`, columns `item, what, evidence_file`: facts, not prose. Fixed rows: old smoke overlap (evidence `results/smoke_overlap.csv`); `val_calibration` and `val_routes` in-sample; hyperparameters picked on validation holding late grains; shuffled-label scores away from 20%; `02_eda` class counts include test; image gate is seed 0 for all three; near-upright is the group's own reading; Perceptron and kNN scores uncalibrated. If `results/test_reads_disclosed.csv` exists and has rows, one row counts the extra reads and names that file as evidence, and one counts rows whose `reason` starts with `code-changed`. The ledger is not copied. A missing ledger or `run_log.csv` is no error.
- `MANIFEST.csv` gains a row for every new output: validation and test tables by lot, angle and angle and lot, `mean_colour_check.csv`, `results/smoke_overlap.csv`, `DISCLOSURES.csv`. The `run_log.csv` row comes with the runner. An `output_file` starting `results/` is read from the repo root, so the committed overlap file is found in smoke too. Handoff row 16 fails if any `exists` is False.

## Fix 6 (harden before go, C3 disclosures)
- `disclosures()`: the "code changed after lock" row is counted from the `changes` list in `C.RES/code_lock.json` (evidence file `results/code_lock.json`), not from the reasons in the read ledger. The "extra read" row still counts the ledger rows.

## Fix 7 (harden before go, C4)
- `MANIFEST.csv` gains three rows, all stage 11: `run_log.csv` (time and memory per runner row), `test_reads_disclosed.csv` (extra test reads, with reasons) and `code_lock.json` (the code digest and any logged code change).

## Fix 8 (harden before go, integrator review)
- `DISCLOSURES`: the old smoke overlap row says the throwaway smoke models were fitted on those grains and scored them, and that no real model or choice used those models. The counts stay in `results/smoke_overlap.csv`.
- `DISCLOSURES`: the shuffled-label row gives the measured reason: a simple model can give a whole tight class one shuffled label, so the stop is `C.CONTROL_CEILING`, not `C.CHANCE_BAND`. Both values come from `common.py`.
- Every `evidence_file` is a path from the repo root.

## Fix 9 (harden before go, smoke row 16)
- `report.py`: `gaps(point, lo, hi)` returns the two error-bar lengths, each floored at 0. Both interval figures use it, for drawing only, so the saved tables keep their values. A bootstrap percentile interval can miss its point, and a bar length cannot be negative.

## Fix 10 (row 2 failed at go)
`DISCLOSURES` gains a fixed `overfit test limit` row: the plan allowed 30 epochs, the first real run reached 92% against the 99% bar, and the limit was raised to 150 after that failure with the 200 grains and the bar unchanged. Evidence file `results/run_log.csv`.
