# Report outputs

Write `src/report.py`, run by `make report`, reading only saved CSVs and predictions, with no GPU and no test gate. It writes `results/SUMMARY.csv` (every model and feature set: accuracy, macro-F1, interval, seed spread), `results/MANIFEST.csv` (requirement, stage, output file, covering R2, R3, R4, generative task, discriminative task), and rebuilds comparison figures. Numbers only, no prose conclusions.

## Fix 2
`SUMMARY.csv` gains `macro_precision` and `macro_recall`, read from the stage 5 and 7 outputs.

## Fix 3: `SUMMARY.csv` and `MANIFEST.csv` are written to `C.RES` in every mode. In `MANIFEST.csv`, the dictionary, validity and outliers row points at `quality/`, and the two shuffled-label rows point at `baselines/control.csv` and `neural/control.csv`.
