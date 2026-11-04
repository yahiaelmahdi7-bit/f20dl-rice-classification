# Context for every stage (F20DL Group 16 rice study)

We are classifying five rice varieties (Arborio, Basmati, Ipsala, Jasmine, Karacadag). Question: do measured numeric features or raw images work better, on identical grains, and what do near-100% scores depend on? Generate Python code only. Do not write README text, report text or other prose.

Python 3.12, `uv` lockfile, scikit-learn, PyTorch on the Mac GPU (`mps`, falling back to CPU). Repo root is `~/Academic/courses/f20dl/`. Raw data and images never enter git. Scripts live in `src/`, with shared loading, seeds and paths in `src/common.py`.

**Shared base, built first.** `data/derived/grains.parquet`: one row per grain. `splits.csv`, committed to git, columns `grain_id, class, lot, file_no, msc_row, split`: 70/15/15 train/validation/test, stratified by class and lot, seed 0. Every model uses these grain IDs.

**Feature sets (one name each).**

| Name | Columns |
|---|---|
| `all106` | All 106, as published, copies included |
| `distinct91` | `all106` minus the 15 `ALLdaub4*` copies |
| `shape16` | `AREA` through `SHAPEFACTOR_4` |
| `colour90` | `meanRR` through `ALLdaub4ZZ` |
| `sizefree7` | `ECCENTRICITY, SOLIDITY, ASPECT_RATIO, ROUNDNESS, COMPACTNESS, SHAPEFACTOR_3, SHAPEFACTOR_4` |
| `sizefree22` | `sizefree7` plus `meanRR, meanRG, meanRB, meanH, meanS, meanV, meanL, meanA, meanB, meanY, meanCb, meanCr, meanXX, meanYY, meanZZ` |
| `selected` | Chosen in stage 3 |

`EXTENT` is left out of the size-free sets: it moved x1.148 in the late lot for reasons we have not explained.

Images: resize the whole 250px crop uniformly to 128px, never crop to the grain. Cached once as a `uint8` `np.memmap` (about 3.7 GB) in `data/derived/`, converted per batch. Minimum 8 GB RAM.

**Test split rule.** A test read is any run that loads test labels or scores test grains. `common.load_split("test")` is the only path to them. It raises unless the caller is stage 7, 8 `--score` or 9 and `results/frozen.json` exists, and it appends caller, time and frozen hash to `results/test_reads.csv`. `make freeze` hashes every configuration, seed, floor, familiarity cut and test pair, after all validation work. Each of those three stages reads test once, every read carries the same hash, and seeds run inside the read. Any rerun is logged and disclosed. `predict.py` is not a read.

Conventions for all scripts:
- Scripts live in `src/`; `src/common.py` owns paths, the seed, the feature-set lists, `load_grains()`, `load_split(name)`, the image cache reader, device choice, prediction save and load, and figure saving. No script reads test grains except through `load_split("test")` or the image reader's equivalent gated path.
- Saved predictions: `results/preds/<model>__<featureset>__seed<k>__<split>.parquet` with columns `grain_id`, `y`, `p0..p4`. Class order is sorted class names.
- Each script writes CSVs to `results/<stage>/` and PNG figures to `figures/`, and prints what it wrote.
- Training stages must never predict on test. Test predictions are created only by stage 7 from saved models.
- Smoke mode: env `RICE_SMOKE=1`. Uses a stratified 2,000-grain slice, 1 epoch, small grids, skips global asserts, lets validation stand in for test with no frozen hash or read log, and writes to `results/smoke/` and `figures/smoke/`.
- Every script is rerunnable and overwrites only its own outputs. Raw data is never modified.
- Keep comments short and factual, stating what a block does. No invented numbers: every value in an output comes from a computation in this run.
