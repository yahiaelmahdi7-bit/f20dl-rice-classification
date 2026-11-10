# F20DL Group 16: classifying five rice varieties from measurements and from images

**DRAFT — pending Group 16 sign-off**

## 1. Purpose and research question

This repository is the code, saved outputs and records for the Group 16 study for F20DL (Heriot-Watt). It classifies five rice varieties (Arborio, Basmati, Ipsala, Jasmine, Karacadag). The question: do measured numeric features or raw images work better, on identical grains, and what do near-100% scores depend on?

Every model uses the same grain IDs and the same train/validation/test split. No result numbers are quoted here, so the README cannot go stale. The numbers are in `results/SUMMARY.csv`, with the full file list in `results/MANIFEST.csv`, and in the group's written report (not part of this repository).

## 2. Datasets

Both datasets are by Murat Koklu and are downloaded from Kaggle:

- Rice MSC dataset (numeric features): <https://www.kaggle.com/datasets/muratkokludataset/rice-msc-dataset>
- Rice Image dataset (images): <https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset>

Papers to cite:

- Koklu, M., Cinar, I., Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture 187:106285. doi:10.1016/j.compag.2021.106285
- Cinar, I., Koklu, M. (2021). Determination of effective and specific physical features of rice varieties by computer vision in exterior quality inspection. Selcuk Journal of Agriculture and Food Sciences 35(3):229-243. doi:10.15316/SJAFS.2021.252

Licence: the Kaggle API (`https://www.kaggle.com/api/v1/datasets/view/muratkokludataset/<dataset>`, no login) returned JSON for both datasets with the licence name "CC0: Public Domain" (retrieved 2026-11-10). It was read from that API response, not from the papers.

## 3. Requirements

- Python 3.12 (`pyproject.toml` pins `>=3.12,<3.13`; `.python-version` is 3.12) and [uv](https://docs.astral.sh/uv/). Dependencies are locked in `uv.lock`.
- Tested platform: macOS on Apple silicon, using the Apple GPU through PyTorch `mps`.
- CPU fallback: `src/common.py` picks the device as `mps` if `torch.backends.mps.is_available()`, otherwise `cpu`. There is no other branch (no CUDA), so an NVIDIA GPU is not used. CPU training has not been timed or tested by the group.
- Windows and Linux are untested.
- Disk: `data/SHA256SUMS` lists only two checksums (`rice-msc.zip`, `rice-image.zip`), no sizes. The Kaggle API reports the two zips at about 142 MB and 267 MB. The unpacked images and the image cache (about 3.7 GB, in `data/derived/`) add to that. A full re-run asserts at least 20 GiB free.

Set up the environment from the repository root:

```bash
uv sync
```

This creates `.venv/`. The Makefile runs everything with `.venv/bin/python` (override with `make <target> PY=<python>`).

## 4. Get and verify the data

```bash
make get-data
```

This runs `src/00_get_data.py`. It looks for `rice-msc.zip` and `rice-image.zip` in `data/raw/` (or in the folder named by the environment variable `RICE_ZIP_DIR`), downloads from Kaggle only if a zip is missing, checks each zip against `data/SHA256SUMS`, unpacks to `data/raw/Rice_MSC_Dataset/` and `data/raw/Rice_Image_Dataset/`, and makes those folders read-only. It prints `raw data verified and read-only` on success and exits 1 on a download failure or a checksum mismatch.

If the download fails (Kaggle may require a login), the script prints the manual steps:

1. Open the two dataset pages in section 2 in a browser.
2. Download both zips and name them exactly `rice-msc.zip` and `rice-image.zip`.
3. Put them in `data/raw/`, or point `RICE_ZIP_DIR` at their folder, then run `make get-data` again.

## 5. What is not in git

| Path | What it is | Filled by |
|---|---|---|
| `data/raw/` | the Kaggle zips and unpacked images | `make get-data` |
| `data/derived/` | `grains.parquet` and the 128 px image cache | `make data` (stages 1 to 4) |
| `models/baselines/`, `models/holdout/` | table-model and hold-out checkpoints | `make train`, `make holdout-train` |
| `results/smoke/`, `figures/smoke/`, `models/smoke/` | smoke-run outputs | `make smoke` |
| `docs/designs/`, `research/`, `.venv/`, `.claude/` | working notes and environment | not needed to reproduce |

In git: the code, `data/splits.csv`, `data/SHA256SUMS`, the models `predict.py` needs (`models/*.pt`, `models/*.joblib`, `models/routing.json`), `results/`, `figures/`, `logs/` and `prompts/`.

## 6. Directory map

- `src/`: numbered stage scripts `00` to `09`, `predict.py` (stage 10), `report.py`, and `common.py` (paths, seeds, feature sets, device, the test-split gate).
- `tools/`: `run_order.py` (stage runner), `freeze.py`, `leak_audit.py`, `stamp_prompts.py`, `check_prompts.py`, `smoke_images.py`, `recheck_r2.py`.
- `prompts/`: the AI prompts each script was generated from.
- `data/`: `splits.csv` and `SHA256SUMS` (tracked); `raw/` and `derived/` (not tracked).
- `models/`: trained models used by `predict.py`.
- `results/`: CSV and JSON outputs of every stage, the run log and the test-read ledgers.
- `figures/`: PNG figures.
- `logs/`: logs of the runs that produced the saved results.
- `tests/`: pytest suite.
- `report/`: the group report (`main.tex`, `refs.bib`, generated `numbers.tex` and `FACTS.md`, built PDF `F20DL-Group16-report.pdf`), `make_numbers.py` (writes every reported number from `results/`), `build.sh` (tectonic build), `check_report.py` (report lint) and `check_readme.py` (checks the commands in this README).

## 7. Reproduce

The only target that needs no training and no test data is:

```bash
make report
```

It runs `src/report.py`, which rebuilds `results/SUMMARY.csv`, `results/MANIFEST.csv`, `results/DISCLOSURES.csv` and the figures in `figures/` from saved CSVs under `results/`. It loads no models, uses no GPU and never calls the test gate. Its inputs are all tracked in git (`results/compare/`, `results/baselines/val_results.csv`, `results/neural/curves.csv`, `results/robustness/late_recall.csv`, `results/test_reads_disclosed.csv`, `results/code_lock.json`), so it needs no git-ignored file. A missing input only skips the matching table or figure and prints a message.

`make report` rewrites tracked files. After it, run `git status`: a clean tree means the saved outputs reproduce the tables and figures.

```bash
git status
```

Full re-run from raw data (long; needs the data from section 4). `tools/run_order.py` is the ordered stage runner; `make all` drives it. Its subcommands are `status`, `next`, `through N`, `table`, `rerun-read N --reason R` and `reopen ROW --reason R`. `status` only reports:

```bash
.venv/bin/python tools/run_order.py status
```

**Warning: the three test reads (stages 7, 8 and 9) are already spent. They are logged in `results/test_reads.csv` and must not be repeated.** Do not run `make compare`, `make score`, `make robustness`, `READS=yes make all` or `next --spend-read` on this repository. Any extra read has to be logged and disclosed (`results/test_reads_disclosed.csv`).

To exercise the whole pipeline without touching the real test split, use the smoke run (`make smoke`: a 2,000-grain slice that writes only to the `smoke` folders).

## 8. Predict one image

`src/predict.py` classifies one 250x250 RGB grain image using the saved models in `models/`. Exit code 2: no argument, or the image is unreadable or not 250x250 RGB. Exit code 3: a trained-model file is missing. Quote the path, because file names contain spaces and brackets.

Example, using the first train-split image in `data/splits.csv` (`Arborio-2`, file number 2; needs the raw data from section 4):

```bash
.venv/bin/python src/predict.py "data/raw/Rice_Image_Dataset/Arborio/Arborio (2).jpg"
```

Real output (run 2026-11-10 on macOS, Apple MPS; two library `UserWarning` lines from PyTorch and scikit-learn may print first and are harmless):

```text
image: Arborio (2).jpg
image side: Arborio (confidence 1.0000)
table side: Arborio (confidence 0.9998)
route: hgb
familiarity: familiar (table gate False, image gate False)
answer: Arborio (confidence 0.9998)
```

The output gives the image-side answer, the table-side answer (only when `data/derived/grains.parquet` exists and the file name has the form `<Class> (<n>).jpg`), the route taken, a familiarity verdict and the final answer. `make predict IMG=<path>` does the same.

## 9. Where things live

- Models: `models/` (`cnn_seed*.pt`, `mlp_seed*.pt`, `hgb_all106.joblib`, familiarity gates, `routing.json`).
- Results: `results/<stage>/` CSVs. `results/SUMMARY.csv` is the headline table; `results/MANIFEST.csv` maps requirements to output files.
- Figures: `figures/`.
- Run history: `results/run_log.csv`, `results/test_reads.csv`, `results/frozen.json`, `logs/`.
- Report: `report/F20DL-Group16-report.pdf` (source `report/main.tex`; rebuild with `bash report/build.sh`, needs tectonic). Every number in it comes from `report/numbers.tex`, regenerated by `.venv/bin/python report/make_numbers.py`.

## 10. Tests

```bash
make test
```

This runs `env -u RICE_SMOKE .venv/bin/python -m pytest -q tests` (pytest settings are in `pyproject.toml`). Acceptance tests gated behind the environment variable `RICE_GO=1` live in `tests/test_awaiting_go.py`; they are not part of a plain run. They were run once for the final acceptance check, with every file under `results/`, `models/`, `src/` and `tools/` fingerprinted before and after (see `docs/FINAL_SUBMISSION_CHECKLIST.md`).

## 11. Known limitations

The group's disclosed limitations are in `results/DISCLOSURES.csv` (13 items, written by `make report`) and in the disclosure table of the report. They are not restated here.

## 12. Contributions

| Member | Contribution |
|---|---|
| TBC | TBC (names to be confirmed by the group) |
| TBC | TBC |
| TBC | TBC |

## 13. Code-generation prompts

Each generated script carries a header recording the AI use, and the full prompts are in `prompts/`: `00-context.md` (shared context), `setup.md`, one file per stage (`00-get-data.md` to `10-predict.md`), `report.md` and `BUILD.md`. `make check` fails if a script header differs from its prompt file.
