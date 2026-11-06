# Build instructions (this file is the prompt given to the build agent)

Build the F20DL rice study code in this folder, from the prompts in `prompts/`.

Order: `prompts/setup.md`, then stages 00 to 10 in order, then `prompts/report.md`. `prompts/00-context.md` applies to every stage. Treat each stage prompt as the specification. Read `research/audit/*.py` as reference only. Do not read or copy prose from `research/` or `docs/` into the repo.

Data already on disk: `data/raw/rice-msc.zip`, `data/raw/rice-image.zip`, unpacked `data/raw/Rice_MSC_Dataset/` and `data/raw/Rice_Image_Dataset/`, checksums at `data/raw/SHA256SUMS`. Copy that file to `data/SHA256SUMS` (the committed copy). Stage 0 must detect verified data and not download. `.venv` exists on Python 3.12; let `uv sync` manage it.

Prompt record:
- Keep `prompts/*.md` text as it is. If you change behaviour to fix something, append the instruction you acted on under `## Fix N` in that stage's prompt, then rerun `tools/stamp_prompts.py`.
- Every script ends the build stamped; `tools/check_prompts.py` passes.

Run for real on the full data: stage 0, stage 1, then `06_neural.py --overfit-test` and `06_neural.py --time-epoch` (one full SmallCNN epoch at the default config). Commands may take up to 10 minutes in the foreground; set the tool timeout to match.

Do NOT run full stages 2 to 9. They run afterwards as a separate job. Prove each script instead through `make smoke`, on a real 2,000-grain slice, which must run every stage from 1 to 10 plus report, including stage 8 `--fit` and `--score`, stage 9 `--train` and `--score`, and `predict.py` on three real images.

Add `tests/` with pytest, run by `make test`, covering at least: `load_split("test")` raises from an unlisted caller; it raises when `results/frozen.json` is missing; a permitted read appends to `results/test_reads.csv` with the frozen hash; train, validation and test grain IDs do not overlap; class and lot shares match across splits; the size-free lists contain no size column; grey-world gains come from mask pixels only; `predict.py` exits 2 on a wrong-size image and 3 on a missing model file.

Do not write a README, notebook, or any prose document. `make help` is the usage text.

DISCIPLINE (non-negotiable):
- Lane: work only in this folder. Do not edit `docs/`, `research/`, `brief/`, `.claude/`, or anything in `data/raw/` beyond stage 0's read-only step. Never delete data.
- Commits: `git init` if needed, atomic commits, each message says what and why. NEVER push or add a remote. Before each commit, confirm `git status` shows no `data/raw`, `data/derived`, `research/` or `docs/designs/` file.
- Gates: run tests and smoke in the FOREGROUND and wait for them. No background shells, no watch modes, no until-loops. Kill anything you started before finishing.
- Claims: done means you ran the gate and pasted its passing output. "It ran" is not a pass: check that output values are plausible and that row counts match.
- No Jev, no external model or API call, and no LLM anywhere in the pipeline. Routing is local statistics.
- The shell is zsh: unquoted `$var` does not word-split, and `rg` is a shell function, so use `grep` inside scripts.
- If blocked: say BLOCKED plus one line why. Do not invent extra scope.

Final message: pass or fail per item, with pasted output for: `uv sync`; stage 1 asserts (75,000 links, 297 drops, 74,703 grains, 3,283 late grains); split shares; overfit test; seconds per epoch on the device used; `make test`; `make smoke`; `check_prompts.py`; `git log --oneline`; `git status`. List every `## Fix` you appended, and anything in a stage prompt you could not meet.

## Resume prompts

### Resume 1, R2 first

You were stopped on purpose. STANDING ORDER FROM YAHIA, which overrides BUILD.md and your earlier plan: no model training until he says go. The data side (brief R2) comes first.

YOU MAY EXECUTE ONLY: src/00, 01, 02, 03, 04, in smoke mode on the 2,000-grain slice. Feature scoring in 03 and clustering in 04 are R2 work and are allowed. Do not run any stage in full mode. The reviewer launches the full data run.

NEVER EXECUTE, on any data, in any mode: src/05_baselines.py, src/06_neural.py (including --overfit-test and --time-epoch), 07, 08, 09, predict.py, or make smoke / wedge / baselines / neural / train / all. Check those files with python -m py_compile only. Any test that fits a model or needs predictions is written and marked skip with reason "awaits go". Do not delete or rewrite anything already under results/.

ORDER OF WORK
1. prompts/01, 02, 03, 04 each now end with a "## Fix 1" block. Implement every item in src/01-04 and smoke those four stages. Confirm each named output file exists and is non-empty in the smoke output.
2. Write the Makefile. `make -n all` must list every stage. Add a target `data` that runs stages 1-4 in full mode, stopping before stage 5. Do not run it.
3. Write stages 07-10, predict.py, report, tools and tests. Write only, under the limits above.
4. Commit in small steps. No data in git. No README or report prose.

When finished, list every command you executed since this resume, so the reviewer can confirm nothing outside the limit ran.

### Resume 2, Fix 2 round

Fix 2 round. Every file below now ends with a `## Fix 2` block: prompts/setup.md, 02-eda.md, 04-cluster.md, 05-baselines.md, 06-neural.md, 07-compare.md, 09-robustness.md, report.md. Implement every item.

HARD LIMITS. These come from Yahia and override BUILD.md and anything else you read. Last round, stages 5 and 6 ran after his stop order. That must not happen again.

YOU MAY EXECUTE ONLY:
- RICE_SMOKE=1 .venv/bin/python src/02_eda.py and RICE_SMOKE=1 .venv/bin/python src/04_cluster.py, on the 2,000-grain slice. Run 01 or 03 in smoke mode only if a shared change breaks them.
- py_compile, env -u RICE_SMOKE .venv/bin/python -m pytest -q tests, tools/stamp_prompts.py, tools/check_prompts.py, make -n, git.

NEVER EXECUTE, on any data, in any mode: src/05_baselines.py, src/06_neural.py (including --overfit-test and --time-epoch), 07, 08, 09, predict.py, report.py, or make data / smoke / wedge / baselines / neural / train / all. Stages 05, 06, 07, 09 and report get py_compile only. No stage runs in full mode: the reviewer launches the full data run.

ORDER OF WORK
1. setup.md Fix 2 first, then stages 02 and 04, and smoke those two. Open the saved cluster_heatmaps.png and eda_outlier_montage.png and confirm the fixes show in the images, not just that the files exist.
2. Stages 05, 06, 07, 09 and report: write only, py_compile only.
3. Re-stamp with tools/stamp_prompts.py, then tools/check_prompts.py must report 0 problems. Commit in small steps, including the prompts/ and docs/designs/rice-classification.md changes already in the working tree. No data, results or brief files in git. No README or report prose.

Do not edit prompts/*.md yourself. If a Fix 2 item cannot be done as written, leave it, and say so in your report.

In your final report, list every command you executed since this resume, so the reviewer can confirm nothing outside the limit ran. Give the new stage 4 ARI at k=5 for all106 and distinct91, and the seed-to-seed range from gmm_stability.csv.

## Fix 3
- `tests/test_contract.py`: read the code and CSV headers, fit no model. Check that a reader never asks for a column its writer does not produce, that a CSV header matches its `ARTEFACTS` entry, and that no stage defines a function with the same name as one in `common.py`. Without `results/` the header check reports xfail with that reason, never a silent skip.
- `tests/contract_known.txt`: one line per real mismatch in a file Foundation does not own. The test passes on a listed mismatch and fails on anything new or on a line that has stopped being true.
- `tests/test_foundation.py`: small model-free tests for the Foundation fixes and for the shared lines the 2,000-grain slice cannot reach.
- `tests/test_data.py`: small model-free tests for stages 0 to 2 and `tools/leak_audit.py`. Stage 2 is read as source and never imported, because importing it runs it.

## Fix 4
- `tests/test_evaluation.py`: model-free tests for stages 7, 8, 9 and `tools/freeze.py`, on made-up rows. Each stage stops before its test read when a needed file is missing, freeze refuses a bad control and hashes each file under its own path, a file changed since freeze stops the read before it is logged, and a thin read still writes its tables. A stage's `main()` runs only to prove a refusal.
- `tests/test_awaiting_go.py`: the hold-out test reads `train_ids` from every hold-out arm and checks that none is a late grain. `test_every_read_is_kept_under_one_hash_and_extras_are_disclosed` checks the read log after go: committed, never edited, one hash that matches `frozen.json`, and every read past a stage's first disclosed with a reason. `test_stage7_tables_reproduce_from_saved_predictions` reruns stage 7's tables on the saved test predictions in a temporary folder and checks each file matches the one the read wrote, so plan step 8 has a check behind it; a crash after a read still costs a second logged read.

## Fix 5 (harden before go)
- `tests/test_harden_seal.py`: model-free tests for the seal fixes and the real gate in smoke. Stage 1 `--subset` keeps only train and val ids and stops on a class missing from a slice split. Real mode keeps today's exact paths; under `RICE_SMOKE=1` the freeze file, the read log and the models folder sit in smoke folders; a smoke read creates nothing outside them. Hand-worked frames only, no fit.
- `tests/test_awaiting_go.py`: `test_stage7_tables_reproduce_from_saved_predictions` moves to validation. It rebuilds the `val_*` tables (metrics, recall, confusion, seed spread, McNemar, by lot, by angle, by angle and lot) from the saved validation predictions through the same `tables()` and compares them with the files stage 7 wrote. It touches no sealed label and adds nothing to the read path. After this no `@GO` test loads a test predictions file; `test_no_go_test_loads_a_test_predictions_file` pins that.
- `tests/test_harden_seal.py`: `report.py` draws the learning-curve and late-lot recall figures from small made-up CSV frames, leaves a model's missing `late_all` cell empty, and draws nothing from a missing or empty file.
- `tests/test_harden_seal.py`: `C.code_digest` does not move when the prompt-record block is rewritten and does move on a one-character edit, for a Python file and a Makefile alike; a hold-out CNN file whose `run` holds an older code digest is retrained.
- `tests/test_harden_seal.py`: `report.main()` runs with `test_reads_disclosed.csv` missing and writes `DISCLOSURES.csv` and `MANIFEST.csv`; with a ledger of two rows, one carrying a `code-changed` reason, the extra-read rows count it and name it; every disclosure evidence file and every new output has a `MANIFEST.csv` row.
