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
