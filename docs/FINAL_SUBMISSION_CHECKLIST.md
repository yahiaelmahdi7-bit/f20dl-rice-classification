DRAFT for Group 16 review, not yet approved by the group.

# Final submission checklist: F20DL Group 16

Checked on 2026-11-10 against the coursework brief (sections 4, 5, 7 and 8). The marking rubric, the CW Handbook and the viva slot were not available when this was written, so nothing here is checked against them.

Status words: **Ready** (done and checked), **Needs group approval** (done, the group must read and agree), **Incomplete** (the group must supply something), **Not verified** (could not be checked).

## 1. Requirements against deliverables

| Requirement (brief section 4) | D2a report | D2b code and outputs | D3 viva |
|---|---|---|---|
| R1 topic, objectives, problem | Introduction: question and objectives O1 to O3 | `README.md` section 1 | slide 1 |
| R2 cleaning, EDA, visualisation, feature selection, clustering with metrics | Section II (linking, cleaning, late Jasmine lot, 16 selected columns, k-means and GMM with ARI, NMI, silhouette, BIC), Fig. 1 and 2 | `src/00` to `src/04`, `results/quality/`, `results/eda/`, `results/features/`, `results/cluster/` | slide 2 |
| R3 at least three baselines, metrics justified, decision trees in practice | Sections III and IV-A, Tables III and IV, Fig. 3 and 6, discussion paragraph on trees | `src/05_baselines.py`, `results/baselines/`, `figures/baselines_tree_depth4.png` | slide 3 |
| R4 MLP and CNN trained, tuned, compared with the baselines | Sections III, IV-B, IV-C, Tables II, IV and V, Fig. 4 and 5 | `src/06_neural.py`, `src/07_compare.py`, `results/neural/`, `results/compare/` | slide 4 |
| Beyond the brief: routing, late-lot hold-out, tint arm, disclosures | Sections IV-D to IV-F, Tables VI and VII | `results/routing/`, `results/robustness/`, `results/DISCLOSURES.csv` | slide 5 |

All four requirements are covered in all three deliverables. Status of each cell: **Needs group approval**.

## 2. Deliverable rows

| Item | Evidence | Status |
|---|---|---|
| Report is a PDF with the six named sections | `report/F20DL-Group16-report.pdf`, source `report/main.tex` | Needs group approval |
| Report is at most 6 pages | Body text ends on page 4, the disclosure table is page 5, the contributions appendix is page 6 (not counted). Built with tectonic. | Ready; **re-check the page count in the tool the group submits from** (Overleaf may break pages differently) |
| Every number in the report comes from a saved result | `report/make_numbers.py` writes 1,240 values to `report/numbers.tex`; `report/FACTS.md` lists each value with its source file and row. `report/check_report.py` passes all six checks. | Ready |
| Report was read end to end against the results | Seven sentences or captions were corrected on 2026-11-10 (interval units, seed spread units, a wrong figure caption, the McNemar family, the tint paragraph, an overclaim in the discussion, a validation-agreement phrase) | Needs group approval |
| References | Four references in `report/refs.bib`, IEEE style | Not verified against the Handbook's citation style |
| Contributions appendix | Report appendix and `README.md` section 12 hold blank rows | **Incomplete: the group fills in names and contributions** |
| Code repository is documented | `README.md` (13 sections); `report/check_readme.py` passes 7 of 7 command lines | Needs group approval |
| Submission runs on another machine | Fresh clone in a scratch folder, `uv sync`, `make report`: `git status` clean afterwards, so the saved outputs rebuild every table and figure exactly. `make test` in the clone: 430 passed, 9 skipped. | Ready on macOS (Apple silicon). Windows and Linux: Not verified |
| Data is accessible | Two public Kaggle datasets, fetched and checksum-verified by `make get-data` | Ready (download itself not re-run on 2026-11-10) |
| Deck of at most 5 slides | `docs/viva/deck.pptx` (5 slides, editable) and `docs/viva/deck.pdf`; built by `docs/viva/make_deck.py` | Needs group approval |
| All members present and answer questions | `docs/viva/SPEAKER_PLAN.md` (5 speaker slots, 7:00 timing), `docs/viva/QA_GUIDE.md` (15 questions with evidence files) | **Incomplete: the group assigns speakers and rehearses** |
| Code demo if asked | Two safe commands in `SPEAKER_PLAN.md`: `make report` and `src/predict.py` | Ready |

## 3. Acceptance run (2026-11-10)

| Check | Result |
|---|---|
| `RICE_GO=1` full test suite (`pytest -rA tests`, `RICE_SMOKE` unset) | **439 passed, 0 failed, 0 skipped, 72 warnings** |
| Plain test suite, committed record `results/checks/pytest_report.txt` | 430 passed, 9 skipped. The 9 skipped tests are the ones that wait for `RICE_GO`; with it set they run and pass. |
| Fingerprint of every file under `results/`, `models/`, `src/`, `tools/` and `data/SHA256SUMS` (449 files, SHA-256), before and after the run | Identical |
| Git commit before and after | Identical (`23c8c5d`) |
| Model training or test reads during this work | None |

## 4. Disclosures: where each one appears

All 13 rows of `results/DISCLOSURES.csv` are in report Table VII (checked by `report/check_report.py`, check c). Five are also discussed in the text:

| Disclosure | Also in report text |
|---|---|
| in-sample calibration and routes | IV-D Routing; Limitations |
| validation holds late grains | Limitations |
| shuffled-label scores | III, baseline settings |
| holdout cnn retrained | Limitations (GPU nondeterminism) |
| tint gain vs area | IV-F Grey-world tint arm |
| the other eight | Table VII only |

## 5. Use of generative AI (brief section 8)

What the brief says, as written: "No GenAI generated content is permitted in your assessment submission; this statement does not include code." GenAI may be used for editing and proofreading, and for code generation, which the group must evaluate itself. "If you use it to generate code, you should explicitly say so in comments, by providing the prompt you used in full. If you use it to prepare part of your report you should have a clearly outlined disclaimer."

What the repository shows:

| Item | State | Status |
|---|---|---|
| Stage code in `src/` and `tools/` | Each script carries its prompt record; full prompts are in `prompts/`; `make check` fails if they differ | Ready |
| New helper scripts (`report/make_numbers.py`, `check_report.py`, `check_readme.py`, `build.sh`, `docs/viva/make_deck.py`) | Each has a header saying it was generated with AI assistance, with its prompt | Ready |
| Report text, README text, slides, speaker plan, Q&A guide | Drafted with AI assistance and marked DRAFT. The report has no AI disclaimer yet. | **Needs group decision: how the final text is written and what disclaimer it carries, in line with section 8** |

## 6. Not checkable from the repository

- Marking rubric and submission requirements ("check your assessment page").
- CW Handbook, including the citation style.
- Viva date, time slot and who attends.
- D4 peer assessment form (each member, privately).
- The brief itself is kept out of git (`brief/` is untracked on purpose).

## 7. Before submitting

1. The group reads the report, README and deck, and agrees or edits.
2. Fill in names and contributions (report appendix, README section 12, slide footers).
3. Settle the section 8 decision and add the disclaimer.
4. Remove the DRAFT lines from each file.
5. Rebuild the report (`bash report/build.sh`), re-run `python3 report/check_report.py`, and confirm the page count in the tool used to submit.
6. Check the rubric, the Handbook and the submission page.
