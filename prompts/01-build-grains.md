# Stage 1

| Stage | Script | Meets | Content |
|---|---|---|---|
| 1 | `01_build_grains.py` | R2 | Tidies `research/audit/` into `src/`. Details below |

**Stage 1 details.** Features come from `Rice_MSC_Dataset.arff`, `?` read as missing; `msc_row` is the zero-based data row. Links are rebuilt by the audit's Otsu measures and one-to-one matching. `grain_id` is `<Class>-<file_no>`. Of each duplicate pair, the lowest file number stays. The 8 rows with `?` stay, so table and image sets match; median imputation and scaling are fitted on train inside each pipeline. Late lot means Jasmine with file number at least 11718, taken from `Jasmine (N).jpg`. Full mode asserts 75,000 links, 297 drops, 74,703 grains, 3,283 late grains, and agreement with committed `splits.csv`. `--subset` mode skips those global asserts.

Method to reproduce, from the audit:
- Per image: grey = 0.2989R + 0.5870G + 0.1140B; Otsu threshold; keep the largest connected component; measure area, major and minor axis, mean R, G, B over the grain mask (scikit-image `regionprops`, scipy `ndimage`). Multiprocessing pool.
- Link per class: divide the 6 measures `AREA, MAJOR_AXIS, MINOR_AXIS, meanRR, meanRG, meanRB` by each table column's standard deviation; `cKDTree` 8 nearest rows for each image; sparse matrix of those distances plus 1e-6; `scipy.sparse.csgraph.min_weight_full_bipartite_matching` for the one-to-one assignment.
- Duplicates: 297 exact duplicate table rows, all Ipsala (compare after rounding to 6 decimals), which line up with 297 duplicate image pairs.
- Reference implementations: `research/audit/extract_images.py`, `link_rows.py`, `build_link.py`, `jasmine_lot.py`.

Write `src/01_build_grains.py`. It writes `data/derived/grains.parquet`, `splits.csv` (only if absent; otherwise it asserts agreement) and the `uint8` image cache.

## Fix 1
Write `results/quality/cleaning_log.csv`, one row per check: `check, scope, rows_affected, action, reason`. Cover image decode and size, `?` cells, exact duplicate pairs, extra specks beside the grain, and link distance. List the 8 rows with `?` in `results/quality/missing_cells.csv` with class, column and file number. Write `results/quality/link_check.csv` and a figure showing table `AREA` against image-measured area per class, and median link distance for the true match against a shuffled control. All values computed in this run.

## Fix 3
- Safe writes. `data/derived/grains.parquet` and the image memmap are written to `.tmp` names in the same folder. Only when both are complete is the memmap renamed into place, then the parquet. A crash before that leaves both old files untouched, and a new parquet is never beside an old memmap. The measures cache (`measures.parquet`) is written the same way. File contents are unchanged.
- `--subset N` without `RICE_SMOKE=1` exits before writing anything. Outside smoke mode it would overwrite the committed `data/splits.csv` and the full-size derived files with a subset.

## Fix 9
- `--subset N` slices only from images whose id (`cls-file_no`) sits in `train` or `val` of the committed `data/splits.csv`. An allow-list, not a block-list: `splits.csv` holds 74,703 of 75,000 grains, and a grain dropped as a duplicate can be the twin of a test grain, so blocking test ids alone would not seal the slice. The slice fraction is taken from the allowed pool, so the slice keeps N images. Reason: the old slice held 298 real test grains (`results/smoke_overlap.csv`).
- After the slice is split, stage 1 stops if any class lacks grains in the slice's train, val or test, so the smaller pool cannot silently lose one. The `full` path is not touched.
