# Stage 0

| Stage | Script | Meets | Content |
|---|---|---|---|
| 0 | `00_get_data.py` | Reuse | Uses zips already in `data/raw/` or `$RICE_ZIP_DIR`; otherwise downloads `muratkokludataset/rice-msc-dataset` and `rice-image-dataset` from `https://www.kaggle.com/api/v1/datasets/download/muratkokludataset/<slug>` (no token needed on 2026-11-04). Verifies committed `data/SHA256SUMS`, unpacks, makes raw read-only. On failure, prints the manual download steps and exits 1 |

Write `src/00_get_data.py`. The committed checksum file is `data/SHA256SUMS` (entries `rice-msc.zip`, `rice-image.zip`). Use streaming GET with a progress line; a HEAD request returns 404 on this URL, so do not use HEAD.
