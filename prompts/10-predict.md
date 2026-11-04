# Stage 10

**Stage 10 `predict.py`.** Input is one image path. Unreadable files, or anything other than 250x250 RGB, exit 2 with a message saying what was expected. If the file name matches a `grain_id`, HistGB `all106` uses that grain's table row; otherwise the table side prints "no table row" and only the image side answers, since colour features for a new image cannot be recomputed. Missing `models/` files exit 3, naming the file and `make train`. Output: each model's answer and confidence, the route, and familiarity. A non-rice image should print `unfamiliar`. No labels are read and no metric computed.

Checks: `predict.py` and `make smoke`. Check: three known test grains predicted live; 250x250 non-rice image prints `unfamiliar`; wrong-size image exits 2; clean clone passes smoke.

Write `src/predict.py`. The familiarity check on the image side uses `models/gate_image.joblib`; the table side uses `models/gate_table.joblib` when a table row exists.

## Fix 1
The table read takes `grain_id` and `all106` only, so no label is loaded. If `data/derived/grains.parquet` is absent, the table side prints "no table row" as for any unknown file. The missing-file message names `make train`, then `make fit`, because the routing and gate files come from `make fit`.
