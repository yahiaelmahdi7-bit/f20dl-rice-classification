# Stage 2

| Stage | Script | Meets | Content |
|---|---|---|---|
| 2 | `02_eda.py` | R2 | Class balance, distributions, correlation clusters, tint by class, Jasmine break, entropy against area, copy-column check, example grains |

Write `src/02_eda.py`. Train and validation only for anything using labels beyond class counts. Check for the figures: title, labelled axes with units, n stated, no clipped text. Include a chart of entropy columns against log area inside a class, and a table confirming each `ALLdaub4*` column against its mean column.

## Fix 1
Add these outputs. Train and validation only for anything using labels beyond class counts.
- `results/quality/data_dictionary.csv`: for each of the 106 columns, its family (read from the real column names), colour channel if any, missing count, min, median, max, the feature sets holding it, and median within-class correlation with log area.
- `results/quality/validity.csv`: infinite values, values outside a column's possible range (negative sizes, ratios outside their bounds), and constant or near-constant columns, with counts. Append any finding to the cleaning log.
- `results/quality/outliers.csv`: per class and column, the count of grains with robust z above 6, using median and MAD. A figure with the 20 most extreme grains as an image montage, labelled by grain ID and column. Default action is keep; record the action taken in the cleaning log.
- Trends: area and mean colour against file number per class as a rolling median in one figure, with lag-1 autocorrelation in `results/eda/file_order.csv`.
- Image side: mean grain image per class at 128 px, and a check that grain area ratios survive the resize to 128 px.
- `results/eda/insights.csv`: `id, finding, value, n, figure`. Short labels and computed values only, no sentences.
