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

## Fix 2
- Outlier figure: all 20 picks came from one column, kurtosisCr, because its tiny spread inflates robust z. Pick at most 2 grains per column and at most 8 per class, strongest first, and skip columns with zero MAD. Label each picture with class, column, value and class median. Write the picks to `results/quality/outlier_picks.csv`: `grain_id, class, column, value, class_median, robust_z, action`, with action `keep`.
- Entropy check: the entropy columns are flagged out of range. A manual check on 40 photos matched `entropyRR` and `entropyRG` within 0.07% with `-sum(s^2 * ln(s^2))` over non-zero pixel values of the whole 250px image, RGB on the 0-255 scale. H and S matched less closely. Write `results/quality/entropy_check.csv` on 200 train grains, for channels RR RG RB (0-255) and H S V (0-1, matplotlib `rgb_to_hsv`): `column, scale, n, pearson, median_ratio, min_ratio, max_ratio`. Report what comes out. Do not tune the formula to force a match, and add no packages. Add the formula and file name to the `validity_out_of_range` reason in the cleaning log.

## Fix 3
- `results/eda/jasmine_lots.csv` names its first column `feature`.
