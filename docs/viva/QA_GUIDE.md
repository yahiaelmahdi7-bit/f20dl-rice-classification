DRAFT for Group 16 review, not yet approved by the group.

# Mini-viva Q&A guide: F20DL Group 16

Fifteen likely questions, grouped by the brief's requirements (R1, the topic and question, is slide 1 and the report introduction). Each has a short answer, one evidence file, and one thing not to say. All paths are relative to the repo root.

## R2: Data analysis (feature selection, clustering, the late lot)

### Q1. How were the 16 features selected?
Short answer: We used mutual information, ANOVA and correlation-cluster pruning. We tried a grid of k with HGB and logistic regression. For each model we took the smallest k within one tenth of a point of its best validation accuracy, then took the larger of the two. That gave 16 columns, with validation accuracy 99.86%.
Evidence: `results/features/selected.json`
Do not say: "The 16 columns are the best possible set." We picked them by a rule on validation, not by a proof.

### Q2. K-means or GMM, and what do ARI, NMI and silhouette mean?
Short answer: K-means makes round clusters of similar size. A GMM allows clusters of different shape and size, so it fits better here: best GMM k=5 with ARI 0.9437, best k-means k=6 with ARI 0.8934. ARI and NMI compare clusters with the true varieties (1 is a perfect match; ARI is corrected for chance). Silhouette needs no labels: it measures how well separated the clusters are, and the GMM silhouette at k=5 is 0.2952. We scored on validation.
Evidence: `results/cluster/heldout.csv`
Do not say: "Clustering proves the varieties are separate groups." It matches the labels well, and that is all.

### Q3. What does the late lot show?
Short answer: A late Jasmine lot of 3283 grains is larger and bluer than early Jasmine: median area x1.81, length x1.35, tint x2.64. Unlabelled k-means at k=6 already puts a share 0.9896 of it in one cluster. The cause is not established.
Evidence: `results/eda/jasmine_lots.csv`
Do not say: "The late lot is a different camera" or "it is a different batch". We do not know.

## Setup and metrics (behind R3 and R4)

### Q4. Why accuracy and macro-F1?
Short answer: Classes are balanced, so accuracy is a fair headline. Macro-F1 averages the F1 of each class equally, so a model that fails badly on one variety cannot hide behind the others. They agree closely here, for example HGB 99.88% accuracy against its macro-F1 in the report table.
Evidence: `results/compare/test_metrics.csv`
Do not say: "Accuracy alone is enough." At this ceiling a handful of grains moves the second decimal.

### Q5. What is a Wilson interval, and why use it?
Short answer: It is a 95% confidence interval for a proportion such as accuracy. It behaves well near 100%, where the simple plus-or-minus formula can go past the ceiling or collapse to zero width. For HGB the test interval is 99.82% to 99.94%, and for the MLP seed mean 99.86% to 99.96%, so they overlap.
Evidence: `results/compare/test_metrics.csv`
Do not say: "The intervals overlap, so the models are identical." Overlap fits a tie; it does not prove one.

### Q6. How was the test data protected?
Short answer: We tuned on validation only. Then we froze the code and a hash of all settings, and made exactly 3 logged single test reads (compare, routing, robustness). Each read is recorded with the same frozen hash. Test reads are spent; they must not be rerun.
Evidence: `results/test_reads.csv`
Do not say: "No leakage" or "contamination-free." We say what we did, and the disclosures list the limits, including one logged code change after the lock.

### Q7. Why exact McNemar, and why Holm?
Short answer: Two models are scored on the same 11206 grains, so the predictions are paired. McNemar looks only at the grains where exactly one model is right. The exact version is needed because those counts are tiny. We fixed three pairs in advance at seed zero, and Holm correction controls the error rate across the three. HGB against MLP gave p = 0.7266, 0.7895 and 0.2266 across seeds: no significant difference.
Evidence: `results/compare/test_mcnemar.csv`
Do not say: "The MLP is significantly better than HGB." It is not, on any seed.

## R3: Baselines

### Q8. Why are HGB and the MLP a tie?
Short answer: HGB makes 13 errors in 11206 test grains, the MLP seeds make 11, 9 and 8. That is a handful of grains, and the Wilson intervals overlap. The paired test is not significant on any seed. So we report a tie and not a winner.
Evidence: `results/compare/test_seed_spread.csv`
Do not say: "MLP beats HGB because 99.92 is bigger than 99.88."

### Q9. What are decision trees good for in practice?
Short answer: A depth-four tree scores 99.03% and its rules can be read and audited. Its top splits use SHAPEFACTOR_3, then MAJOR_AXIS and PERIMETER, which are pixel-scale size measures. A change of camera scale moves grains across those thresholds. So we suggest it as an interpretable first stage with a familiarity check, recalibrated per camera.
Evidence: `results/baselines/tree_depth4.txt`
Do not say: "A depth-four tree is as good as the boosted model." It is clearly lower: 99.03% against 99.88%.

## R4: Neural models and routing

### Q10. Why do the CNN seeds differ?
Short answer: The CNN makes between 15 and 50 errors across three seeds. Seeds can drift on the GPU because it is not fully deterministic, the CNN validation loss jumps from epoch to epoch, and early stopping then ends the runs at different points. The MLP curves are smooth. This fits the picture but we do not claim it is the proven cause.
Evidence: `results/neural/curves.csv`
Do not say: "GPU nondeterminism causes the gap." We only show it is plausible.

### Q11. How does routing work, and why are its numbers optimistic?
Short answer: HGB answers a grain when its confidence is above a floor. The CNN takes grains HGB declines, if its own confidence is above its floor. Two familiarity gates mark grains unlike anything seen as unfamiliar, and the rest are unsure. On seed zero it answers a share 0.9797 of test grains with 2 errors (99.98%), unsure 0.0046, unfamiliar 0.0156. The floors, temperature and cuts were set on the same validation rows, so these figures are optimistic.
Evidence: `results/routing/val_calibration.csv`
Do not say: "The chain is 99.98% accurate on any new data."

## Honesty and limits

### Q12. Why does the hold-out matter more than test accuracy?
Short answer: Late grains are spread across train, validation and test, so test accuracy cannot see the shift. When we train without the lot and test on its 492 late test grains, HGB on all106 recovers 1.42%, at mean confidence 0.9263. The CNN seeds recover 0.41%, 1.42% and 16.06%. Size-free features keep 84.55%. With the lot in training, recall returns.
Evidence: `results/robustness/late_recall.csv`
Do not say: "Size-free features fix the problem." They survived this one test; one dataset, one camera.

### Q13. Why can grey-world not isolate colour for Jasmine?
Short answer: Grey-world removes the overall tint. On the 492 late test grains it takes CNN recall from 99.93% to 4.00%. We pre-set a check that the correction does not track grain area. It passed only for Arborio, and for Jasmine the largest absolute correlation was 0.9293. So colour and size are tangled there.
Evidence: `results/robustness/greyworld_gain_vs_area.csv`
Do not say: "Colour or tint causes the CNN failure." The cause of the late lot is not established.

### Q14. What is the old smoke overlap disclosure?
Short answer: An early smoke-test slice contained some real test grains, and throwaway smoke models were fitted on them. No real model or choice used those models. The slice now comes from real train and validation grains only. The counts by slice split are in the evidence file.
Evidence: `results/smoke_overlap.csv`
Do not say: "There was never any overlap." Say what it was and that no real result used it.

### Q15. How do I run predict.py?
Short answer: Run `python src/predict.py <image.jpg>`, with the path in quotes if it has spaces. It predicts one image. It is not a test read. Do not rerun the experiments: the test reads are spent.
Evidence: `src/predict.py`
Do not say: "I can retrain it live." We do not retrain in the viva.
