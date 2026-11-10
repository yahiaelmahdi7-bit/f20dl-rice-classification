DRAFT for Group 16 review, not yet approved by the group.

# Mini-viva speaker plan: F20DL Group 16

Deck: `docs/viva/deck.pptx` (5 slides). Total talk: 7 minutes. That is about 80 seconds per slide, with 20 seconds spare. Speaker names are left blank on purpose.

Rule for everyone: say the number on the slide, then say what it means in one plain sentence. Do not read the bullets out.

## Slide 1: Problem, data and the question

- Speaker 1: ______
- Time budget: 1:20
- Number to say out loud: **74703 grains**

Talking points:
1. We study five rice varieties: Arborio, Basmati, Ipsala, Jasmine and Karacadag.
2. Our question is simple. Which works better on the same grains: numbers someone measured, or the raw photo? And what do near-perfect scores really depend on?
3. We joined two datasets. One has 75000 images. The other has 106 measured columns for the same grains. We rebuilt all 75000 links one-to-one.
4. We dropped 297 exact duplicate pairs, all Ipsala. That leaves 74703 grains.
5. We split 70/15/15 by class and by lot: train 52292, validation 11205, test 11206.

Handover: "So the data is clean enough to use, but it hides something. Speaker 2 will show what."

## Slide 2: What the data hides

- Speaker 2: ______
- Time budget: 1:20
- Number to say out loud: **x1.81** (median area of the late Jasmine lot against early Jasmine)

Talking points:
1. Plot Jasmine by image file number. Partway through, area and tint jump. That is the late lot: 3283 grains.
2. Late grains are bigger and bluer. Median area is x1.81, length x1.35, tint x2.64.
3. We do not know why. It could be a new batch, a new camera setting, or something else. We say the cause is not established.
4. Clustering with no labels finds it too. At k=6, k-means puts a share 0.9896 of the late lot in one cluster.
5. The best mixture model (GMM) has k=5 and ARI 0.9437. The best k-means has k=6 and ARI 0.8934.

Handover: "Now that we know the data has a break in it, Speaker 3 will show how the models score."

## Slide 3: Setup and baselines

- Speaker 3: ______
- Time budget: 1:20
- Number to say out loud: **99.03%** (depth-four tree; the best baseline, HGB, is 99.88% with 13 errors in 11206 grains)

Talking points:
1. Rules first. We tuned on validation only, froze the code, and read the test set exactly 3 times, each one logged.
2. Almost every model is near the ceiling. HGB is 99.88%, logistic regression 99.83%, kNN 99.79%.
3. Shape alone is lower: HGB 97.99%. Colour alone: HGB 99.14%.
4. The depth-four tree scores 99.03% and you can read its rules. It splits first on SHAPEFACTOR_3, then on MAJOR_AXIS and PERIMETER.
5. MAJOR_AXIS and PERIMETER are measured in pixels. If the camera scale changes, grains cross those thresholds. So use the tree as a readable first stage with a familiarity check, recalibrated per camera.

Handover: "Those are the simple models. Speaker 4 will compare the two neural models."

## Slide 4: MLP against CNN

- Speaker 4: ______
- Time budget: 1:20
- Number to say out loud: **99.92%** (MLP seed mean; HGB is 99.88%)

Talking points:
1. The MLP uses the numbers. It makes 11, 9 and 8 errors on its three seeds. HGB makes 13. We call that a tie: the exact McNemar test finds no significant difference on any seed (p = 0.7266, 0.7895, 0.2266).
2. The CNN uses the images. Its errors run from 15 to 50 across three seeds. Mean accuracy is 99.69%.
3. HGB against the CNN is significant for seed zero and seed two, and not for seed one. So the CNN gap depends on the seed.
4. Look at the learning curves. The CNN validation loss jumps around, the MLP is smooth. That fits what we see, but we do not claim it is the cause.
5. Routing chain: HGB answers when it is confident, the CNN takes what HGB declines, the rest is flagged. On seed zero it answers a share 0.9797 of test grains with 2 errors. We set the floors on validation, so this is optimistic.

Handover: "All of these scores are high. Speaker 5 will show what they do not tell us."

## Slide 5: What the scores do not show

- Speaker 5: ______
- Time budget: 1:20
- Number to say out loud: **1.42%** (HGB recall on the late lot when it never saw the lot)

Talking points:
1. We trained without the late lot, then tested on its 492 late test grains. HGB on all 106 columns gets 1.42% of them, and its mean confidence is still 0.9263. It is wrong and sure.
2. The CNN seeds get 0.41%, 1.42% and 16.06%. HGB on seven size-free shape ratios keeps 84.55%.
3. With the lot in training, recall comes back (HGB all106: 100.00%). So the near-perfect test score mostly says the test set looks like the training set.
4. Removing the overall tint takes CNN recall from 99.93% to 4.00%. But our check that tint correction does not track grain area failed for Jasmine (0.9293). So we cannot separate colour from size there.
5. Limits: one dataset, one camera, cause unknown, routing floors set in-sample. We logged 13 disclosures. Takeaway: add a familiarity check, and do not trust test accuracy alone.

Handover: "That is our summary. We are happy to take questions."

## Timing table

| Slide | Speaker | Topic | Time |
|---|---|---|---|
| 1 | Speaker 1 | Problem, data, question | 1:20 |
| 2 | Speaker 2 | What the data hides | 1:20 |
| 3 | Speaker 3 | Setup and baselines | 1:20 |
| 4 | Speaker 4 | MLP against CNN, routing | 1:20 |
| 5 | Speaker 5 | What scores do not show | 1:20 |
| | | Spare for handovers | 0:20 |
| | | **Total** | **7:00** |

Practise once with a timer. If you run over, cut talking point 5 on slides 1 to 4, never the number to say out loud.

## If the group has fewer than five members

- Four members: merge slides 1 and 2 into one 2:00 talk (Speaker 1). Keep slides 3, 4, 5 as they are with Speakers 2, 3, 4. Drop the cluster numbers from slide 2 if short of time.
- Three members: merge slides 1+2 (Speaker 1, 2:00) and 4+5 (Speaker 3, 2:40 at most). Speaker 2 keeps slide 3.
- Keep the footer on each slide correct: edit the "Speaker n" text box when you re-assign slides. The deck still has 5 slides.

## If asked to demo the code

Only two commands are safe.

1. `make report` rebuilds the tables and figures from saved outputs. It does no training.
2. `python src/predict.py <image.jpg>` predicts the variety of a single image. Put the path in quotes if it has spaces.

Say plainly: the test reads are spent. We made exactly 3 logged test reads and they must not be rerun. Do not run any other make target, and do not retrain.
