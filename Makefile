PY ?= .venv/bin/python
IMG ?= data/raw/Rice_Image_Dataset/Arborio/Arborio (18).jpg
SM = RICE_SMOKE=1
.DEFAULT_GOAL := help
.PHONY: help get-data data grains eda features cluster baselines neural wedge fit freeze compare score holdout-train robustness report predict train all smoke test stamp check

help:  ## list targets (this is the usage text)
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

get-data:  ## stage 0: detect verified raw data, download only if missing
	$(PY) src/00_get_data.py
grains:  ## stage 1: measure, link, dedupe, split, cache images
	$(PY) src/01_build_grains.py
eda:  ## stage 2: exploratory analysis and quality tables
	$(PY) src/02_eda.py
features:  ## stage 3: feature ranking and selected list
	$(PY) src/03_features.py
cluster:  ## stage 4: k-means and Gaussian mixture
	$(PY) src/04_cluster.py
data: grains eda features cluster  ## stages 1-4 on the full data, stops before any model
baselines:  ## stage 5: table models
	$(PY) src/05_baselines.py
neural:  ## stage 6: MLP and SmallCNN
	$(PY) src/06_neural.py
train: baselines neural  ## stages 5 and 6
wedge:  ## HistGB all106, one SmallCNN seed, validation comparison
	$(PY) src/05_baselines.py --only hgb:all106
	$(PY) src/06_neural.py --only cnn --seeds 0
	$(PY) src/07_compare.py --split val
fit:  ## stage 8 --fit on validation
	$(PY) src/08_routing.py --fit
freeze:  ## hash configurations, seeds, floors, cuts, test pairs
	$(PY) tools/freeze.py
compare:  ## stage 7 on test (one logged read)
	$(PY) src/07_compare.py --split test
score:  ## stage 8 --score on test (one logged read)
	$(PY) src/08_routing.py --score
holdout-train:  ## stage 9 --train, no test read
	$(PY) src/09_robustness.py --train
robustness:  ## stage 9 --score on test (one logged read)
	$(PY) src/09_robustness.py --score
predict:  ## stage 10: predict one image, IMG=path
	$(PY) src/predict.py "$(IMG)"
report:  ## summary tables, manifest and figures from saved outputs
	$(PY) src/report.py
all: get-data data train fit freeze compare score holdout-train robustness predict report  ## every stage in order
stamp:  ## copy prompts into file headers
	$(PY) tools/stamp_prompts.py
check:  ## fail if a prompt record differs
	$(PY) tools/check_prompts.py
test:  ## pytest
	env -u RICE_SMOKE $(PY) -m pytest -q tests
smoke:  ## every stage 1-10 and report on a 2,000-grain slice
	$(SM) $(PY) src/01_build_grains.py
	$(SM) $(PY) src/02_eda.py
	$(SM) $(PY) src/03_features.py
	$(SM) $(PY) src/04_cluster.py
	$(SM) $(PY) src/05_baselines.py
	$(SM) $(PY) src/06_neural.py
	$(SM) $(PY) src/07_compare.py --split val
	$(SM) $(PY) src/08_routing.py --fit
	$(SM) $(PY) tools/freeze.py
	$(SM) $(PY) src/07_compare.py --split test
	$(SM) $(PY) src/08_routing.py --score
	$(SM) $(PY) src/09_robustness.py --train
	$(SM) $(PY) src/09_robustness.py --score
	$(SM) $(PY) src/predict.py $$($(SM) $(PY) tools/smoke_images.py 3)
	$(SM) $(PY) src/report.py
