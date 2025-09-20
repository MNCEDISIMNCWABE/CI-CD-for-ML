install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

LATEST_METRIC := $(shell ls -t results/metrics_*.txt | head -1)
LATEST_CM := $(shell ls -t results/confusion_matrix_*.png | head -1)
LATEST_TREND := $(shell ls -t results/performance_trend_*.png | head -1)

eval:
	echo "## Model Metrics" > report.md
	cat $(LATEST_METRIC) >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo "![Confusion Matrix]($(LATEST_CM))" >> report.md
	echo '\n## Performance Trend Plot' >> report.md
	echo "![Performance Trend]($(LATEST_TREND))" >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add results/*.txt results/*.png results/*.json model/*.skops app/*
	git commit -m "Update with new results"
	git push --force origin HEAD:update

hf-login:
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	hf auth login --token $(HF)

push-hub:
	# Upload app files to root
	hf upload MNCEDISIM/Drug-Classification ./app --repo-type=space --commit-message "Sync App files"
	
	# Upload model files to /model/
	for file in model/*.skops; do \
		if [ -f "$$file" ]; then \
			filename=$$(basename "$$file"); \
			hf upload MNCEDISIM/Drug-Classification "model/$$filename" "$$file" --repo-type=space --commit-message "Sync Model Version: $$filename"; \
		fi; \
	done
	
	# Upload results files (metrics, plots, trends) to /results/
	for file in results/*; do \
		if [ -f "$$file" ]; then \
			filename=$$(basename "$$file"); \
			hf upload MNCEDISIM/Drug-Classification "results/$$filename" "$$file" --repo-type=space --commit-message "Sync Results File: $$filename"; \
		fi; \
	done

deploy: hf-login push-hub

all: install format train eval update-branch deploy
