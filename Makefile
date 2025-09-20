install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

LATEST_METRIC := $(shell ls -t results/metrics_*.txt | head -1)
LATEST_CM := $(shell ls -t results/confusion_matrix_*.png | head -1)

eval:
	echo "## Model Metrics" > report.md
	cat $(LATEST_METRIC) >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo "![Confusion Matrix]($(LATEST_CM))" >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	hf auth login --token $(HF)

push-hub:
	hf upload MNCEDISIM/Drug-Classification ./app --repo-type=space --commit-message "Sync App files"
	hf upload MNCEDISIM/Drug-Classification ./model --repo-type=space --commit-message "Sync Model Versions"
	hf upload MNCEDISIM/Drug-Classification ./results --repo-type=space --commit-message "Sync Metrics Versions"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
