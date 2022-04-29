SHELL := /bin/bash
PREP := src/pre_train_dnn.py
PRED := src/train_dnn.py
POST := src/post_train_dnn.py
CPU_LIST := '1-30'
CUDA_VISIBLE_DEVICES := '2'


.PHONY: list venv clean prep pred hpo

list: # list packages
	conda list --prefix venv/

venv: # install dependencies in venv folder
	conda env create --file environment.yml --prefix venv

test: # test
	rm -rf ./data/clean ./data/processed && \
	conda run --prefix venv python src/download_data.py && \
	conda run --prefix venv python src/train_dnn.py test_run=true optuna_config.n_trials=1 && \
	echo 'success'

clean: # uninstall venv folder
	conda uninstall --prefix venv/ -y --all

prep: # preprocess input
	conda run --prefix venv/ python  $(PRE)

pred: # pred
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) taskset --cpu-list $(CPU_LIST) conda run --prefix venv/ python  $(PRED)

post: # pred
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) taskset --cpu-list $(CPU_LIST) conda run --prefix venv/ python  $(POST)
