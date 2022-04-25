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

clean: # uninstall venv folder
	conda uninstall --prefix venv/ -y --all

prep: # preprocess input
	conda run --prefix venv/ python  $(PRE)

pred: # pred
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) taskset --cpu-list $(CPU_LIST) conda run --prefix venv/ python  $(PRED)

post: # pred
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) taskset --cpu-list $(CPU_LIST) conda run --prefix venv/ python  $(POST)

hpo:
	python train_ml.py -m pred_hour=1,3,6 model_name=cb,xgb,rf,lgb station=SF_0002,SF_0003,SF_0004,SF_0005,SF_0006,SF_0007,SF_0008,SF_0009,SF_0011