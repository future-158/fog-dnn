import argparse
import enum
import hashlib
import json

# import torch.distributed as dist
import json
import logging
import multiprocessing
import os
import pickle
import random
import uuid
from dataclasses import dataclass
from itertools import product, zip_longest
from multiprocessing import process
from pathlib import Path
from posix import environ
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence, Union

from hydra.utils import get_original_cwd, to_absolute_path
import hydra
import joblib
import numpy as np
import omegaconf
import optuna
import pandas as pd
import pytorch_lightning as pl

import torch

import torch.nn.functional as F


import torch.onnx
import tqdm
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

from model import dnn, lstm
from utils import calc_metrics


# model_name에 따라 dnn or lstm을 반환
def get_module(model_name):
    if model_name == "dnn":
        return dnn.SeafogDNNClassifier
    elif model_name == "lstm":
        return lstm.SeafogLSTMClassifier
    else:
        raise ValueError("not implemented")


# 목적 함수 생성
def objective_wrapper(config):
    def objective(trial):
        hparams = OmegaConf.create()
        hparams.batch_size = 2 ** trial.suggest_int(
            "batch_exponent", 7, 11
        )  # 128 ~ 2048
        hparams.drop_rate = trial.suggest_discrete_uniform(
            "drop_rate", 0.0, 0.5, 0.1
        )  # end inclusive:
        hparams.label_smoothing = trial.suggest_discrete_uniform(
            "label_smoothing", 0.0, 0.3, 0.1
        )  # end inclusive
        hparams.lr = trial.suggest_loguniform("lr", 1e-4, 5e-1)  # end inclusive

        # gradient를 clip할 상한을 탐색
        hparams.gradient_clip_val = trial.suggest_float(
            "gradient_clip_val", 0.5, 10
        )  # end inclusive

        # lstm은 lstm last layer hidden output에 mlp를 연결하는 형태로 mlp 깊이는 2~5로 탐색함
        # 이와달리 mlp 모델의 경우 이보다 깊은 5~20을 탐색
        if config.model_name == "dnn":
            n_layers = range(trial.suggest_int("n_layers", 5, 20))
        else:
            n_layers = range(trial.suggest_int("n_layers", 2, 5))

        hparams.num_nodes = []
        #  노드사이즈는 8~512를 탐색
        for i in n_layers:
            n_units = 2 ** trial.suggest_int("n_units_l{}".format(i), 3, 9)
            hparams.num_nodes.append(n_units)

        # hidden_size와 num_layers는 lstm 파라미터임.
        if config.model_name == "lstm":
            hparams.hidden_size = 2 ** trial.suggest_int("hidden_exponent", 3, 9)
            hparams.num_layers = trial.suggest_int("num_layers", 2, 5)
            hparams.dropout = trial.suggest_discrete_uniform(
                "dropout ", 0.0, 0.5, 0.1
            )  # end inclusive:
            hparams.bidirectional = trial.suggest_categorical(
                "bidirectional ", [True, False]
            )  # end inclusive:

        #tensorboard 로거
        tb_logger = TensorBoardLogger(
            'tb_logger',
            name=config.optuna_config.study_name,
            default_hp_metric=False,
        )

        # hparams에 config 병합
        hparams.update(config)
        hparams.dataset_path = os.path.join(get_original_cwd(), hparams.dataset_path)
        model = get_module(hparams.model_name)(hparams)

        # model save, earlystop, prune callback 설정
        callbacks = [
            ModelCheckpoint(
                # 'trial.study.study_name',
                monitor="val/fbeta_epoch",
                save_top_k=1,
                mode="max",
            ),
            EarlyStopping(monitor="val/fbeta_epoch", patience=10, mode="max"),
            # 아래는 모델 pruning이 아닌, hpo 과정에서 현재 하이퍼파라미터의 성능이 너무 별로면 epoch 돌다가 중간에 멈추는 callback임
            PyTorchLightningPruningCallback(trial, monitor="val/fbeta_epoch"),
        ]

        # trainer 정의
        trainer = pl.Trainer(
            logger=tb_logger, # or mlflow logger
            auto_lr_find=False,
            stochastic_weight_avg=True,
            # default_root_dir=hparams.result_dest,
            check_val_every_n_epoch=1,
            callbacks=callbacks,
            deterministic=True, # 시드 고정
            gradient_clip_val=0.5,  # loss surface가 irregular해서 gradient가 터지지 않도록 방지
            gradient_clip_algorithm="value",
            limit_train_batches=10 if hparams.test_run else None,
            limit_val_batches=10 if hparams.test_run else None,
            max_epochs=1 if hparams.test_run else hparams.trainer_config.max_epochs, # early stop 안됐을 때 달성할 수 있는 최대 epoch
            num_sanity_val_steps=0,
            # max_steps=10, # early stop 안됐을 때 달성할 수 있는 최대 epoch            gpus=1,
            
        )
        # trainer.tune(model)

        # fit s
        trainer.fit(model)
        # best_model_path = trainer.checkpoint_callback.best_model_path
        # retrieve best model path
        best_model_path = callbacks[0].best_model_path
        best_model_path = Path(best_model_path).resolve().as_posix()

        # validation set 성능 평가
        val_score = trainer.validate(
            dataloaders=[model.val_dataloader()], ckpt_path=best_model_path
        )[0]["val/fbeta_epoch"]
        trial.set_user_attr(key="best_model_path", value=best_model_path)
        trial.set_user_attr(key="hparams", value=OmegaConf.to_container(hparams))

        # logging hyperparameters
        tb_logger.log_hyperparams(OmegaConf.to_container(hparams))
        # tb_logger.log_graph(model, [(1,13,12), (1,5)])
        # tb_logger.log_metrics({'val/fbeta_epoch':val_score}, step=0)

        # del trainer
        # del model
        return val_score

    return objective

# evalutate
def evaluate(cfg: DictConfig, refit=False):
    """
    hpo한 study 파일을 불러와서 가장 좋은 모델 trial과 그 hparam, model_path를 찾은 후 불러와서 test 성능 계산
    """

    # load study
    study = optuna.load_study(
        cfg.optuna_config.study_name, storage=cfg.optuna_config.storage
    )
    trial = study.best_trial
    best_model_path = trial.user_attrs["best_model_path"]
    hparams = OmegaConf.load(
        os.path.dirname(os.path.dirname(best_model_path)) + "/" + "hparams.yaml"
    )

    # load model
    model: LightningModule = get_module(cfg.model_name).load_from_checkpoint(
        study.best_trial.user_attrs["best_model_path"]
    )

    # 모델 gradient update 중지
    model.freeze()

    # load data && make dataloader
    model.prepare_data()  # load, transform data
    model.setup()

    # trainer = pl.Trainer(gpus=1) # gpu 1개 사용
    trainer = pl.Trainer(gpus=0) # gpu 1개 사용
    # test
    trainer.test(model, dataloaders=[model.test_dataloader()])

    # return obs,pred
    return model.obs_pred


# run hpo
def run_hpo(config: DictConfig) -> None:
    """
    - Create a study.
    - Define the objective function.
    - Run the optimization.
    - Save the results.
    """

    # hpo 탐색 reset
    study_summaries = optuna.study.get_all_study_summaries(
        storage=config.optuna_config.storage
    )

    # check study file already exists
    study_exists = len(
        [
            x.study_name
            for x in study_summaries
            if x.study_name == config.optuna_config.study_name
        ]
    )

    # study 파일이 있으면 기존 것 삭제
    if study_exists:
        optuna.delete_study(
            study_name=config.optuna_config.study_name,
            storage=config.optuna_config.storage,
        )

    # 11번 trial부터는 epoch이 5보다 클 때, validation 성능이 이전 trial의 median보다 작으면 더 이상 진행하고 않고 멈추는 것
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5,
    )

    # 11번 trial부터는 epoch이 5보다 클 때, validation 성능이 이전 trial의 median보다 작으면 더 이상 진행하고 않고 멈추는 것
    # tpe는 independent sampling 방법으로 하이퍼파라미터마다 독립적으로  exploration/exploitation 방식으로 샘플링

    # define sampler
    sampler = optuna.samplers.TPESampler()

    # create study
    study = optuna.create_study(
        study_name=config.optuna_config.study_name,
        storage=config.optuna_config.storage,
        load_if_exists=config.optuna_config.load_if_exists,
        direction="maximize",  # csi이므로
        pruner=pruner,
        sampler=sampler,
    )

    # id 생성
    sweep_id = str(uuid.uuid1())

    @dataclass
    class Run_config:
        sweep_id: str = ""

    c1 = Run_config(sweep_id=sweep_id)  # just unique id
    c1 = OmegaConf.structured(c1)
    config = OmegaConf.merge(config, c1)


    # run optimize
    study.optimize(
        objective_wrapper(config),
        timeout=config.optuna_config.time_budget, # 제한 시간 
        n_trials=config.optuna_config.n_trials, # 탐색 횟수
        gc_after_trial=True, 
        n_jobs=1,
        callbacks=[
            MaxTrialsCallback(
                config.optuna_config.n_trials, states=(TrialState.COMPLETE,)
            ),
        ],
    )

    """
    refit이 true면 Train(train+val) + Test 세트 모두를 학습하는 것. 사용하지 않음.
    """

    # get obs_pred
    obs_pred = evaluate(config, refit=False)
    metrics = calc_metrics(obs_pred.obs, obs_pred.pred)
    # save result
    obs_pred.to_csv('obs_pred.csv')


    OmegaConf.save(
        OmegaConf.create(metrics),
        'metrics.yaml'
        )

    study.trials_dataframe().to_csv("trials_dataframe.csv", index=False)

# $PWD/conf/config.yaml 파일을 config로 로드함.
@hydra.main(config_path="../conf", config_name="config")
def main(config: DictConfig):
    run_hpo(config)


if __name__ == "__main__":
    main()
