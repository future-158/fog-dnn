import argparse
import hashlib

# import torch.distributed as dist
import json
import logging
import os
import pickle
import random
import uuid
from itertools import product, zip_longest
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Sequence, Union

import joblib
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch

# import torch.functional as F
import torch.nn.functional as F

# from dask.distributed import Client
import torch.onnx
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import check_array, resample, shuffle
from torch import nn, threshold
from torch.autograd import Variable
from torch.nn import parameter
from torch.nn.functional import linear, relu
from torch.nn.modules import rnn
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.data.sampler import (
    BatchSampler,
    SequentialSampler,
    WeightedRandomSampler,
)


# 데이터 셋 생성.
class LSTMDataset(Dataset):
    def __init__(
        self,
        X_3d: np.ndarray,
        X_2d: np.ndarray,
        y: pd.Series,
        transform: Optional[Callable] = None,
    ):
        X_3d = X_3d.astype(np.float32)  # batch, time, features
        X_2d = X_2d.astype(np.float32)  # batch, features. 시간 변수 및 통계 변수는 X_2d에 들어감
        y = y.astype(np.int64)

        self.len = y.shape[0]
        self.X_3d = X_3d
        self.X_2d = X_2d
        self.y = y

    # when dataset[i] is called, __getitem__ is called.
    def __getitem__(self, idx):
        """
        encoder: lstm으로 multivariate timeseries를 encoding함. last hidden state를 사용
        decoder: encoder output과 시간 변수로 n시간 후 해무 클래스를 예측함

        """
        return {
            "encoder_features": self.X_3d[idx],
            "decoder_features": self.X_2d[idx],
            "y": self.y[idx],
        }

    def __len__(self):
        return self.len


# mlp 레이어 정의
class FCN(nn.Module):
    def __init__(self, hparams):
        super(FCN, self).__init__()
        self.dropout = nn.Dropout(p=hparams.drop_rate)
        self.linear_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        # layers ; 32, 16, 8
        input_dim = (
            hparams.hidden_size + hparams.input_dim
        )  # hidden_size는 lstm last hidden state size이며 input_dim은 time 변수 및 통계 변수 크기
        for num_node in hparams.num_nodes:
            self.linear_layers.append(nn.Linear(input_dim, num_node))
            self.bn_layers.append(nn.BatchNorm1d(num_node))
            input_dim = num_node
        self.out = nn.Linear(self.linear_layers[-1].out_features, 2)

    def forward(self, x):
        x = torch.cat(x, dim=-1)
        # x = self.bn0(x)
        # x = x[-1]
        for linear_layer, bn_layer in zip(self.linear_layers, self.bn_layers):
            x = linear_layer(x)
            x = bn_layer(x)
            x = nn.ReLU()(x)
            x = self.dropout(x)
        x = self.out(x)
        return x


# lstm 레이어 정의
class LSTM(nn.Module):
    def __init__(self, hparams):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=hparams.input_size,
            hidden_size=hparams.hidden_size,
            batch_first=True,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout,
            bidirectional=hparams.bidirectional,
        )

        self.num_layers = hparams.num_layers
        self.hidden_size = hparams.hidden_size
        self.bidirectional = hparams.bidirectional
        # lstm output shape은 time, batch, hidden_size이며 [-1]로 indexing하면 됨
        # bidirectional을 적용할 경우 time, 2, batch, hidden_size이므로 [-1,-2]로 indexing 해야함.
        self.query_layer = (
            [-1] if not hparams.bidirectional else [-1, -2]
        )  # bidirectional일 경우 [forward, backward] dim이 추가되므로 -2로 forward만 가져옴

    def forward(self, x):
        encoder_features, decoder_features = x
        # self.reset_state(x.size(0))
        # _, (h_n, _) = self.lstm(x, (self.h, self.c))
        # D * num_layer, batch, hidden_size
        _, (h_n, _) = self.lstm(encoder_features)

        # bidirectional
        # assert h_n.size() == (self.num_layers, encoder_features.size(0), self.hidden_size)
        # h_all =  h_n.view(self.num_layers,x.size(0), self.hidden_size)
        return h_n[self.query_layer].mean(axis=0), decoder_features


# lstm + mlp를 sequential하게 묶어서 SeafogLSTMClassifier 정의
class SeafogLSTMClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(SeafogLSTMClassifier, self).__init__()
        self.save_hyperparameters(hparams)  # hyperparameter 저장
        self.hparams.lr = self.hparams.get("lr", 0.01)  # set default learning rate
        self.get_info()

        self.encoder = LSTM(hparams)
        self.decoder = FCN(hparams)
        self.model = nn.Sequential(self.encoder, self.decoder)

    # model(X) 했을 때 실행되는 forward 함수
    def forward(self, x):
        return self.model(x)

    # train step
    def training_step(self, batch, batch_idx):
        x = batch["encoder_features"]
        x2 = batch["decoder_features"]
        y = batch["y"]
        pred = self(
            (x, x2)
        )  # self means self.__call__, but in this case self.forward i suppose
        # loss = F.binary_cross_entropy(pred.view(-1), y.view(-1)) # flatten
        # loss = F.nll_loss()
        loss = F.cross_entropy(pred, y)
        self.log("train/loss_step", loss, on_step=True, on_epoch=True, prog_bar=False)
        # self.log('seafog_frac', y.mean() * 100, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    # epoch 끝날 때 마다 실행
    def train_epoch_end(self, outputs):
        mean_train_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.log(
            "train/loss_epoch",
            mean_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # validation step
    def validation_step(self, batch, batch_idx):
        x = batch["encoder_features"]
        x2 = batch["decoder_features"]
        y = batch["y"]
        yhat = torch.argmax(self((x, x2)), axis=-1)
        tp = torch.sum((y) * (yhat))
        tn = torch.sum((1 - y) * (1 - yhat))
        fp = torch.sum((1 - y) * (yhat))
        fn = torch.sum((y) * (1 - yhat))
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    # validation 에폭 끝날 때 마다 실행
    def validation_epoch_end(self, outputs):
        total_tp = torch.stack([x["tp"] for x in outputs]).sum()
        total_tn = torch.stack([x["tn"] for x in outputs]).sum()
        total_fp = torch.stack([x["fp"] for x in outputs]).sum()
        total_fn = torch.stack([x["fn"] for x in outputs]).sum()
        pag = total_tp / (total_tp + total_fp + 1e-6)
        pod = total_tp / (total_tp + total_fn + 1e-6)
        fbeta = 2 * pag * pod / (pag + pod + 1e-6)
        self.log(
            "val/pag_epoch", pag * 100, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/pod_epoch", pod * 100, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/fbeta_epoch", fbeta * 100, on_step=False, on_epoch=True, prog_bar=True
        )

    # model.test(X) 할 때 실행하는 loop 안에서 함수, {obs, pred}를 반환함
    def test_step(self, batch, batch_idx):
        x = batch["encoder_features"]
        x2 = batch["decoder_features"]
        y = batch["y"]
        yhat = torch.argmax(self((x, x2)), axis=-1)
        return {"obs": y, "pred": yhat}

    # model.test(X) 하면 batch step을 다 끝냈을 때 실행하는 함수, 각 batch마다의 {obs, pred}를 각각 묶어서 전체 테스트 세트의 obs, pred를 반환함
    def test_epoch_end(self, outputs):
        obs = torch.cat([x["obs"] for x in outputs]).cpu().numpy()
        pred = torch.cat([x["pred"] for x in outputs]).cpu().numpy()
        self.obs_pred = pd.DataFrame({"obs": obs, "pred": pred}, index=self.test_index)

    @staticmethod
    def download_data(data_dir: Optional[str] = None):
        pass
        # with FileLock(os.path.expanduser("~/.data.lock")):
        # download

    # 모델 구조 정의를 위해 미리 알아야하는 input shape등
    def get_info(self):
        data = joblib.load(self.hparams.dataset_path)
        X = data["x"]

        y = data["y"]
        label_name = f'y_{self.hparams.pred_hour}'  # 1시간은 y_1, 3시간은 y_3, 6시간은 y_6

        # pandas multi-column dataframe을 (sample, time, feature) shape의 numpy array로 변환
        X_3d = np.transpose(X.stack().to_xarray().to_array().values, [1, 2, 0]).astype(
            np.float32
        )
        X_2d = y.filter(like="time").values.astype(np.float32)  # 시간 feature 추출
        self.hparams.input_size = X_3d.shape[-1]
        self.hparams.input_dim = X_2d.shape[-1]

    # 데이터 준비
    def prepare_data(self):
        data = joblib.load(self.hparams.dataset_path)
        X = data["x"]
        scaler = StandardScaler()
        X.loc[:, :] = scaler.fit_transform(
            X
        )  # lag 3시간이라 최대 18개 데이터만 차이나므로 lagging마다 독립적으로 scaler를 학습하여 적용함.

        y = data["y"]
        label_name = f'y_{self.hparams.pred_hour}'

        X_3d = np.transpose(X.stack().to_xarray().to_array().values, [1, 2, 0]).astype(
            np.float32
        )
        X_2d = y.filter(like="time").values.astype(np.float32)
        y = y[label_name]

        # dropna
        nan_mask = (
            np.isnan(X_3d).any(axis=(1, 2))
            | np.isnan(X_2d).any(axis=1)
            | y.isna().values
        )
        X_3d = X_3d[~nan_mask]
        X_2d = X_2d[~nan_mask]
        y = y[~nan_mask]

        self.hparams.input_dim = X_2d.shape[-1]
        self.hparams.input_size = X_3d.shape[1]

        # train / test split
        test_mask = y.index > "2020-07-01"
        test_split = np.flatnonzero(test_mask)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        groups = y.index.year * 366 + y.index.dayofyear
        groups = groups[~test_mask]
        train_split, val_split = list(
            gss.split(y[~test_mask], y[~test_mask], groups=groups)
        )[0]
        splits = [train_split, val_split, test_split]

        # scikit-learn split은 정수 index를 반환하므로 iloc 사용
        self.train_ds = LSTMDataset(X_3d[splits[0]], X_2d[splits[0]], y.iloc[splits[0]])
        self.val_ds = LSTMDataset(X_3d[splits[1]], X_2d[splits[1]], y.iloc[splits[1]])
        self.test_ds = LSTMDataset(X_3d[splits[2]], X_2d[splits[2]], y.iloc[splits[2]])
        pos_label_weight = 1 / y.mean()

        self.weight = np.where(y == 1, pos_label_weight, 1) / 2
        self.weight = self.weight.astype(np.float32)[splits[0]]
        self.test_index = y.index[splits[-1]]

        # self.weight_val = self.weight.astype(np.float32)[splits[1]]
        # self.weight_concat = torch.cat([self.weight, self.weight_val])
        self.num_batch = (
            splits[0].size + splits[1].size
        ) // self.hparams.batch_size + 1

    def setup(self, stage=None):  # foreach gpu
        pass

    # define train dataloader with weighted sampler
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            sampler=WeightedRandomSampler(
                self.weight, len(self.weight), replacement=True
            ),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            # num_workers=1
        )

    # define val dataloader
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # num_workers=1
        )

    # define val dataloader
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # num_workers=1
        )

    # define optimizer and scheduler
    def configure_optimizers(self):
        # optimizers = [torch.optim.Adam(self.model.parameters(), lr=self.lr)]
        optimizers = [
            torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
            # Ranger(self.model.parameters(), lr = self.hparams.lr)
        ]

        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0], T_max=50, last_epoch=-1
                )
                # 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[0], mode='max', factor=0.5, patience=3, min_lr=1e-5),
                # 'monitor': 'val_fbeta',  # Default: val_loss
                # 'interval': 'epoch',
                # 'frequency': 1,
            }
        ]
        return optimizers, schedulers


if __name__ == "__main__":
    pass
