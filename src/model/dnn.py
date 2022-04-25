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
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import (
    BatchSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

# dataset for dnn classifier
class DNNDataset(Dataset):
    def __init__(
        self, X: np.ndarray, y: pd.Series, transform: Optional[Callable] = None
    ):
        self.X = X.astype(np.float32)
        self.y = y.values.astype(np.int64)
        self.len = self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.len


def transform_data():
    pass

# define mlp block
class MLP_BLOCK(torch.nn.Module):
    def __init__(self, drop_rate, input_dim, output_dim):
        super(MLP_BLOCK, self).__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.bn_1 = nn.BatchNorm1d(output_dim)

        self.linear_2 = nn.Linear(input_dim + output_dim, output_dim)
        self.bn_2 = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(drop_rate)

    # when module called
    def forward(self, x):
        """
        x  -> linear  -> batchnorm -> relu -> dropout 통과한 값을 original_x와 concat한 후
        다시 linear -> batchnorm -> relu -> dropout을 통과함
        """
        og = nn.Identity()(x)
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = torch.cat([og, x], axis=-1)

        x = self.linear_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.drop(x)
        return x


# define SeafogDNNClassifier
class SeafogDNNClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(SeafogDNNClassifier, self).__init__()
        self.save_hyperparameters(hparams)  # hparams.yaml 파일에 저장

        self.hparams.lr = self.hparams.get("lr", 0.01)  # set default learning rate

        module_list = []

        self.prepare_data()

        input_dim = self.input_dim
        for output_dim in self.hparams.num_nodes:
            module_list.append(MLP_BLOCK(self.hparams.drop_rate, input_dim, output_dim))
            input_dim = output_dim

        head = nn.Linear(input_dim, 2)  # multiclass classification
        self.model = nn.Sequential(*module_list, head)

    # when module called
    def forward(self, x):
        # return self.model(x.view(x.size(0), -1))
        return self.model(x)

    # for each batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(
            x
        )  # self means self.__call__, but in this case self.forward i suppose
        # loss = F.binary_cross_entropy(pred.view(-1), y.view(-1)) # flatten
        # loss = F.nll_loss()
        loss = F.cross_entropy(pred, y)
        self.log("train/loss_step", loss, on_step=True, on_epoch=True, prog_bar=False)
        # self.log('seafog_frac', y.mean() * 100, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    # for each epoch end
    def train_epoch_end(self, outputs):
        mean_train_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.log(
            "train/loss_epoch",
            mean_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # for each validation batch
    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = torch.argmax(self(x), axis=-1)
        tp = torch.sum((y) * (yhat))
        tn = torch.sum((1 - y) * (1 - yhat))
        fp = torch.sum((1 - y) * (yhat))
        fn = torch.sum((y) * (1 - yhat))
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    # for each validation epoch ends
    def validation_epoch_end(self, outputs):
        """
        aggegrate stats from validation batch output
        """

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

    # when model.predict is called
    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self.forward(batch[0])

    # when model.test(dataloader) is called
    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = torch.argmax(self(x), axis=-1)
        return {"obs": y, "pred": yhat}

    # for each test epoch end
    def test_epoch_end(self, outputs):
        obs = torch.cat([x["obs"] for x in outputs]).cpu().numpy()
        pred = torch.cat([x["pred"] for x in outputs]).cpu().numpy()
        self.obs_pred = pd.DataFrame({"obs": obs, "pred": pred}, index=self.test_index)

    @staticmethod
    def download_data(data_dir: Optional[str] = None):
        pass

    def calc_input_dim(self):
        pass

    # prepare dataset
    def prepare_data(self):
        # data = joblib.load(self.hparams.dataset_path, mmap_mode='r')
        data = joblib.load(
            self.hparams.dataset_path
        )
        X = data["x"]
        scaler = StandardScaler()
        X.loc[:, :] = scaler.fit_transform(X)

        y = data["y"]
        label_name = f'y_{self.hparams.pred_hour}'
        X = X.join(y.filter(like="time"))  # time related feature
        y = y[label_name]  # y_1 | y_3 | y_6
        nan_mask = np.isnan(X).any(axis=1) | y.isna().values

        # dropna
        X = X[~nan_mask].values
        y = y[~nan_mask]
        self.input_dim = X.shape[-1]


        # train / test split
        test_mask = y.index > "2020-07-01"
        test_split = np.flatnonzero(test_mask)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        groups = y.index.year * 366 + y.index.dayofyear
        groups = groups[~test_mask]

        # test set을 제외한 Train(train+val) 세트를 년도로 구분하면 연도에 따라 해무발생 빈도가 많이 다르니 validation set이 train set을 대표하지 못함
        # 그러므로 최소 unit을 하루 단위로 하여 train/valid로 shuffle split함. group shuffle하면 하루가 train/valid로 나누어지지 않음
        train_split, val_split = list(
            gss.split(y[~test_mask], y[~test_mask], groups=groups)
        )[0]
        splits = [train_split, val_split, test_split]

        self.train_ds = DNNDataset(X[splits[0]], y.iloc[splits[0]])
        self.val_ds = DNNDataset(X[splits[1]], y.iloc[splits[1]])
        self.test_ds = DNNDataset(X[splits[2]], y.iloc[splits[2]])
        pos_label_weight = 1 / y.mean()

        # calc weight
        self.weight = (
            np.where(y == 1, pos_label_weight, 1) / 2
        )  # pos_label_weight을 class 비율의 역수로할 때 2로 나누어주면 sample_weight 평균이 1이 되어 나눔
        self.weight = self.weight.astype(np.float32)[splits[0]]
        self.test_index = y.index[splits[-1]]

        self.num_batch = (
            splits[0].size + splits[1].size
        ) // self.hparams.batch_size + 1  # 무시

    def setup(self, stage=None):  # foreach gpu
        pass

    # train dataloader with weighted sampler
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            # sampler=WeightedRandomSampler(self.weight, self.hparams.batch_size, replacement=True),
            sampler=WeightedRandomSampler(
                self.weight, len(self.weight), replacement=True
            ),  # sample weight에 따라 해무를 데이터 비율보다 더 많이 샘플링함.
            batch_size=self.hparams.batch_size,
            drop_last=True,
            # num_workers=1
        )

    # val dataloader
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # num_workers=1
        )

    # test dataloader
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # num_workers=1
        )

    # optimizer and schedulers
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
                )  # T_max의 단위는 epoch임
                # 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[0], mode='max', factor=0.5, patience=3, min_lr=1e-5),
                # 'monitor': 'val_fbeta',  # Default: val_loss
                # 'interval': 'epoch',
                # 'frequency': 1,
            }
        ]
        return optimizers, schedulers


if __name__ == "__main__":
    pass
