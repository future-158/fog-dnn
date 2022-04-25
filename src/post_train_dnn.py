import argparse
import enum
import hashlib
from itertools import chain
import json

# import torch.distributed as dist
import json
import logging
import multiprocessing
import os
import pickle
import random
import uuid
import shutil
from dataclasses import dataclass
from itertools import product, zip_longest, chain
from multiprocessing import process
from pathlib import Path
from posix import environ
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence, Union

import joblib
import numpy as np
from omegaconf.omegaconf import OmegaConf
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
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
from utils import calc_metrics
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import check_array, resample, shuffle
import argparse

cfg = OmegaConf.load('conf/config.yaml')

log_files = list(Path(cfg.log_prefix).glob('**/metrics.yaml'))
for log_file in log_files:

    metric = OmegaConf.load(log_file)

    obs_pred = pd.read_csv(
        log_file.with_name('obs_pred.csv'),
        parse_dates=['datetime'],
        index_col=['datetime']
    )

    trials_dataframe = pd.read_csv(
        log_file.with_name('trials_dataframe.csv'),
        # parse_dates=['datetime'],
        # index_col=['datetime']
    )

    row = {
        **OmegaConf.to_container(metric),
        "cv_score": trials_dataframe.value.max(),
    }
    print(row)
