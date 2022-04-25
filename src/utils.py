import itertools
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    recall_score,
    roc_auc_score,
)


# calcuate metrics(ACC, CSI, etc.)
def calc_metrics(obs, pred):
    usecols = [
        "ACC",
        "CSI",
        "PAG",
        "POD",
        "FAR",
        "f1_score",
        "POFD",
        "POCD",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    # round to 0 or 1
    obs = obs.round()

    # round to 0 or 1
    pred = pred.round()
    tn, fp, fn, tp = confusion_matrix(obs, pred).flatten()

    # get percentage
    POFD = fp / (tn + fp) * 100

    # generate metrics dictionary
    performance = dict(
        ACC=accuracy_score(obs, pred) * 100,
        CSI=tp / (tp + fn + fp) * 100,
        PAG=100 - fp / (tp + fp) * 100,
        POD=recall_score(obs, pred) * 100,
        FAR=fp / (tp + fp) * 100,
        f1_score=fbeta_score(obs, pred, beta=1) * 100,
        POFD=POFD,
        POCD=100 - POFD,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )

    performance = {
        k: int(v) if k in ['tn','fp','fn','tp'] else float(v) for k,v in performance.items()
        }

    

    # return metrics with appropriate name
    return dict(zip(usecols, map(performance.get, usecols)))


if __name__ == "__main__":
    ...
