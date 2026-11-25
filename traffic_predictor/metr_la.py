"""METR-LA data pipeline used by the strongest reference models (DCRNN -> Graph WaveNet -> MTGNN).

Follows DCRNN's scripts/generate_training_data.py and MTGNN's util.load_dataset / trainer:

* Zeros (missing readings) are LEFT IN the inputs. Nothing is imputed.
* Features per step: [speed mph, time of day in 0..1).
* Windows: 12 past steps (t-11..t) -> 12 future steps (t+1..t+12).
* Split the windows in time order: 70% train, 10% val, 20% test.
* Z-score ONLY the speed channel of the inputs, with mean/std of the TRAIN inputs (zeros included).
  Targets stay in mph; the model output is inverse-transformed before the loss.
* Loss and metrics are MASKED: targets equal to 0 are ignored, and the mask is rescaled so the
  score is the average over real readings (DCRNN masked_mae / masked_rmse / masked_mape).

Windows are sliced on the fly instead of materialised (DCRNN's npz files are ~2.7 GB);
the samples are identical.
"""
from __future__ import annotations

import os
import urllib.request

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_IN, SEQ_OUT = 12, 12
TRAIN_RATIO, TEST_RATIO = 0.7, 0.2
NULL_VAL = 0.0
# Horizons reported in the brief: 5, 15, 30 min = step index 0, 2, 5 of the 12-step output.
HORIZONS = {"5 min": 0, "15 min": 2, "30 min": 5}

MIRRORS = {
    "metr-la.h5": "https://media.githubusercontent.com/media/kevinkbc/DCRNN_PyTorch/master/data/metr-la.h5",
    "adj_mx.pkl": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl",
    "graph_sensor_locations.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv",
}


def download(raw_dir: str) -> None:
    """Fetch the DCRNN METR-LA files into raw_dir if they are not there yet."""
    os.makedirs(raw_dir, exist_ok=True)
    for name, url in MIRRORS.items():
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)


def load_speeds(raw_dir: str) -> pd.DataFrame:
    """Speeds in mph as a (time, sensor) DataFrame. Reads the old pandas h5 with h5py."""
    with h5py.File(os.path.join(raw_dir, "metr-la.h5"), "r") as f:
        ids = f["df/axis0"][:].astype(str)
        times = pd.to_datetime(f["df/axis1"][:])
        values = f["df/block0_values"][:]
    return pd.DataFrame(values, index=times, columns=ids)


def build_features(speed: pd.DataFrame) -> np.ndarray:
    """(T, N, 2) float32 array: [speed, time of day], as DCRNN add_time_in_day=True."""
    idx = speed.index.values
    tod = (idx - idx.astype("datetime64[D]")) / np.timedelta64(1, "D")
    tod = np.tile(tod[:, None], (1, speed.shape[1]))
    return np.stack([speed.values, tod], axis=-1).astype(np.float32)


def split_starts(num_steps: int) -> dict[str, np.ndarray]:
    """Index t of the last input step for every window, split like DCRNN (by sample, in order)."""
    t = np.arange(SEQ_IN - 1, num_steps - SEQ_OUT)          # DCRNN: range(min_t, max_t)
    n = len(t)
    n_test, n_train = round(n * TEST_RATIO), round(n * TRAIN_RATIO)
    n_val = n - n_test - n_train
    return {"train": t[:n_train], "val": t[n_train:n_train + n_val], "test": t[-n_test:]}


class StandardScaler:
    def __init__(self, mean: float, std: float):
        self.mean, self.std = float(mean), float(std)

    @classmethod
    def fit_train_inputs(cls, data: np.ndarray, train_t: np.ndarray) -> "StandardScaler":
        """Mean/std over the speed channel of every train INPUT window, zeros included.

        Same as DCRNN/MTGNN: x_train[..., 0].mean() / .std() (overlapping windows counted as such).
        """
        rows = train_t[:, None] + np.arange(-SEQ_IN + 1, 1)[None, :]       # (n, 12)
        x = data[rows, :, 0].astype(np.float64)
        return cls(x.mean(), x.std())

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


class WindowDataset(Dataset):
    """Returns x: (12, N, 2) with speed z-scored, y: (12, N) speed in mph (zeros = missing)."""

    def __init__(self, data: np.ndarray, starts: np.ndarray, scaler: StandardScaler):
        x = data.copy()
        x[..., 0] = scaler.transform(x[..., 0])
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(data[..., 0].copy())
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = int(self.starts[i])
        return self.x[t - SEQ_IN + 1:t + 1], self.y[t + 1:t + 1 + SEQ_OUT]


# ---- masked metrics (DCRNN / Graph WaveNet / MTGNN util.py) ---------------------------------

def _mask(labels: torch.Tensor, null_val: float = NULL_VAL) -> torch.Tensor:
    mask = (labels != null_val).float()
    mask = mask / torch.mean(mask)
    return torch.nan_to_num(mask)


def masked_mae(preds, labels, null_val: float = NULL_VAL):
    loss = torch.abs(preds - labels) * _mask(labels, null_val)
    return torch.mean(torch.nan_to_num(loss))


def masked_mse(preds, labels, null_val: float = NULL_VAL):
    loss = (preds - labels) ** 2 * _mask(labels, null_val)
    return torch.mean(torch.nan_to_num(loss))


def masked_rmse(preds, labels, null_val: float = NULL_VAL):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mape(preds, labels, null_val: float = NULL_VAL):
    loss = torch.abs(preds - labels) / labels * _mask(labels, null_val)
    return torch.mean(torch.nan_to_num(loss, posinf=0.0))


def horizon_report(preds: torch.Tensor, labels: torch.Tensor) -> pd.DataFrame:
    """MAE / RMSE / MAPE at 5, 15, 30 min. preds, labels: (samples, 12, N) in mph."""
    rows = {}
    for name, h in HORIZONS.items():
        p, y = preds[:, h], labels[:, h]
        rows[name] = {"MAE": masked_mae(p, y).item(), "RMSE": masked_rmse(p, y).item(),
                      "MAPE %": 100 * masked_mape(p, y).item()}
    return pd.DataFrame(rows).T.round(3)
