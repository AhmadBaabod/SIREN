# Data

Datasets are not stored in git. Each notebook downloads what it needs into `data/raw/`.

## METR-LA (Student 2 · Traffic Predictor)

Downloaded by `traffic_predictor/traffic_predictor.ipynb` (part A).

| File | What it is | Source |
|---|---|---|
| `METR-LA.zip` → `node_values.npy`, `adj_mat.npy` | Speeds `(34272, 207, 2)` = (time, sensor, [speed mph, time of day]) and the 207×207 road graph. Used by `METRLADatasetLoader`. | PyTorch Geometric Temporal (Box link); fallback mirror: [FelixOpolka/STGCN-PyTorch](https://github.com/FelixOpolka/STGCN-PyTorch) |
| `metr-la.h5` | Same speeds with timestamps (2012-03-01 to 2012-06-27, every 5 min) and sensor IDs. sha256 `64784b76…7c05`. Old pandas format: read it with `h5py`. | DCRNN data (Git LFS copy in [kevinkbc/DCRNN_PyTorch](https://github.com/kevinkbc/DCRNN_PyTorch)) |
| `adj_mx.pkl`, `graph_sensor_locations.csv` | Official DCRNN sensor graph and GPS position of each sensor (for the app map). | [liyaguang/DCRNN](https://github.com/liyaguang/DCRNN) |

**Zeros mean missing.** 575,302 of 7,094,304 readings (8.11%) are 0. 77% of these come from 2,148 time steps where all 207 sensors are 0 at once. The other 23% are single sensors dropping out. Evaluate with masked MAE/RMSE that skip zero targets, as DCRNN does.

`traffic_predictor/data/processed/metr_la_missing_per_sensor.csv` gives the missing share of each sensor.

## PEMS-BAY (second test set, part G)

Downloaded by the notebook (part G).

| File | What it is | Source |
|---|---|---|
| `pems-bay.h5` (136 MB) | Speeds of 325 Bay Area sensors, 2017-01-01 to 2017-06-30, every 5 min, with timestamps and sensor IDs. Key `speed`, same old pandas format as `metr-la.h5`. MD5 `bfa47e6ee7cc2d665e9b62c0c95c0b41`, checked after download. | DCRNN data, mirrored on [Zenodo record 4263971](https://zenodo.org/records/4263971) |

Almost nothing is missing (0.0031% zeros). The time grid has one 65-minute break, which DCRNN's script ignores, so the windows (36,465 / 5,209 / 10,419) match DCRNN's.
