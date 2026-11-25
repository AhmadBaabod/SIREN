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
