# Traffic Predictor (Student 2)

Predicts traffic speed on the METR-LA road network 5, 15 and 30 minutes ahead.
Plan: an LSTM baseline that uses each sensor's own history, then a GCN + GRU graph model (T-GCN style).
Both are scored with masked MAE and RMSE and compared with DCRNN.

Everything for this part stays in this folder.

```
traffic_predictor/
  metr_la.py                    shared data pipeline + masked metrics (same as MTGNN / DCRNN)
  01_get_and_check_data.ipynb   A. download METR-LA, count missing (zero) readings
  02_prepare_data.ipynb         B. windows, split, normalisation, masked metrics, naive reference
  data/README.md                data sources; raw/ is downloaded by the notebook (gitignored)
  data/processed/               small derived files (per-sensor missing share)
  figures/                      plots for the report
  requirements.txt
```

## Run

Colab: open the notebook and run all cells. Locally:

```bash
pip install torch torch-geometric numpy pandas h5py matplotlib
pip install --no-deps torch-geometric-temporal
jupyter notebook 01_get_and_check_data.ipynb
```

## Results so far

| Step | Result |
|---|---|
| A. Data check | 34,272 steps × 207 sensors; **8.11%** of readings are 0 (missing). 77% of these come from outages where all sensors are 0 at once. |
| B. Data pipeline | Same as MTGNN / DCRNN: zeros kept in inputs, 70/10/20 split (23,974 / 3,425 / 6,850 windows), train-only z-score, **masked** MAE/RMSE. |

| Model (test, masked MAE mph) | 5 min | 15 min | 30 min |
|---|---|---|---|
| DCRNN paper | – | 2.77 | 3.15 |
| MTGNN paper (strongest reference) | – | 2.69 | 3.05 |
| Naive: repeat last reading | 2.81 | 3.50 | 4.21 |
| LSTM baseline | next | | |
| GCN + GRU | | | |

![missing data](figures/metr_la_missing.png)
