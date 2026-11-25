# Traffic Predictor (Student 2)

Predicts traffic speed on the METR-LA road network 5, 15 and 30 minutes ahead.
Models: an LSTM baseline that uses each sensor's own history, the brief's GCN + GRU graph model (T-GCN),
and a re-implementation of MTGNN, the reference model with the lowest error. The graph models are trained
with MTGNN's recipe. All are scored with masked MAE and RMSE and compared with DCRNN and MTGNN.

Everything for this part stays in this folder.

```
traffic_predictor/
  traffic_predictor.ipynb   the one notebook, run top to bottom:
                              A. get and check the data   B. prepare the data (MTGNN/DCRNN pipeline)
                              C. LSTM baseline   D. GCN + GRU (T-GCN)   E. MTGNN   F. results
  data/README.md                data sources; raw/ is downloaded by the notebook (gitignored)
  data/processed/               small derived files (per-sensor missing share)
  models/                       trained weights (<name>.pt) and resumable checkpoints (<name>_checkpoint.pt)
  results.csv                   results table written by the notebook (test split, masked MAE / RMSE)
  figures/                      plots for the report
  requirements.txt
```

## Run

Colab: open
[the notebook in Colab](https://colab.research.google.com/github/AhmadBaabod/SIREN/blob/main/traffic_predictor/traffic_predictor.ipynb),
choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. It asks to mount Google Drive and
keeps the data, checkpoints and results in `MyDrive/SIREN/traffic_predictor/`, so after a disconnect just run all
again and training resumes. Parts D and E train 100 epochs each (MTGNN's setting), which needs the GPU.

Locally:

```bash
cd traffic_predictor
pip install -r requirements.txt
pip install --no-deps torch-geometric-temporal
jupyter notebook traffic_predictor.ipynb
```

Install `torch-geometric-temporal` with `--no-deps` as shown. A normal install tries to build `torch_sparse`, which fails on most machines, and the notebook doesn't need it.

Training saves a checkpoint every epoch, so after a Colab disconnect just run the notebook again and it resumes.
Trained weights are in `models/`, so rerunning loads them instead of training again
(delete `models/<name>_checkpoint.pt` to retrain). The LSTM took about 3.5 min per epoch on a 4-core CPU (30 epochs).

## Results so far

| Step | Result |
|---|---|
| A. Data check | 34,272 steps × 207 sensors; **8.11%** of readings are 0 (missing). 77% of these come from outages where all sensors are 0 at once. |
| B. Data pipeline | Same as MTGNN / DCRNN: zeros kept in inputs, 70/10/20 split (23,974 / 3,425 / 6,850 windows), train-only z-score, **masked** MAE/RMSE. |
| C. LSTM baseline | One LSTM shared by all sensors, time history only (51k parameters). Beats naive and the DCRNN paper's FC-LSTM; still behind the graph models. |
| D. GCN + GRU (T-GCN) | Cell from T-GCN's official code on DCRNN's road graph (14k parameters), MTGNN's recipe: Adam 1e-3, weight decay 1e-4, 100 epochs, curriculum learning. Code tested; full training on Colab next. |
| E. MTGNN | Rebuilt from MTGNN's official code: learned graph, dilated inception, mix-hop graph convolution. 405,452 parameters with the 3 unused convolutions, the same as the original. Code tested; full training on Colab next. |

Test split, masked metrics, MAE / RMSE in mph (`results.csv`):

| Model | 5 min | 15 min | 30 min |
|---|---|---|---|
| FC-LSTM (DCRNN paper) | – | 3.44 / 6.30 | 3.77 / 7.23 |
| DCRNN paper | – | 2.77 / 5.38 | 3.15 / 6.45 |
| MTGNN paper (strongest reference) | – | 2.69 / 5.18 | 3.05 / 6.17 |
| Naive: repeat last reading (ours) | 2.81 / 4.63 | 3.50 / 6.42 | 4.20 / 8.05 |
| **LSTM baseline (ours)** | **2.40 / 4.24** | **3.01 / 5.90** | **3.62 / 7.26** |
| GCN + GRU, T-GCN (ours) | train on Colab | | |
| MTGNN re-implementation (ours) | train on Colab | | |

![missing data](figures/metr_la_missing.png)

![LSTM forecast on one sensor](figures/lstm_example_sensor.png)
