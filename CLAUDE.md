# SIREN — Smart Intersection Brain (CS4085 Deep Learning)

Course project for Dr. Naila Marir, Effat College of Engineering. A smart intersection that
(1) detects an approaching ambulance from camera images and siren audio, and
(2) forecasts traffic speed on nearby roads, then suggests the clearest route in one web app.

The user works on **Student 2 · Traffic Predictor** (`traffic_predictor/`).

## Project parts

| Part | Owner | Models | Data | Target to compare |
|---|---|---|---|---|
| Emergency detector (`detector/`) | Student 1 | YOLOv8n (images), CNN on mel spectrograms (siren vs noise), fusion rule | Large-scale Emergency Siren & Road Noise; Roboflow Emergency Vehicle Detection v4 | Asif 2022 (97% audio), Noor 2024 (YOLOv8 P 0.976 / R 0.958) |
| Traffic predictor (`traffic_predictor/`) | Student 2 | LSTM baseline (time only) → GCN + GRU (T-GCN style) | METR-LA first, then PEMS-BAY | DCRNN: METR-LA MAE 2.77 @15 min, 3.15 @30 min (paper); 2.18 / 2.67 / 3.08 @5/15/30 (official README) |
| App (`app/`) | Both | Streamlit + Folium map, NetworkX shortest path | — | Time to alert, CPU FPS, false alarms |

Forecast horizons: **5, 15, 30 min** (= 1, 3, 6 steps of 5 min). Metrics: **MAE and RMSE**.
App must say the ambulance placement on a METR-LA sensor is a simulated scenario.
Deliverables: one Colab notebook per student that runs end to end, short report with
results table (paper vs baseline vs best model), shared GitHub repo, 2-min demo video.

## Repo layout

**Each part is a self-contained top-level folder** (own notebooks, `data/`, `figures/`, README,
requirements). Never put a part's files at the repo root; three more parts will be added later.

```
traffic_predictor/   Student 2 (this user)
  traffic_predictor.ipynb       THE single notebook (brief: one Colab notebook per student, runs start
                                to end). Parts: A data check (done), B MTGNN/DCRNN pipeline (done),
                                C LSTM (done), D GCN + GRU = T-GCN (code done, train on Colab),
                                E MTGNN re-implementation (code done, train on Colab), F results.
                                Add new work as new parts here,
                                not as new notebooks; all code inline so Colab needs no repo clone.
  data/README.md                sources; data/raw/ is gitignored, downloaded by the notebook
  data/processed/               small derived files (*.npy gitignored)
  models/                       <name>.pt best weights + <name>_checkpoint.pt (committed, small)
  results.csv                   summary table written at the end of the notebook
  figures/                      plots for the report
<other parts>/       to be added later (e.g. detector, app)
```

## METR-LA facts (from part A of the notebook)

- 34,272 steps × 207 sensors, 2012-03-01 → 2012-06-27, 5-min grid with no gaps.
- `node_values.npy` is (time, sensor, [speed mph, time-of-day 0–1]); matches `metr-la.h5` exactly.
- **Zero = missing**, not stopped traffic: 575,302 / 7,094,304 readings (8.11%).
  77% of zeros come from 2,148 steps where all 207 sensors are 0. Every sensor has zeros;
  worst sensor 718076 is 20.1% missing. Per-sensor table: `traffic_predictor/data/processed/metr_la_missing_per_sensor.csv`.
- PyG `METRLADatasetLoader` z-scores with zeros included (mean 53.72, std 20.26), so a missing
  reading becomes the constant −2.652 in `loader.X`. Clean stats: mean 58.46, std 13.02.

## Conventions for the traffic predictor (copy MTGNN, the strongest reference model)

User decision: follow the strongest model's pipeline exactly. MTGNN (KDD 2020; METR-LA MAE 2.69 @15 min,
3.05 @30 min) uses DCRNN's data script, so this also matches DCRNN. All of it is in part B of the notebook
(checked to produce DCRNN's exact samples: 23,974 / 3,425 / 6,850 windows).

- Zeros stay in the inputs as 0. No imputation (filling them is only an optional extra experiment).
- Features: speed + time of day. Windows 12 in → 12 out. Windows split in time order 70/10/20.
- Z-score only the speed channel of the inputs with train-input mean/std, zeros included
  (mean 54.406, std 19.494). Targets stay in mph; inverse-transform model output before the loss.
- Loss = masked MAE (`null_val=0`); metrics = masked MAE / RMSE / MAPE at 5/15/30 min
  (output step index 0/2/5). Clip gradients at 5. MTGNN also uses curriculum learning.
- Never score with unmasked metrics or the PyG Temporal example's plain MSE: not comparable.
- Test split has 12.1% zero targets (train 7.1%).

## Training tools (part C1, reused by parts D and E)

- `fit(model, name, max_epochs=30, lr=1e-3, patience=5, clip=5.0, weight_decay=0.0, cl_step=None)`: Adam,
  masked MAE on inverse-transformed output, keeps best val weights. `patience=None` = no early stopping.
  `cl_step` = MTGNN curriculum (loss on first `task_level` steps, +1 every `cl_step` batches; state keeps
  `iter`/`task_level`). Saves `models/<name>_checkpoint.pt` every epoch and resumes from it; best weights in
  `models/<name>.pt`. A finished checkpoint means rerunning the notebook just loads the model.
- `fetch_trained(name)` (called by `fit`) downloads `models/<name>*.pt` from GitHub `main` raw if missing locally,
  so Colab doesn't retrain the LSTM.
- `MTGNN_RECIPE = dict(max_epochs=100, lr=1e-3, weight_decay=1e-4, clip=5.0, patience=None, cl_step=2500)`,
  from MTGNN's train.py/trainer.py; used for parts D and E (user decision: "mimic the lowest-error model").
- On Colab, ROOT = `/content/drive/MyDrive/SIREN/traffic_predictor` (Drive mounted in part A) so checkpoints
  survive disconnects. The user uploads trained `models/*.pt`, `results.csv`, figures back to GitHub.
- `predict(model, loader)` returns mph predictions and targets; `horizon_report` scores them;
  add each model to `results` so the final `summary` / `results.csv` picks it up.
- `QUICK_CHECK = True` runs a tiny smoke test instead of full training.
- Cost here (4-core CPU, no GPU): LSTM about 3.5 min/epoch, 30 epochs ≈ 1 h 50 min including the notebook.
  T-GCN and MTGNN take roughly 10+ min/epoch on this CPU, so 100 epochs need the Colab GPU (user decision).
- To refresh outputs here without training D/E: run only the cells before part D with nbclient
  (`execute_cell` per cell up to the `## D.` markdown cell), leaving D–F without outputs.
- Run the notebook headless with
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 traffic_predictor/traffic_predictor.ipynb`.

## Results so far (test split, masked MAE in mph)

| Model | 5 min | 15 min | 30 min |
|---|---|---|---|
| FC-LSTM (DCRNN paper Table 2) | – | 3.44 (RMSE 6.30) | 3.77 (RMSE 7.23) |
| DCRNN paper / official README | – / 2.18 | 2.77 / 2.67 | 3.15 / 3.08 |
| MTGNN paper | – | 2.69 | 3.05 |
| Naive repeat-last-reading (ours) | 2.81 | 3.50 | 4.21 |
| LSTM baseline (ours, part C) | 2.40 | 3.01 | 3.62 |

Parts D/E (not trained yet): T-GCN cell copied from lehaifeng/T-GCN PyTorch code (hidden 64, 2 inputs,
normalised adjacency D^-1/2 (A+I)^T D^-1/2 from PyG's `adj_mat.npy` = DCRNN `adj_mx.pkl`, same sensor order as the
h5, checked), 13,644 params. MTGNN rebuilt from nnzhan/MTGNN net.py/layer.py with METR-LA defaults (k 20, node
dim 40, 3 layers, 32/32/64/128 channels, dropout 0.3, gcn depth 2, prop alpha 0.05, tanh alpha 3):
402,284 params + 3,168 for unused residual convs = 405,452, the original's count.

Paper numbers were checked against the papers' LaTeX tables. LSTM: one shared LSTM(2→64, 2 layers) + Linear(64→12)
per sensor, 51,468 params, 30 epochs (hit the cap, val still slowly improving), best val 3.29.

## Environment gotchas

- `torch-geometric-temporal` imports `torch_sparse` at package level; install it with `--no-deps`
  and register a stub module if `torch_sparse` is missing (see the notebook's import cell).
  GConvGRU / TGCN layers don't need it.
- Pass an **absolute** `raw_data_dir` to `METRLADatasetLoader` (it double-joins relative paths).
- `metr-la.h5` is an old pandas/py2 file: `pd.read_hdf` fails, read `df/axis0`, `df/axis1`,
  `df/block0_values` with `h5py`.
- In the cloud dev container the Box download URL is blocked; notebooks fall back to GitHub
  mirrors (raw.githubusercontent.com / media.githubusercontent.com for LFS). Local venv: `.venv/`.
- Notebooks set `ROOT` to the `traffic_predictor/` folder (works from inside it, from the repo root,
  or on Colab where ROOT = cwd) and keep all paths under it.
