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
  01_get_and_check_data.ipynb   done
  data/README.md                sources; data/raw/ is gitignored, downloaded by the notebook
  data/processed/               small derived files (*.npy gitignored)
  figures/                      plots for the report
<other parts>/       to be added later (e.g. detector, app)
```

## METR-LA facts (from 01_get_and_check_data.ipynb)

- 34,272 steps × 207 sensors, 2012-03-01 → 2012-06-27, 5-min grid with no gaps.
- `node_values.npy` is (time, sensor, [speed mph, time-of-day 0–1]); matches `metr-la.h5` exactly.
- **Zero = missing**, not stopped traffic: 575,302 / 7,094,304 readings (8.11%).
  77% of zeros come from 2,148 steps where all 207 sensors are 0. Every sensor has zeros;
  worst sensor 718076 is 20.1% missing. Per-sensor table: `traffic_predictor/data/processed/metr_la_missing_per_sensor.csv`.
- PyG `METRLADatasetLoader` z-scores with zeros included (mean 53.72, std 20.26), so a missing
  reading becomes the constant −2.652 in `loader.X`. Clean stats: mean 58.46, std 13.02.

## Conventions for the traffic predictor

- Split in time order 70 / 10 / 20 (train / val / test), as DCRNN does.
- Fit normalisation on the train split only, ignoring zeros.
- Evaluate with **masked** MAE/RMSE (skip targets equal to 0) in mph, so numbers compare to DCRNN.
- Input window 12 steps (1 hour) unless there is a reason to change it.

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
