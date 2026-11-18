# PathSnap – Map-Matching & Route Reconstruction (UE24CS352A Hackathon)

Noisy GPS trajectories are matched to the road network in two stages:

1. **Candidate classifier**: for every GPS point, the K=12 nearest directed road arcs
   (within 150 m) are scored with a trained model, which outputs P(arc is correct).
2. **HMM / Viterbi decoder**: candidate arcs are the hidden states, classifier probabilities
   are the emissions, and network-distance vs GPS-distance gives the transitions. The decoder
   returns one connected, plausible road segment per GPS point.

The system runs on two laptops: a **FastAPI** server (Laptop 1) with the model, road
networks and server-side scoring, and a browser **UI** (Laptop 2).

## Team and contributions
| Member | SRN | Part | Main files |
|---|---|---|---|
| Akshaya M | 50 | Data exploration, preprocessing, feature engineering | `pathsnap/data.py`, `geometry.py`, `labels.py`, `candidates.py`, `features.py`; `scripts/00-02`; `docs/STEP1_SUMMARY.md`, `docs/STEP1B_FEATURES_SUMMARY.md` |
| Anchita Chhibba | 60 | Model comparison (8 models) | `pathsnap/models.py`, `metrics.py`; `scripts/03*`; `docs/STEP2_SUMMARY.md` |
| Akash | 9 | Bayesian tuning and HMM route reconstruction | `pathsnap/hmm.py`; `scripts/04, 05`; `docs/STEP3_SUMMARY.md` |
| Aditya Raj | 33 (PES2UG24CS033) | End-to-end pipeline, API, UI, deployment, plots and maps | `pathsnap/pipeline.py`, `api/`, `ui/`, `scripts/06, 07`, this README |

Per-step summaries with the verified numbers are in `docs/`. The deployment and demo checklist is `docs/DEPLOYMENT_CHECKLIST.md`.

## Results (10 student-test trips, evaluator trips never used)
| Stage | Top-1 | Route F1 | Route continuity |
|---|---|---|---|
| Nearest road (rule) | 0.544 | - | - |
| Nearest road + heading (rule) | 0.857 | - | - |
| Best of 8 models, untuned LightGBM | 0.912 | - | - |
| Tuned LightGBM (Optuna, 40 trials) | 0.917 | 0.958 | 0.950 |
| Tuned LightGBM + HMM (final system) | 0.945 | 0.983 | 0.980 |

The labels are derived by us from the supplied `.route` files (monotone alignment), so these numbers are measured against derived labels; limitations are listed in `docs/`.

## Repository layout
```
pathsnap/            core library
  data.py            raw-file loading, per-trip AEQD projection, duplicate-arc canonicalisation
  labels.py          monotone DP alignment of GPS points to the .route arcs (training labels)
  candidates.py      KD-tree candidate generation
  features.py        27 point/candidate features
  models.py          model zoo (8 models, 6 algorithm families)
  metrics.py         Top-1, macro-F1, log loss, route continuity
  hmm.py             HMM + Viterbi decoder (bounded Dijkstra transitions, gap handling)
  pipeline.py        end-to-end inference + server-side evaluation
scripts/
  01_eda.py          dataset inventory, data-quality checks, plots, Folium maps
  02_build_dataset.py  labels + candidates + features -> parquet
  03_train_models.py   GroupKFold CV + student-test evaluation for every model
  03b_baselines.py     non-learned reference baselines
  04_bayes_opt.py      Optuna (TPE) Bayesian optimisation of LightGBM, tuned vs untuned, OOF preds
  05_hmm_eval.py       decoder parameter selection (train OOF) + classifier vs HMM comparison
  06_plots.py          comparison plots
  07_folium_maps.py    Folium maps of predicted routes
api/main.py          FastAPI server (Laptop 1)
ui/index.html        evaluator interface (Laptop 2)
artifacts/           results: eda/, experiments/, bayes_opt/, hmm/, plots/, maps/, final/ (model + config)
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the experiments
```bash
DATA="/path/to/Student folder"          # contains train/ and student_test/
python scripts/01_eda.py --data "$DATA"
python scripts/02_build_dataset.py --data "$DATA"
python scripts/03_train_models.py --data "$DATA"
python scripts/03b_baselines.py
python scripts/04_bayes_opt.py --trials 30
python scripts/05_hmm_eval.py --data "$DATA"
python scripts/06_plots.py
python scripts/07_folium_maps.py --data "$DATA"
```
Seed: 33 everywhere. The evaluator-testing partition is never read by any training or selection script.

## Deployment (demo)
**Laptop 1 (API)**, with both laptops on the same Wi-Fi:
```bash
source .venv/bin/activate
export PATHSNAP_EVAL_DIR="/path/to/evaluator_test"   # one sub-folder per trip
uvicorn api.main:app --host 0.0.0.0 --port 8000
# find this laptop's IP (macOS): ipconfig getifaddr en0
```
Request logs go to the terminal and to `logs/api_requests.log`.

**Laptop 2 (UI)**:
```bash
cd ui && python3 -m http.server 8080
# open http://localhost:8080, set the API URL to http://<laptop1-ip>:8000, click "Connect"
```
The UI lists the trips found on the server, sends the selected trip id to `POST /predict`,
and draws the GPS points and predicted route along with the server-computed metrics.
Ground-truth routes stay on the server; only scores are returned.

### API
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status |
| GET | `/trips` | – | list of evaluation trip ids |
| POST | `/predict` | `{"trip_id": "00000000"}` | GPS, predicted route segments, classifier-only segments, metrics, timing |

If the laptops can't reach each other: allow incoming connections for Python in the macOS
firewall, and check `curl http://<laptop1-ip>:8000/health` from Laptop 2. If the campus Wi-Fi
isolates clients, use a phone hotspot.

## Windows notes (UI laptop)
The UI is a static page, so Windows only needs Python: `cd ui` then `python -m http.server 8080`, open `http://localhost:8080`, enter the API URL. No packages and no firewall change are needed on this laptop.

## Academic integrity
All results in `artifacts/` are produced by the scripts above. AI assistance was used
during development and is disclosed in the technical report.
