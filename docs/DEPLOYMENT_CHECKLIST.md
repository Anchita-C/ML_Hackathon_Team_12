# Deployment and demo checklist (Laptop 1 = API on Mac, Laptop 2 = UI on Windows)

## Before the demo (do once, at lunch)
1. Laptop 1: clone the repo, `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
2. Check `artifacts/final/model_lgbm.txt` and `config.json` are present (they come from the repo).
3. Put the student-test trips in a folder (one sub-folder per trip) and run the API on them first:
   `PATHSNAP_EVAL_DIR="<folder>" uvicorn api.main:app --host 0.0.0.0 --port 8000`
4. Laptop 1 IP: `ipconfig getifaddr en0`. Allow incoming connections for Python in the macOS firewall if asked.
5. Both laptops on the same Wi-Fi (campus Wi-Fi may isolate devices; a phone hotspot is the fallback).
6. From Laptop 2 browser: `http://<laptop1-ip>:8000/health` should show status ok.
7. Laptop 2: `cd ui`, `python -m http.server 8080`, open `http://localhost:8080`, set API URL, Connect, run one trip.

## At the final evaluation
1. Start the API with `PATHSNAP_EVAL_DIR` pointing at the evaluator trips. Never copy these into the repo or open the ground truth.
2. Check `/health` shows the expected number of trips.
3. Run trips from the UI; the terminal prints a RESULT line per trip and `logs/api_requests.log` keeps the log.
4. If a trip is slow (long trips can take several seconds), wait; do not click twice.

## If something breaks
| Symptom | Fix |
|---|---|
| UI cannot connect | Wrong IP or firewall; test `/health` from the UI laptop's browser; try hotspot |
| /trips is empty | `PATHSNAP_EVAL_DIR` wrong, or trip folders do not contain `<id>.track` |
| 404 on /predict | Trip id not in the list; use ids from /trips |
| Import error on start | Venv not active, or `pip install -r requirements.txt` not run |
