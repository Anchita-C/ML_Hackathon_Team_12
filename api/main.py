"""PathSnap FastAPI server (Laptop 1).

Run:  PATHSNAP_EVAL_DIR="/path/to/evaluator_test" uvicorn api.main:app --host 0.0.0.0 --port 8000
The evaluation folder holds one sub-folder per trip (<id>/<id>.track|.nodes|.arcs|.route).
Only .track/.nodes/.arcs reach the prediction pipeline; .route is read by the
server-side evaluator after prediction. Ground truth is never returned to the UI.
"""
import logging, os, sys, time
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.pipeline import PathSnapPredictor, evaluate_trip, route_from_point_arcs

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "eval_data"))
FINAL_DIR = os.environ.get("PATHSNAP_MODEL_DIR", os.path.join(ROOT, "artifacts", "final"))
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(os.path.join(ROOT, "logs", "api_requests.log"))])
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap Map-Matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = PathSnapPredictor(FINAL_DIR)
log.info(f"Model loaded from {FINAL_DIR}; evaluation trips dir = {EVAL_DIR}")


def list_eval_trips():
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(d for d in os.listdir(EVAL_DIR)
                  if os.path.isfile(os.path.join(EVAL_DIR, d, f"{d}.track")))


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.cfg.get("model_name"), "n_trips": len(list_eval_trips())}


@app.get("/trips")
def trips(request: Request):
    t = list_eval_trips()
    log.info(f"GET /trips from {request.client.host} -> {len(t)} trips")
    return {"trips": t}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    if tid not in list_eval_trips():
        log.warning(f"POST /predict unknown trip '{tid}' from {request.client.host}")
        raise HTTPException(404, f"unknown trip {tid}")
    tdir = os.path.join(EVAL_DIR, tid)
    log.info(f"POST /predict trip={tid} from {request.client.host} -> running pipeline")
    res = predictor.predict(tdir)
    metrics = evaluate_trip(tdir, res)           # server-side only
    trip = res["trip"]
    ll = trip.nodes_ll
    route = route_from_point_arcs(res["hmm_arcs"])
    segs = [[[float(ll[f, 1]), float(ll[f, 0])], [float(ll[t, 1]), float(ll[t, 0])]] for f, t in trip.arcs[route]]
    clf_route = route_from_point_arcs(res["clf_arcs"])
    clf_segs = [[[float(ll[f, 1]), float(ll[f, 0])], [float(ll[t, 1]), float(ll[t, 0])]] for f, t in trip.arcs[clf_route]]
    gps = [[float(lat), float(lon)] for lon, lat in trip.track_ll]
    h, c = metrics["hmm"], metrics["clf"]
    log.info(f"RESULT trip={tid} points={metrics['n_points']} | HMM top1={h['top1_accuracy']:.4f} "
             f"routeF1={h['route_f1_len']:.4f} continuity={h['route_continuity']:.4f} | classifier-only "
             f"top1={c['top1_accuracy']:.4f} | breaks={res['chain_breaks']} | {res['timing_ms']['total']:.0f} ms")
    return {"trip_id": tid, "n_points": metrics["n_points"], "gps": gps,
            "predicted_route_segments": segs, "classifier_only_segments": clf_segs,
            "predicted_arc_ids": [int(a) for a in route],
            "metrics": metrics, "chain_breaks": res["chain_breaks"], "timing_ms": res["timing_ms"],
            "served_at": datetime.now().isoformat(timespec="seconds")}
