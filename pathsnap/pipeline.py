"""End-to-end inference (used by the API) and server-side evaluation.

The predictor NEVER reads the .route file: trips are loaded with with_route=False.
Ground truth is only read by `evaluate_trip`, after prediction, to score it.
"""
import json, os, pickle, time
import numpy as np
from .data import load_trip, canonical_arc_ids
from .features import build_trip_frame, FEATURES
from .hmm import viterbi, RoadGraph
from .labels import align_track_to_route


class PathSnapPredictor:
    def __init__(self, final_dir):
        self.cfg = json.load(open(os.path.join(final_dir, "config.json")))
        txt = os.path.join(final_dir, "model_lgbm.txt")
        if os.path.exists(txt):                    # version-independent LightGBM format
            import lightgbm as lgb
            self.booster = lgb.Booster(model_file=txt)
            self.model = None
        else:
            self.booster = None
            self.model = pickle.load(open(os.path.join(final_dir, "model.pkl"), "rb"))

    def score(self, X):
        if self.booster is not None:
            return self.booster.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, trip_dir):
        t0 = time.time()
        trip = load_trip(trip_dir, with_route=False)
        c = self.cfg
        frame, lens = build_trip_frame(trip, k=c["k"], radius=c["radius"])
        frame["p"] = self.score(frame[FEATURES].values.astype(np.float32)) if len(frame) else []
        t_clf = time.time()
        n = len(trip.xy)
        clf = np.full(n, -1, dtype=np.int64)
        if len(frame):
            best = frame.loc[frame.groupby("pid")["p"].idxmax()]
            clf[best.pid.values] = best.arc.values
        h = c["hmm"]
        hmm, breaks = viterbi(frame, trip, graph=RoadGraph(trip), beta=h["beta"], w_e=h["w_e"],
                              gap_s=h["gap_s"], back_tol=h["back_tol"], unreach=h["unreach"],
                              stop_speed=h.get("stop_speed", 0.0), stop_w=h.get("stop_w", 0.1),
                              signed=h.get("signed", False))
        return dict(trip=trip, clf_arcs=clf, hmm_arcs=hmm, chain_breaks=breaks,
                    timing_ms=dict(features_and_classifier=1000 * (t_clf - t0), hmm=1000 * (time.time() - t_clf),
                                   total=1000 * (time.time() - t0)))


def route_from_point_arcs(arcs):
    """Ordered arc sequence with consecutive duplicates removed."""
    a = arcs[arcs >= 0]
    if len(a) == 0:
        return a
    return a[np.r_[True, a[1:] != a[:-1]]]


def continuity(trip, arcs):
    a = arcs[arcs >= 0]
    if len(a) < 2:
        return float("nan")
    A = trip.arcs
    return float(np.mean((a[1:] == a[:-1]) | (A[a[:-1], 1] == A[a[1:], 0])))


def evaluate_trip(trip_dir, result, max_label_dist=50.0):
    """Server-side scoring against the reference route (never sent to the model/UI)."""
    gt = load_trip(trip_dir, with_route=True)
    canon = canonical_arc_ids(gt.arcs)
    ref, _, rdist = align_track_to_route(gt)
    out = {}
    for name in ("clf", "hmm"):
        pa = result[f"{name}_arcs"]
        ok = (pa >= 0) & (canon[np.maximum(pa, 0)] == canon[ref])
        rel = rdist <= max_label_dist
        pred_set = set(canon[route_from_point_arcs(pa)])
        ref_set = set(canon[gt.route])
        L = np.hypot(*(gt.nodes_xy[gt.arcs[:, 1]] - gt.nodes_xy[gt.arcs[:, 0]]).T)
        inter = pred_set & ref_set
        lp = sum(L[list(pred_set)]); lr = sum(L[list(ref_set)]); li = sum(L[list(inter)])
        prec = li / lp if lp else 0.0; rec = li / lr if lr else 0.0
        out[name] = dict(top1_accuracy=float(ok.mean()), top1_accuracy_reliable_labels=float(ok[rel].mean()),
                         route_precision_len=prec, route_recall_len=rec,
                         route_f1_len=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                         route_continuity=continuity(gt, pa))
    out["n_points"] = int(len(ref)); out["reliable_label_fraction"] = float(np.mean(rdist <= max_label_dist))
    return out
