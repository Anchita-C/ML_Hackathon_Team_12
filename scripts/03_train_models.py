"""Train and compare all models with 5-fold GroupKFold (groups = trips) on the
80 training trips, then evaluate each refitted model on the 10 student-test trips.

Training rows: points whose derived reference arc is within 50 m (reliable labels).
Every model sees the same per-trip point subsample (SUB_FRAC) for fairness and speed.
"""
import argparse, os, sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from pathsnap.features import FEATURES
from pathsnap.models import get_models, SEED
from pathsnap.metrics import top1, classification_report, route_continuity
from pathsnap.data import load_trip, list_trips

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="artifacts")
ap.add_argument("--data", required=True)
ap.add_argument("--sub", type=float, default=0.25)
ap.add_argument("--models", default="all")
args = ap.parse_args()
rng = np.random.default_rng(SEED)
out = os.path.join(args.art, "experiments"); os.makedirs(out, exist_ok=True)

tr = pd.read_parquet(os.path.join(args.art, "features_train.parquet"))
te = pd.read_parquet(os.path.join(args.art, "features_student_test.parquet"))
tr = tr[tr.ref_dist <= 50]
pts = tr[["trip", "pid"]].drop_duplicates()
pts = pts[rng.random(len(pts)) < args.sub]
tr = tr.merge(pts, on=["trip", "pid"])
print("train rows", len(tr), "points", len(pts), "pos rate", tr.y.mean().round(4), flush=True)
X, y, g = tr[FEATURES].values.astype(np.float32), tr.y.values, tr.trip.values
Xte = te[FEATURES].values.astype(np.float32)
n_test_pts = te.groupby("trip").pid.max().add(1).sum()
test_arcs = {t: load_trip(os.path.join(args.data, "student_test", t)).arcs for t in te.trip.unique()}

models = get_models()
if args.models != "all":
    models = {k: v for k, v in models.items() if k in args.models.split(",")}
KNN_CAP = 150_000
rows = []
for name, (family, model) in models.items():
    res = dict(model=name, family=family, config=str(model)[:300])
    fold_f1, fold_top1, fold_ll = [], [], []
    t0 = time.time()
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        if name == "KNN" and len(a) > KNN_CAP:
            a = rng.choice(a, KNN_CAP, replace=False)
        m = pickle.loads(pickle.dumps(model))
        m.fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        cr = classification_report(y[b], p)
        vb = tr.iloc[b].assign(p=p)
        fold_f1.append(cr["macro_f1"]); fold_ll.append(cr["log_loss"]); fold_top1.append(top1(vb)[0])
        print(f"  {name} fold{k} F1={cr['macro_f1']:.4f} top1={fold_top1[-1]:.4f}", flush=True)
    res.update(cv_macro_f1=np.mean(fold_f1), cv_macro_f1_std=np.std(fold_f1), cv_top1=np.mean(fold_top1),
               cv_log_loss=np.mean(fold_ll), cv_time_s=time.time() - t0,
               cv_fold_f1=json.dumps([round(v, 5) for v in fold_f1]))
    # refit on all training rows, evaluate on student test
    a = np.arange(len(X))
    if name == "KNN" and len(a) > KNN_CAP:
        a = rng.choice(a, KNN_CAP, replace=False)
    m = pickle.loads(pickle.dumps(model))
    t1 = time.time(); m.fit(X[a], y[a]); res["train_time_s"] = time.time() - t1
    t1 = time.time(); p = m.predict_proba(Xte)[:, 1]; dt = time.time() - t1
    res["infer_ms_per_point"] = 1000 * dt / n_test_pts
    cr = classification_report(te.y.values, p)
    acc, best = top1(te.assign(p=p), n_points=n_test_pts)
    res.update({f"test_{k}": v for k, v in cr.items()})
    res["test_top1"] = acc
    res["test_route_continuity"] = route_continuity(best, test_arcs)
    res["test_confusion"] = json.dumps(confusion_matrix(te.y.values, (p >= 0.5).astype(int)).tolist())
    pd.DataFrame({"trip": te.trip, "pid": te.pid, "arc": te.arc, "p": p}).to_parquet(os.path.join(out, f"pred_test_{name}.parquet"))
    with open(os.path.join(out, f"model_{name}.pkl"), "wb") as f:
        pickle.dump(m, f)
    if name == "MLP":
        mlp = m[-1]
        json.dump({"train_loss": mlp.loss_curve_, "val_score": mlp.validation_scores_}, open(os.path.join(out, "mlp_history.json"), "w"))
    rows.append(res)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "config"}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"model_comparison{'' if args.models=='all' else '_'+args.models.replace(',','_')}.csv"), index=False)
