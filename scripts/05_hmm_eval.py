"""HMM/Viterbi decoder: parameter selection on TRAINING trips (out-of-fold classifier
probabilities, reliable labels only), then classifier-only vs classifier+HMM on the
10 student-test trips. Writes artifacts/hmm/ and the final inference config.
"""
import argparse, itertools, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import load_trip, canonical_arc_ids
from pathsnap.hmm import viterbi, RoadGraph
from pathsnap.pipeline import route_from_point_arcs, continuity
from pathsnap.features import FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--art", default="artifacts")
ap.add_argument("--oof", default="artifacts/bayes_opt/oof_train_tuned.parquet")
ap.add_argument("--test_pred", default="artifacts/bayes_opt/pred_test_LightGBM_tuned.parquet")
args = ap.parse_args()
out = os.path.join(args.art, "hmm"); os.makedirs(out, exist_ok=True)
GAP_S, UNREACH = 60.0, -30.0


def prepare(part, pred_path):
    f = pd.read_parquet(os.path.join(args.art, f"features_{part}.parquet"),
                        columns=["trip", "pid", "arc", "dist_rank", "proj_t", "arc_len", "y", "ref_dist", "dist"])
    f["p"] = pd.read_parquet(pred_path).p.values
    trips = {}
    for tid, g in f.groupby("trip"):
        tr = load_trip(os.path.join(args.data, part, tid)); tr.canon = canonical_arc_ids(tr.arcs)
        n = len(tr.xy)
        ref = g[g.y == 1].drop_duplicates("pid").set_index("pid").arc.reindex(range(n), fill_value=-1).values
        rd = g.drop_duplicates("pid").set_index("pid").ref_dist.reindex(range(n), fill_value=np.inf).values
        clf = g.loc[g.groupby("pid").p.idxmax()].set_index("pid").arc.reindex(range(n), fill_value=-1).values
        trips[tid] = dict(g=g, tr=tr, G=RoadGraph(tr), ref=ref, rd=rd, clf=clf)
    return trips


def score(d, pred):
    tr, ref = d["tr"], d["ref"]
    ok = (pred >= 0) & (ref >= 0) & (tr.canon[np.maximum(pred, 0)] == tr.canon[np.maximum(ref, 0)])
    return ok, d["rd"] <= 50


train = prepare("train", args.oof)
grid = [dict(beta=b, w_e=w, back_tol=bt, stop_speed=ss, signed=False)
        for b, w, bt, ss in itertools.product([3, 6, 12], [0.5, 1.0], [10, 40], [0.0, 4.0])]
rows = []


def run_grid(configs):
    for hpc in configs:
        c = n = 0; t0 = time.time()
        for d in train.values():
            pred, _ = viterbi(d["g"], d["tr"], graph=d["G"], gap_s=GAP_S, unreach=UNREACH, stop_w=0.1, **hpc)
            ok, rel = score(d, pred); c += ok[rel].sum(); n += rel.sum()
        rows.append(dict(**hpc, train_oof_top1=c / n, sec=time.time() - t0)); print(rows[-1], flush=True)


run_grid(grid)
b0 = max(rows, key=lambda r: r["train_oof_top1"])
run_grid([dict(beta=b0["beta"], w_e=b0["w_e"], back_tol=b0["back_tol"], stop_speed=b0["stop_speed"], signed=True)])  # signed variant
gdf = pd.DataFrame(rows); gdf.to_csv(os.path.join(out, "decoder_grid_train_oof.csv"), index=False)
best = gdf.sort_values("train_oof_top1", ascending=False).iloc[0]
hp = dict(beta=float(best.beta), w_e=float(best.w_e), back_tol=float(best.back_tol), stop_speed=float(best.stop_speed),
          stop_w=0.1, signed=bool(best.signed), gap_s=GAP_S, unreach=UNREACH)
clf_train = sum(score(d, d["clf"])[0][score(d, d["clf"])[1]].sum() for d in train.values()) / sum(d["rd"].__le__(50).sum() for d in train.values())
print("best decoder params", hp, "train OOF clf-only top1", clf_train)

test = prepare("student_test", args.test_pred)
per = []
for tid, d in test.items():
    t0 = time.time()
    pred, br = viterbi(d["g"], d["tr"], graph=d["G"], **hp); dt = time.time() - t0
    for name, pa in [("classifier_only", d["clf"]), ("classifier_hmm", pred)]:
        ok, rel = score(d, pa)
        canon, tr = d["tr"].canon, d["tr"]
        L = np.hypot(*(tr.nodes_xy[tr.arcs[:, 1]] - tr.nodes_xy[tr.arcs[:, 0]]).T)
        ps, rs = set(canon[route_from_point_arcs(pa)]), set(canon[tr.route])
        li = L[list(ps & rs)].sum(); prec = li / max(L[list(ps)].sum(), 1e-9); rec = li / max(L[list(rs)].sum(), 1e-9)
        per.append(dict(trip=tid, method=name, n_points=len(pa), top1=ok.mean(), top1_reliable=ok[rel].mean(),
                        route_precision_len=prec, route_recall_len=rec, route_f1_len=2 * prec * rec / max(prec + rec, 1e-9),
                        continuity=continuity(tr, pa), chain_breaks=br if name == "classifier_hmm" else np.nan,
                        decode_ms=1000 * dt if name == "classifier_hmm" else np.nan))
pdf = pd.DataFrame(per); pdf.to_csv(os.path.join(out, "student_test_clf_vs_hmm_per_trip.csv"), index=False)
w = pdf.assign(c=pdf.top1 * pdf.n_points).groupby("method").agg(points=("n_points", "sum"), c=("c", "sum"),
     route_f1_len=("route_f1_len", "mean"), continuity=("continuity", "mean"))
w["top1_pointweighted"] = w.c / w.points; w = w.drop(columns="c"); w.to_csv(os.path.join(out, "student_test_clf_vs_hmm_summary.csv"))
print(pdf.pivot(index="trip", columns="method", values="top1").round(4)); print(w.round(4))

cfg = dict(model_name="LightGBM (Optuna-tuned)", features=FEATURES, k=12, radius=150.0, hmm=hp,
           label_max_dist=50.0, notes="candidates: K nearest directed arcs within radius (metres, local AEQD projection)")
json.dump(cfg, open(os.path.join(args.art, "final", "config.json"), "w"), indent=2)
print("wrote final/config.json")
