import numpy as np, pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score


def top1(df, score_col="p", n_points=None):
    """Observation-level Top-1: argmax candidate per GPS point vs reference arc.
    Points without any candidate count as errors when n_points is given."""
    idx = df.groupby(["trip", "pid"])[score_col].idxmax()
    best = df.loc[idx]
    correct = best["y"].sum()
    denom = n_points if n_points is not None else len(best)
    return correct / denom, best


def route_continuity(best, trips_arcs):
    """Fraction of consecutive GPS points whose predicted arcs are the same arc
    or directly connected (to-node of previous == from-node of next)."""
    ok, tot = 0, 0
    for tid, g in best.sort_values(["trip", "pid"]).groupby("trip"):
        arcs = trips_arcs[tid]
        a = g["arc"].values
        same = a[1:] == a[:-1]
        conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
        ok += int((same | conn).sum()); tot += len(a) - 1
    return ok / max(tot, 1)


def classification_report(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    out = dict(macro_f1=f1_score(y, yhat, average="macro"),
               precision=precision_score(y, yhat, zero_division=0),
               recall=recall_score(y, yhat, zero_division=0),
               pos_f1=f1_score(y, yhat, zero_division=0))
    try:
        out["log_loss"] = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))
        out["auc"] = roc_auc_score(y, p)
    except ValueError:
        out["log_loss"] = np.nan; out["auc"] = np.nan
    return out
