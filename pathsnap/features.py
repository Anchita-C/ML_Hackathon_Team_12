"""Feature engineering for (GPS point, candidate arc) pairs.

Every feature is computed from the GPS track and road network only; the
reference route is used solely to create the training label `y`.
"""
import numpy as np
import pandas as pd
from .geometry import point_segment, angdiff
from .candidates import ArcIndex

CTX = (-5, -3, -1, 1, 3, 5)  # neighbour offsets for context features

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "head_diff", "head_cos", "head_rel", "speed", "speed_valid",
    "arc_len", "proj_t", "outside_seg", "has_reverse",
    "deg_from", "deg_to", "ctx_dist_mean", "ctx_dist_max", "ctx_rank_mean",
    "ctx_head_diff", "dt_prev", "dt_next", "turn_rate",
    "prog3", "prog10", "prog10_rel", "ctx_dist_far",
]


def track_kinematics(xy, t, w=2):
    """Heading/speed from a centred difference over +-w samples (time-aware)."""
    n = len(xy)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    dxy = xy[i1] - xy[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    spd = np.hypot(dxy[:, 0], dxy[:, 1]) / dt
    head = np.arctan2(dxy[:, 1], dxy[:, 0])
    # carry heading through stops (speed < 1 m/s) from last moving sample
    valid = spd >= 1.0
    if valid.any():
        idx = np.where(valid, np.arange(n), -1)
        idx = np.maximum.accumulate(idx)
        first = np.argmax(valid)
        idx[idx < 0] = first
        head = head[idx]
    dtp = np.r_[np.nan, np.diff(t)]
    dtn = np.r_[np.diff(t), np.nan]
    turn = np.r_[0, angdiff(head[1:], head[:-1])]
    return head, spd, valid, dtp, dtn, turn


def build_trip_frame(trip, k=12, radius=150.0, ref=None, ref_dist=None, index=None):
    index = index or ArcIndex(trip)
    xy, t = trip.xy, trip.t
    n = len(xy)
    cands = index.query(xy, k=k, radius=radius)
    lens = np.array([len(c[0]) for c in cands])
    pid = np.repeat(np.arange(n), lens)
    arc = np.concatenate([c[0] for c in cands]).astype(np.int64)
    dist = np.concatenate([c[1] for c in cands])
    rank = np.concatenate([np.arange(l) for l in lens])
    dmin = np.repeat([c[1][0] if len(c[1]) else np.nan for c in cands], lens)

    head, spd, valid, dtp, dtn, turn = track_kinematics(xy, t)
    a, b = index.a[arc], index.b[arc]
    _, tproj, L = point_segment(xy[pid, 0], xy[pid, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    arc_bear = np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])
    hd = angdiff(head[pid], arc_bear)

    df = pd.DataFrame({"pid": pid, "arc": arc, "dist": dist, "dist_rank": rank,
                       "dist_minus_min": dist - dmin, "dist_ratio": (dist + 1) / (dmin + 1),
                       "n_cand": lens[pid], "head_diff": np.degrees(hd), "head_cos": np.cos(hd),
                       "speed": spd[pid], "speed_valid": valid[pid].astype(np.int8),
                       "arc_len": L, "proj_t": tproj,
                       "outside_seg": ((tproj < 0) | (tproj > 1)).astype(np.int8)})
    df["head_rel"] = df["head_diff"] - df.groupby("pid")["head_diff"].transform("min")

    # graph features
    arcs = trip.arcs
    nn = len(trip.nodes_xy)
    deg_out = np.bincount(arcs[:, 0], minlength=nn)
    deg_in = np.bincount(arcs[:, 1], minlength=nn)
    df["deg_from"] = deg_in[arcs[arc, 0]] + deg_out[arcs[arc, 0]]
    df["deg_to"] = deg_in[arcs[arc, 1]] + deg_out[arcs[arc, 1]]
    key = arcs[:, 0] * (nn + 1) + arcs[:, 1]
    rkey = arcs[arc, 1] * (nn + 1) + arcs[arc, 0]
    df["has_reverse"] = np.isin(rkey, key).astype(np.int8)

    # context: how well does this arc explain neighbouring GPS points?
    cd, ch = [], []
    for o in CTX:
        q = np.clip(pid + o, 0, n - 1)
        d, _, _ = point_segment(xy[q, 0], xy[q, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        cd.append(d)
        ch.append(angdiff(head[q], arc_bear))
    cd = np.vstack(cd)
    df["ctx_dist_mean"] = cd.mean(0)
    df["ctx_dist_max"] = cd.max(0)
    df["ctx_head_diff"] = np.degrees(np.vstack(ch).mean(0))
    df["ctx_rank_mean"] = df.groupby("pid")["ctx_dist_mean"].rank(method="min")
    df["dt_prev"] = np.nan_to_num(dtp[pid], nan=1.0)
    df["dt_next"] = np.nan_to_num(dtn[pid], nan=1.0)
    df["turn_rate"] = np.degrees(turn[pid])

    # signed progress along the candidate arc direction over +-w points (m/s):
    # positive = vehicle moves from arc.from towards arc.to. Robust to slow speeds
    # where instantaneous heading is noise; separates a->b from b->a.
    ux, uy = (b[:, 0] - a[:, 0]) / np.maximum(L, 1e-6), (b[:, 1] - a[:, 1]) / np.maximum(L, 1e-6)
    for w in (3, 10):
        q0, q1 = np.clip(pid - w, 0, n - 1), np.clip(pid + w, 0, n - 1)
        disp = (xy[q1, 0] - xy[q0, 0]) * ux + (xy[q1, 1] - xy[q0, 1]) * uy
        df[f"prog{w}"] = disp / np.maximum(t[q1] - t[q0], 1.0)
    df["prog10_rel"] = df["prog10"] - df.groupby("pid")["prog10"].transform("max")
    far = []
    for o in (-15, -10, 10, 15):
        q = np.clip(pid + o, 0, n - 1)
        far.append(point_segment(xy[q, 0], xy[q, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])[0])
    df["ctx_dist_far"] = np.vstack(far).mean(0)

    if ref is not None:
        canon = trip.canon
        df["y"] = (canon[arc] == canon[ref[pid]]).astype(np.int8)
        df["ref_dist"] = ref_dist[pid]
    df["t"] = t[pid]
    return df, lens
