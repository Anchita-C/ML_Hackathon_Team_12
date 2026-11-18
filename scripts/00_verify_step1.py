"""Independent re-check of the headline dataset statistics, straight from the raw files.
It does NOT reuse the EDA script, so agreement between the two is a real cross-check.

Usage: python scripts/00_verify_step1.py --data "<Student folder>"
Prints a table, writes artifacts/eda/verification_raw_stats.csv
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="artifacts/eda")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
rows = []
for part in ["train", "student_test"]:
    for tid in sorted(os.listdir(os.path.join(args.data, part))):
        p = os.path.join(args.data, part, tid, tid)
        files = {e: os.path.exists(f"{p}.{e}") for e in ["track", "nodes", "arcs", "route"]}
        tr = np.loadtxt(p + ".track", ndmin=2); nd = np.loadtxt(p + ".nodes", ndmin=2)
        ar = np.loadtxt(p + ".arcs", dtype=np.int64, ndmin=2); rt = np.loadtxt(p + ".route", dtype=np.int64, ndmin=1)
        dt = np.diff(tr[:, 2]); S = set(map(tuple, ar))
        # weak components via union-find
        par = np.arange(len(nd))
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for a, b in ar:
            ra, rb = find(a), find(b)
            if ra != rb: par[ra] = rb
        comps = len({find(i) for i in range(len(nd))})
        rows.append(dict(partition=part, trip=tid, all_4_files=all(files.values()),
            gps_points=len(tr), nodes=len(nd), arcs=len(ar), route_len=len(rt),
            nan_cells=int(np.isnan(tr).sum() + np.isnan(nd).sum()),
            non_increasing_time=int((dt <= 0).sum()), gaps_gt10s=int((dt > 10).sum()), max_gap_s=float(dt.max()),
            steps_not_1s=float(np.mean(dt != 1)), lon_in_range=bool(np.all(np.abs(tr[:, 0]) <= 180)),
            lat_in_range=bool(np.all(np.abs(tr[:, 1]) <= 90)), route_ids_valid=bool(rt.min() >= 0 and rt.max() < len(ar)),
            arc_ids_valid=bool(ar.min() >= 0 and ar.max() < len(nd)),
            self_loops=int((ar[:, 0] == ar[:, 1]).sum()), duplicate_arcs=len(ar) - len(S),
            two_way_share=float(np.mean([(b, a) in S for a, b in S])),
            isolated_nodes=int(len(nd) - len(np.unique(ar))), weak_components=comps,
            route_breaks=int((ar[rt[:-1], 1] != ar[rt[1:], 0]).sum())))
d = pd.DataFrame(rows); d.to_csv(os.path.join(args.out, "verification_raw_stats.csv"), index=False)
g = d.groupby("partition")
out = pd.DataFrame({"trips": g.size(), "all_4_files": g.all_4_files.sum(), "gps_points": g.gps_points.sum(),
    "median_points": g.gps_points.median(), "min_arcs": g.arcs.min(), "max_arcs": g.arcs.max(),
    "nan_cells": g.nan_cells.sum(), "non_increasing_time": g.non_increasing_time.sum(),
    "gaps_gt10s": g.gaps_gt10s.sum(), "max_gap_s": g.max_gap_s.max(), "avg_steps_not_1s": g.steps_not_1s.mean(),
    "coords_out_of_range": (~(g.lon_in_range.all() & g.lat_in_range.all())).astype(int),
    "bad_route_ids": (~g.route_ids_valid.all()).astype(int) , "bad_arc_ids": (~g.arc_ids_valid.all()).astype(int),
    "self_loops": g.self_loops.sum(), "duplicate_arcs": g.duplicate_arcs.sum(), "avg_two_way_share": g.two_way_share.mean(),
    "isolated_nodes": g.isolated_nodes.sum(), "max_components": g.weak_components.max(), "route_breaks": g.route_breaks.sum()}).T
print(out.round(4).to_string())
