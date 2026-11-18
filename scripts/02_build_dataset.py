"""Build the candidate/feature table for train and student_test partitions.

Usage: python scripts/02_build_dataset.py --data "<Student folder>" --out artifacts/
"""
import argparse, os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import load_trip, list_trips, canonical_arc_ids
from pathsnap.labels import align_track_to_route
from pathsnap.features import build_trip_frame

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="artifacts")
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--radius", type=float, default=150.0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
stats = []
for part in ["train", "student_test"]:
    frames = []
    for tid in list_trips(os.path.join(args.data, part)):
        t0 = time.time()
        trip = load_trip(os.path.join(args.data, part, tid))
        trip.canon = canonical_arc_ids(trip.arcs)
        ref, _, rdist = align_track_to_route(trip)
        df, lens = build_trip_frame(trip, k=args.k, radius=args.radius, ref=ref, ref_dist=rdist)
        df.insert(0, "trip", tid)
        frames.append(df)
        hit = df.groupby("pid")["y"].max().reindex(range(len(trip.xy)), fill_value=0).values
        stats.append(dict(part=part, trip=tid, n_pts=len(trip.xy), rows=len(df), no_cand=int((lens == 0).sum()),
                          miss_rate=float(1 - hit.mean()), miss_rate_reliable=float(1 - hit[rdist <= 50].mean()),
                          unreliable=float((rdist > 50).mean()), sec=round(time.time() - t0, 2)))
        print(stats[-1], flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(os.path.join(args.out, f"features_{part}.parquet"))
pd.DataFrame(stats).to_csv(os.path.join(args.out, "candidate_stats.csv"), index=False)
