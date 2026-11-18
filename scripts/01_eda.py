"""Dataset exploration: inventory, data-quality checks, network connectivity,
label-derivation quality and plots/maps. Writes to artifacts/eda/.

Partitions used: train (80 trips) and student_test (10 trips). Evaluator data is never read.
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import load_trip, list_trips
from pathsnap.labels import align_track_to_route

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="artifacts/eda")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

rows, dts, speeds = [], {"train": [], "student_test": []}, []
for part in ["train", "student_test"]:
    for tid in list_trips(os.path.join(args.data, part)):
        d = os.path.join(args.data, part, tid)
        tr = load_trip(d)
        dt = np.diff(tr.t); dts[part].append(dt)
        n = len(tr.nodes_ll)
        g = coo_matrix((np.ones(len(tr.arcs)), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(n, n))
        wcc, wl = connected_components(g, directed=True, connection="weak")
        scc, _ = connected_components(g, directed=True, connection="strong")
        deg = np.bincount(tr.arcs.ravel(), minlength=n)
        s = set(map(tuple, tr.arcs))
        r = tr.route
        cont = tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0]
        ref, j, rd = align_track_to_route(tr)
        step = np.hypot(*np.diff(tr.xy, axis=0).T)
        spd = step / np.maximum(dt, 1e-9)
        rows.append(dict(
            partition=part, trip=tid, n_gps=len(tr.t), n_nodes=n, n_arcs=len(tr.arcs), route_len=len(r),
            size_kb=sum(os.path.getsize(os.path.join(d, f"{tid}.{e}")) for e in ["track", "nodes", "arcs", "route"]) // 1024,
            nan_values=int(np.isnan(np.c_[tr.track_ll, tr.t]).sum()), non_increasing_t=int((dt <= 0).sum()),
            t_start=tr.t[0], duration_s=tr.t[-1] - tr.t[0], frac_dt_1s=float(np.mean(dt == 1)),
            n_gaps_gt10s=int((dt > 10).sum()), max_gap_s=float(dt.max()),
            lon_min=tr.track_ll[:, 0].min(), lon_max=tr.track_ll[:, 0].max(),
            lat_min=tr.track_ll[:, 1].min(), lat_max=tr.track_ll[:, 1].max(),
            extent_km=float(np.hypot(*(tr.xy.max(0) - tr.xy.min(0))) / 1000),
            path_km=float(step.sum() / 1000), max_speed_ms=float(spd.max()), n_jump_gt60ms=int((spd > 60).sum()),
            isolated_nodes=int((deg == 0).sum()), weak_components=wcc, strong_components=scc,
            largest_wcc_frac=float(np.bincount(wl).max() / n), self_loops=int((tr.arcs[:, 0] == tr.arcs[:, 1]).sum()),
            duplicate_arcs=len(tr.arcs) - len(s), two_way_frac=float(np.mean([(b, a) in s for a, b in tr.arcs])),
            route_connected_frac=float(cont.mean()), route_breaks=int((~cont).sum()),
            label_dist_median=float(np.median(rd)), label_dist_p95=float(np.percentile(rd, 95)),
            unreliable_label_frac=float(np.mean(rd > 50))))
        speeds.append(spd)
inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(args.out, "dataset_inventory.csv"), index=False)
summ = inv.groupby("partition").agg(trips=("trip", "count"), gps_points=("n_gps", "sum"), gps_median=("n_gps", "median"),
                                   arcs_median=("n_arcs", "median"), nan=("nan_values", "sum"),
                                   gaps_gt10s=("n_gaps_gt10s", "sum"), max_gap=("max_gap_s", "max"),
                                   route_breaks=("route_breaks", "sum"), unreliable=("unreliable_label_frac", "mean"))
summ.to_csv(os.path.join(args.out, "summary_by_partition.csv")); print(summ.T)

# --- plots
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0, 0].hist(inv[inv.partition == "train"].n_gps, bins=30, alpha=.7, label="train"); ax[0, 0].hist(inv[inv.partition == "student_test"].n_gps, bins=30, alpha=.7, label="student_test")
ax[0, 0].set_xscale("log"); ax[0, 0].set_title("GPS points per trip"); ax[0, 0].legend()
alldt = np.concatenate(dts["train"]); ax[0, 1].hist(np.clip(alldt, 0, 30), bins=30); ax[0, 1].set_yscale("log"); ax[0, 1].set_title("Sampling interval (s, clipped at 30) - train")
ax[0, 2].scatter(inv.n_arcs, inv.n_gps, c=(inv.partition == "train"), cmap="coolwarm"); ax[0, 2].set_xscale("log"); ax[0, 2].set_yscale("log"); ax[0, 2].set_title("Network size vs track length")
ax[0, 2].set_xlabel("arcs"); ax[0, 2].set_ylabel("GPS points")
ax[1, 0].hist(inv.label_dist_median, bins=40); ax[1, 0].set_title("Median GPS-to-reference-arc distance (m)")
ax[1, 1].hist(np.clip(np.concatenate(speeds), 0, 50), bins=50); ax[1, 1].set_yscale("log"); ax[1, 1].set_title("Point-to-point speed (m/s, clipped 50)")
ax[1, 2].scatter(inv.lon_min, inv.lat_min, c=(inv.partition == "train"), cmap="coolwarm"); ax[1, 2].set_title("Trip locations (lon, lat) - world-wide"); ax[1, 2].set_xlabel("lon"); ax[1, 2].set_ylabel("lat")
plt.tight_layout(); plt.savefig(os.path.join(args.out, "eda_overview.png"), dpi=120)

# road network + trajectory plot for one trip
tr = load_trip(os.path.join(args.data, "train", "00000005"))
A = tr.nodes_ll; fig, ax = plt.subplots(figsize=(7, 8))
for f, t_ in tr.arcs[::1]:
    ax.plot(A[[f, t_], 0], A[[f, t_], 1], color="lightgray", lw=.6)
r = tr.route
for f, t_ in tr.arcs[r]:
    ax.plot(A[[f, t_], 0], A[[f, t_], 1], color="green", lw=2.5)
ax.scatter(tr.track_ll[:, 0], tr.track_ll[:, 1], s=2, color="blue")
ax.set_title("Trip 00000005: network (grey), reference route (green), GPS (blue)"); plt.savefig(os.path.join(args.out, "trip5_network.png"), dpi=120)


def trip_map(part, tid, path):
    tr = load_trip(os.path.join(args.data, part, tid))
    ref, j, rd = align_track_to_route(tr)
    c = tr.track_ll.mean(0)[::-1]
    m = folium.Map(location=list(c), zoom_start=14, tiles="OpenStreetMap")
    net = folium.FeatureGroup("Road network (near track)", show=True)
    lo, hi = tr.track_ll.min(0) - 0.003, tr.track_ll.max(0) + 0.003
    A = tr.nodes_ll
    inb = np.all((A[tr.arcs[:, 0]] >= lo) & (A[tr.arcs[:, 0]] <= hi), 1)
    for f, t_ in tr.arcs[inb][:6000]:
        folium.PolyLine([A[f][::-1], A[t_][::-1]], color="#999", weight=1, opacity=.6).add_to(net)
    net.add_to(m)
    rt = folium.FeatureGroup("Reference route (.route)")
    for f, t_ in tr.arcs[tr.route]:
        folium.PolyLine([A[f][::-1], A[t_][::-1]], color="green", weight=5, opacity=.8).add_to(rt)
    rt.add_to(m)
    gp = folium.FeatureGroup("Raw GPS")
    for k, (lon, lat) in enumerate(tr.track_ll[::3]):
        bad = rd[k * 3] > 50
        folium.CircleMarker([lat, lon], radius=2, color="red" if bad else "blue", fill=True,
                            tooltip=f"t={tr.t[k*3]:.0f}s, dist to ref arc={rd[k*3]:.1f} m").add_to(gp)
    gp.add_to(m)
    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:999;background:white;padding:8px;border:1px solid #888;font-size:13px">'
        f'<b>Trip {tid} ({part})</b><br><span style="color:blue">&#9679;</span> GPS (label &lt;=50 m)<br>'
        '<span style="color:red">&#9679;</span> GPS far from reference (&gt;50 m)<br><span style="color:green">&#9644;</span> Reference route<br>'
        '<span style="color:#999">&#9644;</span> Road network</div>'))
    m.save(path)

trip_map("train", "00000005", os.path.join(args.out, "map_train_00000005.html"))
trip_map("train", "00000033", os.path.join(args.out, "map_train_00000033_label_issue.html"))
trip_map("student_test", "00000052", os.path.join(args.out, "map_test_00000052.html"))
print("done")
