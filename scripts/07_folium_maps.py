"""Folium maps of predicted routes on student-test trips: raw GPS, nearby road network,
reference route, classifier-only prediction and classifier+HMM prediction (layer control)."""
import argparse, os, sys
import numpy as np, pandas as pd, folium
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.pipeline import PathSnapPredictor, route_from_point_arcs, evaluate_trip

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--art", default="artifacts")
ap.add_argument("--trips", default="00000077,00000056,00000009")
args = ap.parse_args(); out = os.path.join(args.art, "maps"); os.makedirs(out, exist_ok=True)
pred = PathSnapPredictor(os.path.join(args.art, "final"))

for tid in args.trips.split(","):
    tdir = os.path.join(args.data, "student_test", tid)
    res = pred.predict(tdir); met = evaluate_trip(tdir, res)
    from pathsnap.data import load_trip
    gt = load_trip(tdir, with_route=True); ll = gt.nodes_ll; A = gt.arcs
    m = folium.Map(location=list(gt.track_ll.mean(0)[::-1]), zoom_start=15, tiles="OpenStreetMap")
    lo, hi = gt.track_ll.min(0) - .002, gt.track_ll.max(0) + .002
    inb = np.where(np.all((ll[A[:, 0]] >= lo) & (ll[A[:, 0]] <= hi), 1))[0][:8000]
    def seg_layer(name, arcs, color, weight, show=True, dash=None):
        fg = folium.FeatureGroup(name, show=show)
        for f, t in A[arcs]:
            folium.PolyLine([ll[f][::-1], ll[t][::-1]], color=color, weight=weight, opacity=.85, dash_array=dash).add_to(fg)
        fg.add_to(m)
    seg_layer("Road network", inb, "#9e9e9e", 1)
    seg_layer("Reference route (ground truth)", gt.route, "#2ca02c", 7)
    seg_layer("Classifier-only prediction", route_from_point_arcs(res["clf_arcs"]), "#ff7f0e", 3, show=False, dash="5,5")
    seg_layer("Predicted route (classifier + HMM)", route_from_point_arcs(res["hmm_arcs"]), "#d62728", 4)
    fg = folium.FeatureGroup("Raw GPS")
    step = max(1, len(gt.track_ll) // 1500)
    for k in range(0, len(gt.track_ll), step):
        lon, lat = gt.track_ll[k]
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True, tooltip=f"#{k} t={gt.t[k]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker(gt.track_ll[0][::-1].tolist(), tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gt.track_ll[-1][::-1].tolist(), tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([gt.track_ll.min(0)[::-1].tolist(), gt.track_ll.max(0)[::-1].tolist()])
    h, c = met["hmm"], met["clf"]
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:10px;border:1px solid #777;font-size:13px">'
        f'<b>Student-test trip {tid}</b><br>'
        '<span style="color:#1f77b4">&#9679;</span> Raw GPS<br><span style="color:#2ca02c">&#9644;</span> Reference route<br>'
        '<span style="color:#ff7f0e">&#9644;</span> Classifier only (toggle)<br><span style="color:#d62728">&#9644;</span> Classifier + HMM<br>'
        f'Top-1: classifier {c["top1_accuracy"]:.3f} &rarr; HMM {h["top1_accuracy"]:.3f}<br>'
        f'Route F1 (length): {c["route_f1_len"]:.3f} &rarr; {h["route_f1_len"]:.3f}</div>'))
    m.save(os.path.join(out, f"route_map_test_{tid}.html")); print(tid, "saved", round(h["top1_accuracy"], 4))
