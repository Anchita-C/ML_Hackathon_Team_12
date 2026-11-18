"""Candidate road-segment generation.

For each GPS point, candidates are the K nearest directed arcs within
`radius` metres (exact point-to-segment distance). Arcs are straight segments
between consecutive nodes, so we index points sampled every `step` metres along
each arc in a KD-tree, then compute exact distances for the hits.
"""
import numpy as np
from scipy.spatial import cKDTree
from .geometry import point_segment


class ArcIndex:
    def __init__(self, trip, margin=300.0, step=15.0):
        A = trip.nodes_xy
        a, b = A[trip.arcs[:, 0]], A[trip.arcs[:, 1]]
        lo, hi = trip.xy.min(0) - margin, trip.xy.max(0) + margin
        mid_lo = np.minimum(a, b); mid_hi = np.maximum(a, b)
        keep = np.where(np.all(mid_hi >= lo, 1) & np.all(mid_lo <= hi, 1) & (trip.arcs[:, 0] != trip.arcs[:, 1]))[0]
        self.trip, self.a, self.b = trip, a, b
        L = np.hypot(*(b[keep] - a[keep]).T)
        nseg = np.maximum(1, np.ceil(L / step).astype(int))
        arc_rep = np.repeat(keep, nseg + 1)
        frac = np.concatenate([np.linspace(0, 1, n + 1) for n in nseg])
        pts = a[arc_rep] + frac[:, None] * (b[arc_rep] - a[arc_rep])
        self.sample_arc = arc_rep
        self.tree = cKDTree(pts)
        self.step = step

    def query(self, xy, k=12, radius=150.0):
        """Return list of (arc_ids, dists) per point, sorted by distance."""
        out = []
        r = radius + self.step
        hits = self.tree.query_ball_point(xy, r)
        for i, h in enumerate(hits):
            if not h:
                out.append((np.empty(0, np.int64), np.empty(0)))
                continue
            arcs = np.unique(self.sample_arc[h])
            a, b = self.a[arcs], self.b[arcs]
            d, _, _ = point_segment(xy[i, 0], xy[i, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
            m = d <= radius
            arcs, d = arcs[m], d[m]
            o = np.argsort(d, kind="stable")[:k]
            out.append((arcs[o], d[o]))
        return out
