"""HMM + Viterbi route reconstruction.

Hidden state at GPS point i : one of its candidate arcs (candidate index k, arc id
                              and projection position along the arc are kept, so
                              the candidate -> road-segment correspondence is preserved).
Emission log P(z_i | arc)   : w_e * log(classifier probability) (selected model's output).
Transition log P(arc_j|arc_i): -|d_route - d_gps| / beta  (Newson & Krumm 2009), where
    d_route = network distance from the projection on arc_i to the projection on arc_j
              (same arc: signed offset, small backward moves tolerated; different arcs:
               rest of arc_i + shortest directed path + start of arc_j),
    d_gps   = straight-line distance between the two GPS fixes.
    Unreachable (no directed path within the search bound) -> log penalty `unreach`.
GPS gaps (dt > gap_s) or no reachable transition at all -> the chain is broken and
restarted (all transitions uniform), so decoding never fails on disconnected segments.
Emissions of (nearly) stationary points can be down-weighted (stop_speed > 0).
Points with no candidate get no prediction (counted as wrong) and also break the chain.
"""
import heapq
from collections import defaultdict
import numpy as np


class RoadGraph:
    def __init__(self, trip):
        A = trip.nodes_xy
        self.arcs = trip.arcs
        self.len = np.hypot(*(A[trip.arcs[:, 1]] - A[trip.arcs[:, 0]]).T)
        adj = defaultdict(list)
        for k, (u, v) in enumerate(trip.arcs):
            if u != v:
                adj[u].append((v, self.len[k]))
        self.adj = adj
        self._cache = {}

    def dists_from(self, src, cutoff):
        """Bounded Dijkstra; cached per (src) with the largest cutoff seen."""
        c = self._cache.get(src)
        if c is not None and c[0] >= cutoff:
            return c[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        adj = self.adj
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for v, w in adj.get(u, ()):
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[src] = (cutoff, dist)
        return dist


def viterbi(frame, trip, graph=None, beta=8.0, w_e=1.0, gap_s=60.0, unreach=-30.0,
            back_tol=10.0, vmax=60.0, score_col="p", signed=False,
            stop_speed=0.0, stop_w=0.1):
    """frame: candidate rows for ONE trip with columns pid, arc, proj_t, arc_len, <score_col>.
    Returns array (n_points,) of predicted arc ids (-1 = no candidate), and break count."""
    graph = graph or RoadGraph(trip)
    xy, t = trip.xy, trip.t
    n = len(xy)
    frame = frame.sort_values(["pid", "dist_rank"])
    pids = frame["pid"].values
    starts = np.searchsorted(pids, np.arange(n + 1))
    arcs_all = frame["arc"].values
    tp_all = np.clip(frame["proj_t"].values, 0, 1)
    L_all = frame["arc_len"].values
    em_all = w_e * np.log(np.clip(frame[score_col].values, 1e-6, 1.0))
    if stop_speed > 0:
        # Stationary down-weighting: while the vehicle is (nearly) stopped, GPS jitter makes
        # every observation weakly informative but hundreds of them would accumulate and
        # override the route context. Weight emissions by clip(speed / stop_speed, stop_w, 1).
        from .features import track_kinematics
        _, spd, _, _, _, _ = track_kinematics(xy, t)
        em_all = em_all * np.clip(spd[pids] / stop_speed, stop_w, 1.0)
    A = trip.arcs

    pred = np.full(n, -1, dtype=np.int64)
    breaks = 0
    seg_scores, seg_back, seg_pts = [], [], []   # current chain

    def flush():
        if not seg_scores:
            return
        k = int(np.argmax(seg_scores[-1]))
        for j in range(len(seg_pts) - 1, -1, -1):
            s, e = starts[seg_pts[j]], starts[seg_pts[j] + 1]
            pred[seg_pts[j]] = arcs_all[s + k]
            if j > 0:
                k = seg_back[j][k]
        seg_scores.clear(); seg_back.clear(); seg_pts.clear()

    prev = None
    for i in range(n):
        s, e = starts[i], starts[i + 1]
        if e == s:                       # no candidate: break chain
            flush(); prev = None; breaks += 1
            continue
        em = em_all[s:e]
        if prev is None or t[i] - t[prev] > gap_s:
            if prev is not None:
                breaks += 1
            flush()
            seg_scores.append(em.copy()); seg_back.append(None); seg_pts.append(i)
            prev = i
            continue
        ps, pe = starts[prev], starts[prev + 1]
        a_prev, t_prev, L_prev = arcs_all[ps:pe], tp_all[ps:pe], L_all[ps:pe]
        a_cur, t_cur, L_cur = arcs_all[s:e], tp_all[s:e], L_all[s:e]
        d_gps = float(np.hypot(*(xy[i] - xy[prev])))
        dt = max(t[i] - t[prev], 1.0)
        cutoff = max(d_gps * 2.0 + 50.0, min(vmax * dt, 2000.0)) if dt > 1 else d_gps * 2.0 + 50.0
        trans = np.full((len(a_prev), len(a_cur)), unreach)
        for u in range(len(a_prev)):
            ap = a_prev[u]
            rest = (1 - t_prev[u]) * L_prev[u]
            dist = graph.dists_from(A[ap, 1], cutoff)
            for v in range(len(a_cur)):
                ac = a_cur[v]
                if ac == ap:
                    off = (t_cur[v] - t_prev[u]) * L_cur[v]
                    if off < -back_tol:
                        continue
                    # signed along-arc displacement: moving backwards along a directed
                    # arc costs |off - d_gps| = |off| + d_gps, so the decoder cannot
                    # "drive" the reverse carriageway for free; GPS jitter at stops
                    # (|off|, d_gps both small) stays cheap.
                    dr = off if signed else max(off, 0.0)
                else:
                    sp = dist.get(A[ac, 0])
                    if sp is None:
                        continue
                    dr = rest + sp + t_cur[v] * L_cur[v]
                trans[u, v] = -abs(dr - d_gps) / beta
        tot = seg_scores[-1][:, None] + trans
        if trans.max() <= unreach:      # all transitions impossible -> restart chain
            breaks += 1
            flush()
            seg_scores.append(em.copy()); seg_back.append(None); seg_pts.append(i)
        else:
            bk = np.argmax(tot, axis=0)
            seg_scores.append(tot[bk, np.arange(len(a_cur))] + em)
            seg_back.append(bk); seg_pts.append(i)
        prev = i
    flush()
    return pred, breaks
