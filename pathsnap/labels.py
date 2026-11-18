"""Derive a reference arc for every GPS observation.

The supplied .route only lists the arcs traversed, not which GPS point lies on
which arc. We align the track to the route with a monotone dynamic program:
point i is assigned route position j_i with j_0 <= j_1 <= ... (the vehicle
never goes back along the route list), minimising sum of point-to-arc
distance. Skipping route arcs is allowed (short arcs can be crossed within one
1 s step, or during a GPS gap). This is NOT used at inference time.
"""
import numpy as np
from .geometry import point_segment


def align_track_to_route(trip, chunk=4000):
    xy, A, N = trip.xy, trip.nodes_xy, len(trip.xy)
    r = trip.route
    a, b = A[trip.arcs[r, 0]], A[trip.arcs[r, 1]]
    R = len(r)
    back = np.empty((N, R), dtype=np.int32)
    prev = None
    dist_keep = np.empty((N, R), dtype=np.float32) if N * R < 4e7 else None
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        D, _, _ = point_segment(xy[s:e, 0, None], xy[s:e, 1, None], a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1])
        if dist_keep is not None:
            dist_keep[s:e] = D
        for k in range(e - s):
            i = s + k
            if prev is None:
                cur = D[k].astype(np.float64)
                back[i] = np.arange(R)
            else:
                # prefix minimum over j' <= j of prev cost
                pm = np.minimum.accumulate(prev)
                idx = np.arange(R)
                is_new = np.r_[True, prev[1:] < pm[:-1]]
                arg = np.maximum.accumulate(np.where(is_new, idx, 0))
                back[i] = arg
                cur = pm + D[k]
            prev = cur
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    pa, pb = a[j], b[j]
    d, _, _ = point_segment(xy[:, 0], xy[:, 1], pa[:, 0], pa[:, 1], pb[:, 0], pb[:, 1])
    return r[j], j, d
