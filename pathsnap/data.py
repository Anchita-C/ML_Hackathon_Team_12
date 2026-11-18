"""Raw-file loading and per-trip metric projection.

File formats (verified in EDA, see scripts/01_eda.py):
  .track  lon lat t        (whitespace separated, no header)
  .nodes  lon lat          (row index = node id)
  .arcs   from to          (row index = arc id, directed)
  .route  arc_id           (chronological reference arcs)
"""
import os
from dataclasses import dataclass, field
import numpy as np
from pyproj import CRS, Transformer


@dataclass
class Trip:
    trip_id: str
    track_ll: np.ndarray            # (N,2) lon,lat
    t: np.ndarray                   # (N,)  seconds
    nodes_ll: np.ndarray            # (M,2) lon,lat
    arcs: np.ndarray                # (A,2) from,to
    route: np.ndarray | None = None # (R,) arc ids
    xy: np.ndarray = field(default=None, repr=False)        # projected track (m)
    nodes_xy: np.ndarray = field(default=None, repr=False)  # projected nodes (m)
    transformer: Transformer = field(default=None, repr=False)


def _read(path, dtype=float, ndmin=2):
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(trip_dir, with_route=True):
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = os.path.join(trip_dir, tid)
    tr = _read(p + ".track")
    nodes = _read(p + ".nodes")
    arcs = _read(p + ".arcs", dtype=np.int64)
    route = None
    if with_route and os.path.exists(p + ".route"):
        route = _read(p + ".route", dtype=np.int64, ndmin=1)
    trip = Trip(tid, tr[:, :2].copy(), tr[:, 2].copy(), nodes, arcs, route)
    project(trip)
    return trip


def project(trip):
    """Azimuthal-equidistant projection centred on the track.

    Trips are spread world-wide (lon -100..137), so one global CRS is wrong;
    a local AEQD gives metre units with negligible distortion at city scale.
    """
    lon0, lat0 = np.median(trip.track_ll, axis=0)
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +ellps=WGS84")
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    trip.transformer = tf
    trip.xy = np.column_stack(tf.transform(trip.track_ll[:, 0], trip.track_ll[:, 1]))
    trip.nodes_xy = np.column_stack(tf.transform(trip.nodes_ll[:, 0], trip.nodes_ll[:, 1]))
    return trip


def list_trips(root):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def canonical_arc_ids(arcs):
    """Map parallel duplicate arcs (same from,to) to one canonical id (smallest)."""
    key = arcs[:, 0].astype(np.int64) * (arcs.max() + 1) + arcs[:, 1]
    order = np.lexsort((np.arange(len(arcs)), key))
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    canon_sorted = order[np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))]
    canon = np.empty(len(arcs), dtype=np.int64)
    canon[order] = canon_sorted
    return canon
