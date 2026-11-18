import numpy as np


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance. Returns (dist, t_unclamped, seg_len)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    L2s = np.where(L2 > 0, L2, 1.0)
    t = ((px - ax) * dx + (py - ay) * dy) / L2s
    t = np.where(L2 > 0, t, 0.0)
    tc = np.clip(t, 0.0, 1.0)
    qx, qy = ax + tc * dx, ay + tc * dy
    return np.hypot(px - qx, py - qy), t, np.sqrt(L2)


def bearing(dx, dy):
    """Compass-free planar angle in radians (atan2)."""
    return np.arctan2(dy, dx)


def angdiff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)
