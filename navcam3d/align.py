"""Turning Depth Anything's unitless output into metres, and finding ground hazards. numpy/scipy only."""
import numpy as np
from scipy import ndimage

METHODS = ("plain", "weighted", "power")


def fit_align(method, r, z):
    """Fit depth z (metres) from the model's relative inverse depth r.

    plain     1/z = s*r + t       least squares on inverse depth
    weighted  1/z = s*r + t       each pixel's error counted as a % of its depth
    power     z = a * r^k         a straight line in log-log; copes with near ground and far hills in one photo
    """
    r, z = np.asarray(r, float), np.asarray(z, float)
    if method == "plain":
        s, t = np.linalg.lstsq(np.stack([r, np.ones_like(r)], 1), 1 / z, rcond=None)[0]
        return lambda R_: 1 / np.maximum(s * R_ + t, 1e-3)
    if method == "weighted":
        s, t = np.linalg.lstsq(np.stack([r * z, z], 1), np.ones_like(r), rcond=None)[0]
        return lambda R_: 1 / np.maximum(s * R_ + t, 1e-3)
    if method == "power":
        k, c = np.polyfit(np.log(np.maximum(r, 1e-3)), np.log(z), 1)
        return lambda R_: np.exp(c) * np.maximum(R_, 1e-3) ** k
    raise ValueError(method)


def choose_align(r, z, rng=None):
    """Try every method on 80% of the pixels; keep the one with the lowest % error on the other 20%."""
    rng = rng or np.random.default_rng(0)
    r, z = np.asarray(r, float), np.asarray(z, float)
    cut = rng.random(len(r)) < 0.8
    score = lambda m: np.mean(np.abs(fit_align(m, r[cut], z[cut])(r[~cut]) - z[~cut]) / z[~cut])
    best = min(METHODS, key=score)
    return best, fit_align(best, r, z)


def abs_rel(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.mean(np.abs(pred - truth) / truth))


def find_hazards(height, valid, cell, min_height=0.25, max_items=600):
    """Rocks and ledges: connected patches standing at least `min_height` metres above the ground around them.

    height: 2D grid (metres), valid: where height is real, cell: metres per grid cell.
    Returns a list of (row, col, height_above_ground_m, area_m2), tallest first.
    """
    h = np.where(valid, height, 0.0)
    w = ndimage.gaussian_filter(valid.astype(float), 5)                    # ground level from ~1.5 m around
    ground = ndimage.gaussian_filter(h, 5) / np.maximum(w, 1e-6)
    relief = np.where(valid, height - ground, 0.0)
    mask = relief > min_height
    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    peak = ndimage.maximum(relief, labels, idx)
    size = ndimage.sum(mask, labels, idx)
    cy, cx = np.array(ndimage.center_of_mass(mask, labels, idx)).T
    keep = size >= 2                                                       # at least two cells, not a single speck
    items = sorted(zip(cy[keep], cx[keep], peak[keep], size[keep] * cell * cell), key=lambda t: -t[2])
    return [(float(r), float(c), round(float(p), 2), round(float(a), 3)) for r, c, p, a in items[:max_items]]
