"""Native-pixel OBB regions and exact layer1 cell integration.

M2 region order is body, end, side (not the M3 slot order).
"""
import numpy as np

SIZE = 160
REGIONS = ("body", "end", "side")
EDGES = np.r_[-.5, np.arange(2., 155., 4.), 159.5]
YY, XX = np.mgrid[:SIZE, :SIZE]


def soft_regions(g, template=None, shape=(SIZE, SIZE), offset=(0, 0)):
    if g is None:
        return np.zeros((3, *shape), np.float64)
    cx, cy, length, width, angle = g[:5]
    if template is not None:
        length, width = template
    if shape == (SIZE, SIZE) and offset == (0, 0):
        yy, xx = YY, XX
    else:
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        xx, yy = xx + offset[0], yy + offset[1]
    u = (xx-cx)*np.cos(angle) + (yy-cy)*np.sin(angle)
    v = -(xx-cx)*np.sin(angle) + (yy-cy)*np.cos(angle)

    def rectangle(l, w):
        return np.clip(l/2+.5-np.abs(u), 0, 1) * np.clip(w/2+.5-np.abs(v), 0, 1)

    body = rectangle(length, width)
    return np.stack((body, np.maximum(rectangle(1.5*length, width)-body, 0),
                     np.maximum(rectangle(length, 2*width)-body, 0)))


def planned_mass(g, template=None):
    if g is None:
        return np.zeros(3)
    length, width = g[2:4] if template is None else template
    radius = int(np.ceil(np.hypot(1.5*length, 2*width)/2+2))
    start = (int(np.floor(g[0]))-radius, int(np.floor(g[1]))-radius)
    return soft_regions(g, template, (2*radius+1, 2*radius+1), start).sum((-2, -1))


def cell_mass(mask):
    """Integrate constant pixel cells over stride4/origin0 Voronoi cells."""
    points = EDGES + .5
    idx = np.minimum(np.floor(points).astype(int), SIZE-1)
    frac = points - idx

    def integrate_last(a):
        prefix = np.concatenate((np.zeros((*a.shape[:-1], 1)), np.cumsum(a, axis=-1)), -1)
        values = prefix[..., idx] + a[..., idx]*frac
        return np.diff(values, axis=-1)

    out = integrate_last(mask)
    return np.swapaxes(integrate_last(np.swapaxes(out, -1, -2)), -1, -2)


def region_quality(mask, footprint, g, template=None):
    pixel = mask * footprint
    mass = pixel.sum((-2, -1))
    plan = planned_mass(g, template)
    weights = np.stack([cell_mass(np.rot90(pixel, k, (-2, -1))) for k in range(4)])
    neff = mass[None]**2 / np.maximum((weights**2).sum((-2, -1)), 1e-300)
    coverage = np.divide(mass, plan, out=np.zeros_like(mass), where=plan > 0)
    return dict(area=mass, planned=plan, coverage=coverage,
                neff=neff.mean(0), min_neff=neff.min(0),
                nodes=(weights > 0).sum((-2, -1)).min(0))


def rotate_geometry(g, k):
    out = g.copy()
    for _ in range(k % 4):
        out[0], out[1] = out[1], SIZE-1-out[0]
        out[4] -= np.pi/2
    return out
