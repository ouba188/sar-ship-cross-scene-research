"""Exact native-cell geometry and deterministic area-priority cuts.

Coordinates are pixel centres: pixel (0,0) occupies [-.5,.5]^2. Polygon
integration follows the source geometry audit; no feature values enter here.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import torch
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from .contracts import LevelGeometry, ShipGeometry


@dataclass(frozen=True)
class GeometryConfig:
    min_lengths: tuple = (6., 16., 32.)
    max_depth: int = 3
    leaf_budget: int = 4
    min_centers: int = 2
    min_coverage: float = .5
    min_axis_ratio: float = 1.5

    def __post_init__(self):
        if len(self.min_lengths) != 3 or min(self.min_lengths) <= 0:
            raise ValueError('Three positive native region lengths are required')
        if not 0 <= self.max_depth <= 3 or not 1 <= self.leaf_budget <= 4:
            raise ValueError('Supported geometry: depth 0..3 and leaf budget 1..4')


def rectangle(axis, lo=-.5, hi=.5):
    cx, cy, length, width, angle = np.asarray(axis, dtype=float)
    e = np.array([np.cos(angle), np.sin(angle)])
    if e[0] < -1e-10 or (abs(e[0]) <= 1e-10 and e[1] < 0):
        e = -e
    f = np.array([-e[1], e[0]])
    return Polygon([np.array([cx, cy]) + u * length * e + v * width * f
                    for u, v in ((lo, -.5), (hi, -.5), (hi, .5), (lo, .5))])


def mask_polygon(mask):
    """Union of original valid pixels, including internal nodata holes."""
    mask = np.asarray(mask, dtype=bool)
    pieces = []
    for y, row in enumerate(mask):
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            pieces.append(box(left-.5, y-.5, right-.5, y+.5))
    return unary_union(pieces) if pieces else Polygon()


@lru_cache(maxsize=12)
def grid(side, stride):
    offset = stride / 2 - .5
    coords = np.array([(offset+stride*x, offset+stride*y)
                       for y in range(side//stride) for x in range(side//stride)])
    cells = [box(x-stride/2, y-stride/2, x+stride/2, y+stride/2) for x, y in coords]
    return coords, cells


def coverage(poly, side, stride):
    coords, cells = grid(side, stride)
    if poly.is_empty:
        return np.zeros((side//stride, side//stride), np.float32)
    bounds = poly.bounds
    result = np.zeros(len(cells), np.float32)
    # Grid cell bounds are exactly center +/- half-stride. Filter numerically
    # before GEOS calls; the same candidate cells and exact intersections remain.
    half = stride / 2
    candidates = ((coords[:, 0] + half >= bounds[0]) &
                  (coords[:, 0] - half <= bounds[2]) &
                  (coords[:, 1] + half >= bounds[1]) &
                  (coords[:, 1] - half <= bounds[3]))
    for j in np.flatnonzero(candidates):
        result[j] = poly.intersection(cells[j]).area / stride**2
    return result.reshape(side//stride, side//stride)


def build_geometry(axis, valid, config=None, shape_usable=True):
    config = config or GeometryConfig()
    axis = np.asarray(axis, dtype=float)
    valid = np.asarray(valid, dtype=bool).reshape(2, *np.asarray(valid).shape[-2:])
    side = valid.shape[-1]
    if valid.shape[-2] != side or side not in (64, 96, 128):
        raise ValueError('Geometry requires a 64/96/128 native square canvas')
    if axis.shape != (5,) or not np.isfinite(axis).all() or min(axis[2:4]) <= 0:
        raise ValueError('axis must contain finite cx,cy,length,width,angle')
    root = rectangle(axis)
    valids = [mask_polygon(v) for v in valid]
    union = unary_union(valids)
    if root.intersection(union).area <= 0:
        raise ValueError('No valid target observation')
    if not union.buffer(1e-7).covers(root):
        raise ValueError('Truncated target must be handled outside normal classification')
    direction = np.array([np.cos(axis[4]), np.sin(axis[4])])
    if direction[0] < -1e-10 or (abs(direction[0]) <= 1e-10 and direction[1] < 0):
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    axis_ok = bool(shape_usable and axis[2]/axis[3] >= config.min_axis_ratio)
    nodes = {}

    def node(depth, index):
        key = (depth, index)
        if key in nodes:
            return nodes[key]
        lo, hi = -.5+index/2**depth, -.5+(index+1)/2**depth
        poly = rectangle(axis, lo, hi)
        result = dict(poly=poly, area=poly.intersection(union).area,
                      interval=((lo+hi)/2, hi-lo), weights=[], observable=[], fraction=[])
        for level, stride in enumerate((4, 8, 16)):
            coords, _ = grid(side, stride)
            axial = (coords-axis[:2]) @ direction
            lateral = (coords-axis[:2]) @ normal
            owns = (axial >= lo*axis[2]-1e-8) & (np.abs(lateral) <= axis[3]/2+1e-8)
            owns &= (axial <= hi*axis[2]+1e-8) if index == 2**depth-1 else (axial < hi*axis[2]-1e-8)
            weights, obs, fractions = [], [], []
            for v in valids:
                clipped = poly.intersection(v)
                w = coverage(clipped, side, stride)
                fraction = clipped.area/poly.area
                positions = axial[owns & (w.ravel() >= config.min_coverage-1e-8)]
                good = (axis[2]*(hi-lo) >= config.min_lengths[level]-1e-8
                        and fraction >= config.min_coverage-1e-8
                        and len(positions) >= config.min_centers
                        and np.ptp(positions) >= stride-1e-8)
                weights.append(w); obs.append(bool(good)); fractions.append(fraction)
            result['weights'].append(np.stack(weights))
            result['observable'].append(np.array(obs))
            result['fraction'].append(fractions)
        nodes[key] = result
        return result

    leaves, children = [], {}
    def descend(key):
        d, i = key
        node(d, i)
        if not axis_ok or d == config.max_depth:
            leaves.append(key); return
        pair = ((d+1, 2*i), (d+1, 2*i+1))
        a, b = (node(*k) for k in pair)
        if np.any(a['observable'][0] & b['observable'][0]):
            children[key] = pair
            for child in pair:
                descend(child)
        else:
            leaves.append(key)
    descend((0, 0))
    cut = [(0, 0)]
    while len(cut) < config.leaf_budget:
        candidates = [k for k in cut if k in children]
        if not candidates:
            break
        picked = min(candidates, key=lambda k: (-nodes[k]['area'], k))
        cut.remove(picked); cut.extend(children[picked])
    cut.sort(key=lambda k: nodes[k]['interval'][0])
    base, halves = node(0, 0), (node(1, 0), node(1, 1))
    levels = []
    for level, stride in enumerate((4, 8, 16)):
        slots = np.stack([base['weights'][level], halves[0]['weights'][level],
                          halves[1]['weights'][level]], axis=1)
        flags = slots.sum((-2, -1)) > 0
        paired = axis_ok & halves[0]['observable'][level] & halves[1]['observable'][level]
        flags[:, 1:] = paired[:, None]
        slots *= flags[..., None, None]
        region_weights = np.stack([nodes[k]['weights'][level] for k in cut])
        matches = np.stack([nodes[k]['weights'][level].sum((-2, -1)) > 0
                            if k == (0, 0) else nodes[k]['observable'][level] for k in cut])
        v = np.stack([coverage(x, side, stride) for x in valids])[:, None]
        union_slots = np.stack([coverage(rectangle(axis, *limits).intersection(union), side, stride)
                                for limits in ((-.5, .5), (-.5, 0), (0, .5))])
        union_slots *= flags.any(0)[:, None, None]
        union_regions = np.stack([coverage(rectangle(axis, nodes[k]['interval'][0] - nodes[k]['interval'][1]/2,
                                                      nodes[k]['interval'][0] + nodes[k]['interval'][1]/2).intersection(union), side, stride)
                                  for k in cut])
        levels.append(LevelGeometry(torch.from_numpy(v), torch.from_numpy(slots),
                                    torch.from_numpy(flags), torch.from_numpy(region_weights),
                                    torch.from_numpy(matches)))
        if level == 0:
            union_slots_f1, union_regions_f1 = union_slots, union_regions
    return ShipGeometry(tuple(levels), torch.tensor([nodes[k]['area'] for k in cut], dtype=torch.float32),
                        torch.tensor([nodes[k]['interval'] for k in cut], dtype=torch.float32),
                        torch.tensor(np.array([nodes[k]['fraction'] for k in cut]).transpose(0, 2, 1), dtype=torch.float32),
                        bool(axis_ok and base['observable'][0].any()), len(leaves), side,
                        torch.from_numpy(union_slots_f1), torch.from_numpy(union_regions_f1))
