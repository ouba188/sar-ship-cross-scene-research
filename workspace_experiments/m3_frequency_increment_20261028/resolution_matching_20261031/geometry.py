"""Native-160 geometry and linear multiscale readout for resolution matching.

The caller puts sar_m123_current_20261031 on sys.path. No model is loaded here.
"""
from functools import lru_cache

import numpy as np
import torch

from sar_m123 import tokens


_GRIDS = {'coarse_fixed': 20, 'fine_fixed': 40, 'fine_adaptive': 40}


def _grid(mode):
    if mode not in _GRIDS:
        raise ValueError(f'Unknown resolution mode: {mode!r}')
    return _GRIDS[mode]


def _edges(count):
    stride = 160 / count
    return np.r_[-.5, (np.arange(count - 1) + .5) * stride, 159.5]


@lru_cache(maxsize=3)
def _cell_area(count):
    widths = np.diff(_edges(count))
    return widths[:, None] * widths[None, :]


def _adaptive_slots(geometry, footprint, stride):
    cx, cy, length, width, angle = geometry[:5]
    u = (tokens.XX - cx) * np.cos(angle) + (tokens.YY - cy) * np.sin(angle)
    v = -(tokens.XX - cx) * np.sin(angle) + (tokens.YY - cy) * np.cos(angle)
    n_long = int(np.clip(np.floor(length / stride), 1, 4))
    n_wide = int(np.clip(np.floor(width / stride), 1, 2))
    along = np.clip(np.floor((u / length + .5) * n_long).astype(int), 0, n_long - 1)
    across = (v >= 0).astype(int) if n_wide == 2 else np.zeros_like(along)
    side, end = (v >= 0).astype(int), (u >= 0).astype(int)
    regions = tokens.soft_regions(geometry) * footprint
    masks = np.zeros((20, 160, 160), dtype=np.float64)
    for a in range(n_long):
        for b in range(n_wide):
            masks[2 * a + b] = regions[0] * (along == a) * (across == b)
        for b in range(2):
            # Each physical side is W/2 wide and has only one transverse slot.
            masks[8 + 2 * a + b] = regions[2] * (along == a) * (side == b)
    for a in range(2):
        for b in range(n_wide):
            # Each end is L/4 long: never merge the two physical ends together.
            masks[16 + 2 * a + b] = regions[1] * (end == a) * (across == b)
    return masks


def build_masks(geometry, footprint, mode):
    """Return float32 read[20,g,g], write[20,10,10], mass[20].

    Geometry is (cx, cy, L, W, angle_radians, ...) on the native 160 canvas;
    footprint is its 160x160 pixel occupancy. Missing/empty geometry yields
    zeros. Region indices remain tokens.REGIONS (8 body, 8 side, 4 end).

    Fixed modes reuse tokens.slots. Adaptive uses nL=clip(floor(L/4),1,4)
    and nW=clip(floor(W/4),1,2), with uniform physical bins. Active body
    indices are 2*a+b, sides 8+2*a+side, ends 16+2*end+b. Unused indices
    are zero. A side has no additional width subdivision; each end keeps
    its identity even when L/4 < stride. Pixel-center boundary assignment
    follows tokens.slots, so axis reversal may move boundary pixels.

    Read is cell_mass / true cell area, NOT the pixel integral. Its sum is
    the normalization mass. Write uses stride16 occupancy, then divides
    each cell by max(1, sum of all slot occupancies). Regional native area
    is preserved before write overlap normalization.
    """
    count = _grid(mode)
    if geometry is None or geometry[2] <= 0 or geometry[3] <= 0:
        return dict(read=np.zeros((20, count, count), np.float32),
                    write=np.zeros((20, 10, 10), np.float32),
                    mass=np.zeros(20, np.float32))
    footprint = np.asarray(footprint)
    if footprint.shape != (160, 160):
        raise ValueError('footprint must have shape (160, 160)')
    native = (_adaptive_slots(geometry, footprint, 160 // count)
              if mode == 'fine_adaptive' else tokens.slots(geometry, footprint))
    read = (tokens.cell_mass(native, 160 // count) / _cell_area(count)).astype(np.float32)
    write = (tokens.cell_mass(native, 16) / _cell_area(10)).astype(np.float32)
    write /= np.maximum(write.sum(axis=0, keepdims=True), 1)
    return dict(read=read, write=write, mass=read.sum(axis=(-2, -1)))


@lru_cache(maxsize=1)
def _downsample_weights():
    fine, coarse = _edges(40), _edges(20)
    overlap = np.maximum(0, np.minimum(coarse[1:, None], fine[None, 1:])
                         - np.maximum(coarse[:-1, None], fine[None, :-1]))
    return overlap / np.diff(coarse)[:, None]


def _upsample_origin_zero(features):
    # Target center 4*j is source index j/2; extend the final source center.
    positions = torch.arange(40, device=features.device)
    left = positions // 2
    right = (left + 1).clamp_max(19)
    fraction = (positions % 2).to(features.dtype) * .5
    horizontal = (features.index_select(-1, left) * (1 - fraction)
                  + features.index_select(-1, right) * fraction)
    fraction = fraction[:, None]
    return (horizontal.index_select(-2, left) * (1 - fraction)
            + horizontal.index_select(-2, right) * fraction)


def pool_multiscale(f1, f2, read, mode):
    """Linear weighted means, [B,256,40,40]+[B,512,20,20] -> [B,20,768].

    Fine modes interpolate f2 at physical centers 0,4,...,156 from centers
    0,8,...,152 (bilinear, border extension at 156). Coarse mode treats f1
    as constant over its center-Voronoi cells and integrates their physical
    overlap with the 20-grid cells, including clipped canvas edges. This
    is area resampling, not ordinary 2x2 pooling or point-center sampling.

    No projection, nonlinear activation or feature normalization is applied.
    Any channel counts are accepted, so bias-free pointwise Linear commutes
    with this function. With bias, mask empty slots after token projection.
    Empty reads return exact zeros; normalization uses read.sum, never native
    pixel mass. Gradients flow to both features (and floating-point reads).
    """
    count = _grid(mode)
    if (f1.ndim != 4 or f2.ndim != 4 or f1.shape[0] != f2.shape[0]
            or f1.shape[-2:] != (40, 40) or f2.shape[-2:] != (20, 20)):
        raise ValueError('Expected f1[B,C1,40,40] and f2[B,C2,20,20]')
    if read.shape != (f1.shape[0], 20, count, count):
        raise ValueError(f'Expected read[B,20,{count},{count}]')
    if count == 20:
        weights = torch.as_tensor(_downsample_weights(), device=f1.device, dtype=f1.dtype)
        f1 = torch.matmul(torch.matmul(weights, f1), weights.T)
    else:
        f2 = _upsample_origin_zero(f2)
    weights = read.to(f1).flatten(2)
    mass = weights.sum(-1, keepdim=True)
    weights = weights / torch.where(mass > 0, mass, torch.ones_like(mass))
    first = torch.bmm(weights, f1.flatten(2).transpose(1, 2))
    second = torch.bmm(weights.to(f2), f2.flatten(2).transpose(1, 2))
    return torch.cat((first, second), dim=-1)
