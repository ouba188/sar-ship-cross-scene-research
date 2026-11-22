"""Native crop/pad input. Trusted background is explicit, never inferred from OBB complement."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from PIL import Image
import torch
from shapely.geometry import Polygon, box
from .geometry import GeometryConfig, build_geometry, coverage, mask_polygon, rectangle


@dataclass
class InputConfig:
    buckets: tuple = (64, 96, 128)
    protect_margin: float = 8.
    mean: tuple = (0., 0.)
    std: tuple = (1., 1.)

    def __post_init__(self):
        if tuple(self.buckets) != (64, 96, 128) or self.protect_margin < 0:
            raise ValueError('Use native 64/96/128 buckets and nonnegative protection')
        if len(self.mean) != 2 or len(self.std) != 2 or min(self.std) <= 0:
            raise ValueError('Two fixed source normalization values required')


@dataclass
class PreparedShip:
    images: torch.Tensor        # [2,1,S,S], fixed normalized intensity
    valid: torch.Tensor         # [2,1,S,S]
    background: torch.Tensor    # [2,1,S,S], explicitly trusted Q
    geometry: object
    placement: dict
    pair_eligible: bool

    def to(self, device):
        return PreparedShip(self.images.to(device), self.valid.to(device),
                            self.background.to(device), self.geometry.to(device),
                            self.placement, self.pair_eligible)


def _image(path, directory, binary=False):
    path = Path(path)
    with Image.open(path if path.is_absolute() else directory/path) as im:
        if im.mode not in ('L', 'RGB', 'RGBA', '1'):
            raise ValueError('Explicit radiometric preprocessing is required for non-8bit chips')
        arr = np.array(im.convert('L'))
    return arr > 0 if binary else arr


def _axis(spec):
    if spec.get('ship_axis'):
        a = spec['ship_axis']
        return np.array([*a['center'], a['length_px'], a['width_px'], a['angle']], dtype=float)
    det = spec['detections'][spec['target_index']]
    points = np.asarray(det['corners'], float)
    edges = np.roll(points, -1, axis=0)-points
    lengths = np.linalg.norm(edges, axis=1)
    k = lengths.argmax()
    return np.array([*points.mean(0), lengths[k], lengths[(k+1) % 4],
                     np.arctan2(edges[k, 1], edges[k, 0])])


def _shift(native_size, low, high, side):
    if native_size <= side:
        shift = (side-native_size)//2
        return shift if low+shift >= -.5-1e-7 and high+shift <= side-.5+1e-7 else None
    lower = math.ceil(max(0., high-(side-.5))-1e-7)
    upper = math.floor(min(native_size-side, low+.5)+1e-7)
    return -min(max((native_size-side)//2, lower), upper) if lower <= upper else None


def _move(arr, side, tx, ty):
    h, w = arr.shape
    result = np.zeros((side, side), dtype=arr.dtype)
    sx, sy = max(0, -tx), max(0, -ty)
    ex, ey = min(w, side-tx), min(h, side-ty)
    result[sy+ty:ey+ty, sx+tx:ex+tx] = arr[sy:ey, sx:ex]
    return result


def prepare_record(record, directory=Path('.'), input_config=None, geometry_config=None):
    cfg = input_config or InputConfig()
    directory = Path(directory)
    views = [record.get(p) for p in ('vv', 'vh')]
    views = [v if v and v.get('alignment_valid', False) else None for v in views]
    if not any(views):
        raise ValueError('At least one explicitly alignment_valid polarization is required')
    arrays = [_image(v['image'], directory) if v else None for v in views]
    shapes = {x.shape for x in arrays if x is not None}
    if len(shapes) != 1:
        raise ValueError('Dual polarization native coordinate shapes differ; register before use')
    h, w = next(iter(shapes))
    axes = [_axis(v) if v else None for v in views]
    axis = next(a for a in axes if a is not None)
    target = rectangle(axis)
    if all(v is not None for v in views) and not record.get('polarizations_coregistered', False):
        other = rectangle(axes[1])
        if target.intersection(other).area/max(target.union(other).area, 1e-12) < .999:
            raise ValueError('Different VV/VH target geometry needs explicit common-coordinate verification')
    native_valid = [(_image(v['valid_mask'], directory, True) if v.get('valid_mask')
                     else np.ones((h, w), bool)) if v else np.zeros((h, w), bool) for v in views]
    for m in native_valid:
        if m.shape != (h, w):
            raise ValueError('valid mask shape differs from native chip')
    union = mask_polygon(np.any(native_valid, axis=0))
    if not union.buffer(1e-7).covers(target):
        raise ValueError('Original target truncated or nodata; not a natural one-atom ship')
    protection = target.buffer(cfg.protect_margin)
    # Every available polarization needs a complete protected target for paired intervention.
    protected_ok = all(mask_polygon(v).buffer(1e-7).covers(protection)
                       for v, spec in zip(native_valid, views) if spec is not None)
    choice = None
    for bounds, eligible in ((protection.bounds, True), (target.bounds, False)):
        if eligible and not protected_ok:
            continue
        for side in cfg.buckets:
            tx, ty = _shift(w, bounds[0], bounds[2], side), _shift(h, bounds[1], bounds[3], side)
            if tx is not None and ty is not None:
                choice = (side, tx, ty, eligible); break
        if choice:
            break
    if choice is None:
        raise ValueError('Complete target does not fit maximum 128 canvas')
    side, tx, ty, eligible = choice
    pixels, valids, backgrounds = [], [], []
    for p, (v, raw, valid) in enumerate(zip(views, arrays, native_valid)):
        pixel = np.zeros((side, side), np.float32)
        moved_valid = _move(valid, side, tx, ty)
        q = np.zeros((h, w), bool)
        if v:
            pixel = (_move(raw.astype(np.float32)/255., side, tx, ty)-cfg.mean[p])/cfg.std[p]
            pixel *= moved_valid
            if v.get('background_mask') and v.get('background_validated', False):
                q = _image(v['background_mask'], directory, True)
                if q.shape != (h, w):
                    raise ValueError('Trusted Q mask shape differs from native chip')
                # A whole pixel must be outside target protection; partial contact is excluded.
                outside = coverage(protection, h, 1) if h == w else None
                if outside is None:
                    for yy, xx in zip(*np.nonzero(q)):
                        if protection.intersects(box(xx-.5, yy-.5, xx+.5, yy+.5)):
                            q[yy, xx] = False
                else:
                    q &= outside == 0
                q &= valid
        pixels.append(pixel); valids.append(moved_valid); backgrounds.append(_move(q, side, tx, ty))
    shifted_axis = axis.copy(); shifted_axis[:2] += (tx, ty)
    axis_reliable = all(v.get('axis_reliable', True) for v in views if v)
    geom = build_geometry(shifted_axis, np.stack(valids), geometry_config, shape_usable=axis_reliable)
    return PreparedShip(torch.from_numpy(np.stack(pixels)[:, None]),
                        torch.tensor(np.stack(valids)[:, None], dtype=torch.float32),
                        torch.tensor(np.stack(backgrounds)[:, None], dtype=torch.float32), geom,
                        dict(canvas_size=side, shift_x=tx, shift_y=ty, native_hw=(h, w),
                             input_profile='native_crop_pad_bucket_64_96_128_v1'), eligible)


def collate(samples, device='cpu'):
    if len({s.geometry.canvas_size for s in samples}) != 1:
        raise ValueError('Batch only same-canvas ships; do not pad 64 back to 128')
    samples = [s.to(device) for s in samples]
    return dict(images=torch.stack([s.images for s in samples]),
                valid=torch.stack([s.valid for s in samples]),
                background=torch.stack([s.background for s in samples]),
                geometries=[s.geometry for s in samples],
                pair_eligible=torch.tensor([s.pair_eligible for s in samples], device=device))


def background_targets(images, background):
    targets = images.new_zeros((len(images), 2, 2))
    reliable = torch.zeros((len(images), 2), dtype=torch.bool, device=images.device)
    for i in range(len(images)):
        for p in range(2):
            values = images[i, p][background[i, p] > 0]
            if values.numel():
                q = torch.quantile(values, values.new_tensor([.1, .5, .9]))
                targets[i, p] = torch.stack((q[1], q[2]-q[0])); reliable[i, p] = True
    return targets, reliable


def intervene_background(batch, generator=None):
    """Paired-polar source donor, chosen without labels; only validated Q changes.

    Returns an eligible-sample mask. This constructs L_inv views, not proof of
    metadata/class deconfounding; L_leak additionally needs an explicit audit.
    """
    images, q = batch['images'], batch['background'].bool()
    altered = images.clone()
    eligible = torch.zeros(len(images), dtype=torch.bool, device=images.device)
    availability = batch['valid'].flatten(2).sum(-1) > 0
    for i in range(len(images)):
        if not batch['pair_eligible'][i]:
            continue
        active = torch.where(availability[i])[0].tolist()
        if not active or not all(q[i, p].any() for p in active):
            continue
        candidates = [j for j in range(len(images)) if j != i
                      and torch.equal(availability[i], availability[j])
                      and q[j, active].all(0).any()]
        if not candidates:
            continue
        pick = int(torch.randint(len(candidates), (1,), generator=generator))
        j = candidates[pick]
        donor_mask = q[j, active].all(0).flatten()
        donor_ids = torch.where(donor_mask)[0]
        if not len(donor_ids):
            continue
        # One paired donor index field is shared by VV and VH.
        indices = torch.randint(len(donor_ids), (images.shape[-1]**2,), generator=generator).to(images.device)
        for p in active:
            sampled = images[j, p].flatten()[donor_ids[indices]].reshape_as(images[j, p])
            altered[i, p] = torch.where(q[i, p], sampled, images[i, p])
        eligible[i] = True
    return altered, eligible
