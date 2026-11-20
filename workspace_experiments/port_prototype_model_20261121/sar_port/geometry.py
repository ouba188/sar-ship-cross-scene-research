"""Native-pixel chip preparation; coordinates are image x-right, y-down."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def crop_pad(image, masks, size=160):
    h, w = image.shape
    yy, xx = np.nonzero(masks[0])
    bounds = (int(xx.min()), int(yy.min()), int(xx.max()) + 1,
              int(yy.max()) + 1) if len(xx) else None
    if bounds and (bounds[2] - bounds[0] > size or bounds[3] - bounds[1] > size):
        raise ValueError(f'Target extent {bounds} exceeds the trained {size}px canvas; no resizing is allowed.')

    def start(length, lo, hi):
        if length <= size:
            return -(size - length) // 2
        center = (lo + hi) / 2 if lo is not None else length / 2
        return max(0, min(length - size, int(np.floor(center - size / 2))))

    x0 = start(w, bounds[0] if bounds else None, bounds[2] if bounds else None)
    y0 = start(h, bounds[1] if bounds else None, bounds[3] if bounds else None)
    sx, sy = max(0, x0), max(0, y0)
    ex, ey = min(w, x0 + size), min(h, y0 + size)
    dx, dy = sx - x0, sy - y0
    out = np.zeros((size, size), dtype=image.dtype)
    region = np.zeros((3, size, size), dtype=np.float32)
    out[dy:dy+ey-sy, dx:dx+ex-sx] = image[sy:ey, sx:ex]
    region[:, dy:dy+ey-sy, dx:dx+ex-sx] = masks[:, sy:ey, sx:ex]
    info = dict(crop_x=sx, crop_y=sy, crop_width=ex-sx, crop_height=ey-sy,
                pad_left=dx, pad_top=dy, target_present=bool(bounds),
                native_target_pixels=float(masks[0].sum()),
                retained_target_pixels=float(region[0].sum()),
                native_rest_pixels=float(masks[2].sum()),
                retained_rest_pixels=float(region[2].sum()), input_scale=1.0)
    return out, region, info


def prepare_view(spec, directory=Path('.')):
    if spec is None or not spec.get('alignment_valid', True):
        return None
    path = Path(spec['image'])
    if not path.is_absolute():
        path = Path(directory) / path
    with Image.open(path) as im:
        if im.mode not in ('L', 'RGB', 'RGBA'):
            raise ValueError('Expected the trained 8-bit display chip; raw SAR radiometry needs explicit preprocessing.')
        raw = np.array(im.convert('L'))
    h, w = raw.shape
    dets = spec['detections']
    target = -1 if spec.get('target_index') is None else int(spec['target_index'])
    if target < -1 or target >= len(dets):
        raise ValueError('target_index must select one supplied OBB, or be null/-1 when unavailable.')
    masks = np.zeros((3, h, w), dtype=np.float32)
    geometry = None
    for i, det in enumerate(dets):
        corners = np.asarray(det['corners'], dtype=float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError('OBB corners must be four finite (x,y) points in chip coordinates.')
        if not det.get('shape_usable', True) or det['confidence'] < .25:
            continue
        canvas = Image.new('L', (w, h))
        ImageDraw.Draw(canvas).polygon([tuple(x) for x in corners], fill=1)
        masks[0 if i == target else 1] = np.maximum(
            masks[0 if i == target else 1], np.asarray(canvas))
        if i == target:
            edges = np.roll(corners, -1, axis=0) - corners
            lengths = np.linalg.norm(edges, axis=1)
            j = int(lengths.argmax())
            geometry = np.array([*corners.mean(0), lengths[j], lengths[(j+1) % 4],
                                 np.arctan2(edges[j, 1], edges[j, 0]) % np.pi, det['confidence']])
    masks[1] *= 1 - masks[0]
    masks[2] = 1 - np.maximum(masks[0], masks[1])
    axis = spec.get('ship_axis')
    if axis is None and geometry is not None:
        axis = dict(center=geometry[:2].tolist(), length_px=float(geometry[2]),
                    width_px=float(geometry[3]), angle=float(geometry[4]))
    pixels, masks, placement = crop_pad(raw, masks)
    if geometry is not None:
        geometry = geometry.copy()
        geometry[:2] += [placement['pad_left']-placement['crop_x'], placement['pad_top']-placement['crop_y']]
    radar = spec.get('radar_axis_rad')
    if radar is not None and not np.isfinite(radar):
        raise ValueError('radar_axis_rad must be finite or omitted.')
    return dict(pixels=pixels, footprint=masks.sum(0), geometry=geometry, raw=raw,
                placement=placement, masks=masks, ship_axis=axis, radar_axis=radar,
                valid_target=bool(masks[0].any()))


def prepare(record, directory=Path('.')):
    result = [prepare_view(record.get(pol), directory) for pol in ('vv', 'vh')]
    if not any(v is not None and v['valid_target'] for v in result):
        raise ValueError('At least one aligned polarization and target OBB are required.')
    return result

# Native geometric functions retained from sar_m123/tokens.py.
YY, XX = np.mgrid[:160, :160]

def soft_regions(g):
    """Original soft OBB regions, order body/end/side; not input T/O/R masks."""
    if g is None:
        return np.zeros((3, 160, 160), np.float64)
    cx, cy, length, width, angle = g[:5]
    u = (XX-cx)*np.cos(angle)+(YY-cy)*np.sin(angle)
    v = -(XX-cx)*np.sin(angle)+(YY-cy)*np.cos(angle)

    def rectangle(l, w):
        return np.clip(l/2+.5-np.abs(u), 0, 1)*np.clip(w/2+.5-np.abs(v), 0, 1)

    body = rectangle(length, width)
    return np.stack((body, np.maximum(rectangle(1.5*length, width)-body, 0),
                     np.maximum(rectangle(length, 2*width)-body, 0)))

def slots(g, footprint):
    if g is None:
        return np.zeros((20, 160, 160), np.float64)
    cx, cy, length, width, angle = g[:5]
    u = ((XX-cx)*np.cos(angle)+(YY-cy)*np.sin(angle))/length
    v = (-(XX-cx)*np.sin(angle)+(YY-cy)*np.cos(angle))/width
    region = soft_regions(g)*footprint
    ix = np.clip(np.floor((u+.5)*4).astype(int), 0, 3)
    iy, end = (v >= 0).astype(int), (u >= 0).astype(int)
    masks = [region[0]*(ix == a)*(iy == b) for a in range(4) for b in range(2)]
    masks += [region[2]*(ix == a)*(iy == b) for a in range(4) for b in range(2)]
    masks += [region[1]*(end == a)*(iy == b) for a in range(2) for b in range(2)]
    return np.stack(masks)

def cell_mass(mask, stride=4):
    """Pixel-area integral over true center-Voronoi cells (origin zero)."""
    edges = np.r_[-.5, np.arange(stride/2, 160-stride/2, stride), 159.5]
    points = edges+.5
    idx = np.minimum(np.floor(points).astype(int), 159)
    frac = points-idx

    def integral(a):
        prefix = np.concatenate([np.zeros((*a.shape[:-1], 1)), np.cumsum(a, axis=-1)], -1)
        return np.diff(prefix[..., idx]+a[..., idx]*frac, axis=-1)

    return integral(integral(mask).swapaxes(-1, -2)).swapaxes(-1, -2)
