"""Translate/crop native pixels without resampling the image or region masks."""
import numpy as np
import torch


def crop_pad(image, masks, size=96):
    h, w = image.shape
    yy, xx = np.nonzero(masks[0])
    bounds = (int(xx.min()), int(yy.min()), int(xx.max()) + 1,
              int(yy.max()) + 1) if len(xx) else None
    if bounds and (bounds[2] - bounds[0] > size or bounds[3] - bounds[1] > size):
        raise ValueError(f'Target extent {bounds} exceeds {size}px; use a larger canvas, not resizing')

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


def place_native(image, masks, size=160):
    out, region, _ = crop_pad(image, masks, size)
    return torch.from_numpy(out.copy())[None].float()/255, torch.from_numpy(region)
