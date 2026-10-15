"""Coarse head-tail neighborhoods, not vessel segmentation or physical factors."""

import numpy as np
from PIL import Image


def box_mask(shape, box):
    mask = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def region_boxes(row, shape, padding):
    h, w = shape
    values = np.array([row.head_x, row.head_y, row.tail_x, row.tail_y,
                       row.center_x, row.center_y], float)
    hx, hy, tx, ty, cx, cy = values
    points = np.array([[hx-cx+w//2, hy-cy+h//2], [tx-cx+w//2, ty-cy+h//2]])
    length = float(np.hypot(hx-tx, hy-ty))
    # The one nonsquare native chip is scene-edge clipped; centered mapping is unverified there.
    valid = bool(h == w and np.isfinite(values).all() and length > 0 and
                 (points >= 0).all() and (points[:, 0] < w).all() and (points[:, 1] < h).all())
    if not valid:
        return (0, 0, 0, 0), (0, 0, 0, 0), False, False, length, points
    pad = max(3., padding*length)
    low = np.floor(points.min(0)-pad).astype(int)
    high = np.ceil(points.max(0)+pad).astype(int)+1
    low = np.maximum(low, 0)
    high = np.minimum(high, [w, h])
    box = (int(low[0]), int(low[1]), int(high[0]), int(high[1]))
    bw, bh = high-low
    candidates = [(0, 0, int(bw), int(bh)), (w-int(bw), 0, w, int(bh)),
                  (0, h-int(bh), int(bw), h), (w-int(bw), h-int(bh), w, h)]
    center = (low+high)/2
    candidates.sort(key=lambda b: -np.sum((np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])-center)**2))
    mask = box_mask(shape, box)
    for candidate in candidates:
        if not np.any(mask & box_mask(shape, candidate)):
            return box, candidate, True, True, length, points
    return box, (0, 0, 0, 0), True, False, length, points


def apply(pair, mask, kind, seed):
    result = pair.copy()
    if kind == "original" or not mask.any():
        return result
    if kind == "flat":
        for k, a in enumerate(pair):
            border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
            result[k, mask] = int(np.rint(np.median(border)))
    elif kind == "shuffle":
        indices = np.flatnonzero(mask)
        shuffled = np.random.default_rng(seed).permutation(indices)
        result.reshape(2, -1)[:, indices] = pair.reshape(2, -1)[:, shuffled]
    elif kind == "display":
        anchor = np.concatenate([np.asarray(Image.fromarray(a).resize((128, 128), Image.Resampling.BILINEAR)).ravel()
                                 for a in pair])
        low, high = np.quantile(anchor, [.01, .99])
        if high > low:
            result[:, mask] = np.rint(np.clip((pair[:, mask].astype(float)-low)*255./(high-low), 0, 255)).astype(np.uint8)
    else:
        raise ValueError(kind)
    return result


def read_pair(row):
    with Image.open(row.path_vv) as vv, Image.open(row.path_vh) as vh:
        pair = np.stack([np.asarray(vv.convert("L")), np.asarray(vh.convert("L"))])
        np.testing.assert_array_equal(np.asarray(vh.convert("RGB")), np.repeat(pair[1, :, :, None], 3, axis=2))
    return pair


def transform(pair, row, condition, index):
    if condition == "replay_original":
        condition = "original"
    wide = condition.startswith("wide_")
    box, sham, valid, sham_valid, length, points = region_boxes(row, pair.shape[1:], .35 if wide else .20)
    region = box_mask(pair.shape[1:], box)
    parts = condition.removeprefix("wide_").split("_")
    if condition == "original" or not valid:
        mask, kind = np.zeros_like(region), "original"
    else:
        zone, kind = parts
        mask = region if zone == "target" else ~region if zone == "outer" else box_mask(pair.shape[1:], sham)
    result = apply(pair, mask, kind, 20261013 + int(index))
    np.testing.assert_array_equal(result[:, ~mask], pair[:, ~mask])
    info = dict(region_valid=valid, sham_valid=sham_valid, region_area=float(region.mean()),
                edited_area=float(mask.mean()), changed_fraction=float(np.mean(result != pair)),
                change_rms=float(np.sqrt(np.mean((result.astype(float)-pair)**2))),
                headtail_length=length, x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                hx=points[0, 0], hy=points[0, 1], tx=points[1, 0], ty=points[1, 1])
    return result, info
