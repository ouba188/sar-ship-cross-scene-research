"""Label-free native-chip interventions; original images are never overwritten."""

import numpy as np
from PIL import Image


def crop_box(size):
    w, h = size
    nw, nh = max(1, int(np.floor(.75*w))), max(1, int(np.floor(.75*h)))
    left, top = (w-nw)//2, (h-nh)//2
    return left, top, left+nw, top+nh


def transform_pair(vv, vh, condition):
    if condition not in ("original", "display", "crop75", "crop75_display"):
        raise ValueError(condition)
    images = [vv.convert("L"), vh.convert("L")]
    rgb = vh.convert("RGB")
    details = dict(condition=condition, vv_width=vv.width, vv_height=vv.height,
        vh_width=vh.width, vh_height=vh.height, stretch_flat=False, q01=np.nan, q99=np.nan,
        clipped_low_fraction=0., clipped_high_fraction=0.)
    if condition.startswith("crop75"):
        images = [im.crop(crop_box(im.size)) for im in images]
        rgb = rgb.crop(crop_box(rgb.size))
    if condition.endswith("display"):
        anchor = np.concatenate([np.asarray(im.resize((128, 128), Image.Resampling.BILINEAR)).ravel() for im in images])
        low, high = np.quantile(anchor, [.01, .99])
        details.update(q01=float(low), q99=float(high), stretch_flat=bool(high <= low))
        if high > low:
            total = sum(im.width*im.height for im in images)
            details["clipped_low_fraction"] = sum(np.count_nonzero(np.asarray(im)<low) for im in images)/total
            details["clipped_high_fraction"] = sum(np.count_nonzero(np.asarray(im)>high) for im in images)/total

            def stretch(im):
                a = (np.asarray(im, dtype=np.float64)-low)*(255./(high-low))
                return Image.fromarray(np.rint(np.clip(a, 0, 255)).astype(np.uint8))

            images = [stretch(im) for im in images]
            rgb = stretch(rgb)
    details.update(out_vv_width=images[0].width, out_vv_height=images[0].height,
        out_vh_width=images[1].width, out_vh_height=images[1].height)
    pair = np.stack([np.asarray(im.resize((128, 128), Image.Resampling.BILINEAR)) for im in images])
    return pair, rgb, details
