"""Native per-polarization bbox interventions; not fabricated head-tail annotations."""

import numpy as np
from PIL import Image
from transforms_regions import box_mask
from source_region_metadata import bbox_regions


def read_images(row):
    images = []
    for pol in ("vv", "vh"):
        with Image.open(getattr(row, "path_"+pol)) as image:
            a = np.asarray(image.convert("L"))
            if pol == "vh":
                np.testing.assert_array_equal(np.asarray(image.convert("RGB")), np.repeat(a[:, :, None], 3, axis=2))
            images.append(a)
    return images


def boxes(row, padding):
    return [bbox_regions([getattr(row, f"bbox_{k}_{p}") for k in ("x0", "y0", "x1", "y1")],
                         (int(getattr(row, "native_h_"+p)), int(getattr(row, "native_w_"+p))), padding)
            for p in ("vv", "vh")]


def transform_images(images, row, condition, index):
    padding = .35 if condition.startswith("wide_") else .20
    region_info = boxes(row, padding)
    sham_valid = all(info[3] for info in region_info)
    original = condition in ("original", "replay_original")
    if original:
        zone, kind = "none", "original"
    else:
        zone, kind = condition.removeprefix("wide_").split("_")
    low, high = 0., 0.
    if kind == "display":
        anchor = np.concatenate([np.asarray(Image.fromarray(a).resize((128, 128), Image.Resampling.BILINEAR)).ravel()
                                 for a in images])
        low, high = np.quantile(anchor, [.01, .99])
    result, records = [], []
    for a, (box, sham, valid, _) in zip(images, region_info):
        region = box_mask(a.shape, box)
        mask = (region if zone == "target" else ~region if zone == "outer" else
                box_mask(a.shape, sham) if zone == "sham" and sham_valid else np.zeros_like(region))
        b = a.copy()
        if mask.any():
            if kind == "flat":
                border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
                b[mask] = int(np.rint(np.median(border)))
            elif kind == "shuffle":
                loc = np.flatnonzero(mask)
                b.ravel()[loc] = a.ravel()[np.random.default_rng(20261013+int(index)).permutation(loc)]
            elif kind == "display" and high > low:
                b[mask] = np.rint(np.clip((a[mask].astype(float)-low)*255./(high-low), 0, 255)).astype(np.uint8)
        np.testing.assert_array_equal(b[~mask], a[~mask])
        result.append(b)
        records.append(dict(region_area=float(region.mean()), edited_area=float(mask.mean()),
            changed_fraction=float(np.mean(a != b)), change_rms=float(np.sqrt(np.mean((a.astype(float)-b)**2))),
            x0=box[0], y0=box[1], x1=box[2], y1=box[3]))
    info = dict(region_valid=all(x[2] for x in region_info), sham_valid=sham_valid,
        **{k: float(np.mean([r[k] for r in records])) for k in ("region_area", "edited_area", "changed_fraction", "change_rms")})
    for pol, r in zip(("vv", "vh"), records):
        info.update({k+"_"+pol: v for k, v in r.items()})
    # Unsuffixed preview coordinates refer to VH, the second branch's actual input.
    info.update({k: records[1][k] for k in ("x0", "y0", "x1", "y1")})
    return result, info


def prepared(images, transform_rgb):
    pil = [Image.fromarray(a) for a in images]
    pair = np.stack([np.asarray(a.resize((128, 128), Image.Resampling.BILINEAR)) for a in pil])
    return pair, transform_rgb(pil[1].convert("RGB"))
