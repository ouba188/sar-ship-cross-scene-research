"""Bounded numeric replay and all-catalog mask reference for saved moments."""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from osgeo import gdal
from extract_moments import HERE, BASE, SOURCE, Reader, all_obb_background, encode_moments, dump


def independent_bg(points, window):
    x, y, w, h = window
    pad = 12
    image = Image.new('1', (w+2*pad, h+2*pad))
    ink = ImageDraw.Draw(image)
    # No bounding-box candidate filter in this reference, on a bounded local canvas.
    for box in points:
        ink.polygon([tuple(p) for p in box-np.array([x-pad, y-pad])], fill=1)
    mask = binary_dilation(np.asarray(image, bool), structure=np.ones((7, 7), bool))
    return ~mask[pad:pad+h, pad:pad+w]


@torch.inference_mode()
def explicit_moments(reader, image, bg):
    x = torch.tensor(image.astype(np.float32)/255)[None, None].cuda().expand(-1, 3, -1, -1)
    mask = torch.tensor(bg.astype(np.float32))[None, None].cuda()
    first, second = [], []
    for k in range(4):
        z = reader.stem((torch.rot90(x, k, (-2, -1))-reader.mean)/reader.std)
        weights = F.interpolate(torch.rot90(mask, k, (-2, -1)), size=z.shape[-2:], mode='area')
        values = z[0].cpu().numpy().reshape(64, -1).astype(np.float64)
        w = weights[0, 0].cpu().numpy().ravel().astype(np.float64)
        first.append(np.average(values, axis=1, weights=w))
        second.append(np.average(values**2, axis=1, weights=w))
    return np.mean(first, axis=0), np.mean(second, axis=0)


def main():
    started = time.perf_counter()
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64*1024**2)
    with np.load(HERE/'moments.npz') as f:
        saved = {k: f[k] for k in f.files}
    with np.load(SOURCE/'features.npz') as f:
        for key in ('instance_ids', 'mask', 'edge_index'):
            assert np.array_equal(saved[key], f[key])
    assert all(np.isfinite(v).all() for k, v in saved.items() if k != 'instance_ids')
    assert saved['target_mean'].shape == (len(saved['instance_ids']), 2, 64)
    assert saved['neighbor_mean'].shape == (len(saved['instance_ids']), 16, 2, 64)
    cache_seen, repeated = {}, 0
    for i, j in zip(*np.nonzero(saved['mask'])):
        key = int(saved['neighbor_cache_index'][i, j])
        if key in cache_seen:
            a, b = cache_seen[key]
            for name in ('mean', 'second', 'valid', 'bg_pixels', 'weight_sum'):
                assert np.array_equal(saved[f'neighbor_{name}'][i, j], saved[f'neighbor_{name}'][a, b])
            repeated += 1
        else:
            cache_seen[key] = (i, j)
    t = pd.read_csv(SOURCE/'targets.csv').set_index('instance_id')
    edges = pd.read_csv(SOURCE/'neighbors.csv')
    review = pd.read_csv(HERE/'review_cases.csv')
    catalogs = pd.read_csv(BASE/'data_image_review/ports24/fullscene_candidate_catalogs.csv')
    reader = Reader()
    reader.stem.requires_grad_(False)
    rows = []
    for case in review.itertuples():
        row = t.loc[case.instance_id]
        cat = catalogs[catalogs['product'].eq(case.product) & catalogs.pol.eq('vv')].iloc[0]
        det = pd.read_csv(cat.audit_path, usecols=['source_pts', 'conf', 'pixel_status'])
        det = det[det.conf.ge(.25) & det.pixel_status.eq('valid')]
        points = np.array([json.loads(v) for v in det.source_pts])
        bounds = points.min(1), points.max(1)
        if case.kind == 'target':
            ix = int(row.target_index)
        else:
            edge = edges[edges.instance_id.eq(case.instance_id) & edges.detection_id.eq(case.detection_id)].iloc[0]
            ix = (int(edge.target_index), int(edge.slot))
        for pi, pol in enumerate(('vv', 'vh')):
            window = json.loads(getattr(case, f'window_{pol}'))
            bg, selected = all_obb_background(points, bounds, window)
            reference = independent_bg(points, window)
            assert np.array_equal(bg, reference), (case.instance_id, pol, int((bg != reference).sum()))
            assert np.array_equal(bg, np.array(Image.open(HERE/'panels'/getattr(case, f'mask_{pol}'))) > 0)
            ds = gdal.Open(row[f'raster_{pol}'])
            image = ds.ReadAsArray(*window)
            ds = None
            mean, second = explicit_moments(reader, image, bg)
            got_mean, got_second = saved[f'{case.kind}_mean'][ix][pi], saved[f'{case.kind}_second'][ix][pi]
            np.testing.assert_allclose(got_mean, mean, rtol=5e-6, atol=3e-6)
            np.testing.assert_allclose(got_second, second, rtol=5e-6, atol=3e-6)
            assert int(saved[f'{case.kind}_bg_pixels'][ix][pi]) == int(bg.sum())
            rows.append(dict(instance_id=case.instance_id, kind=case.kind, detection_id=case.detection_id, pol=pol,
                             native_mask_exact=True, background_pixels=int(bg.sum()), candidate_obbs=len(selected),
                             mean_max_abs=float(np.max(np.abs(got_mean-mean))), second_max_abs=float(np.max(np.abs(got_second-second))),
                             positive_channel_moment_variance=int((second-mean**2 > 1e-6).sum())))
    # Exercise the fixed threshold on a native 8x8 crop; no performance-based selection.
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    bg = np.ones((8, 8), bool)
    bg[0, 0] = False
    low = encode_moments(reader, image, bg)
    assert not low['valid'] and not low['mean'].any() and not low['second'].any()
    bg[0, 0] = True
    threshold = encode_moments(reader, image, bg)
    assert threshold['valid'] and threshold['bg_pixels'] == 64
    broken = image.astype(np.float32)
    broken[0, 0] = np.nan
    bad = encode_moments(reader, broken, bg)
    assert not bad['valid'] and not bad['mean'].any() and not bad['second'].any()
    pd.DataFrame(rows).to_csv(HERE/'moment_replay.csv', index=False)
    result = dict(saved_arrays_finite=True, source_ids_mask_edge_index_exact=True,
                  unique_neighbor_windows=len(cache_seen), repeated_edges_exact=repeated,
                  replay_cases=len(review), replay_polarizations=len(rows), all_catalog_mask_reference_exact=True,
                  mean_max_abs=max(r['mean_max_abs'] for r in rows), second_max_abs=max(r['second_max_abs'] for r in rows),
                  numeric_reference='float64 NumPy weighted first/second raw moments of explicit feature-map pixels, per C4 then equal average',
                  valid_threshold_63_false_64_true=True, nan_input_returns_zero_false=True,
                  seconds=time.perf_counter()-started)
    dump(HERE/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
