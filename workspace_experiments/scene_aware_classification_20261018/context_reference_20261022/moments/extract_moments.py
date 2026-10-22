"""Native all-OBB-excluded stem moments; no labels, PCA, training or detection."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from osgeo import gdal
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
SOURCE = BASE/'context_expansion_20261022/readout'
sys.path.insert(0, str(BASE/'context_relation_pilot_20261022'))
from prepare_ports24 import Reader, polygon_mask

MIN_BG_PIXELS = 64
POLS = ('vv', 'vh')
REVIEW_TARGETS = {'pair_0031576', 'pair_0069192', 'pair_0065308', 'pair_0030918'}
REVIEW_NEIGHBOR = ('pair_0069192', 'det_01443')


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def all_obb_background(points, bounds, window):
    x, y, w, h = window
    # Candidate bboxes include the 3px dilation fringe, including OBBs outside the crop.
    lo, hi = bounds
    hit = (np.floor(lo[:, 0]) <= x+w+3) & (np.ceil(hi[:, 0]) >= x-3)
    hit &= (np.floor(lo[:, 1]) <= y+h+3) & (np.ceil(hi[:, 1]) >= y-3)
    indices = np.flatnonzero(hit)
    fg = np.zeros((h+6, w+6), bool)
    origin = np.array([x-3, y-3])
    for i in indices:
        fg |= polygon_mask(points[i], origin, fg.shape)
    dilated = F.max_pool2d(torch.from_numpy(fg.astype(np.float32))[None, None], 7, stride=1, padding=3)[0, 0].numpy() > 0
    bg = ~dilated[3:3+h, 3:3+w]
    return bg, indices


@torch.inference_mode()
def encode_moments(reader, image, bg):
    n = int(bg.sum())
    result = dict(mean=np.zeros(64, np.float32), second=np.zeros(64, np.float32),
                  valid=False, bg_pixels=n, weight_sum=np.zeros(4, np.float32), reason='')
    if n < MIN_BG_PIXELS:
        result['reason'] = 'background_pixels_below_64'
        return result
    if not np.isfinite(image).all():
        result['reason'] = 'nonfinite_native_input'
        return result
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)/255)[None, None].cuda().expand(-1, 3, -1, -1)
    mask = torch.from_numpy(bg.astype(np.float32))[None, None].cuda()
    first, second, sums = [], [], []
    for k in range(4):
        z = reader.stem((x.rot90(k, (-2, -1))-reader.mean)/reader.std)
        weights = F.interpolate(mask.rot90(k, (-2, -1)), size=z.shape[-2:], mode='area')
        total = weights.sum()
        sums.append(total)
        first.append((z*weights).sum((-2, -1))/total)
        second.append((z.square()*weights).sum((-2, -1))/total)
    mean = torch.stack(first).mean(0).squeeze(0).cpu().numpy().astype(np.float32)
    raw_second = torch.stack(second).mean(0).squeeze(0).cpu().numpy().astype(np.float32)
    sums = torch.stack(sums).cpu().numpy().astype(np.float32)
    if not (np.isfinite(mean).all() and np.isfinite(raw_second).all() and np.isfinite(sums).all() and (sums > 0).all()):
        result['reason'] = 'nonfinite_or_empty_feature_weight'
        return result
    result.update(mean=mean, second=raw_second, valid=True, weight_sum=sums)
    return result


def make_panel(path, title, images, windows, masks, points, selected, results):
    zoom = 3 if max(max(im.shape) for im in images) <= 128 else 1
    width = max(im.shape[1] for im in images)*zoom+16
    height = max(im.shape[0] for im in images)*zoom+65
    out = Image.new('RGB', (width*4, height*2+48), 'white')
    ink = ImageDraw.Draw(out)
    ink.text((8, 5), title, fill='black')
    ink.text((8, 21), 'Native / all eligible OBBs / BG native mask / k=0 feature area weights; display nearest only.', fill='black')
    for pi, (im, win, bg, ids, result) in enumerate(zip(images, windows, masks, selected, results)):
        feature_shape = ((im.shape[0]+1)//2, (im.shape[1]+1)//2)
        weights = F.interpolate(torch.from_numpy(bg.astype(np.float32))[None, None], size=feature_shape, mode='area')[0, 0].numpy()
        weights = np.asarray(Image.fromarray(weights).resize((im.shape[1], im.shape[0]), Image.Resampling.NEAREST))
        for j in range(4):
            rgb = np.repeat(im[..., None], 3, axis=2).astype(np.float32)
            if j >= 2:
                weight = bg.astype(np.float32) if j == 2 else weights
                alpha = .6*weight[..., None]
                rgb = rgb*(1-alpha)+np.array([20, 220, 255])*alpha
                if j == 2:
                    rgb[~bg] = rgb[~bg]*.4+np.array([255, 40, 40])*.6
            tile = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((im.shape[1]*zoom, im.shape[0]*zoom), Image.Resampling.NEAREST)
            if j == 1:
                draw = ImageDraw.Draw(tile)
                for k in ids:
                    corners = [tuple(p*zoom) for p in points[k]-np.array(win[:2])]
                    draw.line(corners+corners[:1], fill=(255, 160, 20), width=1)
            xx, yy = j*width+8, pi*height+48
            ink.text((xx, yy), f'{POLS[pi].upper()} {("native", "all OBBs", "BG mask", "area weights")[j]} {win}', fill='black')
            ink.text((xx, yy+16), f'BG={result["bg_pixels"]} valid={result["valid"]} boxes={len(ids)}', fill='black')
            out.paste(tile, (xx, yy+40))
    out.save(path)


def pair(reader, datasets, windows, points, bounds):
    images, masks, selected, results = [], [], [], []
    for ds, win in zip(datasets, windows):
        x, y, w, h = win
        assert x >= 0 and y >= 0 and x+w <= ds.RasterXSize and y+h <= ds.RasterYSize
        image = ds.ReadAsArray(x, y, w, h)
        assert image.shape == (h, w)
        bg, ids = all_obb_background(points, bounds, win)
        images.append(image)
        masks.append(bg)
        selected.append(ids)
        results.append(encode_moments(reader, image, bg))
    values = {key: np.stack([r[key] for r in results]) for key in ('mean', 'second', 'weight_sum')}
    values['valid'] = np.array([r['valid'] for r in results], bool)
    values['bg_pixels'] = np.array([r['bg_pixels'] for r in results], np.int32)
    values['excluded_obb_count'] = np.array([len(i) for i in selected], np.int32)
    return values, (images, masks, selected, results)


def arrays(n):
    result = {}
    for prefix, axes in [('target', (n,)), ('neighbor', (n, 16))]:
        for key, shape, dtype in [('mean', (2, 64), np.float32), ('second', (2, 64), np.float32),
                                  ('valid', (2,), bool), ('bg_pixels', (2,), np.int32),
                                  ('excluded_obb_count', (2,), np.int32), ('weight_sum', (2, 4), np.float32)]:
            result[f'{prefix}_{key}'] = np.zeros(axes+shape, dtype=dtype)
    result['neighbor_cache_index'] = np.full((n, 16), -1, np.int32)
    return result


def extract_product(product, targets, edges, catalogs, reader, folder):
    started = time.perf_counter()
    paths = [targets[f'raster_{p}'].unique() for p in POLS]
    assert all(len(v) == 1 for v in paths)
    datasets = [gdal.Open(str(v[0])) for v in paths]
    assert datasets[0].GetGeoTransform() == datasets[1].GetGeoTransform()
    assert datasets[0].GetSpatialRef().IsSame(datasets[1].GetSpatialRef())
    selected_catalog = catalogs[catalogs['product'].eq(product) & catalogs.pol.eq('vv')]
    assert len(selected_catalog) == 1
    catalog_path = selected_catalog.iloc[0].audit_path
    det = pd.read_csv(catalog_path, usecols=['detection_id', 'source_pts', 'conf', 'pixel_status', 'roi_origin', 'source_image'])
    det = det[det.conf.ge(.25) & det.pixel_status.eq('valid')].reset_index(drop=True)
    assert all(json.loads(v) == [0, 0] for v in det.roi_origin)
    assert all(Path(v.replace('\\', '/')).name == Path(str(paths[0][0]).replace('\\', '/')).name for v in det.source_image)
    points = np.array([json.loads(v) for v in det.source_pts], float).reshape(-1, 4, 2)
    bounds = (points.min(1), points.max(1))
    data = arrays(len(targets))
    cache, unique_rows, rows, reviews = {}, [], [], []
    hits = 0
    edge_groups = {key: group for key, group in edges.groupby('instance_id', sort=False)}
    for local_i, row in enumerate(targets.itertuples()):
        for kind, item in [('target', row)]+[('neighbor', e) for e in edge_groups.get(row.instance_id, pd.DataFrame()).itertuples()]:
            windows = [json.loads(getattr(item, f'{"target_window" if kind == "target" else "read_window"}_{p}')) for p in POLS]
            detection_id = row.target_detection_id if kind == 'target' else item.detection_id
            key = (product, detection_id, tuple(tuple(w) for w in windows))
            cached = kind == 'neighbor' and key in cache
            if cached:
                values, cache_id = cache[key]
                hits += 1
            else:
                values, visual = pair(reader, datasets, windows, points, bounds)
                if kind == 'neighbor':
                    cache_id = len(cache)
                    cache[key] = (values, cache_id)
                    unique_rows.append(dict(cache_index_product=cache_id, product=product, detection_id=detection_id,
                                            read_window_vv=json.dumps(windows[0]), read_window_vh=json.dumps(windows[1]),
                                            raster_vv=str(paths[0][0]), raster_vh=str(paths[1][0])))
            ix = local_i if kind == 'target' else (local_i, item.slot)
            for name, value in values.items():
                data[f'{kind}_{name}'][ix] = value
            if kind == 'neighbor':
                data['neighbor_cache_index'][ix] = cache_id
            make_review = (kind == 'target' and row.instance_id in REVIEW_TARGETS) or (kind == 'neighbor' and (row.instance_id, detection_id) == REVIEW_NEIGHBOR)
            if make_review and cached:
                # Cached numeric moments are reused; only selected bounded pixels/masks are reread for display.
                images, masks, selected, results = [], [], [], []
                for pi, (ds, win) in enumerate(zip(datasets, windows)):
                    bg, ids = all_obb_background(points, bounds, win)
                    images.append(ds.ReadAsArray(*win)); masks.append(bg); selected.append(ids)
                    results.append(dict(bg_pixels=int(values['bg_pixels'][pi]), valid=bool(values['valid'][pi])))
                visual = images, masks, selected, results
            if make_review:
                images, masks, selected, results = visual
                panel_name = f'{row.instance_id}_{kind}_{detection_id}.png'
                make_panel(HERE/'panels'/panel_name, f'{row.port} {row.instance_id} {kind} {detection_id}', images, windows, masks, points, selected, results)
                mask_names = []
                for p, bg in zip(POLS, masks):
                    name = f'{row.instance_id}_{kind}_{detection_id}_{p}_bgmask.png'
                    Image.fromarray(bg.astype(np.uint8)*255).save(HERE/'panels'/name)
                    mask_names.append(name)
                reviews.append(dict(instance_id=row.instance_id, kind=kind, detection_id=detection_id, product=product, port=row.port,
                                    panel=str(HERE/'panels'/panel_name), mask_vv=mask_names[0], mask_vh=mask_names[1],
                                    window_vv=json.dumps(windows[0]), window_vh=json.dumps(windows[1]),
                                    excluded_detection_ids_vv=json.dumps(det.iloc[selected[0]].detection_id.tolist()),
                                    excluded_detection_ids_vh=json.dumps(det.iloc[selected[1]].detection_id.tolist()),
                                    bg_pixels_vv=int(values['bg_pixels'][0]), bg_pixels_vh=int(values['bg_pixels'][1])))
            if not cached:
                for pi, p in enumerate(POLS):
                    rows.append(dict(kind=kind, instance_id=row.instance_id, target_index=int(row.target_index),
                                     slot=-1 if kind == 'target' else item.slot, product=product, detection_id=detection_id, pol=p,
                                     source_catalog=str(catalog_path), source_raster=str(paths[pi][0]), source_window=json.dumps(windows[pi]),
                                     bg_pixels=int(values['bg_pixels'][pi]), valid=bool(values['valid'][pi]),
                                     excluded_obb_count=int(values['excluded_obb_count'][pi]),
                                     failure_reason=visual[3][pi]['reason']))
    data['instance_ids'] = targets.instance_id.to_numpy(str)
    data['target_indices'] = targets.target_index.to_numpy(np.int32)
    np.savez_compressed(folder/'moments.npz', **data)
    pd.DataFrame(rows).to_csv(folder/'window_stats.csv', index=False)
    pd.DataFrame(unique_rows, columns=['cache_index_product', 'product', 'detection_id', 'read_window_vv', 'read_window_vh', 'raster_vv', 'raster_vh']).to_csv(folder/'unique_neighbors.csv', index=False)
    pd.DataFrame(reviews, columns=['instance_id', 'kind', 'detection_id', 'product', 'port', 'panel', 'mask_vv', 'mask_vh', 'window_vv', 'window_vh', 'excluded_detection_ids_vv', 'excluded_detection_ids_vh', 'bg_pixels_vv', 'bg_pixels_vh']).to_csv(folder/'review_cases.csv', index=False)
    result = dict(product=product, targets=len(targets), edges=len(edges), unique_neighbor_windows=len(cache), neighbor_cache_hits=hits,
                  seconds=time.perf_counter()-started, catalog_path=str(catalog_path), eligible_detection_obbs=len(det))
    dump(folder/'complete.json', result)
    return result


def main():
    started = time.perf_counter()
    for name in ('products', 'panels'):
        (HERE/name).mkdir(parents=True, exist_ok=True)
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64*1024**2)
    targets = pd.read_csv(SOURCE/'targets.csv', dtype={'mmsi': str}, low_memory=False)
    edges = pd.read_csv(SOURCE/'neighbors.csv')
    with np.load(SOURCE/'features.npz') as source:
        source_ids, mask, edge_index = source['instance_ids'], source['mask'], source['edge_index']
    assert np.array_equal(source_ids, targets.instance_id.to_numpy(str))
    assert np.array_equal(targets.target_index.to_numpy(), np.arange(len(targets)))
    assert len(edges) == mask.sum()
    assert REVIEW_TARGETS.issubset(set(source_ids))
    assert ((edges.instance_id == REVIEW_NEIGHBOR[0]) & (edges.detection_id == REVIEW_NEIGHBOR[1])).any()
    catalogs = pd.read_csv(BASE/'data_image_review/ports24/fullscene_candidate_catalogs.csv')
    reader = Reader()
    reader.stem.requires_grad_(False)
    folders, progress = [], []
    for product, group in targets.groupby('product', sort=True):
        folder = HERE/'products'/product
        folder.mkdir(exist_ok=True)
        folders.append(folder)
        if (folder/'complete.json').exists():
            result = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        else:
            result = extract_product(product, group, edges[edges['product'].eq(product)], catalogs, reader, folder)
        progress.append(result)
        pd.DataFrame(progress).to_csv(HERE/'product_progress.csv', index=False)
        print(json.dumps(dict(products_done=len(progress), products_total=targets['product'].nunique(), **result)), flush=True)
    output = arrays(len(targets))
    unique, stats, reviews = [], [], []
    offset, observed_ids = 0, np.full(len(targets), '', dtype=source_ids.dtype)
    for folder in folders:
        with np.load(folder/'moments.npz') as part:
            ix = part['target_indices']
            observed_ids[ix] = part['instance_ids']
            for key in output:
                values = part[key]
                if key == 'neighbor_cache_index':
                    values = np.where(values >= 0, values+offset, -1)
                output[key][ix] = values
        table = pd.read_csv(folder/'unique_neighbors.csv')
        table['cache_index'] = np.arange(offset, offset+len(table))
        offset += len(table)
        unique.append(table)
        stats.append(pd.read_csv(folder/'window_stats.csv'))
        reviews.append(pd.read_csv(folder/'review_cases.csv'))
    assert np.array_equal(observed_ids, source_ids)
    output.update(instance_ids=source_ids, mask=mask, edge_index=edge_index)
    for prefix in ('target', 'neighbor'):
        valid = output[f'{prefix}_valid']
        for name in ('mean', 'second'):
            values = output[f'{prefix}_{name}']
            assert values.dtype == np.float32 and np.isfinite(values).all() and not values[~valid].any()
        assert (output[f'{prefix}_bg_pixels'][valid] >= MIN_BG_PIXELS).all()
        # Raw second moments must dominate squared first moments (float32 tolerance).
        delta = output[f'{prefix}_second']-output[f'{prefix}_mean']**2
        assert (delta[valid] >= -2e-5*np.maximum(output[f'{prefix}_second'][valid], 1)).all()
    assert not output['neighbor_valid'][~mask].any()
    assert not output['neighbor_mean'][~mask].any() and not output['neighbor_second'][~mask].any()
    assert (output['neighbor_cache_index'][mask] >= 0).all() and (output['neighbor_cache_index'][~mask] == -1).all()
    np.savez_compressed(HERE/'moments.npz', **output)
    pd.concat(unique, ignore_index=True).to_csv(HERE/'unique_neighbors.csv', index=False)
    statistics = pd.concat(stats, ignore_index=True)
    statistics.to_csv(HERE/'window_stats.csv', index=False)
    review = pd.concat(reviews, ignore_index=True)
    review.to_csv(HERE/'review_cases.csv', index=False)
    for prefix in ('target', 'neighbor'):
        valid = output[f'{prefix}_valid']
        active = np.ones(valid.shape[:-1], bool) if prefix == 'target' else mask
        pd.DataFrame([dict(pol=p, active=int(active.sum()), valid=int(valid[..., pi][active].sum()),
                           invalid=int((~valid[..., pi][active]).sum()), min_bg_pixels=int(output[f'{prefix}_bg_pixels'][..., pi][active].min()),
                           median_bg_pixels=float(np.median(output[f'{prefix}_bg_pixels'][..., pi][active])),
                           multiple_candidate_obbs=int((output[f'{prefix}_excluded_obb_count'][..., pi][active] > 1).sum()))
                      for pi, p in enumerate(POLS)]).to_csv(HERE/f'{prefix}_coverage.csv', index=False)
    summary = dict(targets=len(targets), edges=len(edges), context_targets=int(mask.any(1).sum()), products=len(progress),
        unique_neighbor_windows=offset, neighbor_cache_hits=sum(r['neighbor_cache_hits'] for r in progress),
        target_valid_pol=output['target_valid'].sum(0).tolist(), target_invalid_pol=(~output['target_valid']).sum(0).tolist(),
        neighbor_valid_pol=output['neighbor_valid'][mask].sum(0).tolist(), neighbor_invalid_pol=(~output['neighbor_valid'][mask]).sum(0).tolist(),
        invalid_reasons=statistics.loc[~statistics.valid, 'failure_reason'].value_counts().to_dict(),
        all_finite=True, ids_mask_edge_index_unchanged=True, invalid_exact_zero=True,
        shape={k: list(v.shape) for k, v in output.items()}, min_background_native_pixels=MIN_BG_PIXELS,
        mask_definition='union of every same-product valid VV detection OBB with conf>=0.25, dilated by square radius3 native pixels; includes overlapping and dilation-fringe boxes',
        moment_definition='for each pol/rotation: sum(area_weight*z)/sum(area_weight), sum(area_weight*z^2)/sum(area_weight); equal mean of four rotations',
        visual='frozen ImageNet R50 conv1/bn1/relu; native separate VV/VH windows; no image masking/resizing; no PCA',
        source_directory=str(SOURCE), source_files=['targets.csv', 'neighbors.csv', 'features.npz'],
        source_catalogs=str(BASE/'data_image_review/ports24/fullscene_candidate_catalogs.csv'),
        source_inputs_modified=False, new_detection_or_training=False,
        product_seconds=sum(r['seconds'] for r in progress), invocation_seconds=time.perf_counter()-started,
        review_targets=int(review.kind.eq('target').sum()), review_neighbors=int(review.kind.eq('neighbor').sum()),
        limitations=['Background is outside catalog OBBs, not pure sea or a physical component.',
                    '7x7 convolution receptive fields mix OBB boundaries despite output pooling exclusions.',
                    'C4 rotations are repeated views of the same pixels, not four independent observations.',
                    'Second is a raw feature-map second moment, not variance of previously pooled node means.'])
    dump(HERE/'summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
