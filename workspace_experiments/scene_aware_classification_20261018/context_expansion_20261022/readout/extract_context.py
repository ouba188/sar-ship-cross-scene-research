"""Window-only native context extraction; frozen stem, no training or label edits."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
import math
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from osgeo import gdal
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sys.path.insert(0, str(BASE / 'context_relation_pilot_20261022'))
sys.path.insert(0, str(BASE / 'context_region_relation_20261022/readout'))
from prepare_ports24 import Reader, box_geometry, polygon_mask, polygon, crop_world, WaterMap, WC, region_stats
from prepare_regions import region_masks, encode_regions

ARRAY_SHAPES = {'target': (256,), 'target_fg': (256,), 'target_bg': (256,),
                'neighbor': (16, 256), 'neighbor_fg': (16, 256), 'neighbor_bg': (16, 256),
                'geometry': (16, 7), 'mask': (16,), 'fg_valid': (16,), 'bg_valid': (16,),
                'target_fg_valid': (), 'target_bg_valid': (),
                'target_fg_valid_pol': (2,), 'target_bg_valid_pol': (2,),
                'fg_valid_pol': (16, 2), 'bg_valid_pol': (16, 2)}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def frame(rows, columns=None):
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns or ['instance_id', 'product', 'reason'])


def window_polygon(window):
    x, y, w, h = window
    return polygon([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])


def planned_window(ds, world, side):
    gt = ds.GetGeoTransform()
    px, py = gdal.ApplyGeoTransform(gdal.InvGeoTransform(gt), *world)
    w, h = math.ceil(side / abs(gt[1])), math.ceil(side / abs(gt[5]))
    def edge(v):
        return round(v) if abs(v-round(v)) < 1e-6 else math.floor(v)
    return [edge(px-w/2), edge(py-h/2), w, h]


def in_bounds(ds, window):
    x, y, w, h = window
    return min(x, y) >= 0 and min(w, h) > 0 and x+w <= ds.RasterXSize and y+h <= ds.RasterYSize


def encode_pair(reader, images, points, windows):
    masks = [region_masks(points, w) for w in windows]
    encoded = [encode_regions(reader, im, m) for im, m in zip(images, masks)]
    vectors = {name: np.concatenate([e[0][name] for e in encoded]) for name in ('whole', 'fg', 'bg')}
    valid = {name: np.array([bool(e[1][j].sum() > 0) for e in encoded])
             for j, name in enumerate(('fg', 'bg'))}
    details = {}
    for p, m, e in zip(('vv', 'vh'), masks, encoded):
        for j, name in enumerate(('fg', 'bg')):
            details[f'{name}_pixels_{p}'] = int(m[j].sum())
            details[f'{name}_weight_sum_{p}'] = float(e[1][j].sum())
            details[f'{name}_valid_{p}'] = bool(valid[name][('vv', 'vh').index(p)])
    return vectors, valid, details, [e[1] for e in encoded]


def panel(path, title, images, windows, points, footprints):
    # Display nearest-neighbor enlargement only; encoding always uses native arrays.
    zoom = 3 if max(max(im.shape) for im in images) <= 160 else 1
    cell_w = max(im.shape[1] for im in images)*zoom + 16
    cell_h = max(im.shape[0] for im in images)*zoom + 48
    out = Image.new('RGB', (cell_w*3, cell_h*2+48), 'white')
    draw = ImageDraw.Draw(out)
    draw.text((8, 5), title, fill='black')
    draw.text((8, 21), 'Native / FG / BG; orange OBB; area pooling weights. Display nearest only.', fill='black')
    for p, (im, win, weights) in enumerate(zip(images, windows, footprints)):
        for j in range(3):
            rgb = np.repeat(im[..., None], 3, axis=2).astype(np.float32)
            if j:
                w = np.asarray(Image.fromarray(weights[j-1]).resize((im.shape[1], im.shape[0]), Image.Resampling.NEAREST))
                color = np.array([40, 255, 70] if j == 1 else [30, 220, 255])
                alpha = .5*w[..., None]
                rgb = rgb*(1-alpha)+color*alpha
            tile = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((im.shape[1]*zoom, im.shape[0]*zoom), Image.Resampling.NEAREST)
            if j:
                corners = [tuple(v*zoom) for v in points-np.array(win[:2])]
                ImageDraw.Draw(tile).line(corners+corners[:1], fill=(255, 160, 20), width=1)
            x, y = j*cell_w+8, p*cell_h+48
            draw.text((x, y), f'{("VV", "VH")[p]} {("native", "FG", "BG")[j]} {win}', fill='black')
            out.paste(tile, (x, y+28))
    out.save(path)


def empty_arrays(n):
    return {k: np.zeros((n,)+shape, dtype=bool if 'valid' in k or k == 'mask' else np.float32)
            for k, shape in ARRAY_SHAPES.items()}


def process_product(product, group, catalogs, wm, reader, folder, review_ports):
    start = time.perf_counter()
    records, neighbors, excluded, reviews = [], [], [], []
    data = empty_arrays(len(group))
    paths = {p: group[f'full_path_{p}'].unique() for p in ('vv', 'vh')}
    assert all(len(v) == 1 for v in paths.values()), 'Multiple raster versions in product'
    ds = {p: gdal.Open(str(v[0])) for p, v in paths.items()}
    dv, dh = ds['vv'], ds['vh']
    gt = dv.GetGeoTransform()
    assert gt == dh.GetGeoTransform() and dv.GetSpatialRef().IsSame(dh.GetSpatialRef()), 'Unaligned polarization rasters'
    assert not gt[2] and not gt[4] and dv.GetSpatialRef().IsProjected() and np.isclose(abs(gt[1]), abs(gt[5]))
    scale = abs(gt[1])
    cat = catalogs[catalogs['product'].eq(product) & catalogs.pol.eq('vv')]
    assert len(cat) == 1, f'VV catalog ambiguity: {product}'
    detections = pd.read_csv(cat.iloc[0].audit_path, usecols=['detection_id', 'source_pts', 'conf', 'pixel_status', 'roi_origin', 'source_image'])
    for d in detections.itertuples():
        reasons = []
        if not d.conf >= .25:
            reasons.append('confidence_below_0.25')
        if d.pixel_status != 'valid':
            reasons.append('pixel_status_not_valid')
        for reason in reasons:
            excluded.append(dict(scope='catalog', product=product, instance_id='', detection_id=d.detection_id, reason=reason))
    det = detections[detections.conf.ge(.25) & detections.pixel_status.eq('valid')].reset_index(drop=True)
    assert all(json.loads(v) == [0, 0] for v in det.roi_origin)
    assert all(Path(v.replace('\\', '/')).name == Path(str(paths['vv'][0]).replace('\\', '/')).name for v in det.source_image)
    pts = np.asarray([json.loads(v) for v in det.source_pts], dtype=float).reshape(-1, 4, 2)
    centers = pts.mean(1)
    tree = cKDTree(centers) if len(det) else None
    cache, cache_hits, unique_encodes, overlap_checks = {}, 0, 0, 0
    for row in group.to_dict('records'):
        identity = dict(product=product, instance_id=row['instance_id'])
        def reject(reason, k=None, **extra):
            excluded.append(dict(scope='target' if k is None else 'neighbor', **identity,
                                 detection_id='' if k is None else str(det.iloc[k].detection_id), reason=reason, **extra))
        windows = [[int(row[f'{c}_{p}']) for c in ('x', 'y', 'w', 'h')] for p in ('vv', 'vh')]
        if any(not in_bounds(d, w) or max(w[2:]) > 512 for d, w in zip((dv, dh), windows)):
            reject('target_window_outside_or_oversize')
            continue
        left, top, w, h = windows[0]
        center = np.array([left+w/2, top+h/2])
        dself, selfid = tree.query(center) if tree is not None else (float('inf'), -1)
        if dself > 5:
            reject('target_axis_unresolved_distance_gt5px', axis_match_px=float(dself), **{k: row[k] for k in ('class_id', 'class_name', 'mmsi', 'port', 'pass_group')})
            continue
        selfid = int(selfid)
        target_poly = polygon(pts[selfid])
        length, width, axis = box_geometry(pts[selfid], scale)
        cross = np.array([-axis[1], axis[0]])
        ids = sorted(tree.query_ball_point(center, 1000/scale), key=lambda k: (np.linalg.norm(centers[k]-center), int(k)))
        pad = math.ceil(1250/scale)
        lo = np.maximum(0, np.floor(center).astype(int)-pad)
        hi = np.minimum([dv.RasterXSize, dv.RasterYSize], np.ceil(center).astype(int)+pad)
        land, distance = wm.window(dv, (*map(int, lo), *map(int, hi-lo)))
        i, slot = len(records), 0
        counts = Counter()
        selected_review = row['port'] in review_ports
        for k in ids:
            cx, cy = centers[k]
            if k == selfid:
                reject('target_detection', k)
                continue
            if left <= cx < left+w and top <= cy < top+h:
                reject('neighbor_center_inside_target_vv_window', k)
                continue
            counts['outside_candidates'] += 1
            stats = region_stats(land, distance, polygon_mask(pts[k], lo, land.shape))
            if stats['action'] == 'exclude_land_interior':
                counts['land_removed'] += 1
                reject('exclude_land_interior', k, **stats)
                continue
            nl, nw, naxis = box_geometry(pts[k], scale)
            side = max(64*scale, np.ptp(pts[k], axis=0).max()*scale+32*scale)
            if side/scale > 512:
                reject('neighbor_window_oversize', k)
                continue
            world = gdal.ApplyGeoTransform(gt, *centers[k])
            nwin = [planned_window(d, world, side) for d in (dv, dh)]
            if any(not in_bounds(d, win) for d, win in zip((dv, dh), nwin)):
                reject('neighbor_window_outside_raster', k)
                continue
            areas = [float(target_poly.Intersection(window_polygon(win)).GetArea()) for win in nwin]
            if max(areas) > 1e-6:
                counts['target_overlap_removed'] += 1
                reject('neighbor_read_window_intersects_target_obb', k, read_window=json.dumps(nwin[0]), intersection_area_px2=max(areas))
                continue
            if slot >= 16:
                counts['truncated'] += 1
                reject('max16_nearest_truncation', k)
                continue
            if k not in cache:
                images = []
                for d, win in zip((dv, dh), nwin):
                    im, actual = crop_world(d, world, side)
                    assert actual == win and im.shape == (win[3], win[2]) and np.isfinite(im).all()
                    images.append(im)
                cache[k] = encode_pair(reader, images, pts[k], nwin)
                unique_encodes += 1
            else:
                cache_hits += 1
            vectors, valid, details, footprints = cache[k]
            overlap_checks += 2
            rel = (centers[k]-center)*scale
            dot = float(axis@naxis)
            geom = [np.linalg.norm(rel)/1000, abs(rel@axis)/1000, abs(rel@cross)/1000,
                    math.log(max(nl, 1)/max(length, 1)), math.log(max(nw, 1)/max(width, 1)), 2*dot*dot-1, float(det.iloc[k]['conf'])]
            for name, key in [('whole', 'neighbor'), ('fg', 'neighbor_fg'), ('bg', 'neighbor_bg')]:
                data[key][i, slot] = vectors[name]
            for name in ('fg', 'bg'):
                data[f'{name}_valid'][i, slot] = valid[name].all()
                data[f'{name}_valid_pol'][i, slot] = valid[name]
            data['geometry'][i, slot], data['mask'][i, slot] = geom, True
            neighbors.append(dict(**identity, slot=slot, detection_id=det.iloc[k].detection_id,
                source_points=json.dumps(pts[k].tolist()), cx=cx, cy=cy, side_m=side,
                read_window=json.dumps(nwin[0]), read_window_vv=json.dumps(nwin[0]), read_window_vh=json.dumps(nwin[1]),
                target_points=json.dumps(pts[selfid].tolist()), distance_m=float(np.linalg.norm(rel)), land_action=stats['action'],
                overlaps_chip=bool(pts[k, :, 0].min()<left+w and pts[k, :, 0].max()>left and pts[k, :, 1].min()<top+h and pts[k, :, 1].max()>top),
                target_intersection_area_vv=areas[0], target_intersection_area_vh=areas[1],
                raster_vv=str(paths['vv'][0]), raster_vh=str(paths['vh'][0]),
                **{f'g{j}': float(v) for j, v in enumerate(geom)}, **details))
            if selected_review and slot == 0:
                images = [d.ReadAsArray(*win) for d, win in zip((dv, dh), nwin)]
                name = f'{row["instance_id"]}_neighbor_{det.iloc[k].detection_id}.png'
                panel(HERE/'review_panels'/name, f'{row["port"]} {row["instance_id"]} neighbor {det.iloc[k].detection_id}', images, nwin, pts[k], footprints)
                reviews.append(dict(**identity, port=row['port'], kind='neighbor', detection_id=det.iloc[k].detection_id, panel=name, selection='first accepted target per new port, nearest retained neighbor'))
            slot += 1
        images = [d.ReadAsArray(*win) for d, win in zip((dv, dh), windows)]
        assert all(im.shape == (win[3], win[2]) and np.isfinite(im).all() for im, win in zip(images, windows))
        vectors, valid, details, footprints = encode_pair(reader, images, pts[selfid], windows)
        for name, key in [('whole', 'target'), ('fg', 'target_fg'), ('bg', 'target_bg')]:
            data[key][i] = vectors[name]
        for name in ('fg', 'bg'):
            data[f'target_{name}_valid'][i] = valid[name].all()
            data[f'target_{name}_valid_pol'][i] = valid[name]
        for p, im in zip(('vv', 'vh'), images):
            Image.fromarray(im).save(HERE/'native_targets'/f'{row["instance_id"]}_{p}.png')
        if selected_review:
            name = f'{row["instance_id"]}_target.png'
            panel(HERE/'review_panels'/name, f'{row["port"]} {row["instance_id"]} target', images, windows, pts[selfid], footprints)
            reviews.append(dict(**identity, port=row['port'], kind='target', detection_id=det.iloc[selfid].detection_id, panel=name, selection='first accepted target per new port; no prediction selection'))
            review_ports.remove(row['port'])
        records.append(dict(row, target_index_product=i, x=float(center[0]), y=float(center[1]),
            pixel_m=scale, target_window=json.dumps(windows[0]), target_window_vv=json.dumps(windows[0]), target_window_vh=json.dumps(windows[1]),
            target_points=json.dumps(pts[selfid].tolist()), target_detection_id=det.iloc[selfid].detection_id,
            axis_match_px=float(dself), axis_x=float(axis[0]), axis_y=float(axis[1]), target_length_m=float(length), target_width_m=float(width),
            axis_center_rule='recorded VV window center, unchanged pilot criterion',
            raster_vv=str(paths['vv'][0]), raster_vh=str(paths['vh'][0]), original_chip=row['original_chip_vv'],
            date=product.split('_')[4][:8], neighbors=slot,
            outside_candidates=counts['outside_candidates'], land_removed=counts['land_removed'], target_overlap_removed=counts['target_overlap_removed'], truncated=counts['truncated'],
            radius_inside_raster=bool(min(center[0], center[1], dv.RasterXSize-center[0], dv.RasterYSize-center[1])*scale >= 1000), **details))
    data = {k: a[:len(records)] for k, a in data.items()}
    data['instance_ids'] = np.asarray([r['instance_id'] for r in records], dtype=str)
    np.savez_compressed(folder/'features.npz', **data)
    frame(records, list(group.columns)+['neighbors']).to_csv(folder/'targets.csv', index=False)
    frame(neighbors, ['instance_id', 'product', 'slot', 'detection_id']).to_csv(folder/'neighbors.csv', index=False)
    frame(excluded).to_csv(folder/'excluded.csv', index=False)
    frame(reviews, ['instance_id', 'product', 'port', 'kind', 'panel']).to_csv(folder/'review_cases.csv', index=False)
    result = dict(product=product, input_targets=len(group), targets=len(records), edges=len(neighbors),
                  unique_neighbor_encodes=unique_encodes, neighbor_cache_hits=cache_hits, overlap_checks=overlap_checks,
                  seconds=time.perf_counter()-start)
    dump(folder/'complete.json', result)
    return result


def main():
    started = time.perf_counter()
    while not (HERE.parent/'recovery_summary.json').exists():
        print('Waiting for recovery_summary.json', flush=True)
        time.sleep(5)
    for name in ('native_targets', 'products', 'review_panels'):
        (HERE/name).mkdir(parents=True, exist_ok=True)
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64*1024**2)
    linked = pd.read_csv(HERE.parent/'linked_targets.csv', keep_default_na=False, low_memory=False, dtype={'mmsi': str, 'instance_id': str})
    assert len(linked) and linked.instance_id.is_unique
    catalogs = pd.read_csv(BASE/'data_image_review/ports24/fullscene_candidate_catalogs.csv')
    old_ports = set(pd.read_csv(BASE/'context_relation_pilot_20261022/ports24/targets.csv').port)
    review_ports = set(linked.port)-old_ports
    reader = Reader()
    reader.stem.requires_grad_(False)
    wm = WaterMap(sorted(WC.glob('*_Map.tif')))
    progress, folders = [], []
    for product, group in linked.groupby('product', sort=True):
        folder = HERE/'products'/product
        folder.mkdir(exist_ok=True)
        folders.append(folder)
        if (folder/'complete.json').exists():
            result = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            prior = pd.read_csv(folder/'review_cases.csv')
            review_ports.difference_update(prior.port.dropna())
        else:
            result = process_product(product, group, catalogs, wm, reader, folder, review_ports)
        progress.append(result)
        frame(progress).to_csv(HERE/'product_progress.csv', index=False)
        print(json.dumps(dict(products_done=len(progress), products_total=linked['product'].nunique(), **result)), flush=True)
    tables = {}
    for name in ('targets', 'neighbors', 'excluded', 'review_cases'):
        tables[name] = pd.concat([pd.read_csv(f/f'{name}.csv', keep_default_na=False, dtype={'mmsi': str, 'instance_id': str}) for f in folders], ignore_index=True)
    targets, edges, excluded = tables['targets'], tables['neighbors'], tables['excluded']
    targets['target_index'] = np.arange(len(targets))
    by_id = dict(zip(targets.instance_id, targets.target_index))
    edges['target_index'] = edges.instance_id.map(by_id)
    edges['edge_index'] = np.arange(len(edges))
    for name, table in tables.items():
        table.to_csv(HERE/f'{name}.csv', index=False)
    parts = [np.load(f/'features.npz') for f in folders]
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in list(ARRAY_SHAPES)+['instance_ids']}
    for p in parts:
        p.close()
    arrays['edge_index'] = np.full((len(targets), 16), -1, np.int32)
    for edge in edges.itertuples():
        arrays['edge_index'][edge.target_index, edge.slot] = edge.edge_index
    assert len(targets) and np.array_equal(arrays['instance_ids'], targets.instance_id.to_numpy(str))
    assert all(np.isfinite(a).all() for k, a in arrays.items() if k != 'instance_ids')
    assert np.all(np.linalg.norm(arrays['target'], axis=1) > 0)
    mask = arrays['mask']
    assert np.array_equal(mask.sum(1), targets.neighbors.to_numpy()) and int(mask.sum()) == len(edges)
    assert not (edges.target_intersection_area_vv.gt(1e-6) | edges.target_intersection_area_vh.gt(1e-6)).any()
    for name in ('neighbor', 'neighbor_fg', 'neighbor_bg', 'geometry'):
        assert not arrays[name][~mask].any()
    for region in ('fg', 'bg'):
        for pi in range(2):
            sl = slice(pi*128, (pi+1)*128)
            assert not arrays[f'target_{region}'][:, sl][~arrays[f'target_{region}_valid_pol'][:, pi]].any()
            assert not arrays[f'neighbor_{region}'][:, :, sl][~arrays[f'{region}_valid_pol'][:, :, pi]].any()
    original = linked.set_index('instance_id', drop=False).loc[targets.instance_id]
    for col in linked.columns:
        assert np.array_equal(original[col].astype(str).to_numpy(), targets[col].astype(str).to_numpy()), f'Source field changed: {col}'
    np.savez_compressed(HERE/'features.npz', **arrays)
    conflict_rows = linked[linked.mmsi.ne('')].groupby('mmsi').class_id.nunique()
    conflict_ids = conflict_rows[conflict_rows > 1].index
    linked[linked.mmsi.isin(conflict_ids)].to_csv(HERE/'mmsi_class_conflicts.csv', index=False)
    for label, cols in [('class', ['class_id', 'class_name']), ('acquisition', ['pass_group']), ('port', ['port'])]:
        targets.assign(context=mask.any(1)).groupby(cols, dropna=False).agg(targets=('instance_id', 'size'), mmsi=('mmsi', 'nunique'), context_targets=('context', 'sum'), edges=('neighbors', 'sum')).reset_index().to_csv(HERE/f'summary_by_{label}.csv', index=False)
    summary = dict(input_targets=len(linked), targets=len(targets), target_excluded=int(excluded.scope.eq('target').sum()),
        edges=len(edges), context_targets=int(mask.any(1).sum()), classes=targets.class_name.value_counts().to_dict(),
        products=int(targets['product'].nunique()), acquisitions=int(targets.pass_group.nunique()), ports=int(targets.port.nunique()),
        mmsi=int(targets.loc[targets.mmsi.ne(''), 'mmsi'].nunique()), missing_mmsi=int(targets.mmsi.eq('').sum()),
        input_mmsi_cross_class_conflicts=len(conflict_ids), conflict_policy='report only; no row or label cleanup',
        unique_neighbor_encodes=sum(r['unique_neighbor_encodes'] for r in progress), neighbor_cache_hits=sum(r['neighbor_cache_hits'] for r in progress),
        excluded_reasons=excluded.groupby(['scope', 'reason']).size().rename('count').reset_index().to_dict('records'),
        target_fg_invalid=int((~arrays['target_fg_valid']).sum()), target_bg_invalid=int((~arrays['target_bg_valid']).sum()),
        neighbor_fg_invalid=int((mask & ~arrays['fg_valid']).sum()), neighbor_bg_invalid=int((mask & ~arrays['bg_valid']).sum()),
        all_finite=True, source_fields_preserved=True, zero_target_obb_overlap=True, aligned_ids=True,
        feature_shapes={k: list(v.shape) for k, v in arrays.items()},
        product_seconds=sum(r['seconds'] for r in progress), invocation_seconds=time.perf_counter()-started,
        status='frozen_features_only_no_training', review_selection='first accepted target in each port absent from old pilot, plus nearest retained neighbor',
        visual='ImageNet R50 conv1/bn1/relu meanmax C4, separate native VV/VH, no image resize/masking',
        region='same full-raster OBB mapped independently to recorded VV/VH windows; BG beyond square radius3',
        valid_semantics='combined flag requires both polarizations; per-polarization flags also saved',
        limitations=['7x7 stem receptive fields mix boundaries', 'WorldCover 2021 is an external static map', 'axis-unmatched targets explicitly excluded without relaxing 5px rule'])
    dump(HERE/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
