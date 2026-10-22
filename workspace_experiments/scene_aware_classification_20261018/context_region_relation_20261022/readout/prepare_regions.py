"""Native frozen-stem region readout; no training and no writes outside HERE."""
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
OLD = HERE.parent.parent / 'context_relation_pilot_20261022'
sys.path.insert(0, str(OLD))
from prepare_ports24 import Reader, polygon_mask


def region_masks(points, window):
    left, top, width, height = window
    fg = polygon_mask(points, np.array([left, top]), (height, width)).copy()
    # Square/Chebyshev radius 3 in native pixels; no image masking before conv1.
    t = torch.from_numpy(fg.astype(np.float32))[None, None]
    dilated = F.max_pool2d(t, 7, stride=1, padding=3)[0, 0].numpy() > 0
    return fg, ~dilated


def masked_pool(z, weights):
    total = weights.sum()
    if total.item() == 0:
        return torch.zeros((1, 128), dtype=z.dtype, device=z.device)
    mean = (z * weights).sum((-2, -1)) / total
    maximum = z.masked_fill(weights <= 0, -torch.inf).amax((-2, -1))
    return torch.cat([mean, maximum], 1)


@torch.inference_mode()
def encode_regions(reader, image, masks):
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255)[None, None].cuda()
    x = x.expand(-1, 3, -1, -1)
    native_masks = [torch.from_numpy(m.astype(np.float32))[None, None].cuda() for m in masks]
    values = {'whole': [], 'fg': [], 'bg': []}
    footprints = []
    for k in range(4):
        z = reader.stem((x.rot90(k, (-2, -1)) - reader.mean) / reader.std)
        values['whole'].append(torch.cat([z.mean((-2, -1)), z.amax((-2, -1))], 1))
        for name, mask in zip(('fg', 'bg'), native_masks):
            weights = F.interpolate(mask.rot90(k, (-2, -1)), size=z.shape[-2:], mode='area')
            values[name].append(masked_pool(z, weights))
            if k == 0:
                footprints.append(weights[0, 0].cpu().numpy())
    return {name: torch.stack(v).mean(0).squeeze(0).cpu().numpy() for name, v in values.items()}, footprints


def cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0


def make_panel(cases):
    cell_w, row_h = 280, 322
    panel = Image.new('RGB', (cell_w * 6, row_h * len(cases) + 64), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), 'Native TIFF 8-bit; display 3x nearest only. Model input never resized/masked.', fill='black')
    draw.text((8, 23), 'FG green / BG cyan: k=0 area weights (brightness=weight); orange OBB. C4 rotates masks with image.', fill='black')
    for j, name in enumerate(('VV native', 'VV foreground pooling', 'VV background pooling', 'VH native', 'VH foreground pooling', 'VH background pooling')):
        draw.text((j * cell_w + 8, 45), name, fill='black')
    for i, case in enumerate(cases):
        row, ims, masks, weights = case
        y = 64 + i * row_h
        draw.text((8, y), f"edge {row['edge_index']} | {row['instance_id']} / {row['detection_id']} | {row['read_window']} | {row['land_action']}", fill='black')
        draw.text((8, y + 16), f"FG {row['fg_pixels']} px; BG {row['bg_pixels']} px; omitted border {row['guard_pixels']} px; all feature RFs are 7x7.", fill='black')
        points = np.array(json.loads(row['source_points'])) - np.array(json.loads(row['read_window'])[:2])
        for pol_index, im in enumerate(ims):
            for branch in range(3):
                rgb = np.repeat(im[..., None], 3, axis=2).astype(np.uint8)
                if branch:
                    weight = np.asarray(Image.fromarray(weights[branch - 1]).resize((im.shape[1], im.shape[0]), Image.Resampling.NEAREST))
                    color = np.array([40, 255, 70] if branch == 1 else [30, 220, 255])
                    alpha = .50 * weight[..., None]
                    rgb = np.clip(rgb * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
                tile = Image.fromarray(rgb).resize((im.shape[1] * 3, im.shape[0] * 3), Image.Resampling.NEAREST)
                if branch:
                    td = ImageDraw.Draw(tile)
                    corners = [tuple(p * 3) for p in points]
                    td.line(corners + corners[:1], fill=(255, 160, 20), width=1)
                panel.paste(tile, ((pol_index * 3 + branch) * cell_w + 8, y + 38))
    panel.save(HERE / 'region_footprints.png')


def main():
    start = time.perf_counter()
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64 * 1024**2)
    targets = pd.read_csv(OLD / 'ports24/targets.csv')
    edges = pd.read_csv(OLD / 'ports24/neighbors.csv')
    old = np.load(OLD / 'ports24/features.npz')
    mask = old['mask'].copy()
    assert mask.shape == (198, 16) and int(mask.sum()) == 85 and int(mask.any(1).sum()) == 36
    assert len(targets) == 198 and len(edges) == 85 and targets.instance_id.is_unique
    expected_ids = np.repeat(targets.instance_id.to_numpy(), mask.sum(1))
    assert np.array_equal(expected_ids, edges.instance_id.to_numpy()), 'Old edge/target order differs'
    assert np.array_equal(targets.neighbors.to_numpy(), mask.sum(1))
    slots = list(zip(*np.nonzero(mask)))
    result = {name: np.zeros((198, 16, 256), np.float32) for name in ('whole', 'fg', 'bg')}
    valid = {name: np.zeros((198, 16), bool) for name in ('fg', 'bg')}
    edge_index = np.full((198, 16), -1, np.int32)
    reader = Reader()
    reader.stem.requires_grad_(False)
    datasets, cache, stats, cases = {}, {}, [], []
    # Fixed cases, no seed or prediction-based re-selection in this extraction.
    selected = []
    for instance, detection in [('pair_0069192', 'det_01443'), ('pair_0069192', 'det_00273'), ('pair_0069233', None), ('pair_0044888', None)]:
        match = edges.instance_id.eq(instance)
        if detection:
            match &= edges.detection_id.eq(detection)
        selected.append(int(edges.index[match][0]))
    for e, ((i, j), edge) in enumerate(zip(slots, edges.to_dict('records'))):
        target = targets.iloc[i]
        assert target.instance_id == edge['instance_id'] and target['product'] == edge['product']
        window = json.loads(edge['read_window'])
        points = np.asarray(json.loads(edge['source_points']))
        key = (target.raster_vv, target.raster_vh, tuple(window), edge['source_points'])
        if key not in cache:
            ims = []
            for path in (target.raster_vv, target.raster_vh):
                if path not in datasets:
                    datasets[path] = gdal.Open(path)
                im = datasets[path].ReadAsArray(*window)
                assert im.shape == (window[3], window[2]) and np.isfinite(im).all()
                ims.append(im)
            masks = region_masks(points, window)
            encoded = [encode_regions(reader, im, masks) for im in ims]
            vectors = {name: np.concatenate([p[0][name] for p in encoded]) for name in result}
            cache[key] = (vectors, masks, ims, encoded[0][1])
        vectors, masks, ims, weights = cache[key]
        for name in result:
            result[name][i, j] = vectors[name]
        for name, m in zip(('fg', 'bg'), masks):
            valid[name][i, j] = bool(m.any())
        edge_index[i, j] = e
        row = dict(edge_index=e, target_index=int(i), slot=int(j), **edge,
                   raster_vv=target.raster_vv, raster_vh=target.raster_vh,
                   fg_pixels=int(masks[0].sum()), bg_pixels=int(masks[1].sum()),
                   guard_pixels=int((~masks[0] & ~masks[1]).sum()),
                   fg_fraction=float(masks[0].mean()), bg_fraction=float(masks[1].mean()),
                   fg_valid=bool(valid['fg'][i,j]), bg_valid=bool(valid['bg'][i,j]),
                   fg_empty=not bool(valid['fg'][i,j]), bg_empty=not bool(valid['bg'][i,j]),
                   fg_weight_sum=float(weights[0].sum()), bg_weight_sum=float(weights[1].sum()),
                   whole_norm=float(np.linalg.norm(vectors['whole'])), fg_norm=float(np.linalg.norm(vectors['fg'])),
                   bg_norm=float(np.linalg.norm(vectors['bg'])),
                   whole_fg_cos=cosine(vectors['whole'], vectors['fg']), whole_bg_cos=cosine(vectors['whole'], vectors['bg']),
                   replay_max_abs=float(np.abs(vectors['whole'] - old['neighbor'][i,j]).max()))
        stats.append(row)
        if e in selected:
            cases.append((row, ims, masks, weights))
    error = float(np.abs(result['whole'] - old['neighbor']).max())
    assert error <= 1e-5, f'Whole replay failed: {error}'
    for name, values in result.items():
        assert np.isfinite(values).all() and not values[~mask].any()
        if name in valid:
            assert not values[mask & ~valid[name]].any()
    # Empty masks must yield exact zeros; all-one pooling must reproduce whole.
    z = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4).repeat(1, 32, 1, 1)
    assert not masked_pool(z, torch.zeros(1, 1, 3, 4)).any()
    assert torch.equal(masked_pool(z, torch.ones(1, 1, 3, 4)), torch.cat([z.mean((-2,-1)), z.amax((-2,-1))], 1))
    np.savez_compressed(HERE / 'regions.npz', **result, mask=mask,
                        instance_ids=targets.instance_id.to_numpy(dtype=str),
                        fg_valid=valid['fg'], bg_valid=valid['bg'],
                        fg_empty=mask & ~valid['fg'], bg_empty=mask & ~valid['bg'], edge_index=edge_index)
    frame = pd.DataFrame(stats)
    frame.to_csv(HERE / 'region_stats.csv', index=False)
    cases.sort(key=lambda c: selected.index(c[0]['edge_index']))
    make_panel(cases)
    with np.load(HERE / 'regions.npz') as saved:
        assert np.array_equal(saved['mask'], mask)
        assert np.array_equal(saved['instance_ids'], targets.instance_id.to_numpy(dtype=str))
        assert all(np.array_equal(saved[k], v) for k, v in result.items())
    summary = dict(targets=198, edges=85, context_targets=36, unique_windows=len(cache),
                   shape=[198,16,256], whole_replay_max_abs=error,
                   whole_replay_exact=bool(np.array_equal(result['whole'], old['neighbor'])),
                   fg_empty=int((mask & ~valid['fg']).sum()), bg_empty=int((mask & ~valid['bg']).sum()),
                   all_finite=True, node_mask_preserved=True, source_edge_order_preserved=True,
                   empty_pool_zero_check=True, full_pool_check=True, saved_array_roundtrip=True,
                   panel_edges=selected, seconds=time.perf_counter()-start,
                   dilation='square/Chebyshev radius 3 native pixels',
                   limitation='conv1 7x7 RF mixes boundaries; not pure physical separation')
    (HERE / 'readout_checks.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(frame.loc[frame.edge_index.isin(selected), ['edge_index','instance_id','detection_id','fg_fraction','bg_fraction','whole_norm','fg_norm','bg_norm','whole_fg_cos','whole_bg_cos']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
