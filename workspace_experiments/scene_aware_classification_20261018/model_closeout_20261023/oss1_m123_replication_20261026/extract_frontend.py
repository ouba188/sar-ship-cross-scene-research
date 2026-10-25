"""OSS1 native160 C4 frontend and pre-PCA M3 moments, without historical assets.

Run with the existing ClearSAR Python. All writes stay under this directory.
The role table determines observation order; extraction never uses class labels.
M3 slot order: body(8), side(8), end(4). M2 order: body, end, side.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE / 'sar_m123_standalone'))
from sar_m123 import baseline, frequency, regions, tokens
from sar_m123.backbone import Backbone
from sar_m123.input import prepare
from sar_m123.m12 import attention, centered_response

WEIGHTS = Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')
DEFAULT_TABLE = HERE / 'stack_table.csv'
PLACEMENT = ('crop_x', 'crop_y', 'crop_width', 'crop_height', 'pad_left', 'pad_top')


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding='utf-8')


def read_table(path, fold=None):
    table = pd.read_csv(path, dtype={'mmsi': str, 'sample_id': str})
    if fold is not None and 'fold' in table:
        table = table[table.fold.astype(str).eq(str(fold))]
    if not table.obs_id.is_unique:
        raise ValueError('Select one fold: role table must contain one row per obs_id.')
    if len(table) == 0:
        raise ValueError('Empty role table.')
    return table.reset_index(drop=True)


def prepare_inventory(table):
    views = pd.read_csv(HERE / 'view_inventory.csv').set_index(['obs_id', 'pol'])
    radar = pd.read_csv(WORKSPACE / 'oss1_joint_geometry_20261017/geometry.csv',
                        usecols=['sample_id', 'radar_axis_valid', 'radar_axis_rad']).set_index('sample_id')
    records = []
    for row in table.itertuples():
        record = {}
        for pol in ('vv', 'vh'):
            v = views.loc[(row.obs_id, pol)]
            if not bool(v['exists']):
                record[pol] = None
                continue
            detections = []
            axis = None
            if bool(v.detected) and float(v.confidence) >= .25:
                cx, cy, length, width, angle = json.loads(v.raw_xywhr)
                uv = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length/2, width/2]
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                corners = uv @ rotation.T + [cx, cy]
                detections = [dict(corners=corners.tolist(), confidence=float(v.confidence))]
                if bool(v.axis_valid):
                    axis = dict(center=[float(v.cx), float(v.cy)], length_px=float(v.length),
                                width_px=float(v.width), angle=float(v.ship_axis))
            r = radar.loc[row.sample_id]
            record[pol] = dict(image=str(v.path), detections=detections,
                               target_index=0 if detections else None, ship_axis=axis,
                               radar_axis_rad=float(r.radar_axis_rad) if bool(r.radar_axis_valid) else None)
        records.append(record)
    return records


def backbone(device, weights=WEIGHTS):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return Backbone(weights, device)


def global_weight(fp, layer, k):
    f = np.rot90(fp, k)
    return (regions.cell_mass(f) if layer == 'layer1' else
            f.reshape(10, 16, 10, 16).mean((1, 3))).reshape(-1)


@torch.inference_mode()
def extract_one(prepared, maps, net):
    out = baseline.extract(prepared, maps, net)
    out.update(layer1=np.zeros((2, 20, 256), np.float32),
               joint_raw=np.zeros((2, 20, 3, 256), np.float32),
               mass=np.zeros((2, 20), np.float32),
               e_image=np.zeros((2, 20, 24), np.float32),
               coverage=np.zeros((2, 20, 3), np.float32),
               global1=np.zeros((2, 256)), global3=np.zeros((2, 1024)),
               valid=np.zeros(2, bool), token_valid=np.zeros(2, bool),
               footprint=np.zeros((2, 160, 160), np.uint8),
               obb_geometry=np.full((2, 6), np.nan),
               placement=np.zeros((2, 6), np.int32),
               native_shape=np.zeros((2, 2), np.int32),
               geometry=np.zeros((2, 4)), quality=np.zeros((2, 3, 4)),
               region_valid=np.zeros((2, 3), bool))
    for p, view in enumerate(prepared):
        if view is None:
            continue
        fp, g = view['footprint'], view['geometry']
        out['valid'][p] = bool(view['valid_target'])
        out['footprint'][p] = fp
        out['placement'][p] = [view['placement'][key] for key in PLACEMENT]
        out['native_shape'][p] = view['raw'].shape
        for layer, key in (('layer1', 'global1'), ('layer3', 'global3')):
            if not out['valid'][p]:
                continue
            for k in range(4):
                w = global_weight(fp, layer, k)
                z = maps[p][layer][k].flatten(1).double()
                weight = torch.as_tensor(w/w.sum(), device=z.device, dtype=torch.float64)
                out[key][p] += (z @ weight).cpu().numpy()/4
        if g is None or not out['valid'][p]:
            continue
        out['token_valid'][p] = True
        out['obb_geometry'][p] = g
        out['geometry'][p] = np.r_[g[2:4], g[5], 1.]
        q = regions.region_quality(regions.soft_regions(g), fp, g)
        flags = (q['area'] > 0) & (q['min_neff'] >= 4)
        out['quality'][p] = np.stack([q['area'], q['coverage'], q['min_neff'], flags], -1)
        out['region_valid'][p] = flags
        masks = tokens.slots(g, fp)
        spec = frequency.extract(view['raw'], masks, view['placement'])
        out['mass'][p] = masks.sum((-2, -1))
        out['e_image'][p], out['coverage'][p] = spec['e_image'], spec['coverage']
        avg, joint = np.zeros((20, 256)), np.zeros((20, 3, 256))
        for k in range(4):
            rotated = np.rot90(masks, k, (-2, -1))
            w = tokens.cell_mass(rotated).reshape(20, -1)
            energy = np.rot90(spec['energy_field'], k, (-2, -1))
            ew = tokens.cell_mass(rotated[:, None]*energy[None]).reshape(20, 3, -1)
            z = maps[p]['layer1'][k].flatten(1).T.cpu().numpy().astype(np.float64)
            jj, mm = tokens.grid_moments(z, w, ew)
            avg += mm/4
            joint += jj/4
        out['layer1'][p], out['joint_raw'][p] = avg, joint
    return out


def extract_raw(table, outdir, device='cuda', weights=WEIGHTS):
    tick = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    sharddir = outdir / 'raw_shards'
    sharddir.mkdir(exist_ok=True)
    records, ids = prepare_inventory(table), table.obs_id.to_numpy(np.int64)
    net = backbone(device, weights)
    chunks = []
    for first in range(0, len(table), 16):
        path = sharddir / f'{first:06d}.npz'
        expected = ids[first:first+16]
        if path.exists():
            with np.load(path) as z:
                if not np.array_equal(z['obs_id'], expected):
                    raise ValueError(f'Observation order changed in {path}; use a different cache directory.')
                chunk = {key: z[key] for key in z.files}
        else:
            values = []
            for record in records[first:first+16]:
                prepared = prepare(record)
                maps = net(prepared)
                values.append(extract_one(prepared, maps, net))
            chunk = {key: np.stack([v[key] for v in values]) for key in values[0]}
            chunk['obs_id'] = expected
            np.savez_compressed(path, **chunk)
        chunks.append(chunk)
        print('raw', min(first+16, len(table)), '/', len(table), 'seconds', round(time.perf_counter()-tick, 1), flush=True)
    all_data = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    # Early extraction shards retained image-global descriptors for no-OBB views.
    # Target-only readers must use the same mask as standalone ShipClassifier.
    all_data['valid'] = all_data['token_valid'].copy()
    for key in ('global1', 'global3'):
        all_data[key][~all_data['valid']] = 0
    m3keys = ('obs_id', 'layer1', 'joint_raw', 'mass', 'e_image', 'coverage', 'token_valid')
    m3 = {key: all_data[key] for key in m3keys}
    m3['valid'] = all_data['token_valid']
    m3['sample_id'] = table.sample_id.to_numpy(str)
    m3['slot_region'] = np.array(['body']*8 + ['side']*8 + ['end']*4)
    np.savez_compressed(outdir / 'm3_raw.npz', **m3)
    front = {key: value for key, value in all_data.items() if key not in m3keys or key == 'obs_id'}
    front['sample_id'] = table.sample_id.to_numpy(str)
    np.savez_compressed(outdir / 'frontend_raw.npz', **front)
    save_json(outdir / 'raw_complete.json', dict(n=len(table), seconds=time.perf_counter()-tick,
        weights=str(weights), class_labels_used=False, pretrained='ImageNet R50 V2',
        native160=True, resizing=False, c4='four separate forward passes',
        large_maps_saved=False, target_reader_validity='valid_target; no-OBB global features zero',
        invalid_target_polarizations=int((~all_data['valid']).sum()),
        m3='raw256 covariance then linear PCA; source PCA is owned by M3',
        placement_fields=PLACEMENT, token_region_order=['body', 'side', 'end']))
    return all_data


@torch.inference_mode()
def extract_summaries(table, tr, cache, models, outdir, device='cuda', weights=WEIGHTS):
    """Reader-dependent second forward; compact class responses preserve float64."""
    tick = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    response_dir = outdir / 'response_shards'
    response_dir.mkdir(exist_ok=True)
    with np.load(cache / 'frontend_raw.npz') as z:
        index = pd.Index(z['obs_id']).get_indexer(table.obs_id)
        if (index < 0).any():
            raise ValueError('Role table contains observations absent from the cache.')
        data = {k: z[k][index] for k in ('footprint', 'obb_geometry', 'valid', 'quality', 'geometry', 'region_valid')}
    readers = {}
    for layer, dim in (('layer1', 256), ('layer3', 1024)):
        with np.load(models / f'reader_{layer}.npz') as z:
            readers[layer] = z['W'].reshape(4, 2, dim).transpose(1, 0, 2).copy()
    net, records = backbone(device, weights), prepare_inventory(table)
    n = len(table)
    sig = {key: np.zeros((n, 2, 4)) for key in readers}
    for i, record in enumerate(records):
        path = response_dir / f'{int(table.iloc[i].obs_id)}.npz'
        response = {key: np.zeros((2, 4, 4, side*side)) for key, side in (('layer1', 40), ('layer3', 10))}
        prepared = prepare(record)
        # Always recompute reader-dependent responses; a rerun may have new fit rows.
        maps = net(prepared)
        for p in range(2):
            if not data['valid'][i, p]:
                continue
            fp = data['footprint'][i, p]
            for layer in readers:
                wreader = torch.as_tensor(readers[layer][p], device=net.device)
                z = maps[p][layer].flatten(2).double()
                resp = torch.einsum('cd,kdu->kcu', wreader, z)
                response[layer][p] = resp.cpu().numpy()
                for k in range(4):
                    w = global_weight(fp, layer, k)
                    norm = torch.as_tensor((w/w.sum())[None, None], device=net.device, dtype=torch.float64)
                    _, centered = centered_response(resp[k:k+1], norm)
                    sig[layer][i, p] += (centered.square()*norm[:, :, None]).sum(-1).clamp_min(0).sqrt()[0, 0].cpu().numpy()/4
        np.savez_compressed(path, **response)
        if (i+1) % 16 == 0 or i+1 == n:
            print('reader responses', i+1, '/', n, 'seconds', round(time.perf_counter()-tick, 1), flush=True)
    tau = {layer: [float(np.median(sig[layer][tr[data['valid'][tr, p]], p])) for p in range(2)] for layer in readers}
    if not all(np.isfinite(v) and v > 0 for values in tau.values() for v in values):
        raise ValueError('Source-fit response temperature is empty or degenerate.')
    save_json(outdir / 'scales.json', dict(tau=tau, fit_obs_id=table.iloc[tr].obs_id.tolist()))
    np.savez_compressed(outdir / 'scale_sigma.npz', obs_id=table.iloc[tr].obs_id.to_numpy(),
                        **{k: v[tr] for k, v in sig.items()})
    out = dict(mean3=np.zeros((n, 2, 4)), hb3=np.zeros((n, 2, 4, 2)),
               summary1=np.zeros((n, 2, 3, 4, 3)),
               **{k: data[k] for k in ('quality', 'geometry', 'region_valid', 'valid')})
    for i in range(n):
        with np.load(response_dir / f'{int(table.iloc[i].obs_id)}.npz') as response:
            for p in range(2):
                if not data['valid'][i, p]:
                    continue
                fp, g = data['footprint'][i, p], data['obb_geometry'][i, p]
                masks = regions.soft_regions(g if np.isfinite(g).all() else None)*fp
                for k in range(4):
                    for layer in readers:
                        r = torch.tensor(response[layer][p, k:k+1], device=device)
                        w = (global_weight(fp, layer, k)[None] if layer == 'layer3' else
                             regions.cell_mass(np.rot90(masks, k, (-2, -1))).reshape(3, -1))
                        weights_t = torch.tensor(w[None], device=device, dtype=torch.float64)
                        packed = attention(r, weights_t, tau[layer][p], recenter=layer == 'layer3')[0].cpu().numpy()
                        if not np.isfinite(packed).all():
                            raise ValueError('Nonfinite attention summary.')
                        if layer == 'layer3':
                            out['mean3'][i, p] += packed[0, :, 0]/4
                            out['hb3'][i, p] += packed[0, :, 1:]/4
                        else:
                            packed[~data['region_valid'][i, p]] = 0
                            out['summary1'][i, p] += packed/4
        if (i+1) % 100 == 0:
            print('summaries', i+1, '/', n, flush=True)
    with np.load(cache / 'frontend_raw.npz') as z:
        direct = np.einsum('npd,pcd->npc', z['global3'][index], readers['layer3'])
    closure = float(np.max(np.abs(direct-out['mean3'])))
    # Cached global3 uses float32 footprint normalization; attention is float64.
    if closure > 1e-5:
        raise ValueError(f'Reader mean closure failed: {closure}')
    np.savez_compressed(outdir / 'summaries.npz', obs_id=table.obs_id.to_numpy(), **out)
    save_json(outdir / 'summaries_complete.json', dict(n=n, seconds=time.perf_counter()-tick,
              reader_mean_closure=closure, reader_mean_closure_tolerance=1e-5,
              temperature_fit_role='fit_train', class_response_dtype='float64'))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, default=DEFAULT_TABLE)
    parser.add_argument('--fold')
    parser.add_argument('--out', type=Path, default=HERE / 'frontend_cache')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--weights', type=Path, default=WEIGHTS)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(HERE):
        raise ValueError('Outputs must remain inside the authorized task directory.')
    extract_raw(read_table(args.table, args.fold), args.out, args.device, args.weights)


if __name__ == '__main__':
    main()
