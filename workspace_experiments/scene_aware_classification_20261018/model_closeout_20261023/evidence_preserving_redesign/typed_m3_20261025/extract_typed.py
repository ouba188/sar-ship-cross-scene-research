"""Bounded frozen typed M3 descriptors; no fitting, convolution update, or training."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'native_validation'
BASE = HERE.parents[1]
OLD = BASE / 'm2_operator_validation'
UP = BASE / 'regional_utility_pilot/upstream'
PROJECTION = BASE / 'local_evidence_transport/projection_layer1.npz'
CACHE = NATIVE / 'native_cache'
BATCH = 8
QUERY_N = 1741
N = 1805


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_module('typed_m3_prepare', OLD / 'prepare.py')
ops = load_module('typed_m3_moments', HERE.parent / 'operators.py')
oldops = load_module('typed_m3_direct', OLD / 'operators.py')
geo = prep.geo


def update_max(checks, key, value):
    checks[key] = max(checks.get(key, 0.), float(value))


def placed_pixels(table, i, p, placement):
    with Image.open(table.iloc[i]['image_' + geo.POL[p]]) as image:
        raw = np.asarray(image.convert('L'))
    sx, sy, width, height, dx, dy = map(int, placement[i, p])
    out = np.zeros((160, 160), np.uint8)
    out[dy:dy+height, dx:dx+width] = raw[sy:sy+height, sx:sx+width]
    return out


class ReferenceEncoder:
    def __init__(self):
        with np.load(PROJECTION) as z:
            self.mean = torch.tensor(z['mean'], device='cuda')
            self.components = torch.tensor(z['components'], device='cuda')
        assert self.mean.shape == (256,) and self.components.shape == (32, 256)
        with torch.device('meta'):
            self.model = resnet50(weights=None)
        self.model.load_state_dict(torch.load(geo.WEIGHTS, map_location='cuda', weights_only=True), assign=True)
        self.model.eval().requires_grad_(False)
        self.normmean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        self.normstd = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]

    def batch(self, table, ids, pol_valid, placement, bands, checks):
        result = np.zeros((len(ids), 2, 4, 32, 40, 40), np.float16)
        images, addresses = [], []
        for j, i in enumerate(ids):
            for p in range(2):
                if not pol_valid[i, p]:
                    continue
                pix = placed_pixels(table, i, p, placement)
                error = np.abs(bands[j, p].sum(0, dtype=np.float64)-pix.astype(np.float64)/255.).max()
                update_max(checks, 'reference_band_reconstruction_max_abs', error)
                assert error < 2e-7, (i, p, error)
                images.append(pix[None].astype(np.float32)/255.)
                addresses.append((j, p))
        if not images:
            return result
        pixels = torch.tensor(np.stack(images), device='cuda')
        for k in range(4):
            rotated = pixels.rot90(k, (-2, -1)).expand(-1, 3, -1, -1)
            model = self.model
            feature = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(
                (rotated-self.normmean)/self.normstd)))))
            projected = torch.einsum('dc,bcu->bdu', self.components,
                                     feature.flatten(2)-self.mean[None, :, None])
            quantized = projected.reshape(-1, 32, 40, 40).cpu().numpy().astype(np.float16)
            assert np.isfinite(quantized).all()
            for b, (j, p) in enumerate(addresses):
                result[j, p, k] = quantized[b]
        return result


def describe_pol(maps, bands, assignment, regions, flags, p, reader, check, checks):
    """Use precisely the same native pixel weights for every descriptor field."""
    desc = np.zeros((3, 3, 8, 91), np.float64)
    mass = np.zeros((3, 3, 8), np.float64)
    for r, slot_ids in enumerate(prep.ROLE_SLOTS):
        if not flags[r]:
            continue
        slot_mass = assignment[slot_ids].sum((1, 2))
        mass[r, :, :len(slot_ids)] = slot_mass
    tid = slice(p*3, p*3+3)
    for k in range(4):
        a = np.rot90(assignment, k, (-2, -1))
        region = np.rot90(regions, k, (-2, -1))
        band = np.rot90(bands, k, (-2, -1))
        # Sample only pixels carrying valid role mass, at the actual stride-4 centers.
        yy, xx = np.nonzero(region[flags].sum(0) > 0)
        grid = np.stack((2*(xx/4)/39-1, 2*(yy/4)/39-1), -1).astype(np.float32)
        grid = torch.tensor(grid, device='cuda')[None, None]
        coarse = torch.tensor(maps[k].astype(np.float32), device='cuda')[None]
        content = F.grid_sample(coarse, grid, mode='bilinear', padding_mode='border',
                                align_corners=True)[0, :, 0]
        dog = torch.tensor(np.ascontiguousarray(band[:, yy, xx]), device='cuda')
        x = torch.cat((content[None, :7].expand(3, -1, -1), dog[:, None]), 1)
        standardized = (x-reader['mu'][tid, :, None])/reader['scale'][tid, :, None]
        response = torch.einsum('bcd,bdl->bcl', reader['W'][tid], standardized)
        response += reader['bias'][tid, :, None]
        for r, slot_ids in enumerate(prep.ROLE_SLOTS):
            if not flags[r]:
                continue
            keep = region[r, yy, xx] > 0
            weights = torch.tensor(np.ascontiguousarray(a[slot_ids][:, yy[keep], xx[keep]]),
                                   device='cuda', dtype=torch.float64)
            c = content[:, keep].double()
            d = dog[:, keep].double()
            s = response[:, :, keep].double()
            masses = weights.sum(-1)
            normalized = weights / masses[:, None]
            pca_mean = normalized @ c.T
            dog_mean = torch.einsum('sl,bl->bs', normalized, d)
            dog_abs = torch.einsum('sl,bl->bs', normalized, d.abs())
            dog_square = torch.einsum('sl,bl->bs', normalized, d.square())
            cross = torch.einsum('sl,bl,cl->bsc', normalized, d, c)
            covariance = cross-dog_mean[:, :, None]*pca_mean[None]
            stats = ops.moment_pool(s, weights[None].expand(3, -1, -1), reader['tau'][tid].double())
            class_moments = ops.decode_moments(stats)
            value = torch.cat((pca_mean[None].expand(3, -1, -1), class_moments.flatten(2),
                               dog_mean[:, :, None], dog_abs[:, :, None], dog_square[:, :, None],
                               covariance), -1)
            desc[r, :, :len(slot_ids)] += value.cpu().numpy()/4
            if check:
                direct_weight = torch.tensor(region[r, yy[keep], xx[keep]], device='cuda',
                                             dtype=torch.float64)[None, None].expand(3, -1, -1)
                direct = oldops.class_summary(s, direct_weight, reader['tau'][tid].double())
                error = (ops.decode_moments(stats, merge=True)-direct).abs().max().item()
                update_max(checks, 'source4_slot_merge_direct_class_max_abs', error)
                assert error < 2e-8, ('class merge', r, k, error)
                partition = (weights.sum(0)-direct_weight[0, 0]).abs().max().item()
                update_max(checks, 'source4_pixel_partition_max_abs', partition)
                assert partition < 1e-10
                q = masses/masses.sum()
                whole = direct_weight[0, 0]/direct_weight[0, 0].sum()
                cm = c @ whole
                dm = d @ whole
                direct_cov = torch.einsum('l,bl,cl->bc', whole, d, c)-dm[:, None]*cm[None]
                merged_cov = torch.einsum('s,bsc->bc', q, covariance+dog_mean[:, :, None]*pca_mean[None])
                merged_cov -= (dog_mean @ q)[:, None]*(q @ pca_mean)[None]
                error = max((q @ pca_mean-cm).abs().max().item(),
                            (dog_mean @ q-dm).abs().max().item(),
                            (dog_abs @ q-d.abs() @ whole).abs().max().item(),
                            (dog_square @ q-d.square() @ whole).abs().max().item(),
                            (merged_cov-direct_cov).abs().max().item())
                update_max(checks, 'source4_merged_linear_cross_max_abs', error)
                assert error < 2e-8, ('linear/cross merge', r, k, error)
                error = np.abs(masses.cpu().numpy()-mass[r, 0, :len(slot_ids)]).max()
                update_max(checks, 'source4_c4_slot_mass_max_abs', error)
                assert error < 1e-8
                checks['source4_role_rotation_checks'] += 1
    return desc.reshape(9, 8, 91).astype(np.float32), mass.reshape(9, 8).astype(np.float32)


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    HERE.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                   batch_objects=BATCH, threads=2, no_fitting=True, no_training=True, no_git=True)
    receipt_path = HERE / 'extraction_receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    table = pd.read_csv(NATIVE / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    upstream = pd.read_csv(UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    query = pd.read_csv(OLD / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    assert len(table) == N and len(query) == QUERY_N
    up = table.up_index.to_numpy(int)
    assert len(np.unique(up)) == N
    pd.testing.assert_frame_equal(table[upstream.columns].reset_index(drop=True),
                                  upstream.iloc[up].reset_index(drop=True))
    pd.testing.assert_frame_equal(table.iloc[:QUERY_N][query.columns].reset_index(drop=True), query)
    with np.load(PROJECTION) as z:
        bank = z['bank'].copy()
    np.testing.assert_array_equal(up[QUERY_N:], bank.ravel())
    assert table.role.iloc[QUERY_N:].eq('strategy_fit').all()
    with np.load(CACHE / 'metadata.npz') as z:
        np.testing.assert_array_equal(z['up_index'], up)
        np.testing.assert_array_equal(z['bank_up_index'], bank)
        placement = z['placement'].copy()
    with np.load(UP / 'summaries.npz') as z:
        flags = z['region_valid'][up].astype(bool)
        pol_valid = z['valid'][up].astype(bool)
        np.testing.assert_array_equal(z['indices'][up], table.row_index)
    with np.load(OLD / 'metadata.npz') as z:
        np.testing.assert_array_equal(z['up_index'], up[:QUERY_N])
        np.testing.assert_array_equal(z['region_valid'], flags[:QUERY_N])
    np.testing.assert_array_equal(np.load(CACHE / 'pol_valid.npy'), pol_valid)
    geometry = np.load(CACHE / 'geometry.npy')
    geometry_valid = np.load(CACHE / 'geometry_valid.npy')
    assert not flags[~pol_valid].any() and not flags[~geometry_valid].any()
    with np.load(NATIVE / 'reader.npz') as z:
        reader = {k: torch.tensor(z[k], dtype=torch.float32, device='cuda') for k in z.files}
    assert reader['W'].shape == (6, 8, 8) and (reader['scale'] > 0).all()
    candidates = np.flatnonzero(table.role.eq('utility_train').to_numpy() & flags.all((1, 2)))
    source4 = set(map(int, candidates[:4]))
    assert len(source4) == 4
    checks = dict(identity_correspondence=True, source4_role_rotation_checks=0,
                  source4_cases=[dict(index=i, up_index=int(up[i]), obs_id=int(table.iloc[i].obs_id))
                                 for i in sorted(source4)])
    yy, xx = torch.meshgrid(torch.arange(160, device='cuda'), torch.arange(160, device='cuda'), indexing='ij')
    grid = torch.stack((2*(xx/4)/39-1, 2*(yy/4)/39-1), -1)[None]
    ramp = torch.arange(40, device='cuda', dtype=torch.float32)[None, None, None].expand(1, 1, 40, 40)
    sampled = F.grid_sample(ramp, grid, align_corners=True, padding_mode='border')
    checks['stride4_origin0_center_max_abs'] = float((sampled[0, 0, ::4, ::4]-ramp[0, 0]).abs().max())
    assert checks['stride4_origin0_center_max_abs'] < 1e-5
    desc = np.zeros((N, 18, 8, 91), np.float32)
    mass = np.zeros((N, 18, 8), np.float32)
    encoder = None
    last = tick
    timing = dict(reference_encoding=0., descriptors=0.)
    with torch.inference_mode():
        # Split at the query/reference boundary so no batch mixes cache lineages.
        batches = [(first, min(first+BATCH, end)) for start, end in ((0, QUERY_N), (QUERY_N, N))
                   for first in range(start, end, BATCH)]
        for first, stop in batches:
            ids = np.arange(first, stop)
            bmap = np.load(CACHE / 'native_bands.npy', mmap_mode='r')
            vmap = np.load(CACHE / 'valid_pixels.npy', mmap_mode='r')
            bands = np.array(bmap[first:stop])
            footprints = np.array(vmap[first:stop])
            bmap._mmap.close()
            vmap._mmap.close()
            assert bands.shape == (len(ids), 2, 3, 160, 160)
            assert np.isfinite(bands).all() and not bands[~pol_valid[ids]].any()
            if first < QUERY_N:
                maps = np.load(OLD / 'x.npy', mmap_mode='r')
                assert maps.shape == (QUERY_N, 2, 4, 32, 40, 40) and maps.dtype == np.float16
                content = np.array(maps[first:stop])
                maps._mmap.close()
            else:
                stamp = time.perf_counter()
                if encoder is None:
                    encoder = ReferenceEncoder()
                content = encoder.batch(table, ids, pol_valid, placement, bands, checks)
                timing['reference_encoding'] += time.perf_counter()-stamp
            assert np.isfinite(content).all() and not content[~pol_valid[ids]].any()
            stamp = time.perf_counter()
            for j, i in enumerate(ids):
                for p in range(2):
                    if not flags[i, p].any():
                        continue
                    a, regions = prep.slots(geometry[i, p], footprints[j, p])
                    if i in source4:
                        pixels = placed_pixels(table, i, p, placement)
                        error = np.abs(bands[j, p].sum(0, dtype=np.float64)-pixels.astype(np.float64)/255.).max()
                        update_max(checks, 'source4_band_reconstruction_max_abs', error)
                        assert error < 2e-7
                    dd, mm = describe_pol(content[j, p], bands[j, p], a, regions,
                                          flags[i, p], p, reader, i in source4, checks)
                    desc[i, p*9:(p+1)*9] = dd
                    mass[i, p*9:(p+1)*9] = mm
            timing['descriptors'] += time.perf_counter()-stamp
            assert np.isfinite(desc[first:stop]).all() and np.isfinite(mass[first:stop]).all()
            if time.perf_counter()-last >= 20 or stop in (QUERY_N, N):
                receipt.update(processed=stop, seconds=time.perf_counter()-tick, checks=checks)
                receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
                print('EXTRACT', stop, '/', N, 'seconds', round(time.perf_counter()-tick, 1),
                      'gpu_peak_MB', round(torch.cuda.max_memory_allocated()/1e6, 1), flush=True)
                last = time.perf_counter()
    type_pol = np.repeat(np.arange(2), 9)
    type_region = np.tile(np.repeat(np.arange(3), 3), 2)
    type_band = np.tile(np.arange(3), 6)
    type_valid = flags[:, type_pol, type_region]
    slot_count = np.where(type_region == 1, 4, 8)
    slot_valid = type_valid[:, :, None] & (np.arange(8)[None, None] < slot_count[None, :, None])
    assert not desc[~slot_valid].any() and not mass[~slot_valid].any()
    assert (mass[slot_valid] > 0).all()
    assert checks['source4_role_rotation_checks'] == 4*2*3*4
    checks.update(finite=True, invalid_type_exact_zero=True, padded_end_slots_exact_zero=True,
                  valid_slot_positive_mass=True, missing_pol_exact_zero=True)
    np.savez_compressed(HERE / 'features.npz', desc=desc, mass=mass,
                        up_index=up, obs_id=table.obs_id.to_numpy(), old_index=table.old_index.to_numpy(),
                        instance_ids=table.instance_id.to_numpy(dtype=str),
                        type_id=np.arange(18), type_pol=type_pol, type_region=type_region, type_band=type_band,
                        slot_count=slot_count, type_valid=type_valid, region_valid=flags, pol_valid=pol_valid,
                        pol_names=np.array(geo.POL), region_names=np.array(geo.REGIONS),
                        band_names=np.array(['fine', 'middle', 'low']),
                        statistic_names=np.array(['mu', 'h', 'b']),
                        bank_up_index=bank, bank_cache_index=np.arange(QUERY_N, N).reshape(8, 8))
    coverage = {}
    for role, frame in table.groupby('role', sort=False):
        ix = frame.index.to_numpy()
        coverage[role] = dict(n=len(ix), ships=int(frame.mmsi.nunique()), ports=int(frame.port.nunique()),
                              valid_pol=pol_valid[ix].sum(0).tolist(), valid_regions=flags[ix].sum(0).tolist(),
                              valid_types=type_valid[ix].sum(0).tolist())
    receipt.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                   seconds=time.perf_counter()-tick, timing=timing, n=N, query_n=QUERY_N, reference_n=64,
                   shapes=dict(desc=list(desc.shape), mass=list(mass.shape)), dtype='float32', coverage=coverage,
                   checks=checks, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                   output_bytes=(HERE / 'features.npz').stat().st_size,
                   descriptor_fields={'0:32': 'native-interpolated PCA32 weighted mean',
                                      '32:56': '8 classes x (mu,h,b), class-major flatten',
                                      '56:59': 'raw signed native band mean, abs mean, square mean',
                                      '59:91': 'population weighted Cov(raw native band,PCA32)'},
                   type_order='polarization(2), region(body/end/side), band(fine/middle/low)',
                   slot_order='original prepare.ROLE_SLOTS within each region; end slots4:8 are zero padding',
                   class_projection='float32 frozen source reader: standardized PCA[:7]+native band; W/bias; no new convolution',
                   moments='moment_pool/decode_moments float64; response centered over the WHOLE role, then across classes',
                   c4='each rotated R50/PCA view independently interpolated and pooled; nonlinear and cross moments calculated BEFORE C4 averaging',
                   content_grid='stride4 origin0; bilinear grid_sample border padding align_corners=True; native160 crop/pad, no ship resize',
                   reference_encoding='same frozen R50 layer1/source PCA pipeline as query cache; float16 quantization before float32 interpolation',
                   masks='same native-pixel prepare.slots weights for all91 fields; validity copied from upstream summaries for all1805',
                   sources=dict(table=str(NATIVE / 'table.csv'), native_cache=str(CACHE), query_maps=str(OLD / 'x.npy'),
                                reader=str(NATIVE / 'reader.npz'), projection=str(PROJECTION), backbone=str(geo.WEIGHTS)),
                   limitations=['Extraction only: no accuracy, new scaler, learned metric, or training result.',
                                'PCA fields are bilinear interpolation of stride4 features, not native-resolution backbone measurements.',
                                'Low band is Gaussian lowpass; fine/middle are signed differences. No raw-band scale normalization.',
                                'C4 averaging discards between-view variability after each view has preserved its nonlinear moments.',
                                'Source4 checks are deterministic bounded algebra checks, not generalization evidence.',
                                'Existing source-trained PCA/readers and upstream geometry/validity are reused without refitting.'],
                   raw_dataset_models_features_not_for_git=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('status', 'seconds', 'shapes', 'checks', 'output_bytes')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
