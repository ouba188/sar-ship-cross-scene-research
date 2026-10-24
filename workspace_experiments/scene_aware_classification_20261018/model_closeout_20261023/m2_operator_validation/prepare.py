"""Extract frozen native C4 maps for M2; no fitting or label-based decisions."""
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
import torch
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
UP = HERE.parent / 'regional_utility_pilot/upstream'
TOKENS = HERE.parent / 'local_evidence_transport'
ROOT = HERE.parents[1] / 'eight_class_evidence_20261023'
spec = importlib.util.spec_from_file_location('m2_geometry', ROOT / 'obb_dynamic_readout/common.py')
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)
sys.path.insert(0, str(UP))
spec = importlib.util.spec_from_file_location('m2_upstream_extraction', UP / 'extract_source.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

ROLE_SLOTS = (np.arange(8), np.arange(16, 20), np.arange(8, 16))
SLOT_REGION = np.r_[np.zeros(8, int), np.full(8, 2), np.ones(4, int)]
SIGN_PERMUTATION = np.r_[np.arange(7, -1, -1), np.arange(15, 7, -1), np.arange(19, 15, -1)]
CENTERS = np.array([(u, v) for u in (-.375, -.125, .125, .375) for v in (-.25, .25)] * 2
                   + [(u, v) for u in (-.625, .625) for v in (-.25, .25)])
SHAPES = dict(x=32, response=8, region=3, a0=20)
DTYPES = dict(x=np.float16, response=np.float32, region=np.float32, a0=np.float32)
BATCH = 8
REPLAY_TOL = 2e-4
MASS_TOL = 2e-5


def slots(g, footprint):
    """Positive soft pixel assignments, normalized separately within each role."""
    region = geo.soft_regions(g) * footprint
    out = np.zeros((20, geo.SIZE, geo.SIZE), np.float64)
    if g is None:
        return out, region
    cx, cy, length, width, angle = g[:5]
    u = ((geo.XX-cx)*np.cos(angle)+(geo.YY-cy)*np.sin(angle))/length
    v = (-(geo.XX-cx)*np.sin(angle)+(geo.YY-cy)*np.cos(angle))/width
    for r, ids in enumerate(ROLE_SLOTS):
        # Evaluate only observed role pixels; zeros outside the role stay exact.
        yy, xx = np.nonzero(region[r] > 0)
        if not len(yy):
            continue
        score = -((u[yy, xx][None]-CENTERS[ids, 0, None])/.35)**2
        score -= ((v[yy, xx][None]-CENTERS[ids, 1, None])/.5)**2
        weight = np.exp(score-score.max(0, keepdims=True))
        weight /= weight.sum(0, keepdims=True)
        assert (weight > 0).all()
        out[ids[:, None], yy[None], xx[None]] = weight*region[r, yy, xx]
    return out, region


def update_max(checks, name, value):
    checks[name] = max(checks.get(name, 0.), float(value))


def store_batch(name, first, values):
    # Closing each mapping releases resident pages instead of accumulating 4 GB.
    target = np.lib.format.open_memmap(HERE / (name+'.npy'), mode='r+')
    target[first:first+len(values)] = values
    target.flush()
    target._mmap.close()


def main():
    tick = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    table = pd.read_csv(UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    assert len(table) == 3387
    selected = table.role.isin(('utility_train', 'selection', 'evaluation'))
    up_index = np.flatnonzero(selected.to_numpy())
    table = table.loc[selected].copy().reset_index(drop=True)
    table.insert(0, 'up_index', up_index)
    counts = table.role.value_counts().to_dict()
    assert counts == dict(utility_train=644, selection=612, evaluation=485), counts
    table.to_csv(HERE / 'table.csv', index=False)
    n = len(table)
    with np.load(UP / 'summaries.npz') as z:
        meta = {key: z[key][up_index] for key in
                ('indices', 'mean3', 'hb3', 'summary1', 'quality', 'region_valid', 'valid')}
        meta['upstream_geometry'] = z['geometry'][up_index]
    np.testing.assert_array_equal(meta['indices'], table.row_index.to_numpy())
    valid = meta['valid'].astype(bool)
    region_valid = meta['region_valid'].astype(bool)
    assert not region_valid[~valid].any()
    wanted = set(table.obs_id)
    snapshots = {s['obs_id']: s for s in json.loads(
        (geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
        if s['obs_id'] in wanted}
    assert len(snapshots) == len(wanted)
    with np.load(TOKENS / 'projection_layer1.npz') as z:
        mean = torch.tensor(z['mean'], device='cuda')
        components = torch.tensor(z['components'], device='cuda')
    assert mean.shape == (256,) and components.shape == (32, 256)
    with np.load(UP / 'models/reader_layer1.npz') as z:
        reader = torch.tensor(z['W'].reshape(8, 2, 256).transpose(1, 0, 2),
                              device='cuda', dtype=torch.float64)
    tau = torch.tensor(json.loads((UP / 'scales.json').read_text())['tau']['layer1'],
                       device='cuda', dtype=torch.float64)
    # Construct on meta to avoid a second resident CPU copy of the backbone.
    with torch.device('meta'):
        model = resnet50(weights=None)
    model.load_state_dict(torch.load(geo.WEIGHTS, map_location='cuda', weights_only=True), assign=True)
    model = model.eval().requires_grad_(False)
    normmean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    normstd = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    shapes = {key: (n, 2, 4, channels, 40, 40) for key, channels in SHAPES.items()}
    for key, shape in shapes.items():
        arr = np.lib.format.open_memmap(HERE / (key+'.npy'), mode='w+', dtype=DTYPES[key], shape=shape)
        arr.flush()
        arr._mmap.close()
    geometry = np.zeros((n, 2, 6), np.float64)
    theta = np.zeros((n, 2, 4), np.float64)
    geometry_valid = np.zeros((n, 2), bool)
    seen = np.zeros((n, 2), bool)
    mass = np.zeros((n, 2, 4, 3), np.float64)
    slot_mass = np.zeros((n, 2, 4, 20), np.float64)
    cycle_valid = np.zeros((n, 2, 4, 3), bool)
    replay = np.zeros_like(meta['summary1'])
    static_mu = np.zeros((n, 2, 3, 8), np.float64)
    checks = dict(tolerances=dict(summary_replay=REPLAY_TOL, cell_rowmass=MASS_TOL,
                                 pixel_rowmass=1e-10, theta_pi=1e-9, integrated_mass=1e-8),
                  missing_pol_all_zero=True, invalid_summary_all_zero=True)
    timing = dict(native_geometry_cpu=0., backbone_projection_response=0., replay=0., write=0.)
    last = tick
    with torch.inference_mode():
        for first in range(0, n, BATCH):
            stop = min(first+BATCH, n)
            bs = stop-first
            buffers = {key: np.zeros((bs, *shape[1:]), dtype=DTYPES[key]) for key, shape in shapes.items()}
            addresses, images = [], []
            stamp = time.perf_counter()
            for i in range(first, stop):
                for p in range(2):
                    if not valid[i, p]:
                        continue
                    pix, fp, g, place = geo.native(table, snapshots, i, p)
                    assert fp.sum() > 0
                    images.append(pix[None].astype(np.float32)/255.)
                    addresses.append((i, p))
                    a, region = slots(g, fp)
                    q = geo.region_quality(geo.soft_regions(g), fp, g)
                    flags = (q['area'] > 0) & (q['min_neff'] >= 4)
                    np.testing.assert_array_equal(flags, region_valid[i, p])
                    quality = np.stack((q['area'], q['coverage'], q['min_neff'], flags), -1)
                    qerror = np.max(np.abs(quality-meta['quality'][i, p]))
                    update_max(checks, 'quality_max_abs_error', qerror)
                    assert qerror < 1e-8
                    if g is not None:
                        geometry[i, p] = g
                        geometry_valid[i, p] = True
                        theta[i, p] = g[4]-np.arange(4)*np.pi/2
                        np.testing.assert_allclose(np.r_[g[2:4], g[5], 1.], meta['upstream_geometry'][i, p], atol=1e-10, rtol=0)
                        flipped = g.copy()
                        flipped[4] += np.pi
                        other, _ = slots(flipped, fp)
                        error = np.max(np.abs(a-other[SIGN_PERMUTATION]))
                        update_max(checks, 'theta_pi_slot_permutation_max_abs_error', error)
                        assert error < 1e-9
                        del other
                    for r, ids in enumerate(ROLE_SLOTS):
                        error = np.max(np.abs(a[ids].sum(0)-region[r]))
                        update_max(checks, 'pixel_rowmass_max_abs_error', error)
                        assert error < 1e-10
                    for k in range(4):
                        r64 = geo.cell_mass(np.rot90(region, k, (-2, -1)))
                        a64 = geo.cell_mass(np.rot90(a, k, (-2, -1)))
                        # Prefix-sum subtraction can leave roundoff below zero.
                        minimum = min(float(r64.min()), float(a64.min()))
                        update_max(checks, 'negative_roundoff_clipped_max', -minimum)
                        assert minimum > -1e-9
                        r32 = np.maximum(r64, 0).astype(np.float32)
                        a32 = np.maximum(a64, 0).astype(np.float32)
                        for r, ids in enumerate(ROLE_SLOTS):
                            rowmass = a32[ids].sum(0, dtype=np.float64)
                            error = np.max(np.abs(rowmass-r32[r]))
                            update_max(checks, 'stored_cell_rowmass_max_abs_error', error)
                            assert error < MASS_TOL
                            total_error = abs(r64[r].sum()-region[r].sum())
                            update_max(checks, 'integrated_region_mass_max_abs_error', total_error)
                            assert total_error < 1e-8
                            positive = (a32[ids].reshape(len(ids), -1) > 0).astype(np.int32)
                            shared = positive @ positive.T
                            np.fill_diagonal(shared, 0)
                            has_cycle = bool((shared >= 2).any())
                            cycle_valid[i, p, k, r] = has_cycle
                            if region_valid[i, p, r]:
                                assert has_cycle and (a32[ids].sum((1, 2), dtype=np.float64) > 0).all(), (i, p, k, r)
                        buffers['region'][i-first, p, k] = r32
                        buffers['a0'][i-first, p, k] = a32
                        mass[i, p, k] = r32.sum((1, 2), dtype=np.float64)
                        slot_mass[i, p, k] = a32.sum((1, 2), dtype=np.float64)
            timing['native_geometry_cpu'] += time.perf_counter()-stamp
            if addresses:
                pixels = torch.tensor(np.stack(images), device='cuda')
                pol = torch.tensor([p for i, p in addresses], device='cuda')
                for k in range(4):
                    stamp = time.perf_counter()
                    rotated = pixels.rot90(k, (-2, -1)).expand(-1, 3, -1, -1)
                    feature = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1((rotated-normmean)/normstd)))))
                    flat = feature.flatten(2)
                    projected = torch.einsum('dc,bcu->bdu', components, flat-mean[None, :, None])
                    response64 = torch.einsum('bcd,bdu->bcu', reader[pol], flat.double())
                    projected_np = projected.reshape(-1, 32, 40, 40).cpu().numpy().astype(np.float16)
                    response_np = response64.reshape(-1, 8, 40, 40).cpu().numpy().astype(np.float32)
                    assert np.isfinite(projected_np).all() and np.isfinite(response_np).all()
                    timing['backbone_projection_response'] += time.perf_counter()-stamp
                    stamp = time.perf_counter()
                    weights_np = np.stack([buffers['region'][i-first, p, k] for i, p in addresses])
                    weights = torch.tensor(weights_np.reshape(len(addresses), 3, -1), device='cuda', dtype=torch.float64)
                    response = torch.tensor(response_np.reshape(len(addresses), 8, -1), device='cuda', dtype=torch.float64)
                    packed = old.attention(response, weights, tau[pol, None, None, None]).cpu().numpy()
                    direct = old.attention(response64, weights, tau[pol, None, None, None]).cpu().numpy()
                    update_max(checks, 'response_float32_summary_max_abs_error', np.max(np.abs(packed-direct)))
                    # Replay pooling through the stored assignment, including slot masses.
                    assignment = np.stack([buffers['a0'][i-first, p, k] for i, p in addresses])
                    aw = torch.tensor(assignment.reshape(len(addresses), 20, -1), device='cuda', dtype=torch.float64)
                    numerator = torch.einsum('bsu,bcu->bsc', aw, response)
                    aden = aw.sum(-1)
                    for r, ids in enumerate(ROLE_SLOTS):
                        static = numerator[:, ids].sum(1)/aden[:, ids].sum(1)[:, None].clamp_min(1e-300)
                        static = static.cpu().numpy()
                        for b, (i, p) in enumerate(addresses):
                            if region_valid[i, p, r]:
                                static_mu[i, p, r] += static[b]/4
                    for b, (i, p) in enumerate(addresses):
                        buffers['x'][i-first, p, k] = projected_np[b]
                        buffers['response'][i-first, p, k] = response_np[b]
                        packed[b, ~region_valid[i, p]] = 0
                        replay[i, p] += packed[b]/4
                    timing['replay'] += time.perf_counter()-stamp
                for i, p in addresses:
                    seen[i, p] = True
                del pixels, feature, flat, projected, response64, response, weights, aw, numerator
            error = float(np.max(np.abs(replay[first:stop]-meta['summary1'][first:stop])))
            update_max(checks, 'old_summary1_c4_max_abs_error', error)
            assert error < REPLAY_TOL, ('summary1', first, error)
            error = float(np.max(np.abs(static_mu[first:stop]-meta['summary1'][first:stop, ..., 0])))
            update_max(checks, 'static_a0_mean_c4_max_abs_error', error)
            assert error < REPLAY_TOL, ('static_a0_mean', first, error)
            stamp = time.perf_counter()
            for key, values in buffers.items():
                assert np.isfinite(values).all()
                assert not values[~valid[first:stop]].any()
                store_batch(key, first, values)
            timing['write'] += time.perf_counter()-stamp
            del buffers, images
            if time.perf_counter()-last >= 20:
                print('extract', stop, '/', n, 'seconds', round(time.perf_counter()-tick, 1),
                      'summary_error', checks['old_summary1_c4_max_abs_error'], flush=True)
                last = time.perf_counter()
    np.testing.assert_array_equal(seen, valid)
    assert not replay[~region_valid].any()
    assert not geometry[~valid].any() and not theta[~valid].any()
    meta.update(up_index=up_index, geometry=geometry, theta=theta, geometry_valid=geometry_valid,
                region_mass=mass, slot_mass=slot_mass, cycle_valid=cycle_valid,
                slot_region=SLOT_REGION, slot_centers=CENTERS, sign_permutation=SIGN_PERMUTATION,
                replay_summary1=replay, static_a0_mu=static_mu,
                instance_ids=table.instance_id.to_numpy(dtype=str), pol_names=np.array(geo.POL),
                region_names=np.array(geo.REGIONS))
    np.savez_compressed(HERE / 'metadata.npz', **meta)
    coverage = {}
    for role, frame in table.groupby('role', sort=False):
        ids = frame.index.to_numpy()
        coverage[role] = dict(n=len(frame), ships=int(frame.mmsi.nunique()), ports=int(frame.port.nunique()),
                              passes=int(frame.pass_group.nunique()), valid_pol=valid[ids].sum(0).tolist(),
                              valid_regions=region_valid[ids].sum((0, 1)).tolist())
    receipt = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter()-tick, timing=timing, n=n, upstream_n=3387, batch_objects=BATCH,
        gpu=torch.cuda.get_device_name(), gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        shapes={key: list(shape) for key, shape in shapes.items()},
        dtypes={key: np.dtype(dtype).name for key, dtype in DTYPES.items()}, checks=checks,
        source_coverage=coverage, source_columns=table.columns.tolist(),
        missing_pol=int((~valid).sum()), valid_pol=int(valid.sum()),
        geometry_missing_on_valid_pol=int((valid & ~geometry_valid).sum()),
        all_valid_region_rotations_have_cycles=bool(cycle_valid[np.broadcast_to(region_valid[:, :, None], cycle_valid.shape)].all()),
        upstream=str(UP), projection=str(TOKENS / 'projection_layer1.npz'),
        frozen_reader=str(UP / 'models/reader_layer1.npz'), frozen_backbone=str(geo.WEIGHTS),
        inputs='geo.native native160 crop/pad; each rotated input independently traverses frozen R50 layer1',
        projection_rule='x=(F-mean256) projected by components32x256; no unit normalization; float16 only for new convolution',
        response_rule='response[p,c,u]=sum_d W[c,p,d]*F[p,d,u]; no intercept; float64 dot stored float32; never PCA reconstruction',
        region_order=['body', 'end', 'side'], slot_order=['body0:8', 'side8:16', 'end16:20'],
        assignment='exp(-((u-cu)/.35)^2-((v-cv)/.5)^2), normalized within role, multiplied by exact observed pixel region, rotated then geo.cell_mass',
        deviation='Full positive within-role assignment uses soft geometric localization, not strict neighboring-only or two-nearest truncation; cycles checked on stored float32 assignments.',
        theta_rule='theta[i,p,k]=geometry[i,p,4]-k*pi/2; missing geometry/pol remains zero',
        validity='valid copied from upstream summaries; region_valid masks old readout only; region/a0 retain exact observed mass even for subthreshold nonempty regions',
        labels='Original table label columns copied verbatim; never inspected for selection, extraction, geometry, checks, or parameter decisions; no evaluation labels loaded into tensors.',
        reference_summary='metadata.summary1 is original upstream subset; metadata.replay_summary1 uses stored float32 response/region; metadata.static_a0_mu uses actual stored a0',
        no_fitting=True, no_git_commits=True)
    (HERE / 'extraction_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
