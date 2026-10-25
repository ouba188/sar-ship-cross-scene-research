"""Frozen typed91 descriptors on the historical 20 hard slots over soft regions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
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
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
TYPED = HERE.parent / 'typed_m3_20261025/extract_typed.py'
LOCAL = HERE.parents[1] / 'local_evidence_transport'
N, QUERY_N, BATCH = 1805, 1741, 8
ROLE_SLOTS = (np.arange(8), np.arange(16, 20), np.arange(8, 16))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


typed = load_module('anchored_typed_descriptors', TYPED)
hard = load_module('anchored_legacy_hard_slots', LOCAL / 'extract_tokens.py')


def describe_pol(maps, bands, assignment, regions, flags, p, reader):
    """Typed describe_pol arithmetic, with exact zero output for empty hard slots."""
    desc = np.zeros((3, 3, 8, 91), np.float64)
    mass = np.zeros((3, 3, 8), np.float64)
    for r, slot_ids in enumerate(ROLE_SLOTS):
        if flags[r]:
            mass[r, :, :len(slot_ids)] = assignment[slot_ids].sum((1, 2))
    tid = slice(p*3, p*3+3)
    for k in range(4):
        a = np.rot90(assignment, k, (-2, -1))
        region = np.rot90(regions, k, (-2, -1))
        band = np.rot90(bands, k, (-2, -1))
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
        for r, slot_ids in enumerate(ROLE_SLOTS):
            if not flags[r]:
                continue
            keep = region[r, yy, xx] > 0
            weights = torch.tensor(np.ascontiguousarray(a[slot_ids][:, yy[keep], xx[keep]]),
                                   device='cuda', dtype=torch.float64)
            c = content[:, keep].double()
            d = dog[:, keep].double()
            s = response[:, :, keep].double()
            masses = weights.sum(-1)
            # A cropped role can be available while individual hard bins are empty.
            normalized = weights / torch.where(masses > 0, masses, torch.ones_like(masses))[:, None]
            pca_mean = normalized @ c.T
            dog_mean = torch.einsum('sl,bl->bs', normalized, d)
            dog_abs = torch.einsum('sl,bl->bs', normalized, d.abs())
            dog_square = torch.einsum('sl,bl->bs', normalized, d.square())
            cross = torch.einsum('sl,bl,cl->bsc', normalized, d, c)
            covariance = cross-dog_mean[:, :, None]*pca_mean[None]
            stats = typed.ops.moment_pool(s, weights[None].expand(3, -1, -1), reader['tau'][tid].double())
            class_moments = typed.ops.decode_moments(stats)
            value = torch.cat((pca_mean[None].expand(3, -1, -1), class_moments.flatten(2),
                               dog_mean[:, :, None], dog_abs[:, :, None], dog_square[:, :, None],
                               covariance), -1)
            desc[r, :, :len(slot_ids)] += value.cpu().numpy()/4
    return desc.reshape(9, 8, 91).astype(np.float32), mass.reshape(9, 8).astype(np.float32)


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    HERE.mkdir(parents=True, exist_ok=True)
    receipt_path = HERE / 'extraction_receipt.json'
    receipt = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                   threads=2, batch_objects=BATCH, no_fitting=True, no_training=True, no_git=True)

    def save_receipt():
        receipt['seconds'] = time.perf_counter()-tick
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')

    save_receipt()
    try:
        for expected, actual in zip(ROLE_SLOTS, typed.prep.ROLE_SLOTS):
            np.testing.assert_array_equal(actual, expected)
        table = pd.read_csv(typed.NATIVE / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
        upstream = pd.read_csv(typed.UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
        query = pd.read_csv(typed.OLD / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
        assert len(table) == N and len(query) == QUERY_N
        up = table.up_index.to_numpy(int)
        assert len(np.unique(up)) == N
        pd.testing.assert_frame_equal(table[upstream.columns], upstream.iloc[up].reset_index(drop=True))
        pd.testing.assert_frame_equal(table.iloc[:QUERY_N][query.columns], query)
        with np.load(typed.PROJECTION) as z:
            bank = z['bank'].copy()
        np.testing.assert_array_equal(up[QUERY_N:], bank.ravel())
        assert table.role.iloc[QUERY_N:].eq('strategy_fit').all()
        with np.load(typed.CACHE / 'metadata.npz') as z:
            np.testing.assert_array_equal(z['up_index'], up)
            np.testing.assert_array_equal(z['bank_up_index'], bank)
            placement = z['placement'].copy()
        with np.load(typed.UP / 'summaries.npz') as z:
            flags = z['region_valid'][up].astype(bool)
            pol_valid = z['valid'][up].astype(bool)
            np.testing.assert_array_equal(z['indices'][up], table.row_index)
        with np.load(typed.OLD / 'metadata.npz') as z:
            np.testing.assert_array_equal(z['up_index'], up[:QUERY_N])
            np.testing.assert_array_equal(z['region_valid'], flags[:QUERY_N])
        np.testing.assert_array_equal(np.load(typed.CACHE / 'pol_valid.npy'), pol_valid)
        geometry = np.load(typed.CACHE / 'geometry.npy')
        geometry_valid = np.load(typed.CACHE / 'geometry_valid.npy')
        assert not flags[~pol_valid].any() and not flags[~geometry_valid].any()
        with np.load(LOCAL / 'tokens.npz') as z:
            legacy_mass = z['mass'][up]
            np.testing.assert_array_equal(z['valid'][up], geometry_valid)
            np.testing.assert_array_equal(z['instance_ids'][up], table.instance_id.to_numpy(dtype=str))
            np.testing.assert_array_equal(z['regions'], typed.prep.SLOT_REGION)
        with np.load(typed.NATIVE / 'reader.npz') as z:
            reader = {k: torch.tensor(z[k], dtype=torch.float32, device='cuda') for k in z.files}
        assert reader['W'].shape == (6, 8, 8) and (reader['scale'] > 0).all()
        candidates = np.flatnonzero(table.role.eq('utility_train').to_numpy() & flags.all((1, 2)))
        source4 = set(map(int, candidates[:4]))
        assert len(source4) == 4
        checks = dict(identity_correspondence=True, source4_role_rotation_checks=0,
                      raw_legacy_mass_max_abs=0., raw_legacy_mass_mismatch_count=0,
                      raw_region_partition_max_abs=0., typed_helper_desc_max_abs=0.,
                      source4_cases=[dict(index=i, up_index=int(up[i]), obs_id=int(table.iloc[i].obs_id))
                                     for i in sorted(source4)])
        receipt['checks'] = checks
        desc = np.zeros((N, 18, 8, 91), np.float32)
        mass = np.zeros((N, 18, 8), np.float32)
        raw_mass = np.zeros((N, 2, 20), np.float32)
        empty_cases = []
        encoder = None
        last = tick
        timing = dict(reference_encoding=0., descriptors_and_validation=0.)
        batches = [(first, min(first+BATCH, end)) for start, end in ((0, QUERY_N), (QUERY_N, N))
                   for first in range(start, end, BATCH)]
        with torch.inference_mode():
            for first, stop in batches:
                ids = np.arange(first, stop)
                bmap = np.load(typed.CACHE / 'native_bands.npy', mmap_mode='r')
                vmap = np.load(typed.CACHE / 'valid_pixels.npy', mmap_mode='r')
                bands, footprints = np.array(bmap[first:stop]), np.array(vmap[first:stop])
                bmap._mmap.close()
                vmap._mmap.close()
                assert bands.shape == (len(ids), 2, 3, 160, 160)
                assert np.isfinite(bands).all() and not bands[~pol_valid[ids]].any()
                if first < QUERY_N:
                    maps = np.load(typed.OLD / 'x.npy', mmap_mode='r')
                    assert maps.shape == (QUERY_N, 2, 4, 32, 40, 40) and maps.dtype == np.float16
                    content = np.array(maps[first:stop])
                    maps._mmap.close()
                else:
                    stamp = time.perf_counter()
                    if encoder is None:
                        encoder = typed.ReferenceEncoder()
                    content = encoder.batch(table, ids, pol_valid, placement, bands, checks)
                    timing['reference_encoding'] += time.perf_counter()-stamp
                assert np.isfinite(content).all() and not content[~pol_valid[ids]].any()
                stamp = time.perf_counter()
                for j, i in enumerate(ids):
                    for p in range(2):
                        g = geometry[i, p] if geometry_valid[i, p] else None
                        # Call the historical hard-bin implementation, never prepare.slots.
                        a = hard.slots(g, footprints[j, p])
                        regions = hard.geo.soft_regions(g)*footprints[j, p]
                        raw_mass[i, p] = a.sum((1, 2))
                        for r, slot_ids in enumerate(ROLE_SLOTS):
                            error = np.abs(a[slot_ids].sum(0)-regions[r]).max()
                            typed.update_max(checks, 'raw_region_partition_max_abs', error)
                            assert error < 1e-10, ('region partition', int(i), p, r, error)
                            empty = slot_ids[raw_mass[i, p, slot_ids] == 0]
                            if flags[i, p, r] and len(empty):
                                empty_cases.append(dict(index=int(i), up_index=int(up[i]),
                                                        obs_id=int(table.iloc[i].obs_id),
                                                        polarization=typed.geo.POL[p], region=typed.geo.REGIONS[r],
                                                        legacy_slot_ids=empty.tolist()))
                        difference = np.abs(raw_mass[i, p]-legacy_mass[i, p])
                        typed.update_max(checks, 'raw_legacy_mass_max_abs', difference.max())
                        checks['raw_legacy_mass_mismatch_count'] += int(np.count_nonzero(difference))
                        np.testing.assert_array_equal(raw_mass[i, p], legacy_mass[i, p])
                        if not flags[i, p].any():
                            continue
                        dd, mm = describe_pol(content[j, p], bands[j, p], a, regions, flags[i, p], p, reader)
                        if i in source4:
                            assert (raw_mass[i, p] > 0).all()
                            expected, expected_mass = typed.describe_pol(
                                content[j, p], bands[j, p], a, regions, flags[i, p], p, reader, True, checks)
                            typed.update_max(checks, 'typed_helper_desc_max_abs', np.abs(dd-expected).max())
                            np.testing.assert_array_equal(dd, expected)
                            np.testing.assert_array_equal(mm, expected_mass)
                            pixels = typed.placed_pixels(table, i, p, placement)
                            error = np.abs(bands[j, p].sum(0, dtype=np.float64)-pixels/255.).max()
                            typed.update_max(checks, 'source4_band_reconstruction_max_abs', error)
                            assert error < 2e-7
                        desc[i, p*9:(p+1)*9], mass[i, p*9:(p+1)*9] = dd, mm
                timing['descriptors_and_validation'] += time.perf_counter()-stamp
                assert np.isfinite(desc[first:stop]).all() and np.isfinite(mass[first:stop]).all()
                if time.perf_counter()-last >= 20 or stop in (QUERY_N, N):
                    receipt.update(processed=stop)
                    save_receipt()
                    print('EXTRACT_HARD', stop, '/', N, 'seconds', round(receipt['seconds'], 1), flush=True)
                    last = time.perf_counter()

        type_pol = np.repeat(np.arange(2), 9)
        type_region = np.tile(np.repeat(np.arange(3), 3), 2)
        type_band = np.tile(np.arange(3), 6)
        type_valid = flags[:, type_pol, type_region]
        slot_count = np.where(type_region == 1, 4, 8)
        legacy_slot_ids = np.full((18, 8), -1, int)
        expected_mass = np.zeros_like(mass)
        ungated_mass = np.zeros_like(mass)
        for t, (p, r) in enumerate(zip(type_pol, type_region)):
            slot_ids = ROLE_SLOTS[r]
            legacy_slot_ids[t, :len(slot_ids)] = slot_ids
            ungated_mass[:, t, :len(slot_ids)] = legacy_mass[:, p, slot_ids]
            expected_mass[:, t] = ungated_mass[:, t]*flags[:, p, r, None]
        np.testing.assert_array_equal(mass, expected_mass)
        slot_valid = mass > 0
        assert not desc[~slot_valid].any()
        assert not mass[~type_valid].any()
        assert checks['source4_role_rotation_checks'] == 96
        suppressed = (~flags[:, :, typed.prep.SLOT_REGION]) & (legacy_mass > 0)
        suppressed_records = []
        for i, p, s in np.argwhere(suppressed):
            suppressed_records.append(dict(index=int(i), up_index=int(up[i]),
                                           obs_id=int(table.iloc[i].obs_id), polarization=typed.geo.POL[p],
                                           legacy_slot_id=int(s), raw_mass=float(legacy_mass[i, p, s])))
        checks.update(finite=True, empty_and_padded_slots_exact_zero=True, invalid_types_exact_zero=True,
                      output_flagged_legacy_mass_max_abs=float(np.abs(mass-expected_mass).max()),
                      output_ungated_legacy_mass_max_abs=float(np.abs(mass-ungated_mass).max()),
                      output_ungated_legacy_mass_mismatch_count=int(np.count_nonzero(mass != ungated_mass)),
                      regionflag_suppressed_nonzero_slots=int(suppressed.sum()),
                      regionflag_suppressed_mass=float(legacy_mass[suppressed].sum(dtype=np.float64)),
                      regionflag_suppressed_slots=suppressed_records,
                      active_regions_with_empty_hard_slots=empty_cases,
                      active_empty_hard_slot_count=sum(len(c['legacy_slot_ids']) for c in empty_cases))
        output = HERE / 'features_hard.npz'
        np.savez_compressed(output, desc=desc, mass=mass, raw_slot_mass=raw_mass, slot_valid=slot_valid,
                            legacy_slot_ids=legacy_slot_ids, up_index=up, obs_id=table.obs_id.to_numpy(),
                            old_index=table.old_index.to_numpy(), instance_ids=table.instance_id.to_numpy(dtype=str),
                            type_id=np.arange(18), type_pol=type_pol, type_region=type_region, type_band=type_band,
                            slot_count=slot_count, type_valid=type_valid, region_valid=flags, pol_valid=pol_valid,
                            pol_names=np.array(typed.geo.POL), region_names=np.array(typed.geo.REGIONS),
                            band_names=np.array(['fine', 'middle', 'low']), statistic_names=np.array(['mu', 'h', 'b']),
                            bank_up_index=bank, bank_cache_index=np.arange(QUERY_N, N).reshape(8, 8))
        # Reopen the deliverable to detect incomplete serialization before declaring success.
        with np.load(output) as z:
            for key, expected in (('desc', desc), ('mass', mass), ('up_index', up)):
                np.testing.assert_array_equal(z[key], expected)
        checks['saved_arrays_exact_roundtrip'] = True
        receipt.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                       n=N, query_n=QUERY_N, reference_n=64, timing=timing,
                       shapes=dict(desc=list(desc.shape), mass=list(mass.shape)), dtype='float32',
                       output_bytes=output.stat().st_size, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                       type_order='polarization(VV,VH), region(BODY,ENDS,SIDES), band(fine,middle,low)',
                       slot_order='BODY 0:8; ENDS 16:20 padded to8; SIDES 8:16; exact historical ordering',
                       regionflag_behavior='Upstream flags suppress descriptor and output mass only; raw_slot_mass preserves all historical mass. Empty hard slots remain zero even in an available region; slot_valid=mass>0.',
                       descriptor_fields={'0:32': 'PCA32 weighted mean', '32:56': '8 classes x (mu,h,b), class-major',
                                          '56:59': 'native band signed mean, abs mean, square mean',
                                          '59:91': 'population Cov(native band,PCA32)'},
                       class_projection='Frozen source reader on standardized PCA[:7]+native band; no new convolution',
                       c4='Same typed per-view interpolation, float64 moments and covariance BEFORE C4 averaging',
                       empty_slot_handling='Only change to typed descriptor arithmetic: zero denominator replaced by1 for zero-weight slots; no epsilon, redistribution, soft assignment or global monkeypatch',
                       sources=dict(table=str(typed.NATIVE / 'table.csv'), hard_slots=str(LOCAL / 'extract_tokens.py'),
                                    legacy_tokens=str(LOCAL / 'tokens.npz'), typed_descriptor=str(TYPED),
                                    query_maps=str(typed.OLD / 'x.npy'), native_cache=str(typed.CACHE),
                                    reader=str(typed.NATIVE / 'reader.npz'), projection=str(typed.PROJECTION),
                                    backbone=str(typed.geo.WEIGHTS)),
                       reference_encoding='Same typed.ReferenceEncoder frozen R50 layer1/source PCA; float16 before interpolation',
                       limitations=['Extraction only; no training or accuracy claim.',
                                    'PCA32 is interpolated stride4 content, not native-resolution backbone output.',
                                    'Source4 algebra/helper parity checks are bounded; hard mass and region partitions checked on all1805.'])
        save_receipt()
        print(json.dumps({k: receipt[k] for k in ('status', 'seconds', 'shapes', 'checks', 'output_bytes')},
                         ensure_ascii=False), flush=True)
    except Exception as error:
        receipt.update(status='failed', error=repr(error))
        save_receipt()
        raise


if __name__ == '__main__':
    main()
