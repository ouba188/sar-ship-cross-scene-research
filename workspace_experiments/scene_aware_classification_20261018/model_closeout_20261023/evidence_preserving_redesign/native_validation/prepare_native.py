"""Cache separate native VV/VH DoG bands, before geo.native crop/pad."""
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

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'native_cache'
CLOSEOUT = HERE.parents[1]
OLD = CLOSEOUT / 'm2_operator_validation'
UP = CLOSEOUT / 'regional_utility_pilot/upstream'
BANK = CLOSEOUT / 'local_evidence_transport/projection_layer1.npz'
GEO_PATH = CLOSEOUT.parent / 'eight_class_evidence_20261023/obb_dynamic_readout/common.py'
BATCH = 32
PLACEMENT_FIELDS = ('crop_x', 'crop_y', 'crop_width', 'crop_height', 'pad_left', 'pad_top')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geo = load_module('native_cache_geometry', GEO_PATH)
operators = load_module('native_cache_operators', HERE.parent / 'operators.py')


def place_array(array, place):
    """Apply the authoritative geo.native placement to trailing spatial axes."""
    sx, sy, width, height, dx, dy = (int(place[k]) for k in PLACEMENT_FIELDS)
    out = np.zeros((*array.shape[:-2], geo.SIZE, geo.SIZE), dtype=array.dtype)
    out[..., dy:dy+height, dx:dx+width] = array[..., sy:sy+height, sx:sx+width]
    return out


def write_batch(name, first, values):
    target = np.lib.format.open_memmap(CACHE / (name+'.npy'), mode='r+')
    target[first:first+len(values)] = values
    target.flush()
    target._mmap.close()


def main():
    tick = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    dtype = {'mmsi': str, 'instance_id': str}
    query = pd.read_csv(OLD / 'table.csv', dtype=dtype)
    upstream = pd.read_csv(UP / 'table.csv', dtype=dtype)
    assert len(query) == 1741 and len(upstream) == 3387
    assert query.role.value_counts().to_dict() == dict(utility_train=644, selection=612, evaluation=485)
    with np.load(BANK) as z:
        bank = z['bank'].copy()
    assert bank.shape == (8, 8) and len(np.unique(bank)) == 64
    refs = upstream.iloc[bank.ravel()].copy()
    assert refs.role.eq('strategy_fit').all()
    pd.testing.assert_frame_equal(query.drop(columns='up_index'),
                                  upstream.iloc[query.up_index].reset_index(drop=True))
    isolation = {field: len(set(refs[field]) & set(query[field])) for field in ('mmsi', 'pass_group')}
    assert not any(isolation.values()), isolation
    refs.insert(0, 'up_index', bank.ravel())
    table = pd.concat((query, refs), ignore_index=True)
    table.insert(0, 'old_index', np.r_[np.arange(len(query)), np.full(64, -1)])
    n = len(table)
    assert n == 1805 and table.up_index.is_unique
    with np.load(UP / 'summaries.npz') as z:
        pol_valid = z['valid'][table.up_index].astype(bool)
        np.testing.assert_array_equal(z['indices'][table.up_index], table.row_index)
    snapshot_path = geo.OLD / 'local_condition_module/view_snapshot.json'
    wanted = set(table.obs_id)
    snapshots = {int(s['obs_id']): s for s in json.loads(snapshot_path.read_text(encoding='utf-8'))
                 if int(s['obs_id']) in wanted}
    assert len(snapshots) == len(wanted)
    CACHE.mkdir(parents=True, exist_ok=True)
    receipt_path = HERE / 'extraction_receipt.json'
    receipt_path.write_text(json.dumps(dict(status='running', started_utc=started, n=n)), encoding='utf-8')
    table.to_csv(HERE / 'table.csv', index=False)
    shapes = dict(native_bands=(n, 2, 3, 160, 160), valid_pixels=(n, 2, 160, 160))
    dtypes = dict(native_bands=np.float32, valid_pixels=np.uint8)
    for name, shape in shapes.items():
        arr = np.lib.format.open_memmap(CACHE / (name+'.npy'), mode='w+', dtype=dtypes[name], shape=shape)
        arr.flush()
        arr._mmap.close()
    geometry = np.zeros((n, 2, 6), np.float64)
    geometry_valid = np.zeros((n, 2), bool)
    placement = np.zeros((n, 2, 6), np.int32)
    raw_shape = np.zeros((n, 2, 2), np.int32)
    checks = dict(raw_reconstruction_max_abs_error=0., placed_float32_reconstruction_max_abs_error=0.,
                  geo_native_pixel_max_abs_error=0., footprint_replay_exact=True,
                  zero_outside_footprint=True, missing_pol_zero=True)
    timing = dict(read_place_filter_check=0., write=0.)
    modes, replay_cases, replay_kinds, missing_geometry = {}, [], set(), []
    last = tick
    for first in range(0, n, BATCH):
        stop = min(first+BATCH, n)
        buffers = {name: np.zeros((stop-first, *shape[1:]), dtypes[name]) for name, shape in shapes.items()}
        stamp = time.perf_counter()
        for i in range(first, stop):
            row = table.iloc[i]
            for p, pol in enumerate(geo.POL):
                if not pol_valid[i, p]:
                    continue
                # pixels=False still computes the exact legacy masks, placement and geometry.
                _, footprint, g, place = geo.native(table, snapshots, i, p, pixels=False)
                with Image.open(row['image_'+pol]) as image:
                    assert image.format == 'PNG'
                    modes[image.mode] = modes.get(image.mode, 0)+1
                    raw = np.array(image.convert('L'))
                view = snapshots[int(row.obs_id)]['images'][pol]
                assert raw.ndim == 2 and raw.dtype == np.uint8
                assert raw.shape == (view['h'], view['w'])
                raw_shape[i, p] = raw.shape
                placement[i, p] = [place[k] for k in PLACEMENT_FIELDS]
                assert place['input_scale'] == 1.0
                expected_footprint = place_array(np.ones_like(raw, dtype=np.uint8), place)
                np.testing.assert_array_equal(expected_footprint, footprint)
                assert footprint.sum() > 0
                bands = operators.native_bands(raw)
                error = float(np.max(np.abs(bands.sum(0)-raw.astype(np.float64)/255.)))
                checks['raw_reconstruction_max_abs_error'] = max(checks['raw_reconstruction_max_abs_error'], error)
                assert error < 1e-12
                placed = place_array(bands, place).astype(np.float32)
                assert np.isfinite(placed).all() and not placed[:, footprint == 0].any()
                placed_raw = place_array(raw, place)
                error = float(np.max(np.abs(placed.sum(0, dtype=np.float64)-placed_raw.astype(np.float64)/255.)))
                checks['placed_float32_reconstruction_max_abs_error'] = max(checks['placed_float32_reconstruction_max_abs_error'], error)
                assert error < 2e-7
                buffers['native_bands'][i-first, p] = placed
                buffers['valid_pixels'][i-first, p] = expected_footprint
                if g is not None:
                    geometry[i, p] = g
                    geometry_valid[i, p] = True
                else:
                    missing_geometry.append(dict(index=i, obs_id=int(row.obs_id), polarization=pol))
                # One real-image replay for each role/polarization/crop-pad case.
                kind = (row.role, pol, raw.shape[0] > 160, raw.shape[1] > 160,
                        raw.shape[0] < 160, raw.shape[1] < 160)
                if kind not in replay_kinds:
                    pix, fp, replay_g, replay_place = geo.native(table, snapshots, i, p)
                    np.testing.assert_array_equal(pix, placed_raw)
                    np.testing.assert_array_equal(fp, footprint)
                    assert replay_place == place
                    if g is None:
                        assert replay_g is None
                    else:
                        np.testing.assert_array_equal(g, replay_g)
                    replay_kinds.add(kind)
                    replay_cases.append(dict(index=i, obs_id=int(row.obs_id), polarization=pol,
                                             path=str(row['image_'+pol]), raw_shape=list(raw.shape), placement=place))
        timing['read_place_filter_check'] += time.perf_counter()-stamp
        stamp = time.perf_counter()
        for name, values in buffers.items():
            assert not values[~pol_valid[first:stop]].any()
            write_batch(name, first, values)
        timing['write'] += time.perf_counter()-stamp
        if time.perf_counter()-last >= 15:
            print('extract', stop, '/', n, 'seconds', round(time.perf_counter()-tick, 2), flush=True)
            last = time.perf_counter()
    for name, values in dict(geometry=geometry, geometry_valid=geometry_valid, pol_valid=pol_valid).items():
        np.save(CACHE / (name+'.npy'), values, allow_pickle=False)
        shapes[name] = values.shape
        dtypes[name] = values.dtype
    np.savez_compressed(CACHE / 'metadata.npz', placement=placement, placement_fields=np.array(PLACEMENT_FIELDS),
                        raw_shape=raw_shape, up_index=table.up_index.to_numpy(), old_index=table.old_index.to_numpy(),
                        bank_up_index=bank, bank_cache_index=np.arange(1741, 1805).reshape(8, 8),
                        pol_names=np.array(geo.POL), band_names=np.array(['fine', 'middle', 'low']))
    coverage = {}
    for role, frame in table.groupby('role', sort=False):
        ids = frame.index.to_numpy()
        coverage[role] = dict(n=len(ids), valid_pol=pol_valid[ids].sum(0).tolist(),
                             missing_pol=(~pol_valid[ids]).sum(0).tolist(),
                             missing_geometry_on_valid_pol=(pol_valid[ids] & ~geometry_valid[ids]).sum(0).tolist())
    receipt = dict(status='complete', started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                   seconds=time.perf_counter()-tick, timing=timing, n=n, query_n=1741, reference_n=64,
                   shapes={k: list(v) for k, v in shapes.items()},
                   dtypes={k: np.dtype(v).name for k, v in dtypes.items()},
                   pol_order=list(geo.POL), band_order=['fine', 'middle', 'low'],
                   geometry_fields=['cx', 'cy', 'length', 'width', 'angle_radians_mod_pi', 'confidence'],
                   geometry_validity='geometry_valid.npy; absent geometry is zero; pol_valid is independent',
                   pixel_validity='1 for retained original chip pixels, 0 for padding or absent polarization',
                   missing_pol=int((~pol_valid).sum()), valid_pol=int(pol_valid.sum()),
                   geometry_missing_on_valid_pol=len(missing_geometry), missing_geometry_cases=missing_geometry,
                   coverage=coverage, checks=checks, png_modes=modes, geo_native_replay_cases=replay_cases,
                   source=dict(old_table=str(OLD / 'table.csv'), upstream_table=str(UP / 'table.csv'),
                               polarization_validity=str(UP / 'summaries.npz'), bank=str(BANK),
                               snapshot=str(snapshot_path), geometry_implementation=str(GEO_PATH),
                               bands_implementation=str(HERE.parent / 'operators.py')),
                   table_rule='First 1741 rows preserve old table order; next 64 are bank.ravel() order. old_index=-1 for references; up_index addresses upstream table.',
                   filter_rule='Each 2D uint8 PNG converted to L exactly as geo.native; /255; native_bands float64 DoG sigma1/2 reflect on original chip; THEN identical crop/pad to 160; stored float32.',
                   boundary_rule='scipy gaussian_filter default truncate=4; sigma1 radius4 and sigma2 radius8 on original chip, reflected at its edges; crop and zero pad only after filtering',
                   reference_isolation_intersections=isolation,
                   no_resize=True, no_filter_on_padded_input=True, no_c4_cache=True, no_native_slot_cache=True,
                   labels='Copied table metadata only; no label-based selection, predictions, fitting or parameter decisions.',
                   no_fitting=True, no_git=True, raw_bands_not_uploaded=True,
                   cache_bytes=sum(p.stat().st_size for p in CACHE.iterdir() if p.suffix in ('.npy', '.npz')))
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('status', 'seconds', 'n', 'missing_pol',
                     'geometry_missing_on_valid_pol', 'shapes', 'dtypes', 'checks', 'cache_bytes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
