"""CPU-only geometry recovery for the fixed local-slot cohort; no network."""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACK = Path('E:/Hermes/paper_tracking')
sys.path.insert(0, str(ROOT / 'ship_radar_geometry_design_20261016'))
sys.path.insert(0, str(TRACK / 'sensitivity/oof_reliability_check'))
sys.path.insert(0, str(TRACK / 'pixel_correction_review'))
import geometry_metadata as old
from ship_axis_proxy import measure_axis, world_axis, combine_axes
from relative_radar_geometry import range_azimuth
from join_chip_incidence import interpolate, KINDS


@lru_cache(None)
def local_grid(product):
    path = old.GEOMETRY / 'radar_grid' / (product + '.json')
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('product') != product or value.get('status') != 'ok':
        return None
    return value


def radar_recovery(grid, longitude, latitude, geo, width, height):
    """Ship validity is deliberately not an argument."""
    value = range_azimuth(grid, longitude, latitude)
    if value['status'] != 'ok':
        return {}, value['status']
    return dict(radar_axis_rad=old.image_axis(value['axial_azimuth_deg'], longitude,
                latitude, geo, width, height), radar_valid=1,
                radar_status='recovered_independent_local_product_grid',
                nearest_grid_distance_m=value['nearest_grid_distance_m']), 'recovered'


def measure_missing_pair(row, georefs):
    views, dimensions = {}, {}
    for pol in ('vv', 'vh'):
        geo = georefs.get(row['source_tiff_' + pol])
        if geo is None:
            return {}, 'not_computed_missing_exact_georef'
        path = Path(row['image_relpath_' + pol])
        if not path.exists():
            return {}, 'not_computed_missing_chip'
        cx, cy = row['target_chip_center_x_' + pol], row['target_chip_center_y_' + pol]
        px, py = row['target_center_x_' + pol], row['target_center_y_' + pol]
        if not np.isfinite([cx, cy, px, py]).all():
            return {}, 'not_computed_missing_target_center'
        with Image.open(path) as image:
            w, h = image.size
            array = np.asarray(image.convert('L').resize((128, 128), Image.Resampling.BILINEAR))
        value = measure_axis(array, ((cx + .5)*128/w-.5, (cy + .5)*128/h-.5))
        if not value['accepted']:
            return {}, 'rejected_' + pol + '_' + value['status']
        angle = np.deg2rad(value['angle_pixel_deg'])
        native = np.rad2deg(np.arctan2(np.sin(angle)*h/128, np.cos(angle)*w/128))
        value.update(world_axis(geo, px, py, native))
        views[pol], dimensions[pol] = value, (w, h)
    axis, disagreement = combine_axes(views['vv']['axis_true_north_deg'], views['vh']['axis_true_north_deg'])
    if disagreement > 15.:
        return {}, 'rejected_polarization_disagreement'
    vv = views['vv']
    elongation = min(v['elongation'] for v in views.values())
    result = dict(ship_valid=1, ship_axis_rad=old.image_axis(axis, vv['longitude'], vv['latitude'],
                  georefs[row['source_tiff_vv']], *dimensions['vv']),
                  ship_confidence=(elongation**2-1)/(elongation**2+1),
                  ship_disagreement_deg=disagreement, ship_min_elongation=elongation,
                  ship_status='accepted_remeasured_paired_axis_in_vv_frame')
    return result, 'recovered_same_estimator_and_thresholds'


def fixed_tasks():
    pools = json.loads(old.SOURCE.read_text(encoding='utf-8'))['pools']
    target_root = ROOT / 'prepool_local_probe_20261015/body_context_expansion'
    for pool in pools:
        meta = pd.read_csv(Path(pool['source_folder']) / 'metadata.csv')
        for split, tasks in pool['splits'].items():
            yield pool['pool'], split, meta, [t for t in tasks if t['budget'] == 32]
        folder = target_root / pool['pool'].replace(' ', '_')
        meta = pd.read_csv(folder / 'metadata.csv')
        tasks = json.loads((folder / 'tasks.json').read_text(encoding='utf-8'))
        yield pool['pool'], 'target', meta, [t for t in tasks if t['budget'] == 32]


def masks(frame):
    return dict(triple=frame[['ship_valid', 'radar_valid', 'incidence_valid']].all(axis=1).to_numpy(),
                ship_required=frame.ship_valid.astype(bool).to_numpy())


def coverage(before, after):
    roles, tasks_out = [], []
    for port, split, meta, tasks in fixed_tasks():
        origins = sorted({t['origin'] for t in tasks})
        for origin in origins:
            subset = [t for t in tasks if t['origin'] == origin]
            for role in ('s', 'q'):
                ids = sorted({i for t in subset for i in t[role]})
                rows = meta.iloc[ids].global_row.to_numpy(int)
                for version, frame in (('old', before), ('recovered', after)):
                    selected = frame.iloc[rows]
                    roles.append(dict(pool=port, split=split, origin=origin, role=role, version=version,
                        samples=len(rows), **{k: int(selected[k].sum()) for k in
                        ('ship_valid', 'radar_valid', 'incidence_valid')},
                        triple=int(masks(selected)['triple'].sum())))
            for version, frame in (('old', before), ('recovered', after)):
                valid = masks(frame)
                for criterion, mask in valid.items():
                    enabled_queries, enabled_supports = set(), set()
                    count = events = 0
                    for t in subset:
                        s = meta.iloc[t['s']].global_row.to_numpy(int)
                        q = meta.iloc[t['q']].global_row.to_numpy(int)
                        if mask[s].sum() >= 2 and mask[q].any():
                            count += 1
                            events += int(mask[q].sum())
                            enabled_queries.update(q[mask[q]].tolist())
                            enabled_supports.update(s[mask[s]].tolist())
                    tasks_out.append(dict(pool=port, split=split, origin=origin, version=version,
                        criterion=criterion, tasks=len(subset), eligible_tasks=count,
                        enabled_query_samples=len(enabled_queries), enabled_support_samples=len(enabled_supports),
                        enabled_query_events=events))
    return pd.DataFrame(roles), pd.DataFrame(tasks_out)


def verify_preservation(before, after, needed):
    for field in ('ship', 'radar', 'incidence'):
        valid = before[field + '_valid'].astype(bool)
        columns = [c for c in before if c.startswith(field + '_')]
        pd.testing.assert_frame_equal(before.loc[valid, columns], after.loc[valid, columns], check_dtype=False)
    untouched = ~before.global_row.isin(needed)
    pd.testing.assert_frame_equal(before.loc[untouched], after.loc[untouched, before.columns], check_dtype=False)
    for field, angle in (('ship', 'ship_axis_rad'), ('radar', 'radar_axis_rad'), ('incidence', 'incidence_rad')):
        valid = after[field + '_valid'].astype(bool)
        assert after.loc[valid, angle].notna().all()
        assert after.loc[~valid, angle].isna().all()
        upper = np.pi / 2 if field == 'incidence' else np.pi
        assert after.loc[valid, angle].between(0, upper, inclusive='left').all()


def main():
    before = pd.read_csv(ROOT / 'ship_radar_geometry_design_20261016/geometry_metadata.csv')
    manifest = pd.read_csv(old.MANIFEST, low_memory=False)
    np.testing.assert_array_equal(before.global_row, np.arange(len(manifest)))
    np.testing.assert_array_equal(before.instance_id, manifest.instance_id)
    caches = sorted((ROOT / 'geometry_relation_slots_20261016/slots_cache').glob('*.npz'))
    needed = np.unique(np.concatenate([np.load(path)['rows'] for path in caches]))
    georefs = old.load_georefs()
    pairs = pd.read_csv(old.GEOMETRY / 'paired_axes.csv').set_index('instance_id')
    incidence = pd.read_csv(old.INCIDENCE).set_index('instance_id')
    records = before.to_dict('records')
    for n, index in enumerate(needed):
        row = manifest.iloc[index]
        record = records[index]
        sid = row.instance_id
        for field in ('ship', 'radar', 'incidence'):
            record[field + '_recovery_status'] = 'preserved_valid' if record[field + '_valid'] else 'not_computed'
            record[field + '_provenance'] = 'old_geometry_metadata.csv' if record[field + '_valid'] else ''
        if not record['ship_valid']:
            if sid in pairs.index:
                pair = pairs.loc[sid]
                record['ship_recovery_status'] = 'previously_rejected_' + str(pair.status)
                record['ship_provenance'] = 'paired_axes.csv'
            else:
                updates, status = measure_missing_pair(row, georefs)
                record.update(updates)
                record['ship_recovery_status'] = status
                record['ship_provenance'] = 'actual_VV_VH_PNG_same_ship_axis_proxy' if updates else ''
        geo = georefs.get(row.source_tiff_vv)
        px, py = row.target_center_x_vv, row.target_center_y_vv
        coordinate = None
        if geo is not None and np.isfinite([px, py]).all():
            # Use only the geographic origin returned by the existing affine helper.
            value = world_axis(geo, px, py, 0.)
            coordinate = (value['longitude'], value['latitude'])
        elif sid in incidence.index:
            inc = incidence.loc[sid]
            if inc.coordinate_kind in KINDS and np.isfinite([inc.longitude, inc.latitude]).all():
                coordinate = (float(inc.longitude), float(inc.latitude))
        grid = local_grid(row['product'])
        if not record['radar_valid']:
            status = 'not_computed_missing_exact_georef' if geo is None else 'not_computed_missing_coordinate'
            if geo is not None and coordinate is not None:
                status = 'not_computed_missing_local_product_grid'
                if grid is not None:
                    path = Path(row.image_relpath_vv)
                    status = 'not_computed_missing_chip_header'
                    if path.exists():
                        with Image.open(path) as image:
                            updates, status = radar_recovery(grid, *coordinate, geo, *image.size)
                        record.update(updates)
                        record['radar_provenance'] = str(old.GEOMETRY / 'radar_grid' / (row['product'] + '.json'))
            record['radar_recovery_status'] = status
        if not record['incidence_valid']:
            status = 'not_computed_missing_coordinate' if coordinate is None else 'not_computed_missing_local_product_grid'
            if coordinate is not None and grid is not None:
                points = [[p['latitude'], p['longitude'], p['incidenceAngle']] for p in grid['points']]
                value = interpolate(points, [[coordinate[1], coordinate[0]]])[0]
                status = 'outside_grid_hull' if value is None else 'recovered'
                if value is not None:
                    record.update(incidence_rad=np.deg2rad(value['incidence_deg']), incidence_valid=1,
                                  incidence_status='recovered_local_product_grid_interpolation')
                    record['incidence_provenance'] = str(old.GEOMETRY / 'radar_grid' / (row['product'] + '.json'))
            record['incidence_recovery_status'] = status
        if n % 1000 == 0:
            print('RECOVER', n, '/', len(needed), flush=True)
    after = pd.DataFrame(records)
    verify_preservation(before, after, needed)
    after.to_csv(HERE / 'recovered_geometry.csv', index=False)
    roles, tasks = coverage(before, after)
    roles.to_csv(HERE / 'recovered_geometry_role_coverage.csv', index=False)
    tasks.to_csv(HERE / 'recovered_geometry_task_coverage.csv', index=False)
    selected = after.iloc[needed]
    by_port = []
    for port, ids in manifest.iloc[needed].groupby('port').groups.items():
        for version, frame in (('old', before), ('recovered', after)):
            group = frame.loc[ids]
            by_port.append(dict(port=port, version=version, samples=len(ids),
                **{k: int(group[k].sum()) for k in ('ship_valid', 'radar_valid', 'incidence_valid')},
                triple=int(masks(group)['triple'].sum())))
    pd.DataFrame(by_port).to_csv(HERE / 'recovered_geometry_port_coverage.csv', index=False)
    summary = dict(manifest_rows=len(after), bounded_rows=len(needed),
        old_counts={k: int(before.iloc[needed][k].sum()) for k in ('ship_valid', 'radar_valid', 'incidence_valid')},
        recovered_counts={k: int(selected[k].sum()) for k in ('ship_valid', 'radar_valid', 'incidence_valid')},
        statuses={k: selected[k+'_recovery_status'].value_counts().to_dict() for k in ('ship', 'radar', 'incidence')},
        preserved_old_valid=True, preserved_outside_cohort=True, network_requests=0, gpu_used=False)
    (HERE / 'recovered_geometry_coverage.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
