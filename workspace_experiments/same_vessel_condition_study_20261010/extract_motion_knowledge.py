"""Bounded source-only AIS motion extraction; no training or nearest-time fallback."""
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
AUDIT = BASE / 'knowledge_train_only_design_20261011'
OUT = BASE / 'motion_knowledge_20261011'
MANIFEST = BASE.parent / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv'
FIELDS = ('sog_knots', 'cog_deg', 'true_heading_deg')
PRODUCT = re.compile(r'S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4}')


def finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else float('nan')
    except (ValueError, TypeError):
        return float('nan')


def decode_motion(sog, cog, heading):
    s, c, h = map(finite, (sog, cog, heading))
    return (s / 10 if 0 <= s < 1023 else float('nan'),
            c / 10 if 0 <= c < 3600 else float('nan'),
            h if 0 <= h < 360 else float('nan'))


def epoch_utc(value):
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('AIS time must include a timezone')
    return t.timestamp()


def midpoint(product):
    return sum(datetime.strptime(t, '%Y%m%dT%H%M%S').replace(
        tzinfo=timezone.utc).timestamp() for t in product.split('_')[4:6]) / 2


def stored_coordinates(obj):
    pairs = []
    for container in (obj, obj.get('meta', {})):
        for lonkey, latkey in (('lon', 'lat'), ('longitude', 'latitude'),
                               ('ais_lon', 'ais_lat'), ('ais_longitude', 'ais_latitude')):
            if lonkey in container and latkey in container:
                lon, lat = finite(container[lonkey]), finite(container[latkey])
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    pairs.append((lon, lat))
    return sorted(set(pairs))


def resolve(candidates, coordinates):
    eligible = [r for r in candidates if all(
        abs(r['lon'] - lon) <= 1e-6 and abs(r['lat'] - lat) <= 1e-6
        for lon, lat in coordinates)]
    if not candidates:
        return None, 'exact_time_not_found'
    if not eligible:
        return None, 'stored_coordinate_mismatch'
    variants = {(r['lon'], r['lat'], *r['raw_motion']) for r in eligible}
    if len(variants) != 1:
        return None, 'ambiguous_exact_records'
    return eligible[0], 'matched'


def self_test():
    assert decode_motion('0', '0', '0') == (0., 0., 0.)
    assert all(math.isnan(x) for x in decode_motion('1023', '3600', '511'))
    assert all(math.isnan(x) for x in decode_motion('-1', '-1', '-1'))
    assert all(math.isnan(x) for x in decode_motion('', 'NaN', 'inf'))
    assert decode_motion('1022', '3599', '359') == (102.2, 359.9, 359.)
    assert epoch_utc('2025-01-01T00:00:00Z') == epoch_utc('2025-01-01T08:00:00+08:00')
    record = dict(lon=1., lat=2., raw_motion=('0', '0', '0'))
    index = {('P', '123', 100.): [record]}
    assert not index.get(('P', '123', 101.), [])
    assert not index.get(('Q', '123', 100.), [])
    assert not index.get(('P', '456', 100.), [])
    assert resolve([record], [(1., 2.)])[1] == 'matched'
    assert resolve([record], [(3., 2.)])[1] == 'stored_coordinate_mismatch'
    other = dict(record, lon=3.)
    assert resolve([record, other], [])[1] == 'ambiguous_exact_records'
    assert resolve([record, other], [(1., 2.)])[0] is record
    assert resolve([record, dict(record)], [])[1] == 'matched'


def main():
    self_test()
    obs = pd.read_csv(BASE / 'observations.csv', dtype={'mmsi': str})
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    with np.load(BASE / 'incidence_transport_20261011/source_cohorts.npz') as z:
        splits = {s: z[s + '_obs_ids'].copy() for s in ('fit', 'held')}
    ids = np.concatenate(list(splits.values()))
    assert len(ids) == len(set(ids)) == 2119
    assert len(splits['fit']) == 1296 and len(splits['held']) == 823
    cohort = obs.set_index('obs_id').loc[ids].copy()
    assert not set(cohort.loc[splits['fit'], 'mmsi']) & set(cohort.loc[splits['held'], 'mmsi'])
    cohort['split'] = ['fit'] * len(splits['fit']) + ['held'] * len(splits['held'])
    aligned = manifest.iloc[cohort.manifest_row.to_numpy()].copy()
    aligned.index = cohort.index
    for key in ('instance_id', 'mmsi', 'product', 'port'):
        assert aligned[key].tolist() == cohort[key].astype(str).tolist(), key

    views = pd.read_csv(AUDIT / 'object_dimension_audit.csv', dtype=str, keep_default_na=False)
    objects, perobs, view_rows = {}, defaultdict(list), []
    for v in views.to_dict('records'):
        oid = int(v['obs_id'])
        assert oid in cohort.index
        row = cohort.loc[oid]
        assert v['uid'] == aligned.loc[oid, 'obj_uid_' + v['pol']]
        path = v['local_json']
        if path not in objects:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            byuid = defaultdict(list)
            for obj in data.get('objects', []):
                byuid[str(obj.get('obj_uid') or obj.get('source_obj_uid') or '')].append(obj)
            objects[path] = (data.get('image', ''), byuid)
        image, byuid = objects[path]
        matches = byuid[v['uid']]
        status = 'ok'
        obj = matches[0] if len(matches) == 1 else {}
        meta = obj.get('meta', {})
        time = meta.get('ais_time_utc', '')
        epoch = float('nan')
        if len(matches) != 1:
            status = 'ambiguous_or_missing_uid'
        elif row['product'] not in image:
            status = 'json_product_mismatch'
        elif str(meta.get('mmsi', '')) != row.mmsi:
            status = 'json_mmsi_mismatch'
        else:
            try:
                epoch = epoch_utc(time)
            except (ValueError, TypeError):
                status = 'invalid_json_time'
        entry = dict(obs_id=oid, split=row['split'], pol=v['pol'], uid=v['uid'],
                     local_json=path, mmsi=str(meta.get('mmsi', '')), product=row['product'],
                     ais_time_utc=time, epoch=epoch, status=status,
                     source=meta.get('source', ''), coordinates=stored_coordinates(obj),
                     position_method=meta.get('ais_position_method', ''),
                     position_uncertainty_m=meta.get('ais_position_uncertainty_m', ''))
        perobs[oid].append(entry)
        view_rows.append(entry)

    required = {(v['product'], v['mmsi'], v['epoch']) for v in view_rows if v['status'] == 'ok'}
    paths = list(dict.fromkeys(json.loads((AUDIT / 'read_source_txt_paths.json').read_text(encoding='utf-8'))))
    raw, files = defaultdict(list), []
    for name in paths:
        path = Path(name)
        product = PRODUCT.search(path.name).group()
        count = matched = 0
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for line, fields in enumerate(csv.reader(stream), 1):
                count += 1
                if len(fields) != 18:
                    continue
                key = (product, fields[1].strip(), finite(fields[0]))
                if key not in required:
                    continue
                lon, lat = finite(fields[6]), finite(fields[7])
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    continue
                raw[key].append(dict(product=product, mmsi=key[1], epoch=key[2],
                    lon=lon, lat=lat, raw_motion=tuple(fields[j].strip() for j in (4, 8, 9)),
                    source_txt=name, source_line=line))
                matched += 1
        files.append(dict(path=name, bytes=path.stat().st_size, rows_read=count, exact_key_rows=matched))

    output, evidence = [], []
    for oid, row in cohort.iterrows():
        vv = perobs.get(oid, [])
        coordinates = sorted({tuple(p) for v in vv for p in v['coordinates']})
        times = {v['epoch'] for v in vv if v['status'] == 'ok'}
        epoch = next(iter(times)) if len(times) == 1 else float('nan')
        candidates = raw.get((row['product'], row.mmsi, epoch), [])
        chosen = None
        if not vv:
            status = 'no_audited_uid_json'
        elif any(v['status'] != 'ok' for v in vv):
            status = 'invalid_uid_json_view'
        elif len(times) != 1:
            status = 'conflicting_view_times'
        else:
            chosen, status = resolve(candidates, coordinates)
        result = dict(obs_id=int(oid), split=row['split'], instance_id=row.instance_id,
            mmsi=row.mmsi, product=row['product'], match_status=status, ais_epoch=epoch,
            ais_time_utc=vv[0]['ais_time_utc'] if len(times) == 1 else '',
            abs_time_delta_seconds=abs(epoch - midpoint(row['product'])),
            stored_coordinates_available=bool(coordinates),
            stored_coordinates_json=json.dumps(coordinates),
            coordinate_check=('matched' if chosen else status) if coordinates else 'not_available',
            candidate_count=len(candidates), uid_view_count=len(vv),
            uids_json=json.dumps([v['uid'] for v in vv]),
            source_json_paths=json.dumps(sorted({v['local_json'] for v in vv})),
            source_records_json=json.dumps([dict(path=r['source_txt'], line=r['source_line']) for r in candidates]),
            raw_lon=chosen['lon'] if chosen else float('nan'),
            raw_lat=chosen['lat'] if chosen else float('nan'))
        values = decode_motion(*chosen['raw_motion']) if chosen else (float('nan'),) * 3
        for field, value in zip(FIELDS, values):
            result[field] = value
            result[field + '_valid'] = math.isfinite(value)
        for field, value in zip(('raw_sog', 'raw_cog', 'raw_true_heading'), chosen['raw_motion'] if chosen else ('',) * 3):
            result[field] = value
        output.append(result)
        evidence.extend(dict(obs_id=int(oid), split=row['split'], **r) for r in candidates)

    table = pd.DataFrame(output)
    arrays = {}
    for split, split_ids in splits.items():
        selected = table[table.split.eq(split)]
        assert np.array_equal(selected.obs_id.to_numpy(), split_ids)
        arrays[split + '_obs_ids'] = split_ids
        for field in FIELDS:
            values = selected[field].to_numpy(dtype=np.float64)
            mask = selected[field + '_valid'].to_numpy(dtype=bool)
            assert np.array_equal(np.isfinite(values), mask)
            arrays[split + '_' + field] = values
            arrays[split + '_' + field + '_valid'] = mask
        for field in ('ais_epoch', 'abs_time_delta_seconds', 'raw_lon', 'raw_lat'):
            arrays[split + '_' + field] = selected[field].to_numpy(dtype=np.float64)
        for field in ('match_status', 'mmsi', 'product', 'coordinate_check'):
            arrays[split + '_' + field] = selected[field].to_numpy(dtype=str)
        arrays[split + '_sog'] = arrays[split + '_sog_knots']
        arrays[split + '_sog_valid'] = arrays[split + '_sog_knots_valid']
    summary = dict(scope='source fit and held only; no target, training, nearest-time fallback or navigation status',
        self_tests='passed: zero, sentinels, ranges, timezone, exact time/MMSI/product keys, position and ambiguity',
        join='manifest UID -> unique JSON object -> MMSI/product/exact UTC epoch -> raw TXT',
        coordinate_tolerance_degrees=1e-6,
        coordinate_limit='Object/meta explicit lon/lat aliases only; crop pixel coordinates are not geographic coordinates',
        coverage_limit='Only previously audited UID views and listed TXT files; not full recoverable coverage',
        semantics=dict(sog_knots='column 4 / 10; 0 <= raw < 1023',
                       cog_deg='column 8 / 10; 0 <= raw < 3600; course over ground, not body heading',
                       true_heading_deg='column 9; 0 <= raw < 360; rejects 511'),
        no_navigation_status='Column 5 is position accuracy, not navigation status; not extracted',
        sources=dict(observations=str(BASE/'observations.csv'), manifest=str(MANIFEST),
            cohorts=str(BASE/'incidence_transport_20261011/source_cohorts.npz'),
            views=str(AUDIT/'object_dimension_audit.csv'), txt_paths=str(AUDIT/'read_source_txt_paths.json')),
        json_files_read=len(objects), txt_files_read=len(files), txt_bytes=sum(f['bytes'] for f in files),
        coverage={s: dict(observations=len(t), match_status=t.match_status.value_counts().to_dict(),
            valid={f: int(t[f+'_valid'].sum()) for f in FIELDS},
            valid_zero_sog=int((t.sog_knots_valid & t.sog_knots.eq(0)).sum()),
            stored_coordinates_available=int(t.stored_coordinates_available.sum()))
            for s, t in table.groupby('split')})
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT/'motion_knowledge.csv', index=False)
    pd.DataFrame(view_rows).to_csv(OUT/'uid_view_audit.csv', index=False)
    pd.DataFrame(files).to_csv(OUT/'read_sources.csv', index=False)
    np.savez_compressed(OUT/'motion_knowledge.npz', **arrays)
    np.savez_compressed(OUT/'attributes.npz', **arrays)
    with np.load(OUT/'attributes.npz', allow_pickle=False) as saved:
        for split, split_ids in splits.items():
            values, mask = saved[split+'_sog'], saved[split+'_sog_valid']
            assert values.ndim == mask.ndim == saved[split+'_obs_ids'].ndim == 1
            assert values.shape == mask.shape == split_ids.shape
            assert np.array_equal(saved[split+'_obs_ids'], split_ids)
            assert mask.dtype == np.bool_ and np.array_equal(np.isfinite(values), mask)
            assert np.array_equal(values, arrays[split+'_sog'], equal_nan=True)
    with (OUT/'exact_raw_records.jsonl').open('w', encoding='utf-8') as stream:
        for record in evidence:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT/'README.md').write_text(
        '# Source motion knowledge\n\n'
        'Run ../extract_motion_knowledge.py with the existing ClearSAR Python environment.\n'
        'fit means sourcefit; held is held source vessels, never a target dataset.\n'
        'Main consumer: attributes.npz; required 1D keys fit_obs_ids, held_obs_ids, fit_sog, held_sog, fit_sog_valid, held_sog_valid.\n'
        'sog aliases sog_knots (knots); validity masks are boolean. attributes.npz and motion_knowledge.npz have identical content.\n'
        'motion_knowledge.npz uses fit_ and held_ prefixes, in original cohort order.\n'
        'Keys: obs_ids, sog_knots, cog_deg, true_heading_deg; each motion field has a _valid mask.\n'
        'Additional keys: ais_epoch, abs_time_delta_seconds, raw_lon, raw_lat, match_status, mmsi, product, coordinate_check.\n'
        'Missing/invalid/ambiguous motion values are NaN. Zero speed is valid.\n'
        'motion_knowledge.csv contains one row per observation and provenance; uid_view_audit.csv preserves JSON evidence.\n'
        'exact_raw_records.jsonl retains all exact-key candidates and source line numbers.\n'
        'read_sources.csv bounds the TXT reads; summary.json records masks and limitations.\n'
        'COG is not true heading. No navigation-status field is produced.\n'
        'Use fit masks for auxiliary training only; held metadata is audit-only. SAR-only inference needs none of these fields.\n',
        encoding='utf-8')
    print(json.dumps(summary['coverage'], indent=2))


if __name__ == '__main__':
    main()
