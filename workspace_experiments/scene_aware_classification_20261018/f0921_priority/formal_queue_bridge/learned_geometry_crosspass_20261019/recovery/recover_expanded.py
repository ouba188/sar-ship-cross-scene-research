"""Recover the frozen 269 rows using ownership gates and full annotations.

Run with ClearSAR/python.exe -B recover_expanded.py --download.
All writes stay here. Existing external XML and old experiments are read-only.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent.parent
OLD_SCRIPT = BRIDGE / 'orbit_geometry_readout_20261019/recover_orbit.py'
OLD_CSV = OLD_SCRIPT.with_name('orbit_geometry.csv')
OWNERSHIP_CSV = BRIDGE / 'axis_radar_readout_20261019/geometry.csv'
RAW_CACHE = Path('E:/Hermes/paper_tracking/sensitivity/f0921_orbit_metadata_20261019')
MAX_FILES = 4
MAX_FILE_BYTES = 8 * 1024**2
MAX_TOTAL_BYTES = 32 * 1024**2
spec = importlib.util.spec_from_file_location('frozen_orbit_recovery', OLD_SCRIPT)
orbit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orbit)


def eligible(row):
    return bool(row.pixel_product_unique and row.native_window_matches_manifest_raster)


def product_key(product):
    parts = product.split('_')
    return '|'.join((parts[0], parts[4], parts[-1]))


def public_url(product):
    parts = product.split('_')
    date = datetime.strptime(parts[4], '%Y%m%dT%H%M%S')
    if parts[:4] != ['S1A', 'IW', 'GRDH', '1SDV']:
        raise orbit.Unavailable('unsupported_public_annotation_product')
    return (f'https://sentinel-s1-l1c.s3.amazonaws.com/GRD/{date.year}/'
            f'{date.month}/{date.day}/IW/DV/{product}/annotation/iw-vv.xml')


def full_annotation(product, group, download, budget):
    url = public_url(product)
    candidates = [(RAW_CACHE / (product + '.xml'), 'existing_external_raw_cache'),
                  (HERE / 'raw_xml' / (product + '.xml'), 'existing_recovery_raw_cache')]
    candidates += [(Path(p), 'existing_local_full_annotation')
                   for p in group.annotation_path.unique() if Path(p).suffix == '.xml']
    for path, source in candidates:
        if path.is_file():
            return path, dict(source=source, bytes=path.stat().st_size, public_url=url)
    if not download:
        raise orbit.Unavailable('annotation_not_cached_download_disabled')
    if budget['attempted_files'] >= MAX_FILES or budget['bytes'] >= MAX_TOTAL_BYTES:
        raise orbit.Unavailable('bounded_download_budget_exhausted')
    budget['attempted_files'] += 1
    limit = min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - budget['bytes'])
    request = urllib.request.Request(url, headers={'User-Agent': 'bounded-orbit-recovery/1.0'})
    # Public HTTPS only: no authentication handler, credentials, or proxy lookup.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=25) as response:
        data = response.read(limit)
        budget['bytes'] += len(data)
        if len(data) >= limit:
            raise orbit.Unavailable('annotation_reached_download_limit')
    root = ET.fromstring(data)
    if root.tag != 'product':
        raise orbit.Unavailable('download_not_annotation_product')
    path = HERE / 'raw_xml' / (product + '.xml')
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    budget['downloaded_files'] += 1
    return path, dict(source='public_full_annotation_download', bytes=len(data), public_url=url)


def recover_row(row, annotation, transformer=None, affine=None):
    if not eligible(row):
        out = orbit.empty_record(row)
        reasons = []
        if not row.pixel_product_unique:
            reasons.append('multiple_or_nonmatching_product_footprints')
        if not row.native_window_matches_manifest_raster:
            reasons.append('original_current_native_window_mismatch')
        out['orbit_status'] = 'ownership_gate:' + ';'.join(reasons)
    else:
        # Adapt only the obsolete gate in memory; retain original provenance below.
        adapted = row.copy()
        adapted['geometry_valid'] = True
        out = orbit.recover_row(adapted, annotation, transformer, affine)
        out['old_geometry_valid'] = bool(row.geometry_valid)
        out['old_geometry_status'] = row.geometry_status
    out.update(geometry_eligible=eligible(row),
               pixel_product_unique=bool(row.pixel_product_unique),
               native_window_matches_manifest_raster=bool(row.native_window_matches_manifest_raster),
               source_tif=row.source_tif, source_manifest=row.source_manifest,
               contributor_observation_keys_json=row.contributor_observation_keys_json)
    return out


def load_inputs():
    cohort = pd.read_csv(BRIDGE / 'cohort.csv')
    old = pd.read_csv(OWNERSHIP_CSV)
    baseline = pd.read_csv(OLD_CSV)
    for frame in (cohort, old, baseline):
        if len(frame) != 269 or frame.row.duplicated().any():
            raise ValueError('Expected frozen unique 269-row inputs')
    old = old.set_index('row').loc[cohort.row].reset_index()
    baseline = baseline.set_index('row').loc[cohort.row].reset_index()
    for key in ('row', 'clean_row', 'observation_key', 'source_tif', 'lon', 'lat', 'left', 'top', 'side'):
        np.testing.assert_array_equal(old[key], cohort[key], err_msg=key)
    for key in ('row', 'clean_row', 'observation_key', 'resolved_product'):
        np.testing.assert_array_equal(old[key], baseline[key], err_msg=key)
    for row in old.itertuples():
        if product_key(row.resolved_product) != row.observation_key:
            raise ValueError(f'Product/observation identity mismatch: {row.row}')
        if eligible(row) and json.loads(row.contributor_observation_keys_json) != [row.observation_key]:
            raise ValueError(f'Eligible row has inconsistent contributor identity: {row.row}')
    return cohort, old, baseline


def verify_output(frame, old, baseline):
    for key in ('row', 'clean_row', 'observation_key', 'resolved_product'):
        np.testing.assert_array_equal(frame[key], baseline[key], err_msg=key)
    mask = old.pixel_product_unique & old.native_window_matches_manifest_raster
    np.testing.assert_array_equal(frame.geometry_eligible, mask)
    if (frame.orbit_valid & ~mask).any():
        raise AssertionError('Ownership-rejected row became valid')
    if not frame.loc[~frame.orbit_valid, orbit.NUMERIC_COLUMNS].isna().all().all():
        raise AssertionError('Missing geometry was filled with numeric defaults')
    previous = baseline.orbit_valid
    if int(previous.sum()) != 132 or not frame.loc[previous, 'orbit_valid'].all():
        raise AssertionError('Lost old 132 valid rows')
    differences = {}
    for column in orbit.NUMERIC_COLUMNS:
        a, b = frame.loc[previous, column], baseline.loc[previous, column]
        atol = 1e-6 if column == 'azimuth_time_epoch_s' else 1e-9
        np.testing.assert_allclose(a, b, rtol=0, atol=atol, equal_nan=True, err_msg=column)
        differences[column] = float((a-b).abs().max())
    for column in ('azimuth_time_utc', 'orbit_status', 'old_geometry_valid', 'old_geometry_status'):
        np.testing.assert_array_equal(frame.loc[previous, column], baseline.loc[previous, column])
    return dict(status='passed', old_valid_replayed=132, numeric_max_abs_difference=differences,
                numeric_absolute_tolerance=1e-9, epoch_absolute_tolerance_s=1e-6,
                all_rows_identity_and_gate=True, rejected_numeric_values_all_missing=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    cohort, old, baseline = load_inputs()
    mask = old.pixel_product_unique & old.native_window_matches_manifest_raster
    annotations, metadata = {}, []
    budget = dict(attempted_files=0, downloaded_files=0, bytes=0)
    for product, group in old[mask].groupby('resolved_product', sort=False):
        entry = dict(product=product, observation_key=product_key(product),
                     eligible_rows=len(group), eligible_row_ids=group.row.tolist())
        try:
            path, source = full_annotation(product, group, args.download, budget)
            entry.update(source, annotation_path=str(path))
            # Read every grid field from the full original, never the truncated cache.
            a = orbit.load_annotation(path, product, path)
            annotations[product] = a
            entry.update(status='ready', identity_verified=True, polarisation='VV',
                         orbit_vectors=len(a['vectors']), grid_points=len(a['points']),
                         start_utc=orbit.iso(a['start']), stop_utc=orbit.iso(a['stop']))
        except (orbit.Unavailable, urllib.error.URLError, TimeoutError, ET.ParseError, OSError) as exc:
            entry.update(status='unavailable', reason=str(exc))
        metadata.append(entry)
        print(f'{product[-4:]}: {entry["status"]} {entry.get("reason", "")}', flush=True)
    failures = {m['product']: m['reason'] for m in metadata if m['status'] != 'ready'}
    headers, records = {}, []
    for _, row in old.iterrows():
        annotation = annotations.get(row.resolved_product)
        if eligible(row) and annotation is not None:
            if row.source_tif not in headers:
                ds = gdal.Open(row.source_tif, gdal.GA_ReadOnly)
                gt = ds.GetGeoTransform()
                headers[row.source_tif] = (Transformer.from_crs(4326, ds.GetProjection(), always_xy=True),
                                           np.array([[gt[1], gt[2]], [gt[4], gt[5]]]))
                ds = None
            out = recover_row(row, annotation, *headers[row.source_tif])
        else:
            out = recover_row(row, None)
            if eligible(row) and row.resolved_product in failures:
                out['orbit_status'] = 'annotation_unavailable:' + failures[row.resolved_product]
        records.append(out)
    frame = pd.DataFrame(records)
    frame = frame[list(baseline.columns) + [c for c in frame if c not in baseline.columns]]
    checks = verify_output(frame, old, baseline)
    frame.to_csv(HERE / 'expanded_geometry.csv', index=False, na_rep='NaN')
    joined = frame[['row', 'orbit_valid', 'geometry_eligible']].merge(
        cohort[['row', 'date', 'class_raw', 'mmsi']], on='row', validate='one_to_one')
    joined['old_valid'] = baseline.orbit_valid.to_numpy()
    by_date = joined.groupby('date').agg(rows=('row', 'size'), old_valid=('old_valid', 'sum'),
                                        eligible=('geometry_eligible', 'sum'), valid=('orbit_valid', 'sum')).reset_index()
    by_class = joined.groupby('class_raw').agg(rows=('row', 'size'), old_valid=('old_valid', 'sum'),
                                              valid=('orbit_valid', 'sum')).reset_index()
    receipt = dict(rows=len(frame), eligible=int(mask.sum()),
                   old_orbit_valid=int(baseline.orbit_valid.sum()), orbit_valid=int(frame.orbit_valid.sum()),
                   newly_valid=int((frame.orbit_valid & ~baseline.orbit_valid).sum()),
                   missing=int((~frame.orbit_valid).sum()), statuses=frame.orbit_status.value_counts().to_dict(),
                   eligible_unrecovered_row_ids=frame.loc[mask & ~frame.orbit_valid, 'row'].tolist(),
                   newly_valid_row_ids=frame.loc[frame.orbit_valid & ~baseline.orbit_valid, 'row'].tolist(),
                   by_date=by_date.to_dict('records'), by_class=by_class.to_dict('records'),
                   annotation_metadata=metadata, downloads=budget,
                   download_limits=dict(files=MAX_FILES, per_file_bytes=MAX_FILE_BYTES, total_bytes=MAX_TOTAL_BYTES),
                   gate='pixel_product_unique AND native_window_matches_manifest_raster',
                   grid_source='Full original annotation for all rows; old 3-column/grid caches not used for computation',
                   identity='Frozen row/clean_row/observation key; unique contributor key; XML mission/start/stop/orbit/datatake/VV',
                   old_axis_difference='NaN for newly eligible rows: no old safe radar-axis reference exists',
                   code_reused=str(OLD_SCRIPT), ownership_input=str(OWNERSHIP_CSV), baseline_input=str(OLD_CSV),
                   read_only_raw_cache=str(RAW_CACHE), write_root=str(HERE), checks=checks,
                   limitations=['Frozen ownership/window evidence reused, no new pixel ownership audit',
                                '269 development rows; geometry coverage is not fold-specific legal training capacity',
                                'Annotation orbit and WGS84 height=0 proxy; inherited interpolation/accuracy limits'],
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   raster_headers_only=True, imagery_read=False, training=False, git_operations=False)
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Geometry recovery coverage', '',
             f'Valid: {receipt["orbit_valid"]}/269; previous: 132/269; added: {receipt["newly_valid"]}.',
             f'Ownership eligible: {receipt["eligible"]}; missing: {receipt["missing"]}.', '',
             '| Date | Rows | Previous | Eligible | Recovered |', '| --- | ---: | ---: | ---: | ---: |']
    lines += [f'| {r.date} | {r.rows} | {r.old_valid} | {r.eligible} | {r.valid} |' for r in by_date.itertuples()]
    lines += ['', 'Missing reasons:', ''] + [f'- {k}: {v}' for k, v in receipt['statuses'].items() if k != 'usable_annotation_orbit']
    lines += ['', 'The 108 newly eligible rows previously had incidence_only metadata from old caches without native pixel/line.',
              'Mixed/nonmatching product windows remain missing. No default angles or pass-derived signs are assigned.',
              'All original orbit CSV columns are retained. axis_difference_deg remains NaN on newly valid rows because the old reference angle is missing.',
              'The old 132 valid rows are recomputed from full XML and replay-checked; see receipt.json for per-column differences.',
              'This is metadata coverage only, not an evaluation of the learner or independent evidence of classification gains.',
              f'Public XML downloads this run: {budget["downloaded_files"]} files, {budget["bytes"]} bytes. No imagery or credentials read.',
              'Writes are confined to recovery/. The old experiments and external raw cache are read-only. No Git operations.', '']
    (HERE / 'COVERAGE.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('rows', 'eligible', 'old_orbit_valid', 'orbit_valid', 'newly_valid', 'missing', 'downloads')}, indent=2))


if __name__ == '__main__':
    main()
