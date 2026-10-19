"""Bounded annotation-orbit readout; no imagery, classifiers, or git writes.

Run with python -B recover_orbit.py [--download]. Existing geometry.csv is the
frozen whole-crop ownership gate. Raw annotation downloads live outside research.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
RECOVERY = Path('E:/Hermes/paper_tracking/sensitivity/f0921_incidence_recovery_20261010')
RAW = Path('E:/Hermes/paper_tracking/sensitivity/f0921_orbit_metadata_20261019')
TO_ECEF = Transformer.from_crs(4979, 4978, always_xy=True)
GEOD = Geod(ellps='WGS84')
MAX_FILE_BYTES = 8 * 1024**2
MAX_DOWNLOAD_BYTES = 64 * 1024**2
MAX_DOWNLOAD_FILES = 16


class Unavailable(ValueError):
    """An observed metadata/coverage condition prevents this readout."""


def epoch(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).replace(
        tzinfo=timezone.utc).timestamp()


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec='microseconds')


def local_basis(lon, lat):
    lon, lat = np.radians([lon, lat])
    return np.array([[-np.sin(lon), np.cos(lon), 0.],
                     [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                     [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])


def look_geometry(position, velocity, lon, lat):
    target = np.array(TO_ECEF.transform(lon, lat, 0.))
    basis = local_basis(lon, lat)
    delta = np.asarray(position) - target
    los = basis @ (delta / np.linalg.norm(delta))
    flight = basis @ np.asarray(velocity)
    if (not np.isfinite(np.r_[los, flight]).all() or los[2] <= 0
            or np.linalg.norm(los[:2]) == 0 or np.linalg.norm(flight[:2]) == 0):
        raise Unavailable('invalid_or_below_horizon_los')
    return los, flight, float(np.degrees(np.arctan2(np.linalg.norm(los[:2]), los[2])))


def enu_to_map_jacobian(lon, lat, transformer):
    # Central 1 m geodesic tangents preserve UTM meridian convergence and scale.
    columns = []
    for bearing in (90., 0.):
        plus = GEOD.fwd(lon, lat, bearing, 1.)[:2]
        minus = GEOD.fwd(lon, lat, bearing + 180., 1.)[:2]
        columns.append((np.array(transformer.transform(*plus)) -
                        np.array(transformer.transform(*minus))) / 2.)
    return np.column_stack(columns)


def image_direction(horizontal, map_jacobian, affine):
    mapped = map_jacobian @ np.asarray(horizontal)
    image = np.linalg.solve(affine, mapped)
    if not np.isfinite(image).all() or np.linalg.norm(image) == 0:
        raise Unavailable('invalid_image_direction')
    image = image / np.linalg.norm(image)
    mapped = mapped / np.linalg.norm(mapped)
    angle = float(np.arctan2(image[1], image[0]) % (2*np.pi))
    return mapped, image, angle


def axis_difference(a, b):
    return float(np.degrees(abs((a - b + np.pi/2) % np.pi - np.pi/2)))


def interpolate_grid(xy, times, query, values):
    """Barycentric interpolation in projected coordinates, never extrapolation."""
    tri = Delaunay(xy)
    simplex = int(tri.find_simplex(query))
    if simplex < 0:
        raise Unavailable('outside_geolocation_grid')
    ids = tri.simplices[simplex]
    weights = tri.transform[simplex, :2] @ (query - tri.transform[simplex, 2])
    weights = np.r_[weights, 1 - weights.sum()]
    # Subtract the epoch before interpolation to retain submillisecond precision.
    t = float(times[ids[0]] + weights @ (times[ids] - times[ids[0]]))
    if not np.isfinite(t) or not np.isfinite(values[ids]).all():
        raise Unavailable('nonfinite_grid_metadata')
    return t, weights @ values[ids]


def orbit_at(vectors, t, acquisition_start, acquisition_stop):
    if not np.isfinite(t) or not acquisition_start <= t <= acquisition_stop:
        raise Unavailable('outside_actual_acquisition_time')
    if len(vectors) < 2 or not np.isfinite(vectors).all():
        raise Unavailable('missing_or_nonfinite_orbit')
    if not np.all(np.diff(vectors[:, 0]) > 0):
        raise Unavailable('nonmonotonic_orbit_time')
    if not vectors[0, 0] <= t <= vectors[-1, 0]:
        raise Unavailable('outside_orbit_time_coverage')
    origin = vectors[0, 0]
    spline = CubicHermiteSpline(vectors[:, 0] - origin, vectors[:, 1:4],
                               vectors[:, 4:7], extrapolate=False)
    return spline(t-origin), spline(t-origin, 1)


def xml_root(path):
    data = path.read_bytes()
    return ET.fromstring(gzip.decompress(data) if path.suffix == '.gz' else data)


def grid_arrays(root):
    points = root.findall('.//geolocationGridPoint')
    if len(points) < 3:
        raise Unavailable('missing_geolocation_grid')
    names = ('longitude', 'latitude', 'pixel', 'line', 'incidenceAngle', 'height')
    values = np.array([[float(p.findtext(k)) for k in names] for p in points])
    times = np.array([epoch(p.findtext('azimuthTime')) for p in points])
    return values, times


def load_annotation(path, product, grid_path):
    root = xml_root(path)
    header = root.find('adsHeader')
    if header is None:
        raise Unavailable('missing_annotation_header')
    parts = product.split('_')
    start, stop = [epoch(header.findtext(k)) for k in ('startTime', 'stopTime')]
    actual_key = (header.findtext('missionId'),
                  datetime.fromtimestamp(start, timezone.utc).strftime('%Y%m%dT%H%M%S'),
                  datetime.fromtimestamp(stop, timezone.utc).strftime('%Y%m%dT%H%M%S'),
                  int(header.findtext('absoluteOrbitNumber')),
                  int(header.findtext('missionDataTakeId')))
    expected = (parts[0], parts[4], parts[5], int(parts[6]), int(parts[7], 16))
    if actual_key != expected or header.findtext('polarisation') != 'VV':
        raise Unavailable('annotation_product_identity_mismatch')
    points, times = grid_arrays(root)
    used_grid = path
    if grid_path.suffix == '.gz':
        cached_points, cached_times = grid_arrays(xml_root(grid_path))
        if (cached_points.shape != points.shape or
                not np.allclose(cached_points, points, rtol=0, atol=1e-8) or
                not np.allclose(cached_times, times, rtol=0, atol=1e-6)):
            raise Unavailable('cached_grid_annotation_mismatch')
        points, times, used_grid = cached_points, cached_times, grid_path
    info = root.find('imageAnnotation/imageInformation')
    if info is None:
        raise Unavailable('missing_image_time_bounds')
    first, last = [epoch(info.findtext(k)) for k in
                   ('productFirstLineUtcTime', 'productLastLineUtcTime')]
    interval = float(info.findtext('azimuthTimeInterval'))
    if max(abs(first-start), abs(last-stop)) > 1e-6:
        raise Unavailable('header_image_time_disagreement')
    # XML grid boundary times may differ slightly from first/last line metadata.
    # Accept at most one native line here; query time below has strict bounds.
    if times.min() < first-interval or times.max() > last+interval:
        raise Unavailable('grid_outside_acquisition_time')
    orbits = root.findall('generalAnnotation/orbitList/orbit')
    if any(o.findtext('frame') != 'Earth Fixed' for o in orbits):
        raise Unavailable('unsupported_orbit_frame')
    vectors = np.array([[epoch(o.findtext('time'))] +
                        [float(o.findtext(f'{kind}/{axis}'))
                         for kind in ('position', 'velocity') for axis in 'xyz']
                        for o in orbits], dtype=float).reshape(-1, 7)
    orbit_at(vectors, (first+last)/2, first, last)
    if vectors[0, 0] > first or vectors[-1, 0] < last:
        raise Unavailable('orbit_does_not_cover_acquisition')
    return dict(points=points, times=times, vectors=vectors, start=first, stop=last,
                grid_path=str(used_grid), annotation_path=str(path),
                grid_boundary_start_offset_s=float(times.min()-first),
                grid_boundary_stop_offset_s=float(times.max()-last),
                azimuth_time_interval_s=interval)


VECTOR_COLUMNS = [f'{prefix}_{axis}_{unit}' for prefix, axes, unit in
                  [('los', ('east', 'north', 'up'), 'unit'),
                   ('flight', ('east', 'north', 'up'), 'mps'),
                   ('los_map', ('easting', 'northing'), 'unit'),
                   ('los_image', ('x', 'y'), 'unit'),
                   ('flight_map', ('easting', 'northing'), 'unit'),
                   ('flight_image', ('x', 'y'), 'unit')] for axis in axes]
NUMERIC_COLUMNS = VECTOR_COLUMNS + [
    'los_angle_image_rad', 'los_axis_image_rad', 'flight_angle_image_rad',
    'flight_axis_image_rad', 'orbit_incidence_deg', 'axis_difference_deg',
    'incidence_difference_deg', 'azimuth_time_epoch_s', 'time_since_start_s',
    'time_until_stop_s', 'orbit_left_margin_s', 'orbit_right_margin_s',
    'native_range_pixel', 'native_azimuth_line', 'grid_incidence_deg',
    'interpolated_grid_height_m', 'los_dot_flight_unit']


def empty_record(row):
    out = {k: row[k] for k in ('row', 'clean_row', 'observation_key', 'resolved_product')}
    out.update({k: np.nan for k in NUMERIC_COLUMNS})
    out.update(orbit_valid=False, azimuth_time_utc=np.nan, annotation_path=np.nan,
               grid_path=np.nan, orbit_status='not_processed',
               old_geometry_valid=bool(row.geometry_valid),
               old_geometry_status=row.geometry_status)
    return out


def recover_row(row, annotation, transformer, affine):
    out = empty_record(row)
    if not (row.geometry_valid and row.pixel_product_unique and
            row.native_window_matches_manifest_raster):
        out['orbit_status'] = 'prior_geometry_gate:' + row.geometry_status
        return out
    if annotation is None:
        out['orbit_status'] = 'missing_annotation_orbit'
        return out
    out.update(annotation_path=annotation['annotation_path'], grid_path=annotation['grid_path'])
    try:
        points, times = annotation['points'], annotation['times']
        xy = np.column_stack(transformer.transform(points[:, 0], points[:, 1]))
        query = np.array(transformer.transform(row.lon, row.lat))
        t, values = interpolate_grid(xy, times, query, points[:, 2:])
        position, velocity = orbit_at(annotation['vectors'], t, annotation['start'], annotation['stop'])
        los, flight, incidence = look_geometry(position, velocity, row.lon, row.lat)
        map_jac = enu_to_map_jacobian(row.lon, row.lat, transformer)
        los_map, los_image, angle = image_direction(los[:2], map_jac, affine)
        flight_map, flight_image, flight_angle = image_direction(flight[:2], map_jac, affine)
        vector_values = np.r_[los, flight, los_map, los_image, flight_map, flight_image]
        out.update(dict(zip(VECTOR_COLUMNS, vector_values.tolist())))
        out.update(orbit_valid=True, orbit_status='usable_annotation_orbit',
                   los_angle_image_rad=angle, los_axis_image_rad=angle % np.pi,
                   flight_angle_image_rad=flight_angle, flight_axis_image_rad=flight_angle % np.pi,
                   orbit_incidence_deg=incidence,
                   axis_difference_deg=axis_difference(angle, row.radar_angle_image_rad),
                   incidence_difference_deg=incidence-values[2],
                   azimuth_time_epoch_s=t, azimuth_time_utc=iso(t),
                   time_since_start_s=t-annotation['start'], time_until_stop_s=annotation['stop']-t,
                   orbit_left_margin_s=t-annotation['vectors'][0, 0],
                   orbit_right_margin_s=annotation['vectors'][-1, 0]-t,
                   native_range_pixel=values[0], native_azimuth_line=values[1],
                   grid_incidence_deg=values[2], interpolated_grid_height_m=values[3],
                   los_dot_flight_unit=float(los @ (flight / np.linalg.norm(flight))))
    except Unavailable as exc:
        out['orbit_status'] = str(exc)
    return out


def fetch_annotation(product, url, download, budget):
    path = RAW / (product + '.xml')
    if path.exists():
        return path, dict(source='external_raw_cache', bytes=path.stat().st_size)
    if not download:
        raise Unavailable('annotation_not_cached_download_disabled')
    if not isinstance(url, str) or not url.startswith('https://sentinel-s1-l1c.s3.amazonaws.com/'):
        raise Unavailable('missing_public_annotation_url')
    if budget['files'] >= MAX_DOWNLOAD_FILES or budget['bytes'] >= MAX_DOWNLOAD_BYTES:
        raise Unavailable('bounded_download_budget_exhausted')
    budget['files'] += 1
    limit = min(MAX_FILE_BYTES, MAX_DOWNLOAD_BYTES-budget['bytes'])
    request = urllib.request.Request(url, headers={'User-Agent': 'bounded-orbit-metadata/1.0'})
    with urllib.request.urlopen(request, timeout=25) as response:
        data = response.read(limit+1)
    budget['bytes'] += len(data)
    if len(data) > limit:
        raise Unavailable('annotation_exceeds_download_limit')
    root = ET.fromstring(data)
    if root.tag != 'product':
        raise Unavailable('download_not_annotation_product')
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, dict(source='public_annotation_download', url=url, bytes=len(data))


def stats(series):
    series = series.dropna()
    if series.empty:
        return dict(count=0)
    return dict(count=len(series), min=float(series.min()), median=float(series.median()),
                mean=float(series.mean()), p95=float(series.quantile(.95)), max=float(series.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='allow bounded public annotation XML downloads')
    args = parser.parse_args()
    # Read only raster headers; neither pixels nor PAM sidecars are used/written.
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    cohort = pd.read_csv(BRIDGE / 'cohort.csv')
    old = pd.read_csv(BRIDGE / 'axis_radar_readout_20261019/geometry.csv')
    if len(cohort) != 269 or cohort.row.duplicated().any() or old.row.duplicated().any():
        raise ValueError('Expected unique frozen 269-row cohort and geometry keys')
    old = old.set_index('row').loc[cohort.row].reset_index()
    for key in ('row', 'clean_row', 'observation_key', 'source_tif', 'lon', 'lat', 'left', 'top', 'side'):
        if not np.array_equal(old[key].to_numpy(), cohort[key].to_numpy()):
            raise ValueError('Frozen cohort/geometry mismatch: ' + key)
    inventory = pd.read_csv(RECOVERY / 'download_inventory.csv')
    urls = inventory.drop_duplicates('product').set_index('product').url.to_dict()
    budget = dict(files=0, bytes=0)
    annotations, metadata = {}, []
    eligible = old[old.geometry_valid & old.pixel_product_unique & old.native_window_matches_manifest_raster]
    for product, group in eligible.groupby('resolved_product', sort=False):
        grid_path = Path(group.iloc[0].annotation_path)
        entry = dict(product=product, eligible_rows=len(group))
        try:
            if grid_path.suffix == '.xml':
                path, source = grid_path, dict(source='existing_local_annotation', bytes=grid_path.stat().st_size)
            else:
                path, source = fetch_annotation(product, urls.get(product), args.download, budget)
            entry.update(source, annotation_path=str(path))
            annotations[product] = load_annotation(path, product, grid_path)
            a = annotations[product]
            entry.update(status='ready', orbit_vectors=len(a['vectors']), grid_points=len(a['points']),
                         start_utc=iso(a['start']), stop_utc=iso(a['stop']),
                         orbit_start_utc=iso(a['vectors'][0, 0]), orbit_stop_utc=iso(a['vectors'][-1, 0]),
                         grid_boundary_start_offset_s=a['grid_boundary_start_offset_s'],
                         grid_boundary_stop_offset_s=a['grid_boundary_stop_offset_s'],
                         azimuth_time_interval_s=a['azimuth_time_interval_s'])
        except (Unavailable, urllib.error.URLError, TimeoutError, ET.ParseError, OSError) as exc:
            entry.update(status='unavailable', reason=str(exc))
        metadata.append(entry)
        print(product[-4:] + ': ' + entry['status'] + ' ' + entry.get('reason', ''), flush=True)
    headers, records = {}, []
    failures = {m['product']: m.get('reason') for m in metadata if m['status'] != 'ready'}
    for _, row in old.iterrows():
        a = annotations.get(row.resolved_product)
        if not row.geometry_valid or a is None:
            record = recover_row(row, None, None, None)
            if row.geometry_valid and row.resolved_product in failures:
                record['orbit_status'] = 'annotation_unavailable:' + failures[row.resolved_product]
        else:
            if row.source_tif not in headers:
                ds = gdal.Open(row.source_tif, gdal.GA_ReadOnly)
                gt = ds.GetGeoTransform()
                affine = np.array([[gt[1], gt[2]], [gt[4], gt[5]]])
                transformer = Transformer.from_crs(4326, ds.GetProjection(), always_xy=True)
                headers[row.source_tif] = transformer, affine
                ds = None
            record = recover_row(row, a, *headers[row.source_tif])
        records.append(record)
    frame = pd.DataFrame(records)
    valid = frame[frame.orbit_valid]
    if not frame.row.equals(cohort.row) or not frame.loc[~frame.orbit_valid, NUMERIC_COLUMNS].isna().all().all():
        raise ValueError('Output alignment or missing-value contract violated')
    frame.to_csv(HERE / 'orbit_geometry.csv', index=False, na_rep='NaN')
    receipt = dict(rows=len(frame), old_geometry_valid=int(old.geometry_valid.sum()),
                   orbit_valid=int(frame.orbit_valid.sum()), orbit_invalid=int((~frame.orbit_valid).sum()),
                   statuses=frame.orbit_status.value_counts().to_dict(),
                   axis_difference_deg=stats(valid.axis_difference_deg),
                   incidence_difference_deg=stats(valid.incidence_difference_deg),
                   absolute_los_dot_flight=stats(valid.los_dot_flight_unit.abs()),
                   time_since_start_s=stats(valid.time_since_start_s),
                   time_until_stop_s=stats(valid.time_until_stop_s),
                   by_product=frame.groupby('resolved_product', sort=False).agg(
                       rows=('row', 'size'), valid=('orbit_valid', 'sum'),
                       axis_difference_median_deg=('axis_difference_deg', 'median'),
                       axis_difference_max_deg=('axis_difference_deg', 'max')).reset_index().where(
                           lambda x: x.notna(), None).to_dict('records'),
                   annotation_metadata=metadata, downloads=budget,
                   raw_metadata_directory=str(RAW), created_utc=datetime.now(timezone.utc).isoformat(),
                   interpolation='Projected geolocationGridPoint barycentric azimuthTime; ECEF position/velocity CubicHermiteSpline, no extrapolation',
                   direction='target->sensor; atan2(image_dy,image_dx) in [0,2pi); axis modulo pi; no pass-based sign',
                   mapping='ENU geodesic +/-1m differential to source UTM; inverse raster affine; square uniform chip resize preserves angle',
                   time_coverage='Header=image first/last times; grid boundary tolerance <=1 native line; strict query acquisition bounds; orbit covers full acquisition and query',
                   ownership='Frozen geometry_valid AND pixel_product_unique AND native_window_matches_manifest_raster; no eligibility expansion or new pixel audit',
                   limitations=['WGS84 ellipsoid height=0 sea-level proxy, not geoid sea level or ship height',
                                'Product annotation orbit, not independently obtained precise orbit',
                                'Finite grid interpolation and finite-difference projection; not a range-Doppler terrain solution',
                                'Old native-grid unsigned axis and LOS ground projection are distinct quantities',
                                'Public URL/cache and header establish product lineage; no independent orbit accuracy validation'],
                   reference_status='web.run returned empty in this branch; no official specification read online; implementation grounded in raw XML and existing recover_metadata.py',
                   tests=dict(status='not_recorded'), imagery_read=False, training=False, git_operations=False)
    # JSON null for unavailable summaries, unlike the CSV explicit NaN contract.
    receipt['by_product'] = json.loads(pd.DataFrame(receipt['by_product']).to_json(orient='records'))
    (HERE / 'orbit_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('rows', 'orbit_valid', 'statuses', 'axis_difference_deg',
                                             'incidence_difference_deg', 'downloads')}, indent=2))


if __name__ == '__main__':
    main()
