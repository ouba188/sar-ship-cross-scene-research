"""Recover bounded local geometry; no downloads, training, or source mutation."""
import gzip
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from osgeo import gdal, osr
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
PRIORITY = BRIDGE.parent
TRACK = Path('E:/Hermes/paper_tracking')
RECOVERY = TRACK / 'sensitivity/f0921_incidence_recovery_20261010'
NAME = re.compile(r'_(\d{8}T\d{6})_.*_(S1[ABC])_([A-F0-9]{4})\.tif$')


def product_key(product):
    parts = product.split('_')
    return '|'.join([parts[0], parts[4], parts[-1]])


def transform_for(wkt):
    geo = osr.SpatialReference()
    geo.ImportFromEPSG(4326)
    target = osr.SpatialReference(wkt=wkt)
    for srs in (geo, target):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(geo, target)


def local_geometry(xy, incidence, native, query, affine):
    """Piecewise-affine native pixel/line -> projected map -> chip directions."""
    tri = Delaunay(xy)
    simplex = int(tri.find_simplex(query))
    if simplex < 0:
        return dict(status='outside_annotation_grid')
    ids = tri.simplices[simplex]
    weights = tri.transform[simplex, :2] @ (query - tri.transform[simplex, 2])
    weights = np.r_[weights, 1 - weights.sum()]
    result = dict(incidence=float(weights @ incidence[ids]), status='incidence_only')
    if native is None:
        return result
    # Native coordinates are (range pixel, azimuth line), never map row/column.
    design = np.c_[native[ids] - native[ids[0]], np.ones(3)]
    jac = np.linalg.solve(design, xy[ids] - xy[ids[0]])[:2].T
    image_jac = np.linalg.solve(affine, jac)
    range_vector, azimuth_vector = image_jac[:, 0], image_jac[:, 1]
    result.update(angle=float(np.arctan2(range_vector[1], range_vector[0]) % np.pi),
                  azimuth=float(np.arctan2(azimuth_vector[1], azimuth_vector[0]) % np.pi),
                  native_pixel=float(weights @ native[ids, 0]),
                  native_line=float(weights @ native[ids, 1]), status='native_grid_axes')
    return result


def contributor_keys(manifest, row):
    keys = set()
    count = 0
    for source in manifest['sources']:
        window = source['destination_window']
        x, y = window['column_offset'], window['row_offset']
        overlaps = (x < row.left + row.side and x + window['width'] > row.left
                    and y < row.top + row.side and y + window['height'] > row.top)
        if not overlaps or source['pixels_written'] == 0:
            continue
        match = NAME.search(source['filename'])
        keys.add('|'.join([match[2], match[1], match[3]]) if match else 'UNKNOWN')
        count += 1
    return sorted(keys), count


def read_annotation(product, annotations):
    full = annotations.get(product_key(product))
    raw = RECOVERY / 'xml_grid' / (product + '.xml.gz')
    heading = np.nan
    if full is not None or raw.exists():
        path = full if full is not None else raw
        root = ET.parse(path).getroot() if full is not None else ET.fromstring(gzip.decompress(raw.read_bytes()))
        points = root.findall('.//geolocationGridPoint')
        columns = ['longitude', 'latitude', 'incidenceAngle', 'pixel', 'line']
        values = np.array([[float(p.findtext(k)) for k in columns] for p in points])
        if root.findtext('.//platformHeading') is not None:
            heading = float(root.findtext('.//platformHeading'))
        return values[:, :2], values[:, 2], values[:, 3:], str(path), heading
    path = TRACK / 's1_grid' / (product + '.json')
    values = np.array(json.loads(path.read_text(encoding='utf-8')), dtype=float)
    # This older cache discarded line/pixel; never reconstruct them from list order.
    return values[:, [1, 0]], values[:, 2], None, str(path), heading


def main():
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64 * 1024**2)
    cohort = pd.read_csv(BRIDGE / 'cohort.csv', dtype={'mmsi': str, 'code': str})
    assert cohort.row.tolist() == list(range(269))
    catalog = json.loads((RECOVERY / 'catalog_matches.json').read_text(encoding='utf-8'))
    annotations = {}
    paths = list((PRIORITY / 'scene_5ffe').glob('*-vv-*.xml'))
    paths += list((PRIORITY / 'recovery_31cc_2ec7/annotation').glob('*/*-vv-*.xml'))
    for path in paths:
        root = ET.parse(path).getroot()
        start = root.findtext('.//adsHeader/startTime').split('.')[0].replace('-', '').replace(':', '')
        sensor = root.findtext('.//adsHeader/missionId')
        if path.parent.name.startswith('S1'):
            key = product_key(path.parent.name)
        else:
            candidates = cohort.loc[(cohort.sensor == sensor) & (cohort.obs_time == start), 'observation_key'].unique()
            assert len(candidates) == 1
            key = candidates[0]
        annotations[key] = path
    datasets, manifests, transforms, grids = {}, {}, {}, {}
    output = []
    for row in cohort.itertuples():
        candidates = {a['product'] for a in catalog.get(row.observation_key, [])}
        if pd.notna(row.resolved_product):
            candidates.add(row.resolved_product)
        assert len(candidates) == 1, (row.row, candidates)
        product = next(iter(candidates))
        assert product_key(product) == row.observation_key
        if row.source_tif not in datasets:
            current = gdal.Open(row.source_tif, gdal.GA_ReadOnly)
            original_path = Path(row.source_tif.replace('F_0921_relabel_det_final', 'F_0921'))
            original = gdal.Open(str(original_path), gdal.GA_ReadOnly)
            manifest_path = original_path.with_suffix('.manifest.json')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            assert manifest['overlap_method'] == 'first' and manifest['resampling'] == 'nearest'
            assert original.GetGeoTransform() == current.GetGeoTransform()
            assert original.GetProjection() == current.GetProjection()
            datasets[row.source_tif] = current, original
            manifests[row.source_tif] = manifest, manifest_path
            transforms[row.source_tif] = transform_for(current.GetProjection())
        current, original = datasets[row.source_tif]
        manifest, manifest_path = manifests[row.source_tif]
        transform = transforms[row.source_tif]
        gt = current.GetGeoTransform()
        affine = np.array([[gt[1], gt[2]], [gt[4], gt[5]]])
        assert np.array_equal(affine, [[10., 0.], [0., -10.]])
        keys, count = contributor_keys(manifest, row)
        unique_product = keys == [row.observation_key]
        # Only these 269 bounded native windows are read, not the full mosaics.
        window = (int(row.left), int(row.top), int(row.side), int(row.side))
        same_pixels = np.array_equal(current.ReadAsArray(*window), original.ReadAsArray(*window), equal_nan=True)
        grid_key = product, row.source_tif
        if grid_key not in grids:
            lonlat, incidence, native, path, heading = read_annotation(product, annotations)
            projected = np.array(transform.TransformPoints(lonlat.tolist()))[:, :2]
            grids[grid_key] = projected, incidence, native, path, heading
        xy, incidence, native, path, heading = grids[grid_key]
        position = np.array(transform.TransformPoint(row.lon, row.lat)[:2])
        geometry = local_geometry(xy, incidence, native, position, affine)
        record_valid = geometry['status'] == 'native_grid_axes'
        valid = record_valid and unique_product and same_pixels
        reasons = []
        if not unique_product:
            reasons.append('multiple_or_nonmatching_product_footprints')
        if not same_pixels:
            reasons.append('original_current_native_window_mismatch')
        if not record_valid:
            reasons.append(geometry['status'])
        output.append(dict(
            row=row.row, radar_angle_image_rad=geometry.get('angle', np.nan) if valid else np.nan,
            incidence_deg=geometry.get('incidence', np.nan) if valid else np.nan,
            geometry_valid=bool(valid), record_radar_angle_image_rad=geometry.get('angle', np.nan),
            record_incidence_deg=geometry.get('incidence', np.nan), record_geometry_valid=record_valid,
            record_azimuth_angle_image_rad=geometry.get('azimuth', np.nan),
            platform_heading_deg=heading, platform_heading_scope='product_annotation_not_local_chip_heading',
            observation_key=row.observation_key, resolved_product=product, clean_row=row.clean_row,
            mmsi=row.mmsi, source_tif=row.source_tif, source_fid=row.source_fid,
            lon=row.lon, lat=row.lat, left=row.left, top=row.top, side=row.side,
            chip_m_per_pixel=10. * row.side / 224.,
            chip_x0_center_easting=gt[0] + 10. * (row.left + .5 * row.side / 224.),
            chip_y0_center_northing=gt[3] - 10. * (row.top + .5 * row.side / 224.),
            image_axes='x=UTM_easting_positive;y=UTM_northing_negative;not_true_EN_or_SAR_axes',
            angle_convention='atan2(image_dy,image_dx)_mod_pi;range_pixel_increasing',
            geometry_status='usable_product_exclusive_window' if valid else ';'.join(reasons),
            ownership_evidence='all_intersecting_written_source_windows_share_record_product' if unique_product else 'unresolved_multiple_product_footprints',
            pixel_product_unique=unique_product, native_window_matches_manifest_raster=same_pixels,
            contributor_observation_keys_json=json.dumps(keys), contributor_source_window_count=count,
            source_manifest=str(manifest_path), annotation_path=path,
            native_range_pixel=geometry.get('native_pixel', np.nan), native_azimuth_line=geometry.get('native_line', np.nan),
            uncertainty='local_piecewise_affine_geolocation_grid;no_per_pixel_source_chip_owner;not_3D_LOS_or_AIS_heading'))
    frame = pd.DataFrame(output)
    assert frame.row.tolist() == list(range(269))
    assert frame.loc[~frame.geometry_valid, ['radar_angle_image_rad', 'incidence_deg']].isna().all().all()
    valid_rows = frame[frame.geometry_valid]
    assert valid_rows.radar_angle_image_rad.between(0, np.pi, inclusive='left').all()
    assert valid_rows.incidence_deg.between(0, 90, inclusive='neither').all()
    frame.to_csv(HERE / 'geometry.csv', index=False)
    receipt = dict(rows=len(frame), products=frame.resolved_product.nunique(),
                   geometry_valid=int(frame.geometry_valid.sum()),
                   record_geometry_valid=int(frame.record_geometry_valid.sum()),
                   record_incidence_available=int(frame.record_incidence_deg.notna().sum()),
                   product_exclusive_windows=int(frame.pixel_product_unique.sum()),
                   native_window_matches=int(frame.native_window_matches_manifest_raster.sum()),
                   product_heading_available=int(frame.platform_heading_deg.notna().sum()),
                   statuses=frame.geometry_status.value_counts().to_dict(),
                   by_product=frame.groupby('resolved_product').agg(rows=('row','size'),
                       usable=('geometry_valid','sum'),record_axes=('record_geometry_valid','sum'),
                       incidence=('record_incidence_deg','count')).reset_index().to_dict('records'),
                   annotation_files=sorted(frame.annotation_path.unique().tolist()),
                   join='row against unchanged 269-row cohort, check clean_row/observation_key; catalog uses sensor/start/code',
                   interpolation='Delaunay within projected annotation hull; local native (pixel,line)->UTM Jacobian; inverse mosaic affine; angle modulo pi',
                   ownership='Conservative whole-crop possible-contributor bound from recorded destination windows and positive total pixels_written. Multiple product candidates remain unknown. Same-product overlapping source chips are allowed. Exact original/current native window equality checked.',
                   coordinate_mapping='For stored224 pixel centers x,y: E=gt0+10*(left+(x+.5)*side/224); N=gt3-10*(top+(y+.5)*side/224). Uniform square resizing preserves angles.',
                   limits=['Record metadata is separate from safe model columns.',
                           'Older 3-column grids contain rounded lat/lon/incidence only; list order is not a radar-axis contract.',
                           'Manifest contributor bounds establish product exclusivity, not exact source-chip ownership or independent class truth.',
                           'Platform heading is scene-level metadata only; model direction comes from native grid differential.',
                           'Finite-grid interpolation is an approximation, not exact 3D look geometry.'],
                   downloads=False,training=False,source_writes=False)
    (HERE / 'geometry_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('by_product','annotation_files')}, indent=2))


if __name__ == '__main__':
    main()
