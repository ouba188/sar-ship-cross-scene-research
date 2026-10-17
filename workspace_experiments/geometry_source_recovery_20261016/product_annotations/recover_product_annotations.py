"""Recover exact-product VV grids for the fixed 787-row annotation gap."""

import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = ROOT / 'visual_reliable_baseline/artifacts/safe841_metadata_20260917/products'
BASE = 'https://sentinel-s1-l1c.s3.amazonaws.com/'
FIELDS = ('line', 'pixel', 'latitude', 'longitude', 'incidenceAngle')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def source_url(product):
    date = product.split('_')[4][:8]
    return (f'{BASE}GRD/{date[:4]}/{int(date[4:6])}/{int(date[6:8])}/'
            f'IW/DV/{product}/annotation/iw-vv.xml')


def parse_annotation(path, product):
    root = ET.parse(path).getroot()
    parts = product.split('_')
    expected = dict(missionId=parts[0], productType='GRD', polarisation='VV',
                    mode=parts[1], absoluteOrbitNumber=str(int(parts[6])),
                    missionDataTakeId=str(int(parts[7], 16)))
    header = root.find('adsHeader')
    for field, value in expected.items():
        if header.findtext(field) != value:
            raise ValueError(f'Product identity mismatch: {field}')
    for field, value in [('startTime', parts[4]), ('stopTime', parts[5])]:
        actual = header.findtext(field)[:19].replace('-', '').replace(':', '')
        if actual != value:
            raise ValueError(f'Product time mismatch: {field}')
    if root.findtext('generalAnnotation/productInformation/projection') != 'Ground Range':
        raise ValueError('Expected Ground Range projection')
    listing = root.find('geolocationGrid/geolocationGridPointList')
    points = []
    for node in listing.findall('geolocationGridPoint'):
        point = {key: (int(node.findtext(key)) if key in ('line', 'pixel')
                       else float(node.findtext(key))) for key in FIELDS}
        points.append(point)
    if len(points) != int(listing.attrib['count']) or not points:
        raise ValueError('Incomplete grid')
    if not all(math.isfinite(p[k]) for p in points for k in FIELDS):
        raise ValueError('Nonfinite grid')
    if len({(p['line'], p['pixel']) for p in points}) != len(points):
        raise ValueError('Duplicate grid indices')
    if not all(p['line'] >= 0 and p['pixel'] >= 0 and
               -90 <= p['latitude'] <= 90 and -180 <= p['longitude'] <= 180
               and 0 <= p['incidenceAngle'] <= 90 for p in points):
        raise ValueError('Invalid grid coordinates')
    lines = {p['line'] for p in points}
    if len(lines) < 2 or any(len({p['pixel'] for p in points if p['line'] == line}) < 2
                             for line in lines):
        raise ValueError('Grid cannot support same-line range chords')
    return points, dict(expected, startTime=header.findtext('startTime'),
                       stopTime=header.findtext('stopTime'))


def recover_product(product, count):
    source = LOCAL / (product + '.SAFE') / 'annotation/iw-vv.xml'
    destination = HERE / 'annotations' / (product + '.xml')
    grid_path = HERE / 'radar_grid' / (product + '.json')
    result = dict(product=product, local_annotation_path='', source_url=source_url(product),
                  status='failed', target_rows=count, local_grid_path='', grid_points=0,
                  annotation_bytes=0, original_local_annotation_path=str(source), reason='')
    try:
        points, identity = parse_annotation(source, product)
        objects = json.loads((source.parents[1] / 'source_objects.json').read_text())
        matches = [obj for obj in objects if BASE + obj['key'] == source_url(product)]
        assert len(matches) == 1 and matches[0]['size'] == source.stat().st_size
        temporary = destination.with_suffix('.xml.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        grid = dict(status='ok', product=product, source_url=source_url(product), points=points,
                    quality_flags=['product_header_checked', 'full_annotation_parsed',
                                   'historical_exact_product_object_record_checked'],
                    local_annotation_path=str(destination),
                    original_local_annotation_path=str(source), identity=identity,
                    bytes_read=source.stat().st_size,
                    recovery_method='reuse_existing_full_exact_product_annotation')
        write_json(grid_path, grid)
        result.update(local_annotation_path=str(destination), local_grid_path=str(grid_path),
                      status='recovered_local_exact_product', grid_points=len(points),
                      annotation_bytes=destination.stat().st_size)
    except (OSError, ValueError, AssertionError, ET.ParseError) as error:
        result['reason'] = type(error).__name__ + ': ' + str(error)
    return result


def main():
    geometry = read_csv(ROOT / 'geometry_targeted_fix_20261016/recovered_geometry.csv')
    manifest = read_csv(ROOT / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
    caches = sorted((ROOT / 'geometry_relation_slots_20261016/slots_cache').glob('*.npz'))
    needed = set()
    for path in caches:
        with np.load(path) as cache:
            needed.update(int(x) for x in cache['rows'])
    assert len(geometry) == len(manifest) == 38091
    assert len(needed) == 8622
    targets = []
    for row in geometry:
        if row['radar_recovery_status'] != 'not_computed_missing_local_product_grid':
            continue
        index = int(row['global_row'])
        assert index in needed and manifest[index]['instance_id'] == row['instance_id']
        targets.append(dict(global_row=index, instance_id=row['instance_id'],
                            product=manifest[index]['product'],
                            original_status=row['radar_recovery_status']))
    assert len(targets) == 787
    counts = Counter(row['product'] for row in targets)
    assert len(counts) == 4
    (HERE / 'annotations').mkdir(exist_ok=True)
    (HERE / 'radar_grid').mkdir(exist_ok=True)
    write_csv(HERE / 'target_rows.csv', targets)
    results = []
    for product, count in sorted(counts.items()):
        result = recover_product(product, count)
        results.append(result)
        print(json.dumps(result), flush=True)
    manifest_path = HERE / 'manifest.csv'
    combined = {r['product']: r for r in read_csv(manifest_path)} if manifest_path.exists() else {}
    combined.update({r['product']: r for r in results})
    write_csv(manifest_path, list(combined.values()))
    summary = dict(full_table_rows=len(geometry), fixed_cohort_rows=len(needed),
                   target_rows=len(targets), products=len(counts),
                   recovered_products=sum(x['status'].startswith('recovered') for x in results),
                   covered_target_rows=sum(x['target_rows'] for x in results if x['status'].startswith('recovered')),
                   original_tables_modified=False, pixel_angle_conversion_performed=False,
                   measurement_bytes_downloaded=0)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return 0 if summary['recovered_products'] == len(counts) else 1


if __name__ == '__main__':
    sys.exit(main())
