"""Audit external incidence using explicit ship IDs and acquisition metadata."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import re
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
from lxml import etree

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
TRACK = ROOT.parent
OUT = ROOT/'external_knowledge_20261010'
TARGETS = {'f0921': 'f0921_detector_recrop_20261007',
           'oss2': 'opensarship2_transfer_20261007'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def interpolate(grid, latlon):
    grid, latlon = np.asarray(grid, float), np.asarray(latlon, float)
    if not (np.isfinite(grid).all() and ((grid[:, 2] > 0) & (grid[:, 2] < 90)).all()):
        raise ValueError('Invalid incidence grid')
    lat0, lon0 = grid[:, 0].mean(), grid[0, 1]
    def plane(points):
        return np.c_[((points[:, 1]-lon0+180) % 360-180)*np.cos(np.deg2rad(lat0)),
                     points[:, 0]-lat0]
    xy, q = plane(grid), plane(latlon)
    tri = Delaunay(xy)
    simplex = tri.find_simplex(q)
    answer = np.full(len(q), np.nan)
    good = simplex >= 0
    first = np.einsum('nij,nj->ni', tri.transform[simplex[good], :2],
                      q[good]-tri.transform[simplex[good], 2])
    weight = np.c_[first, 1-first.sum(1)]
    answer[good] = (weight*grid[tri.simplices[simplex[good]], 2]).sum(1)
    return answer


def oss2(frame):
    records, cache, unit_records = [], {}, []
    for row in frame.itertuples(index=False):
        scene = row.scene_id
        parent = Path(row.source_path_vv).parent.parent
        if scene not in cache:
            # Native files contain malformed closing tags and invalid ship-name bytes.
            # Recover XML structure, then require unique numeric center/MMSI matches.
            parser = etree.XMLParser(recover=True, encoding='utf-8')
            xml_text = (parent/'Ship.xml').read_bytes().decode('utf-8', errors='replace')
            ship = etree.fromstring(xml_text.encode('utf-8'), parser)
            slots = defaultdict(list)
            for item in ship.findall('ship'):
                sar, ais = item.find('SARShipInformation'), item.find('AISShipInformation')
                key = (int(sar.findtext('Center_x')), int(sar.findtext('Center_y')))
                slots[key].append((float(sar.findtext('Incidence')), ais.findtext('MMSI'),
                                  float(sar.findtext('Center_longitude')),
                                  float(sar.findtext('Center_latitude'))))
            metadata_path = parent/'Metadata.xml'
            if not metadata_path.exists():
                metadata_path = parent/(scene+'.xml')
            units = []
            if metadata_path.exists():
                metadata = ET.parse(metadata_path).getroot()
                units = [e.attrib for e in metadata.iter('attrib')
                         if e.get('name') in ('incidence_near', 'incidence_far')]
                if any(u.get('unit') != 'deg' for u in units):
                    raise ValueError(f'Inconsistent incidence unit: {scene}')
            unit_records.append(dict(scene=scene, units=units, file=str(metadata_path),
                                     ship_xml_recovery=[str(e) for e in parser.error_log]))
            cache[scene] = slots
        match = re.search(r'_x(\d+)_y(\d+)_vv', Path(row.source_path_vv).name)
        if match is None:
            raise ValueError('Missing chip center key')
        slots = cache[scene][tuple(map(int, match.groups()))]
        candidates = [v for v in slots if str(v[1]) == str(row.mmsi_or_ship_id)]
        angle, status, distance = np.nan, 'ambiguous_or_missing_ship', np.nan
        if len(candidates) == 1:
            value, _, lon, lat = candidates[0]
            distance = 111320*np.hypot((lon-row.lon)*np.cos(np.deg2rad(lat)), lat-row.lat)
            if distance > 10:
                raise ValueError('Ship center and manifest coordinates differ by >10m')
            if 0 < value < 90:
                angle, status = value, 'ok'
            else:
                status = 'invalid_incidence'
        records.append(dict(sample_id=row.sample_id, scene=scene, product=scene,
                            incidence_deg=angle, status=status, unit='deg',
                            metadata_file=str(parent/'Ship.xml'), coordinate_error_m=distance,
                            method='scene+chip_center_xy+MMSI; SARShipInformation/Incidence'))
    if not any(item['units'] for item in unit_records):
        raise ValueError('No native degree-unit evidence')
    write(OUT/'oss2_units.json', unit_records)
    return pd.DataFrame(records)


def acquisition_key(record):
    return record['sensor'], record['t'], record['code']


def product_key(product):
    parts = product.removesuffix('.SAFE').split('_')
    return parts[0], parts[4], parts[-1]


def f0921(frame):
    # Coordinates in the chip manifest are rounded to five decimal degrees.
    wanted = set(frame.scene)
    points = defaultdict(lambda: defaultdict(set))
    acquisitions = json.loads((TRACK/'f0921_tile_acq.json').read_text(encoding='utf-8'))
    for item in acquisitions:
        if item['tile'] in wanted:
            pos = round(float(item['lon']), 5), round(float(item['lat']), 5)
            points[item['tile']][pos].add(acquisition_key(item))
    del acquisitions
    index = defaultdict(set)
    grids = {p.stem: p for p in (TRACK/'s1_grid').glob('*.json')}
    for product in grids:
        index[product_key(product)].add(product)
    old_map = json.loads((TRACK/'f0921_tile_products.json').read_text())
    for item in old_map.values():
        index[product_key(item['product'])].add(item['product'])
    for filename in ('f0921_products.txt', 'f0921_product_list.txt'):
        for product in (TRACK/filename).read_text().splitlines():
            if product.startswith('S1'):
                index[product_key(product)].add(product)
    records, by_product = [], defaultdict(list)
    for row in frame.itertuples(index=False):
        pos = round(row.lon, 5), round(row.lat, 5)
        keys = points[row.scene].get(pos, set())
        products = index[next(iter(keys))] if len(keys) == 1 else set()
        product = next(iter(products)) if len(products) == 1 else ''
        status = ('ambiguous_acquisition' if len(keys) > 1 else 'missing_acquisition')
        if len(keys) == 1:
            status = 'missing_product' if not product else ('pending' if product in grids else 'missing_grid')
        old_product = old_map.get(row.scene, {}).get('product', '')
        rec = dict(sample_id=row.sample_id, row=row.row, scene=row.scene, product=product,
                   incidence_deg=np.nan, status=status, unit='deg', lon=row.lon, lat=row.lat,
                   acquisition_candidates=len(keys), old_tile_product=old_product,
                   differs_from_tile_first=bool(product and old_product != product),
                   metadata_file=str(grids[product]) if product in grids else '',
                   method='tile+rounded_ship_lonlat -> sensor/start/code -> product grid; convex interpolation')
        if product in grids:
            by_product[product].append(len(records))
        records.append(rec)
    result = pd.DataFrame(records)
    for i, (product, rows) in enumerate(by_product.items()):
        grid = json.loads(grids[product].read_text())
        angles = interpolate(grid, result.loc[rows, ['lat', 'lon']].to_numpy())
        result.loc[rows, 'incidence_deg'] = angles
        result.loc[rows, 'status'] = np.where(np.isfinite(angles), 'ok', 'outside_grid')
        if i % 200 == 0:
            print(f'METADATA f0921 {i}/{len(by_product)} product grids', flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coverage_path = OUT/'metadata_coverage.json'
    summary = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    for dataset in ('oss2', 'f0921'):
        if (OUT/f'{dataset}_incidence.csv').exists():
            print('PRESERVE original incidence sidecar '+dataset, flush=True)
            continue
        frame = pd.read_csv(ROOT/TARGETS[dataset]/'target_manifest.csv',
                            dtype={'mmsi_or_ship_id': str}, keep_default_na=False,
                            usecols=lambda c: not c.startswith(('class', 'label')))
        side = globals()[dataset](frame)
        np.testing.assert_array_equal(side.sample_id, frame.sample_id)
        if side.sample_id.duplicated().any():
            raise ValueError('Duplicate target IDs')
        side.to_csv(OUT/f'{dataset}_incidence.csv', index=False)
        good = side.incidence_deg.notna()
        summary[dataset] = dict(rows=len(side), usable=int(good.sum()), fraction=float(good.mean()),
            statuses=side.status.value_counts().to_dict(),
            min_median_max_deg=side.loc[good, 'incidence_deg'].quantile([0, .5, 1]).tolist(),
            heading_used=False, manufactured_angles=False)
        if dataset == 'f0921':
            summary[dataset]['old_tile_first_wrong_product_rows'] = int(side.differs_from_tile_first.sum())
        side.groupby(['scene', 'status']).size().rename('rows').to_csv(OUT/f'{dataset}_scene_coverage.csv')
        print(dataset, summary[dataset], flush=True)
        write(OUT/'metadata_coverage.json', summary)


if __name__ == '__main__':
    main()
