"""Bounded native-row and local SAFE audit after the first incidence extraction."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from osgeo import ogr

from external_knowledge_metadata import ROOT, TRACK, OUT, TARGETS, product_key, interpolate, write


def main():
    ogr.UseExceptions()
    side = pd.read_csv(OUT/'f0921_incidence.csv', keep_default_na=False)
    frame = pd.read_csv(ROOT/TARGETS['f0921']/'target_manifest.csv',
                        usecols=['sample_id', 'row', 'row_in_shp', 'scene', 'mmsi', 'lon', 'lat'],
                        dtype={'mmsi': str})
    np.testing.assert_array_equal(frame.sample_id, side.sample_id)
    grids = {p.stem: p for p in (TRACK/'s1_grid').glob('*.json')}
    index = defaultdict(set)
    for product in grids:
        index[product_key(product)].add(product)
    product_meta = json.loads((TRACK/'s1_product_incidence.json').read_text())
    for product in product_meta:
        index[product_key(product)].add(product)
    roots = [Path('E:/Hermes')/s for s in (
        'round4_safe_metadata_from_zip', 'round5_safe_collect_light_from_attachment/round5_safe_collect_light',
        'round5_safe_collect_light_new_only')]
    safe_audit = []
    for root in roots:
        folders = list(root.glob('*.SAFE'))
        annotations = [p for d in folders for p in (d/'annotation').glob('*.xml')]
        safe_audit.append(dict(root=str(root), safe_directories=len(folders), annotation_xml=len(annotations)))
    # Fixed positional samples validate the tile/coordinate path; duplicate positions
    # use the original row_in_shp and verify MMSI/coordinates before reading acquisition.
    sample = np.linspace(0, len(side)-1, 24, dtype=int)
    ambiguous = np.flatnonzero(side.status.to_numpy() == 'ambiguous_acquisition')
    chosen = np.unique(np.r_[sample, ambiguous])
    records, recovered = [], []
    for scene, group in frame.iloc[chosen].groupby('scene'):
        date = scene.split('_')[3]
        path = Path('D:/F_0921_relabel_det')/date[:4]/date/(scene+'.shp')
        ds = ogr.Open(str(path))
        layer = ds.GetLayer(0)
        for ix, row in group.iterrows():
            feature = layer.GetFeature(int(row.row_in_shp))
            lon, lat = float(feature.GetField('lon')), float(feature.GetField('lat'))
            if str(feature.GetField('mmsi')) != row.mmsi or max(abs(lon-row.lon), abs(lat-row.lat)) > .0000051:
                records.append(dict(row=int(row['row']), sample_id=row.sample_id,
                    status='native_row_identity_mismatch', shapefile=str(path), row_in_shp=int(row.row_in_shp),
                    identity_verified=False, native_mmsi=str(feature.GetField('mmsi')),
                    manifest_mmsi=row.mmsi, native_lon=lon, native_lat=lat))
                continue
            key = tuple(str(feature.GetField(k)) for k in ('sensor', 'obs_time', 'code'))
            products = index[key]
            product = next(iter(products)) if len(products) == 1 else ''
            record = dict(row=int(row['row']), sample_id=row.sample_id, status=side.loc[ix, 'status'],
                          native_sensor=key[0], native_time=key[1], native_code=key[2], product=product,
                          shapefile=str(path), row_in_shp=int(row.row_in_shp), identity_verified=True)
            if side.loc[ix, 'product'] and product:
                if side.loc[ix, 'product'] != product:
                    raise ValueError('Coordinate/acquisition lookup disagrees with native row')
            if ix in ambiguous and product in grids:
                angle = interpolate(json.loads(grids[product].read_text()), [[lat, lon]])[0]
                if np.isfinite(angle):
                    recovered.append(dict(index=int(ix), sample_id=row.sample_id, product=product,
                                          incidence_deg=float(angle), metadata_file=str(grids[product]),
                                          shapefile=str(path), row_in_shp=int(row.row_in_shp)))
            records.append(record)
    pd.DataFrame(records).to_csv(OUT/'f0921_native_row_audit.csv', index=False)
    write(OUT/'f0921_native_recovered.json', recovered)
    write(OUT/'local_safe_audit.json', dict(roots=safe_audit, native_rows_checked=len(records),
        ambiguous_rows=len(ambiguous), additional_supported_rows=len(recovered),
        native_row_identity_mismatches=sum(not r['identity_verified'] for r in records),
        note='SAFE directories contain manifests/AIS light metadata but no annotation geolocation grids; no midpoint substitution. Stale shapefile row IDs are not trusted.'))
    print('NATIVE AUDIT', len(records), 'additional supported', len(recovered), safe_audit, flush=True)


if __name__ == '__main__':
    main()
