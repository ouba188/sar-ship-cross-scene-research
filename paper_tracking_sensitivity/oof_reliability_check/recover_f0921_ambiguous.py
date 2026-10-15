"""Resolve only acquisition ambiguities using native identity, not stale row IDs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import ogr
from external_knowledge_metadata import ROOT, TRACK, OUT as OLD, interpolate, write
from recover_f0921_incidence_online import OUT


def main():
    ogr.UseExceptions()
    side = pd.read_csv(OLD/'f0921_incidence.csv', keep_default_na=False)
    frame = pd.read_csv(ROOT/'f0921_detector_recrop_20261007/target_manifest.csv',
        usecols=['sample_id', 'scene', 'mmsi', 'lon', 'lat'], dtype={'mmsi': str})
    np.testing.assert_array_equal(side.sample_id, frame.sample_id)
    candidates = pd.read_csv(OLD/'recovery_inventory/ambiguous_candidates.csv', keep_default_na=False)
    possible = {sid: set(zip(g.sensor, g.acquisition_start, g.product_suffix))
                for sid, g in candidates.groupby('sample_id')}
    results = []
    for scene, group in frame[side.status == 'ambiguous_acquisition'].groupby('scene'):
        date = scene.split('_')[3]
        path = Path('D:/F_0921_relabel_det')/date[:4]/date/(scene+'.shp')
        native = ogr.Open(str(path))
        index = {}
        for feature in native.GetLayer(0):
            key = (str(feature.GetField('mmsi')), round(float(feature.GetField('lon')), 5),
                   round(float(feature.GetField('lat')), 5))
            acq = tuple(str(feature.GetField(k)) for k in ('sensor', 'obs_time', 'code'))
            index.setdefault(key, set()).add(acq)
        for row in group.itertuples():
            matches = index.get((row.mmsi, round(row.lon, 5), round(row.lat, 5)), set())
            common = matches & possible[row.sample_id]
            # Require the entire native identity lookup to be unique, not merely a subset.
            resolved = len(matches) == 1 and len(common) == 1
            acq = next(iter(common)) if resolved else ('', '', '')
            results.append(dict(sample_id=row.sample_id, native_candidates=len(matches),
                resolved=resolved, sensor=acq[0], acquisition_start=acq[1], product_suffix=acq[2],
                shapefile=str(path), method='Unique MMSI + rounded lon/lat in native acquisition records'))
    table = pd.DataFrame(results)
    table.to_csv(OUT/'ambiguous_identity_resolution.csv', index=False)
    write(OUT/'ambiguous_identity_summary.json', dict(rows=len(table), resolved=int(table.resolved.sum()),
        native_candidate_counts=table.native_candidates.value_counts().to_dict()))
    print(table.groupby(['resolved', 'native_candidates']).size().to_string(), flush=True)


if __name__ == '__main__':
    main()
