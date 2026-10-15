"""Read-only input inventory for a separate online incidence recovery task."""
import json
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from external_knowledge_metadata import TRACK, OUT, acquisition_key, write


def annotation_grid(path):
    """Read only true incidence from the exact product annotation, never heading."""
    root = ET.parse(path).getroot()
    values = []
    for point in root.iter():
        if point.tag.rsplit('}', 1)[-1] != 'geolocationGridPoint':
            continue
        fields = {e.tag.rsplit('}', 1)[-1]: e.text for e in point}
        values.append([float(fields[k]) for k in ('latitude', 'longitude', 'incidenceAngle')])
    grid = np.asarray(values, float)
    if grid.ndim != 2 or grid.shape[1] != 3 or not np.isfinite(grid).all():
        raise ValueError('No valid geolocation grid in the exact scene annotation')
    if not ((grid[:, 2] > 0) & (grid[:, 2] < 90)).all():
        raise ValueError('Invalid degree incidence values')
    return grid


def main():
    side = pd.read_csv(OUT/'f0921_incidence.csv', keep_default_na=False)
    missing = side[side.status != 'ok']
    wanted = set(missing.scene)
    points = defaultdict(lambda: defaultdict(set))
    source = TRACK/'f0921_tile_acq.json'
    acquisition = json.loads(source.read_text())
    for row in acquisition:
        if row['tile'] in wanted:
            points[row['tile']][round(float(row['lon']), 5), round(float(row['lat']), 5)].add(acquisition_key(row))
    records, ambiguous = [], []
    for row in missing.itertuples(index=False):
        keys = points[row.scene].get((round(row.lon, 5), round(row.lat, 5)), set())
        key = next(iter(keys)) if len(keys) == 1 else ('', '', '')
        records.append(dict(sample_id=row.sample_id, row=row.row, tile=row.scene, lon=row.lon, lat=row.lat,
            status=row.status, sensor=key[0], acquisition_start=key[1], product_suffix=key[2],
            known_product=row.product, candidate_count=len(keys), acquisition_source=str(source)))
        if len(keys) > 1:
            for candidate in sorted(keys):
                ambiguous.append(dict(sample_id=row.sample_id, row=row.row, tile=row.scene,
                    lon=row.lon, lat=row.lat, sensor=candidate[0], acquisition_start=candidate[1],
                    product_suffix=candidate[2]))
    destination = OUT/'recovery_inventory'
    destination.mkdir(exist_ok=True)
    rows = pd.DataFrame(records)
    rows.to_csv(destination/'missing_rows.csv', index=False)
    unique = rows[rows.candidate_count == 1].groupby(
        ['sensor', 'acquisition_start', 'product_suffix', 'known_product', 'status'], dropna=False).agg(
        target_rows=('sample_id', 'size'), tiles=('tile', 'nunique')).reset_index()
    unique.to_csv(destination/'acquisition_requests.csv', index=False)
    pd.DataFrame(ambiguous).to_csv(destination/'ambiguous_candidates.csv', index=False)
    write(destination/'README.json', dict(rows=len(rows), requests=len(unique),
        statuses=rows.status.value_counts().to_dict(), original_low_coverage_sidecar_unchanged=True,
        source=str(source), product_lookup='Exact sensor + acquisition_start + product_suffix; never tile-first product',
        annotation='Exact SAFE annotation geolocationGridPoint latitude/longitude/incidenceAngle; degree units',
        interpolation='external_knowledge_metadata.interpolate(grid, query_latlon); outside convex hull stays NaN',
        warning='139 ambiguous coordinate matches need disambiguation; seven verified native-row recoveries are separately saved in f0921_native_recovered.json'))
    print(dict(rows=len(rows), requests=len(unique), path=str(destination)), flush=True)


if __name__ == '__main__':
    main()
