"""Join paired body-axis proxies to exact-product local ground-range axes."""

import concurrent.futures
import json

import numpy as np
import pandas as pd

from extract_ship_axis_proxy import OUT, SOURCE
from relative_radar_geometry import fetch_grid, range_azimuth
from ship_axis_proxy import axial_difference
from sar_joint_completion import write_json


def main():
    if (OUT/'RELATIVE_COMPLETE.json').exists():
        raise ValueError('Preserve completed relative conditions')
    pairs = pd.read_csv(OUT/'paired_axes.csv')
    supported = pairs[pairs.accepted].copy()
    products = sorted(supported['product'].unique())
    grids = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(fetch_grid, product, OUT/'radar_grid'): product for product in products}
        for future in concurrent.futures.as_completed(jobs):
            product = jobs[future]
            grids[product] = future.result()
            if len(grids) % 20 == 0 or len(grids) == len(products):
                print(json.dumps(dict(products=len(grids), total=len(products),
                    available=sum(v['status'] == 'ok' for v in grids.values()))), flush=True)
    pd.DataFrame([{k: v for k, v in result.items() if k != 'points'} for result in grids.values()]).to_csv(
        OUT/'radar_fetch_status.csv', index=False)
    records = []
    for row in supported.to_dict('records'):
        value = range_azimuth(grids[row['product']], row['longitude'], row['latitude'])
        result = dict(instance_id=row['instance_id'], product=row['product'], body_axis_deg=row['body_axis_deg'],
                      disagreement_deg=row['disagreement_deg'], longitude=row['longitude'], latitude=row['latitude'],
                      relative_status=value['status'], nearest_grid_distance_m=value['nearest_grid_distance_m'])
        if value['status'] == 'ok':
            result['range_axis_deg'] = value['axial_azimuth_deg']
            result['relative_acute_deg'] = float(axial_difference(row['body_axis_deg'], value['axial_azimuth_deg']))
            result['relative_signed_deg'] = (row['body_axis_deg']-value['axial_azimuth_deg']) % 180.
        records.append(result)
    relative = pd.DataFrame(records)
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    frame = frame.merge(relative, on=['instance_id', 'product'], how='left', validate='one_to_one', sort=False)
    frame['relative_status'] = frame.relative_status.fillna('body_axis_unavailable')
    frame.to_csv(OUT/'relative_conditions.csv', index=False)
    frame.assign(available=frame.relative_status.eq('ok')).groupby(['port', 'class_id']).available.agg(
        ['sum', 'count']).to_csv(OUT/'relative_coverage.csv')
    write_json(OUT/'RELATIVE_COMPLETE.json', dict(manifest_rows=len(frame),
        relative_rows=int(frame.relative_status.eq('ok').sum()),
        status_counts=frame.relative_status.value_counts().to_dict(), products_requested=len(products),
        products_available=sum(v['status'] == 'ok' for v in grids.values()),
        range_direction='local grid ground-range axis approximation, not exact satellite beam azimuth',
        body_direction='central bright-component axis, not true heading',
        target_labels_used_for_features=False, target_rows_removed=False))
    print((OUT/'RELATIVE_COMPLETE.json').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    main()
