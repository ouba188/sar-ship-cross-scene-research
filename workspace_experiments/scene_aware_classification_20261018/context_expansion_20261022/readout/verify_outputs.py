"""Check persisted extraction alignment and inspect bounded new-port windows."""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from osgeo import gdal
from extract_context import HERE, BASE, polygon, window_polygon, region_masks, panel, dump


def footprints(points, window):
    shape = ((window[3]+1)//2, (window[2]+1)//2)
    return [F.interpolate(torch.from_numpy(m.astype(np.float32))[None, None], size=shape, mode='area')[0, 0].numpy()
            for m in region_masks(points, window)]


def main():
    started = time.perf_counter()
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64*1024**2)
    targets = pd.read_csv(HERE/'targets.csv', keep_default_na=False, dtype={'mmsi': str})
    neighbors = pd.read_csv(HERE/'neighbors.csv')
    excluded = pd.read_csv(HERE/'excluded.csv', keep_default_na=False)
    linked = pd.read_csv(HERE.parent/'linked_targets.csv', keep_default_na=False, low_memory=False, dtype={'mmsi': str})
    with np.load(HERE/'features.npz') as saved:
        features = {k: saved[k] for k in saved.files}
    ids = targets.instance_id.to_numpy(str)
    assert len(ids) > 0 and targets.instance_id.is_unique and np.array_equal(features['instance_ids'], ids)
    assert all(np.isfinite(v).all() for k, v in features.items() if k != 'instance_ids')
    assert np.array_equal(features['mask'].sum(1), targets.neighbors.to_numpy())
    rejected = excluded[excluded.scope.eq('target')]
    assert rejected.instance_id.is_unique
    assert not set(ids).intersection(rejected.instance_id)
    assert set(ids).union(rejected.instance_id) == set(linked.instance_id)
    original = linked.set_index('instance_id').loc[ids]
    for col in ('class_id', 'class_name', 'mmsi', 'product', 'port', 'pass_group'):
        assert np.array_equal(original[col].astype(str).to_numpy(), targets[col].astype(str).to_numpy())
    seen_slots, max_area, max_radius = set(), 0., 0.
    for edge in neighbors.itertuples():
        row = targets.iloc[edge.target_index]
        assert row.instance_id == edge.instance_id and row['product'] == edge.product
        assert (edge.target_index, edge.slot) not in seen_slots
        seen_slots.add((edge.target_index, edge.slot))
        assert features['edge_index'][edge.target_index, edge.slot] == edge.edge_index
        assert features['mask'][edge.target_index, edge.slot]
        np.testing.assert_allclose(features['geometry'][edge.target_index, edge.slot], [getattr(edge, f'g{j}') for j in range(7)], rtol=1e-6, atol=1e-7)
        points = json.loads(row.target_points)
        target_poly = polygon(points)
        lx, ty, w, h = json.loads(row.target_window_vv)
        assert not (lx <= edge.cx < lx+w and ty <= edge.cy < ty+h)
        radius = float(np.linalg.norm(np.array([edge.cx-row.x, edge.cy-row.y]))*row.pixel_m)
        assert radius <= 1000+1e-6 and row.axis_match_px <= 5
        max_radius = max(max_radius, radius)
        for name in ('read_window_vv', 'read_window_vh'):
            area = float(target_poly.Intersection(window_polygon(json.loads(getattr(edge, name)))).GetArea())
            assert area <= 1e-6
            max_area = max(max_area, area)
    assert len(seen_slots) == int(features['mask'].sum())
    old_ports = set(pd.read_csv(BASE/'context_relation_pilot_20261022/ports24/targets.csv').port)
    candidates = targets[~targets.port.isin(old_ports) & targets.neighbors.gt(0)].sort_values(['port', 'instance_id'])
    selected = candidates.groupby('port', sort=True).head(1).head(4)
    cases = []
    for row in selected.itertuples():
        ds = [gdal.Open(row.raster_vv), gdal.Open(row.raster_vh)]
        target_windows = [json.loads(row.target_window_vv), json.loads(row.target_window_vh)]
        edge = neighbors[neighbors.instance_id.eq(row.instance_id)].sort_values('slot').iloc[0]
        for kind, windows, points in [('target', target_windows, np.array(json.loads(row.target_points))),
                                     ('neighbor', [json.loads(edge.read_window_vv), json.loads(edge.read_window_vh)], np.array(json.loads(edge.source_points)))]:
            images = [d.ReadAsArray(*win) for d, win in zip(ds, windows)]
            weights = [footprints(points, win) for win in windows]
            name = f'context_{row.instance_id}_{kind}.png'
            detection_id = row.target_detection_id if kind == 'target' else edge.detection_id
            panel(HERE/'review_panels'/name, f'{row.port} {row.instance_id} {kind} / {detection_id}', images, windows, points, weights)
            cases.append(dict(instance_id=row.instance_id, port=row.port, kind=kind, detection_id=detection_id,
                              panel=str(HERE/'review_panels'/name), windows=json.dumps(windows),
                              selection='first ID with context per new port; first four ports alphabetically; slot0'))
        ds.clear()
    pd.DataFrame(cases).to_csv(HERE/'context_review_cases.csv', index=False)
    checks = dict(targets=len(targets), edges=len(neighbors), target_excluded=len(rejected),
                  all_linked_ids_accounted=True, labels_and_identities_unchanged=True, saved_arrays_finite=True,
                  slots_and_geometry_aligned=True, max_target_obb_intersection_px2=max_area,
                  max_neighbor_distance_m=max_radius, new_port_context_panels=len(cases),
                  new_ports_reviewed=selected.port.tolist(), seconds=time.perf_counter()-started)
    dump(HERE/'verification.json', checks)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
