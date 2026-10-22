"""Bounded F0921 linkage preparation; existing evidence only, no training.

Run with the existing ClearSAR environment. Outputs stay beside this script.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from osgeo import gdal, osr
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'f0921_priority'
BRIDGE = ROOT / 'formal_queue_bridge'
IOU_THRESHOLD = 0.25
RADIUS_M = 1000


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def yes(value):
    return str(value).lower() == 'true'


def srs(wkt):
    result = osr.SpatialReference(wkt=wkt)
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def main():
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    cohort = read_csv(BRIDGE / 'cohort.csv')
    geometry_path = BRIDGE / 'axis_radar_readout_20261019/geometry.csv'
    geometry = {r['row']: r for r in read_csv(geometry_path)}
    expanded_path = BRIDGE / 'learned_geometry_crosspass_20261019/recovery/expanded_geometry.csv'
    for r in read_csv(expanded_path):
        old = geometry[r['row']]
        assert all(old[k] == r[k] for k in ['clean_row', 'resolved_product', 'source_tif'])
        old.update(r)
        old['geometry_valid'] = r['orbit_valid']
    objects = read_json(ROOT / 'land_mask/objects_retained.json')
    old_edges = read_json(ROOT / 'land_mask/neighbor_edges_retained.json')
    retained_ids = {r['object_id'] for r in objects}
    assert all(r['object_id'] in retained_ids and r['neighbor_id'] in retained_ids for r in old_edges)
    inputs = {r['object_id']: r for r in read_csv(ROOT / 'recovery_31cc_2ec7/five_scene_object_inputs.csv')}
    policy = {r['object_id']: r for r in read_csv(ROOT / 'land_mask/object_policy.csv')}
    detections = {(str(r['row']), r['pol']): r['boxes'] for r in read_json(BRIDGE / 'detections.json')}
    assert len(cohort) == 269 and len(objects) == 990
    assert sum(yes(r['geometry_valid']) for r in geometry.values()) == 240
    assert len({r['clean_row'] for r in cohort}) == 269
    by_product = defaultdict(list)
    rasters = {}
    for obj in objects:
        by_product[obj['product']].append(obj)
        for member in obj['members']:
            rasters.setdefault((obj['product'], member['pol']), member['source'])
    datasets = {}

    def raster(path):
        if path not in datasets:
            datasets[path] = gdal.Open(str(path), gdal.GA_ReadOnly)
        return datasets[path]

    def world(ds, points):
        return [gdal.ApplyGeoTransform(ds.GetGeoTransform(), x, y) for x, y in points]

    # All neighbor geometry is in the product VV projected CRS, in metres.
    obj_shapes, centers, scene_ds = {}, {}, {}
    for product, items in by_product.items():
        ds = scene_ds[product] = raster(rasters[(product, 'VV')])
        assert abs(srs(ds.GetProjection()).GetLinearUnits() - 1) < 1e-9
        for obj in items:
            anchor_ds = raster(rasters[(product, obj['anchor_pol'])])
            assert srs(anchor_ds.GetProjection()).IsSame(srs(ds.GetProjection()))
            obj_shapes[obj['object_id']] = Polygon(world(anchor_ds, obj['source_corners']))
            centers[obj['object_id']] = Point(world(anchor_ds, [obj['source_center']])[0])

    tiles = defaultdict(list)
    input_paths = [ROOT / folder / 'INPUTS.json' for folder in
                   ['object_bridge', 'object_bridge_b529', 'object_bridge_31cc_2ec7']]
    for path in input_paths:
        for tile in read_json(path)['tiles']:
            if tile['pol'] != 'VV':
                continue
            x, y = tile['origin']
            ds = raster(tile['source'])
            tiles[tile['product']].append(Polygon(world(ds, [(x, y), (x+tile['width'], y),
                (x+tile['width'], y+tile['height']), (x, y+tile['height'])])))
    coverage = {p: unary_union(parts) for p, parts in tiles.items()}
    links, targets = [], []
    for target in cohort:
        geo = geometry[target['row']]
        assert all(target[k] == geo[k] for k in ['clean_row', 'mmsi', 'source_tif', 'source_fid'])
        product = geo['resolved_product']
        lineage = all(yes(target[k]) for k in ['vv_exact', 'vh_exact', 'lineage_unique'])
        ownership = yes(geo['pixel_product_unique']) and yes(geo['native_window_matches_manifest_raster'])
        base = {k: target[k] for k in ['row', 'clean_row', 'mmsi', 'class_raw', 'label', 'date',
                                      'source_tif', 'source_fid', 'png_vv', 'png_vh']}
        base.update(product=product, scene_suffix=product[-4:], pixel_lineage=lineage,
                    product_ownership_unique=ownership, geometry_valid=yes(geo['geometry_valid']),
                    ownership_evidence=geo['ownership_evidence'],
                    contributor_observation_keys_json=geo['contributor_observation_keys_json'],
                    training_truth_verified=False)
        target_result = dict(base, candidate_count=0, qualifying_count=0, status='pending')
        targets.append(target_result)
        if not lineage or not ownership or product not in by_product:
            target_result['status'] = ('unresolved_pixel_lineage' if not lineage else
                'unresolved_product_ownership' if not ownership else 'outside_five_scene_inventory')
            links.append(dict(base, object_id='', evidence_tier=target_result['status']))
            continue
        source = raster(target['source_tif'])
        transform = osr.CoordinateTransformation(srs(source.GetProjection()), srs(scene_ds[product].GetProjection()))
        left, top, side = (float(target[k]) for k in ['left', 'top', 'side'])
        origin = world(source, [(left + side/448, top + side/448)])[0]
        assert abs(origin[0] - float(geo['chip_x0_center_easting'])) < 1e-6
        assert abs(origin[1] - float(geo['chip_y0_center_northing'])) < 1e-6

        def to_scene(points):
            projected = world(source, points)
            return [transform.TransformPoint(x, y)[:2] for x, y in projected]

        window = Polygon(to_scene([(left, top), (left+side, top), (left+side, top+side), (left, top+side)]))
        anchor224 = Point(float(target['anchor_x']) / side * 224, float(target['anchor_y']) / side * 224)
        selected = {}
        for pol, name in enumerate(['VV', 'VH']):
            boxes = [b for b in detections[(target['row'], pol)] if Polygon(b['corners']).covers(anchor224)]
            if len(boxes) == 1:
                selected[name] = Polygon(to_scene([(left+x*side/224, top+y*side/224) for x, y in boxes[0]['corners']]))
        for obj in by_product[product]:
            oid = obj['object_id']
            if not window.intersects(obj_shapes[oid]):
                continue
            overlap = {}
            for pol, target_box in selected.items():
                member = obj.get('pol_candidates', {}).get(pol)
                if member:
                    ds = raster(rasters[(product, pol)])
                    assert srs(ds.GetProjection()).IsSame(srs(scene_ds[product].GetProjection()))
                    shape = Polygon(world(ds, member['source_corners']))
                    overlap[pol] = target_box.intersection(shape).area / target_box.union(shape).area
            qualifies = bool(overlap) and max(overlap.values()) >= IOU_THRESHOLD
            neighbors = [n for n in by_product[product] if n['object_id'] != oid
                         and centers[oid].distance(centers[n['object_id']]) <= RADIUS_M
                         and not window.covers(centers[n['object_id']])]
            disk = centers[oid].buffer(RADIUS_M)
            neighbor_ids = [n['object_id'] for n in neighbors]
            full_coverage = coverage[product].covers(disk)
            record = dict(base, object_id=oid, evidence_tier='box_overlap_candidate' if qualifies else 'window_overlap_only',
                qualifies_geometry=qualifies, iou_vv=overlap.get('VV', ''), iou_vh=overlap.get('VH', ''),
                selected_target_polarizations=';'.join(selected), object_pair_status=obj['pair_status'],
                scene_crs_wkt=scene_ds[product].GetProjection(),
                scene_raster_vv=rasters[(product, 'VV')], scene_raster_vh=rasters[(product, 'VH')],
                land_action=policy[oid]['action'], object_native_vv=inputs[oid]['native_vv'],
                object_native_vh=inputs[oid]['native_vh'], target_window_world_json=json.dumps(list(window.exterior.coords)),
                object_polygon_world_json=json.dumps(list(obj_shapes[oid].exterior.coords)),
                neighbors_outside_chip_1km=len(neighbors), neighbor_ids_json=json.dumps(neighbor_ids),
                neighbor_overlap_target_window_count=sum(window.intersects(obj_shapes[n]) for n in neighbor_ids),
                context_1km_complete=full_coverage,
                context_1km_observed_fraction=min(1., coverage[product].intersection(disk).area/disk.area),
                reciprocal_unique_candidate=False)
            links.append(record)
            target_result['candidate_count'] += 1
            target_result['qualifying_count'] += int(qualifies)
        if target_result['candidate_count'] == 0:
            links.append(dict(base, object_id='', evidence_tier='no_object_intersects_native_chip'))
        target_result['status'] = ('no_object_intersects_native_chip' if target_result['candidate_count'] == 0 else
                                   'no_qualifying_box_overlap' if target_result['qualifying_count'] == 0 else 'box_candidates_pending')

    eligible = [r for r in links if r.get('qualifies_geometry')]
    target_degree = Counter(r['row'] for r in eligible)
    object_degree = Counter(r['object_id'] for r in eligible)
    for r in eligible:
        r['reciprocal_unique_candidate'] = target_degree[r['row']] == 1 and object_degree[r['object_id']] == 1
        if r['reciprocal_unique_candidate']:
            r['evidence_tier'] = 'geometry_one_to_one_candidate'
    unique = [r for r in eligible if r['reciprocal_unique_candidate']]
    unique_rows = {r['row'] for r in unique}
    for r in targets:
        if r['row'] in unique_rows:
            r['status'] = 'geometry_one_to_one_candidate'
    summaries = []
    for product in sorted(by_product):
        subset = [r for r in targets if r['product'] == product]
        candidates = [r for r in unique if r['product'] == product]
        summaries.append(dict(product=product, scene_suffix=product[-4:], inventory_objects=len(by_product[product]),
            cohort_targets=len(subset), cohort_distinct_mmsi=len({r['mmsi'] for r in subset}),
            exclusive_targets=sum(r['product_ownership_unique'] for r in subset),
            one_to_one_candidates=len(candidates), candidate_distinct_mmsi=len({r['mmsi'] for r in candidates}),
            candidate_classes_json=json.dumps(dict(Counter(r['class_raw'] for r in candidates))),
            candidates_with_external_neighbors=sum(r['neighbors_outside_chip_1km'] > 0 for r in candidates),
            candidates_with_complete_1km=sum(r['context_1km_complete'] for r in candidates), confirmed_training_targets=0))
    summary = dict(cohort_rows=len(cohort), distinct_mmsi=len({r['mmsi'] for r in cohort}),
        geometry_valid=240, retained_objects=len(objects), five_scene_record_products=sum(r['product'] in by_product for r in targets),
        inherited_neighbor_edges=len(old_edges),
        status_counts=dict(Counter(r['status'] for r in targets)), candidate_edges=len(eligible),
        one_to_one_candidates=len(unique), candidate_distinct_mmsi=len({r['mmsi'] for r in unique}),
        confirmed_training_targets=0, iou_threshold=IOU_THRESHOLD, neighbor_radius_m=RADIUS_M, scenes=summaries,
        evidence_sources=[str(BRIDGE/'cohort.csv'), str(geometry_path), str(expanded_path), str(BRIDGE/'detections.json'),
            str(ROOT/'land_mask/objects_retained.json'), str(ROOT/'recovery_31cc_2ec7/five_scene_object_inputs.csv')],
        limitations=['Inherited pixel replay is crop-to-mosaic, not an independently verified vessel label.',
          'Record product may be outside these five scenes; overlapping mosaic ownership stays unresolved.',
          'Box IoU >= 0.25 plus reciprocal uniqueness yields a candidate only; no nearest-point or MMSI assignment.',
          '224-coordinate detection boxes are geometrically mapped to native pixels; no resized image is used as native input.',
          'Neighbor counts are retained local detections, not verified vessels or full-scene density.',
          'The 3604 inherited edges are inventoried; 1km external-chip neighbors are recomputed from retained object geometry.',
          'Coverage is union of existing VV detection windows; it is not detector recall or dual-pol completeness.',
          'No native pixel equality between newly linked object and target is asserted. No new visual confirmation.'])
    assert len(targets) == 269 and all(not r['training_truth_verified'] for r in links)
    assert len(unique_rows) == len(unique) == len({r['object_id'] for r in unique})
    write_csv('target_link_candidates.csv', links)
    write_csv('target_status.csv', targets)
    write_csv('scene_summary.csv', summaries)
    class_rows = []
    for product in sorted(by_product):
        for label in sorted({r['class_raw'] for r in cohort}):
            queue = [r for r in targets if r['product'] == product and r['class_raw'] == label]
            candidates = [r for r in unique if r['product'] == product and r['class_raw'] == label]
            class_rows.append(dict(product=product, class_raw=label, queue_targets=len(queue),
                distinct_queue_mmsi=len({r['mmsi'] for r in queue}),
                exclusive_targets=sum(r['product_ownership_unique'] for r in queue),
                candidates=len(candidates), distinct_candidate_mmsi=len({r['mmsi'] for r in candidates}),
                with_external_neighbors=sum(r['neighbors_outside_chip_1km'] > 0 for r in candidates)))
    write_csv('scene_class_coverage.csv', class_rows)
    (HERE/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ['# F0921 target-to-scene linkage preparation', '',
        'All new links remain candidates. Confirmed training targets: 0.', '',
        'Run: `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -B build_linkage.py`', '',
        'Inputs are read-only; raster access reads metadata only. Outputs are confined to this script directory.', '',
        '| Scene | Objects | Queue targets | Unique candidates | MMSI | With external neighbors | Complete 1km |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        report.append('| {scene_suffix} | {inventory_objects} | {cohort_targets} | {one_to_one_candidates} | {candidate_distinct_mmsi} | {candidates_with_external_neighbors} | {candidates_with_complete_1km} |'.format(**r))
    report += ['', '## Target disposition', '', '```json', json.dumps(summary['status_counts'], indent=2), '```', '',
        '## Evidence and limits', ''] + ['- '+s for s in summary['limitations']]
    report += ['', 'Next: inspect native VV/VH target and object windows for the candidate list, preserving competing boxes and source identity uncertainty. Freeze evaluation inputs only after that evidence supports the specific target association.',
        '', 'No training, download, shared-file edits, credential access, or Git synchronization was performed. The current directory-only scope takes precedence over the standing archive workflow.']
    (HERE/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['scenes','evidence_sources','limitations']}, indent=2))


if __name__ == '__main__':
    main()
