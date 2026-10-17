"""Fill all product-grid gaps among radar-invalid rows in the fixed cohort."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from recover_product_annotations import (
    HERE, ROOT, read_csv, recover_product, write_csv, write_json,
)

OLD_GRID = Path('E:/Hermes/paper_tracking/sensitivity/relative_observation_geometry_20261009/radar_grid')


def main():
    geometry = read_csv(ROOT / 'geometry_targeted_fix_20261016/recovered_geometry.csv')
    source = read_csv(ROOT / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
    fixed = set()
    for path in (ROOT / 'geometry_relation_slots_20261016/slots_cache').glob('*.npz'):
        with np.load(path) as cache:
            fixed.update(map(int, cache['rows']))
    assert len(fixed) == 8622 and len(geometry) == len(source) == 38091
    targets = []
    for index in sorted(fixed):
        row = geometry[index]
        assert int(row['global_row']) == index and row['instance_id'] == source[index]['instance_id']
        if int(row['radar_valid']) == 0:
            targets.append(dict(global_row=index, instance_id=row['instance_id'],
                                product=source[index]['product'],
                                original_status=row['radar_recovery_status']))
    counts = Counter(row['product'] for row in targets)
    old_success = set()
    for product in counts:
        path = OLD_GRID / (product + '.json')
        if path.exists():
            grid = json.loads(path.read_text(encoding='utf-8'))
            if grid.get('status') == 'ok' and grid.get('product') == product:
                old_success.add(product)
    manifest_path = HERE / 'manifest.csv'
    existing = read_csv(manifest_path)
    existing_success = {r['product'] for r in existing if r['status'].startswith('recovered')}
    pending = sorted(set(counts) - old_success - existing_success)
    results = []
    for index, product in enumerate(pending, 1):
        result = recover_product(product, counts[product])
        results.append(result)
        if result['status'] == 'failed' or index % 20 == 0 or index == len(pending):
            print(json.dumps(dict(completed=index, total=len(pending), product=product,
                                  status=result['status'], reason=result['reason'])), flush=True)
    extended_path = HERE / 'extended_products.csv'
    extended = {r['product']: r for r in read_csv(extended_path)} if extended_path.exists() else {}
    extended.update({r['product']: r for r in results})
    if extended:
        write_csv(extended_path, list(extended.values()))
        write_csv(HERE / 'extended_target_rows.csv', [r for r in targets if r['product'] in extended])

    # Publish only after all referenced XML and JSON files have been finalized.
    combined = {r['product']: r for r in existing}
    combined.update({r['product']: r for r in results})
    write_csv(manifest_path, list(combined.values()))
    loaded = read_csv(manifest_path)
    loaded_by_product = {r['product']: r for r in loaded}
    assert len(loaded_by_product) == len(loaded)
    for previous in existing:
        if previous['status'].startswith('recovered'):
            assert loaded_by_product[previous['product']] == previous

    # Check the real downstream grid contract before reporting usable products.
    sys.dont_write_bytecode = True
    sys.path.insert(0, 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check')
    from relative_radar_geometry import _validate
    success = set()
    for row in loaded:
        if not row['status'].startswith('recovered'):
            continue
        grid = json.loads(Path(row['local_grid_path']).read_text(encoding='utf-8'))
        assert grid['status'] == 'ok' and grid['product'] == row['product']
        _validate(grid['points'])
        assert len(grid['points']) == int(row['grid_points'])
        success.add(row['product'])
    unresolved = set(counts) - old_success - success
    summary = dict(fixed_cohort_rows=len(fixed), missing_radar_rows=sum(counts.values()),
                   missing_radar_products=len(counts), excluded_old_success_products=len(old_success),
                   preexisting_success_products=len(existing_success),
                   newly_attempted_products=len(results),
                   newly_recovered_products=sum(r['status'].startswith('recovered') for r in results),
                   extended_products=len(extended),
                   extended_target_rows=sum(counts[p] for p in extended),
                   manifest_products=len(loaded), successful_manifest_products=len(success),
                   covered_missing_radar_rows=sum(counts[p] for p in counts if p in success or p in old_success),
                   unresolved_products=sorted(unresolved),
                   new_local_annotation_bytes=sum(int(r['annotation_bytes']) for r in results),
                   network_requests=0, network_bytes_downloaded=0,
                   measurement_bytes_downloaded=0, original_tables_modified=False,
                   georef_or_pixel_angles_computed=False, prior_manifest_rows_preserved=True,
                   manifest_publish_method='same-directory temporary file then atomic replace')
    write_json(HERE / 'extended_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if unresolved else 0


if __name__ == '__main__':
    sys.exit(main())
