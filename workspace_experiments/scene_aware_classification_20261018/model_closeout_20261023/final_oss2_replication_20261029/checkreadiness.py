"""Small metadata/index/path readiness checks. Does not import torch or numpy."""
from collections import Counter
from itertools import combinations
import sys

from protocol import (HERE, SOURCE, INIT, MODEL, CLASS_INDICES, CLASSES, COUNTS,
                      ROLES, BUDGET, cohorts, read, write, assert_rows, class_weights4)
from converter import csv_rows, flag, corners


def check():
    rows = cohorts()
    records = read(HERE/'manifests/all.json')
    table = csv_rows(SOURCE/'stack_table.csv')
    indexes = read(SOURCE/'stack_indices.json')
    assert len(records) == len(table) == 1519
    assert len({r['id'] for r in records}) == 1519
    for i, (record, original) in enumerate(zip(records, table)):
        assert record['id'] == original['sample_id']
        assert record['indices']['table_index'] == i
        for key, value in record['indices'].items():
            assert value == int(original[key]), (record['id'], key)
        assert record['label'] == int(original['class_id'])
        assert record['original_class_id'] == int(original['original_class_id'])
        assert record['class_name'] == CLASSES[record['label']] == original['class_name']
        assert record['mmsi'] == original['mmsi'] and record['pass_group'] == original['pass_group']
        assert record['scene_id'] == original['scene_id'] and record['role'] == original['role']
        assert record['split'] == ROLES[record['role']]
        for pol in ('vv', 'vh'):
            view = record[pol]
            assert (view is not None) == flag(original['valid_'+pol])
            if view:
                assert view['target_index'] == 0 and view['radar_axis_rad'] is None
                assert view['detections'][0]['confidence'] >= .25
                assert view['detections'][0]['corners'] == corners(view['raw_xywhr'])
        assert record['vv'] or record['vh']
    expected_pols = {'source_fit': (444, 11, 23), 'validation': (206, 6, 16),
                     'source_adapt': (227, 5, 9), 'selection': (177, 5, 12),
                     'evaluation': (348, 12, 18)}
    summary = {}
    for role, split in ROLES.items():
        group = rows[split]
        assert_rows(group, split)
        assert len(group) == COUNTS[split]
        assert group == [records[i] for i in indexes[role]]
        pols = Counter((r['vv'] is not None, r['vh'] is not None) for r in group)
        assert tuple(pols[key] for key in ((True, True), (True, False), (False, True))) == expected_pols[split]
        classes = Counter(r['label'] for r in group)
        assert set(classes) == set(range(4))
        summary[split] = dict(count=len(group), classes=dict(classes),
            both=pols[(True, True)], vv_only=pols[(True, False)], vh_only=pols[(False, True)])
    for a, b in combinations(COUNTS, 2):
        for key in ('id', 'mmsi', 'pass_group'):
            assert not ({r[key] for r in rows[a]} & {r[key] for r in rows[b]}), (a, b, key)
    refs = rows['references']
    ref_source = csv_rows(SOURCE/'m3_results/reference_bank.csv')
    assert len(refs) == len({r['mmsi'] for r in refs}) == 32
    assert Counter(r['label'] for r in refs) == dict.fromkeys(range(4), 8)
    for r, original in zip(refs, ref_source):
        assert r == records[r['indices']['table_index']] and r['split'] == 'source_fit'
        assert r['indices']['obs_id'] == int(original['obs_id']) and r['mmsi'] == original['mmsi']
        assert r['label'] == int(original['class_id'])
    for stage, split in (('fit', 'source_fit'), ('adapt', 'source_adapt')):
        weights = class_weights4(rows[split], stage)
        assert abs(sum(weights)-len(rows[split])) < 1e-8
        epochs, batch_size, steps = BUDGET[stage]
        assert epochs*((len(weights)+batch_size-1)//batch_size) == steps
    # A small existence sample detects broken root mappings without re-reading the dataset.
    selected = {g[0]['id']: g[0] for g in rows.values()}
    for pol in ('vv', 'vh'):
        r = next(r for r in records if r[pol] is None)
        selected[r['id']] = r
    sampled_paths = []
    for r in selected.values():
        for pol in ('vv', 'vh'):
            if r[pol]:
                from pathlib import Path
                path = Path(r[pol]['image'])
                assert path.is_file(), path
                sampled_paths.append(str(path))
    assert INIT.is_file() and MODEL.is_file()
    return dict(status='passed_cpu_metadata_only', counts=summary, references=32,
        class_indices=list(CLASS_INDICES), sampled_existing_paths=sampled_paths,
        raw_images_opened=0, input_tensors_loaded=False, torch_imported='torch' in sys.modules,
        gpu_cache='NOT_EXECUTED', gpu_checks='NOT_EXECUTED', training='NOT_EXECUTED')


if __name__ == '__main__':
    result = check()
    assert not result['torch_imported']
    write(HERE/'cpu_readiness.json', result)
    print(result['status'], {k: v['count'] for k, v in result['counts'].items()}, 'refs=32')
